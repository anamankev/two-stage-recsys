"""Two-tower retrieval model (PyTorch, CPU).

User tower: user-id embedding, mean of the embeddings of the user's last ``history_length``
positives (only those *before* the example's target), and gender/age/occupation embeddings,
fed through an MLP. Item tower: item-id embedding, genre multi-hot projection and release-year
bucket embedding, fed through an MLP. Both outputs are L2-normalised; the score is their dot
product divided by a (learned) temperature.

Loss: softmax over in-batch items (each other example's positive is a negative) with
* **log-Q correction** (Yi et al., RecSys 2019): popular items appear as in-batch negatives more
  often, which biases the model against them; subtracting ``log q_j`` (item ``j``'s sampling
  probability) from its logit corrects the bias;
* **accidental-hit masking**: an in-batch item that is also a training positive of the row's user
  is not treated as a negative.

``negatives`` selects the variant used in the ablations: ``in_batch`` (no correction),
``in_batch_logq`` (default), ``uniform`` (items sampled uniformly from the catalog) and
``mixed`` (in-batch plus uniform, corrected by each item's expected number of draws).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F

from .data import AGE_BUCKETS, N_OCCUPATIONS, MovieLens, Split, Stage
from .evaluation import RankingResult, rank_targets
from .features import year_bucket

I64 = NDArray[np.int64]
NEGATIVE_MODES = ("in_batch", "in_batch_logq", "uniform", "mixed")


@dataclass(frozen=True)
class TwoTowerConfig:
    """Model and training hyperparameters."""

    embedding_dim: int = 64
    hidden_dim: int = 128
    history_length: int = 50
    use_side_features: bool = True
    negatives: str = "in_batch_logq"
    n_uniform_negatives: int = 1024
    batch_size: int = 1024
    max_epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    temperature: float = 0.05
    learn_temperature: bool = True
    patience: int = 3
    eval_k: int = 100
    seed: int = 0
    num_threads: int = 4
    max_minutes: float = 20.0

    def __post_init__(self) -> None:
        if self.negatives not in NEGATIVE_MODES:
            raise ValueError(f"negatives must be one of {NEGATIVE_MODES}")
        if min(self.embedding_dim, self.hidden_dim, self.history_length, self.batch_size) < 1:
            raise ValueError("sizes must be positive")
        if not 0 < self.temperature <= 1:
            raise ValueError("temperature must be in (0, 1]")


@dataclass
class SideInfo:
    """Static user and item features as tensors."""

    gender: torch.Tensor
    age: torch.Tensor
    occupation: torch.Tensor
    genres: torch.Tensor
    year: torch.Tensor
    n_year_buckets: int = field(default=0)

    @classmethod
    def from_movielens(cls, ml: MovieLens) -> SideInfo:
        """Extract side information from the parsed dataset."""
        years = year_bucket(ml.years)
        return cls(  # torch.tensor copies: the parsed arrays may be read-only views
            gender=torch.tensor(ml.gender),
            age=torch.tensor(ml.age),
            occupation=torch.tensor(ml.occupation),
            genres=torch.tensor(ml.genres),
            year=torch.tensor(years),
            n_year_buckets=int(years.max()) + 1,
        )


class UserTower(nn.Module):
    """User id + mean history embedding (+ demographics) -> MLP -> unit vector."""

    def __init__(self, n_users: int, n_items: int, config: TwoTowerConfig) -> None:
        super().__init__()
        d = config.embedding_dim
        self.side = config.use_side_features
        self.padding = n_items
        self.user = nn.Embedding(n_users, d)
        self.history = nn.Embedding(n_items + 1, d, padding_idx=n_items)
        extra = 0
        if self.side:
            self.gender = nn.Embedding(2, 8)
            self.age = nn.Embedding(len(AGE_BUCKETS), 8)
            self.occupation = nn.Embedding(N_OCCUPATIONS, 8)
            extra = 24
        self.mlp = nn.Sequential(
            nn.Linear(2 * d + extra, config.hidden_dim), nn.ReLU(), nn.Linear(config.hidden_dim, d)
        )
        for emb in (self.user, self.history):
            nn.init.normal_(emb.weight, std=0.05)

    def forward(self, users: torch.Tensor, history: torch.Tensor, side: SideInfo) -> torch.Tensor:
        """``users``: (B,); ``history``: (B, L) item ids padded with ``n_items``."""
        mask = (history != self.padding).unsqueeze(-1).float()
        pooled = (self.history(history) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        parts = [self.user(users), pooled]
        if self.side:
            parts += [
                self.gender(side.gender[users]),
                self.age(side.age[users]),
                self.occupation(side.occupation[users]),
            ]
        return F.normalize(self.mlp(torch.cat(parts, dim=1)), dim=1)


class ItemTower(nn.Module):
    """Item id (+ genres, year bucket) -> MLP -> unit vector."""

    def __init__(self, n_items: int, n_genres: int, n_years: int, config: TwoTowerConfig) -> None:
        super().__init__()
        d = config.embedding_dim
        self.side = config.use_side_features
        self.item = nn.Embedding(n_items, d)
        nn.init.normal_(self.item.weight, std=0.05)
        extra = 0
        if self.side:
            self.genre = nn.Linear(n_genres, 16)
            self.year = nn.Embedding(n_years, 8)
            extra = 24
        self.mlp = nn.Sequential(
            nn.Linear(d + extra, config.hidden_dim), nn.ReLU(), nn.Linear(config.hidden_dim, d)
        )

    def forward(self, items: torch.Tensor, side: SideInfo) -> torch.Tensor:
        """``items``: (B,) item ids."""
        parts = [self.item(items)]
        if self.side:
            parts += [self.genre(side.genres[items]), self.year(side.year[items])]
        return F.normalize(self.mlp(torch.cat(parts, dim=1)), dim=1)


class TwoTower(nn.Module):
    """Both towers plus the (optionally learned) softmax temperature."""

    def __init__(self, n_users: int, n_items: int, side: SideInfo, config: TwoTowerConfig) -> None:
        super().__init__()
        self.config = config
        self.side = side
        self.user_tower = UserTower(n_users, n_items, config)
        self.item_tower = ItemTower(n_items, side.genres.shape[1], side.n_year_buckets, config)
        log_t = torch.tensor(float(np.log(config.temperature)))
        if config.learn_temperature:
            self.log_temperature = nn.Parameter(log_t)
        else:
            self.register_buffer("log_temperature", log_t)

    def temperature(self) -> torch.Tensor:
        """Current temperature, kept in [0.01, 1] for numerical stability."""
        return self.log_temperature.exp().clamp(0.01, 1.0)


def padded_histories(indptr: I64, items: I64, users: I64, ends: I64, length: int, pad: int) -> I64:
    """For each (user, end position) the last ``length`` items before ``end``, left-aligned.

    ``ends`` are positions within each user's sequence (0 = nothing before), so a training example
    at position k only sees the user's first k positives.
    """
    stop = indptr[users] + ends
    start = np.maximum(indptr[users], stop - length)
    offsets = start[:, None] + np.arange(length)[None, :]
    valid = offsets < stop[:, None]
    gathered = items[np.minimum(offsets, max(items.shape[0] - 1, 0))] if items.size else offsets
    return np.where(valid, gathered, pad).astype(np.int64)


class Trainer:
    """Trains a :class:`TwoTower` model with early stopping on validation Recall@k."""

    def __init__(self, split: Split, ml: MovieLens, config: TwoTowerConfig) -> None:
        if config.use_side_features and not ml.demographics:
            raise ValueError(
                "use_side_features needs user demographics, which this dataset does not have"
            )
        self.split = split
        self.config = config
        self.side = SideInfo.from_movielens(ml)
        torch.manual_seed(config.seed)
        torch.set_num_threads(config.num_threads)
        self.rng = np.random.default_rng(config.seed)
        self.model = TwoTower(split.n_users, split.n_items, self.side, config)
        counts = np.diff(split.train_indptr)
        self.ex_user = np.repeat(np.arange(split.n_users), counts).astype(np.int64)
        self.ex_pos = np.arange(split.train_items.shape[0]) - split.train_indptr[self.ex_user]
        self.ex_item = split.train_items
        freq = np.bincount(self.ex_item, minlength=split.n_items).astype(np.float64)
        self.q = torch.from_numpy(np.maximum(freq / freq.sum(), 1e-12)).float()
        self.positives = torch.zeros((split.n_users, split.n_items), dtype=torch.bool)
        self.positives[torch.from_numpy(self.ex_user), torch.from_numpy(self.ex_item)] = True
        self.log: list[dict[str, Any]] = []

    def _loss(self, users: I64, ends: I64, items: I64) -> torch.Tensor:
        cfg, split = self.config, self.split
        history = padded_histories(
            split.train_indptr, split.train_items, users, ends, cfg.history_length, split.n_items
        )
        u = self.model.user_tower(torch.from_numpy(users), torch.from_numpy(history), self.side)
        user_t = torch.from_numpy(users)
        pos_items = torch.from_numpy(items)
        b = users.shape[0]
        columns: list[torch.Tensor] = []  # item ids of each logit column
        if cfg.negatives in ("in_batch", "in_batch_logq", "mixed"):
            columns.append(pos_items)
        if cfg.negatives in ("uniform", "mixed"):
            columns.append(
                torch.from_numpy(self.rng.integers(0, split.n_items, cfg.n_uniform_negatives))
            )
        col_items = torch.cat(columns)
        v = self.model.item_tower(col_items, self.side)
        logits = u @ v.T / self.model.temperature()
        if cfg.negatives == "uniform":
            pos = (u * self.model.item_tower(pos_items, self.side)).sum(1, keepdim=True)
            logits = torch.cat([pos / self.model.temperature(), logits], dim=1)
            col_items = torch.cat([pos_items[:1].new_full((1,), -1), col_items])
        elif cfg.negatives == "in_batch_logq":
            logits = logits - self.q[col_items].log()[None, :]
        elif cfg.negatives == "mixed":
            expected = b * self.q[col_items] + cfg.n_uniform_negatives / split.n_items
            logits = logits - expected.log()[None, :]
        # Accidental hits: a column item that is a training positive of the row's user (other
        # than the row's own target column) must not be pushed down.
        target_col = (
            torch.arange(b) if cfg.negatives != "uniform" else torch.zeros(b, dtype=torch.long)
        )
        if cfg.negatives == "uniform":
            hits = torch.zeros_like(logits, dtype=torch.bool)
            hits[:, 1:] = self.positives[user_t[:, None], col_items[None, 1:]]
        else:
            hits = self.positives[user_t[:, None], col_items[None, :]]
            hits[torch.arange(b), target_col] = False
        logits = logits.masked_fill(hits, float("-inf"))
        return F.cross_entropy(logits, target_col)

    def fit(self, *, verbose: bool = True) -> TwoTower:
        """Train until validation Recall@eval_k stops improving; returns the best model."""
        cfg = self.config
        optimiser = torch.optim.Adam(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        best_score, best_state, stale = -1.0, copy.deepcopy(self.model.state_dict()), 0
        started = time.perf_counter()
        n = self.ex_item.shape[0]
        for epoch in range(1, cfg.max_epochs + 1):
            self.model.train()
            order = self.rng.permutation(n)
            total, steps = 0.0, 0
            for start in range(0, n, cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                loss = self._loss(self.ex_user[idx], self.ex_pos[idx], self.ex_item[idx])
                optimiser.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimiser.step()
                total += loss.item()
                steps += 1
            val = evaluate(self.model, self.split, "val", k_max=cfg.eval_k).metrics_at(cfg.eval_k)
            entry = {
                "epoch": epoch,
                "loss": total / steps,
                "seconds": time.perf_counter() - started,
                "temperature": self.model.temperature().item(),
                **val,
            }
            self.log.append(entry)
            if verbose:
                print(
                    {k: round(v, 4) if isinstance(v, float) else v for k, v in entry.items()},
                    flush=True,
                )
            if val[f"recall@{cfg.eval_k}"] > best_score:
                best_score, best_state, stale = (
                    val[f"recall@{cfg.eval_k}"],
                    copy.deepcopy(self.model.state_dict()),
                    0,
                )
            else:
                stale += 1
            if stale >= cfg.patience or entry["seconds"] > 60 * cfg.max_minutes:
                break
        self.model.load_state_dict(best_state)
        self.model.eval()
        return self.model


@dataclass(frozen=True)
class Evaluation:
    """Ranking of one stage plus the embeddings that produced it."""

    result: RankingResult
    user_vectors: NDArray[np.float32]
    item_vectors: NDArray[np.float32]

    def metrics_at(self, k: int) -> dict[str, float]:
        """Recall@k and NDCG@10 for early stopping and logging."""
        return {
            f"recall@{k}": float((self.result.ranks <= k).mean()),
            "ndcg@10": self.result.metrics()["ndcg@10"],
        }


@torch.no_grad()
def embed(
    model: TwoTower, split: Split, stage: Stage
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """User vectors for every user (history before the ``stage`` target) and all item vectors."""
    model.eval()
    indptr, items = split.history_csr(stage)
    users = np.arange(split.n_users, dtype=np.int64)
    history = padded_histories(
        indptr, items, users, np.diff(indptr), model.config.history_length, split.n_items
    )
    u = model.user_tower(torch.from_numpy(users), torch.from_numpy(history), model.side).numpy()
    v = model.item_tower(torch.arange(split.n_items), model.side).numpy()
    return u.astype(np.float32), v.astype(np.float32)


def evaluate(model: TwoTower, split: Split, stage: Stage, *, k_max: int = 100) -> Evaluation:
    """Exact full-catalog ranking by dot product of the tower outputs."""
    u, v = embed(model, split, stage)
    targets, _ = split.targets(stage)
    indptr, items = split.history_csr(stage)
    result = rank_targets(
        lambda users: u[users] @ v.T,
        split.users,
        targets,
        indptr,
        items,
        split.n_items,
        k_max=k_max,
    )
    return Evaluation(result, u, v)
