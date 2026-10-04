"""Full-catalog ranking evaluation with bootstrap confidence intervals.

Every method produces a score for every (user, catalog item) pair, or an explicit ranking. Items
in the user's history are excluded. With one held-out target per user:

* ``Recall@K`` = 1 if the target is in the top K (the hit rate).
* ``NDCG@K`` = ``1 / log2(rank + 1)`` if ``rank <= K`` else 0 (the ideal DCG is 1).
* ``MRR`` = ``1 / rank`` over the full ranking (no cut-off).
* ``coverage@K`` = share of the catalog appearing in at least one user's top K.

Ranks are 1-based. Ties in scores are broken by ascending item id, the same rule used to build
top-K lists, so metrics always describe the lists a user would actually see.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

I64 = NDArray[np.int64]
F64 = NDArray[np.float64]
ScoreFn = Callable[[I64], NDArray[np.floating[Any]]]


@dataclass(frozen=True)
class RankingResult:
    """Per-user outcomes of one method on one stage."""

    ranks: I64  # 1-based rank of each user's target
    top_k: I64  # (n_users, k_max) recommended item ids
    n_items: int

    def metrics(self) -> dict[str, float]:
        """Point estimates of all metrics."""
        return summarize(self.ranks, self.top_k, self.n_items)


@dataclass(frozen=True)
class Interval:
    """A point estimate with a 95% bootstrap confidence interval."""

    value: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        """JSON-friendly form."""
        return {"value": self.value, "low": self.low, "high": self.high}


@dataclass(frozen=True)
class BootstrapReport:
    """Bootstrap intervals for every metric of one method."""

    intervals: dict[str, Interval] = field(default_factory=dict)

    def as_dict(self) -> dict[str, dict[str, float]]:
        """JSON-friendly form."""
        return {name: iv.as_dict() for name, iv in self.intervals.items()}


def recall_at_k(ranks: I64, k: int) -> F64:
    """Per-user hit indicator ``rank <= k``."""
    return (ranks <= k).astype(np.float64)


def ndcg_at_k(ranks: I64, k: int) -> F64:
    """Per-user NDCG@k for a single relevant item."""
    out = np.zeros(ranks.shape[0], dtype=np.float64)
    hit = ranks <= k
    out[hit] = 1.0 / np.log2(ranks[hit] + 1.0)
    return out


def reciprocal_rank(ranks: I64) -> F64:
    """Per-user ``1 / rank``."""
    return 1.0 / ranks.astype(np.float64)


def coverage(top_k: I64, n_items: int, k: int) -> float:
    """Fraction of the catalog that appears in at least one user's top ``k``."""
    return float(np.unique(top_k[:, :k]).shape[0] / n_items)


def summarize(ranks: I64, top_k: I64, n_items: int) -> dict[str, float]:
    """All headline metrics as point estimates."""
    return {
        "recall@10": float(recall_at_k(ranks, 10).mean()),
        "recall@50": float(recall_at_k(ranks, 50).mean()),
        "ndcg@10": float(ndcg_at_k(ranks, 10).mean()),
        "mrr": float(reciprocal_rank(ranks).mean()),
        "coverage@10": coverage(top_k, n_items, 10),
    }


def rank_targets(
    score_fn: ScoreFn,
    users: I64,
    targets: I64,
    history_indptr: I64,
    history_items: I64,
    n_items: int,
    *,
    k_max: int = 50,
    batch_size: int = 512,
) -> RankingResult:
    """Rank every target against the full catalog, excluding each user's history.

    Args:
        score_fn: maps a batch of user ids to a ``(batch, n_items)`` score matrix (higher is
            better).
        users: evaluated user ids.
        targets: held-out item for each user.
        history_indptr, history_items: CSR of each user's history (excluded from ranking).
        n_items: catalog size.
        k_max: length of the stored top-K lists.
        batch_size: users scored per call.
    """
    ranks = np.empty(users.shape[0], dtype=np.int64)
    top_k = np.empty((users.shape[0], k_max), dtype=np.int64)
    item_ids = np.arange(n_items)
    for start in range(0, users.shape[0], batch_size):
        batch = users[start : start + batch_size]
        scores = np.asarray(score_fn(batch), dtype=np.float64).copy()
        if scores.shape != (batch.shape[0], n_items):
            raise ValueError(f"score_fn returned shape {scores.shape}")
        for row, user in enumerate(batch):
            history = history_items[history_indptr[user] : history_indptr[user + 1]]
            scores[row, history] = -np.inf
        for row in range(batch.shape[0]):
            target = targets[start + row]
            s = scores[row]
            st = s[target]
            if not np.isfinite(st):
                raise ValueError(f"target {target} of user {batch[row]} is in its history")
            better = np.count_nonzero(s > st) + np.count_nonzero((s == st) & (item_ids < target))
            ranks[start + row] = better + 1
            top_k[start + row] = top_k_ids(s, k_max)
    return RankingResult(ranks, top_k, n_items)


def top_k_ids(scores: NDArray[np.floating[Any]], k: int) -> I64:
    """Ids of the ``k`` highest scores, ties broken by ascending id."""
    k = min(k, scores.shape[0])
    candidates = np.argpartition(-scores, k - 1)[:k]
    kth = scores[candidates].min()
    pool = np.flatnonzero(scores >= kth)  # include every item tied with the k-th score
    order = np.lexsort((pool, -scores[pool]))
    return pool[order[:k]].astype(np.int64)


def rank_with_candidates(
    candidates: I64,
    fallback_fn: ScoreFn,
    users: I64,
    targets: I64,
    history_indptr: I64,
    history_items: I64,
    n_items: int,
    *,
    k_max: int = 50,
) -> RankingResult:
    """Ranks for a two-stage system: the ordered candidate list, then everything else.

    A target inside the candidate list gets its position there. Otherwise its rank is the list
    length plus its rank among the remaining (non-history, non-candidate) items under
    ``fallback_fn`` scores (the retrieval model's exact scores), so MRR stays defined over the full
    catalog. ``top_k`` is the first ``k_max`` candidates.
    """
    n_cand = candidates.shape[1]
    ranks = np.empty(users.shape[0], dtype=np.int64)
    item_ids = np.arange(n_items)
    for row, user in enumerate(users):
        target = targets[row]
        position = np.flatnonzero(candidates[row] == target)
        if position.size:
            ranks[row] = position[0] + 1
            continue
        s = np.asarray(fallback_fn(np.array([user]))[0], dtype=np.float64).copy()
        s[history_items[history_indptr[user] : history_indptr[user + 1]]] = -np.inf
        s[candidates[row][candidates[row] >= 0]] = -np.inf
        st = s[target]
        if not np.isfinite(st):
            raise ValueError(f"target {target} of user {user} is in its history")
        better = np.count_nonzero(s > st) + np.count_nonzero((s == st) & (item_ids < target))
        ranks[row] = n_cand + better + 1
    return RankingResult(ranks, candidates[:, :k_max].copy(), n_items)


def ranks_from_lists(ranked: I64, targets: I64) -> I64:
    """1-based position of each target in explicit full rankings (``n_users x n_items``)."""
    positions = np.argmax(ranked == targets[:, None], axis=1)
    if not (ranked[np.arange(ranked.shape[0]), positions] == targets).all():
        raise ValueError("some targets are missing from their rankings")
    return positions.astype(np.int64) + 1


def bootstrap(
    result: RankingResult, *, n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05
) -> BootstrapReport:
    """95% percentile bootstrap intervals over users for the per-user mean metrics.

    Coverage is deliberately excluded: it counts distinct items across *all* users' lists, and
    resampling users with replacement repeats about 37% of them, which mechanically shrinks the
    set of distinct items. Its bootstrap distribution is therefore biased low (intervals can miss
    the point estimate entirely), so coverage is reported as a point value only.
    """
    rng = np.random.default_rng(seed)
    n = result.ranks.shape[0]
    per_user = {
        "recall@10": recall_at_k(result.ranks, 10),
        "recall@50": recall_at_k(result.ranks, 50),
        "ndcg@10": ndcg_at_k(result.ranks, 10),
        "mrr": reciprocal_rank(result.ranks),
    }
    samples: dict[str, list[float]] = {name: [] for name in per_user}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        for name, values in per_user.items():
            samples[name].append(float(values[idx].mean()))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return BootstrapReport(
        {
            name: Interval(
                float(per_user[name].mean()),
                float(np.percentile(s, lo)),
                float(np.percentile(s, hi)),
            )
            for name, s in samples.items()
        }
    )


def paired_difference(
    a: RankingResult,
    b: RankingResult,
    metric: str = "ndcg@10",
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> Interval:
    """Bootstrap interval of ``metric(a) - metric(b)`` resampling the same users for both."""
    per_user_fn: dict[str, Callable[[I64], F64]] = {
        "recall@10": lambda r: recall_at_k(r, 10),
        "recall@50": lambda r: recall_at_k(r, 50),
        "ndcg@10": lambda r: ndcg_at_k(r, 10),
        "mrr": reciprocal_rank,
    }
    if a.ranks.shape != b.ranks.shape:
        raise ValueError("paired comparison needs the same users in the same order")
    diff = per_user_fn[metric](a.ranks) - per_user_fn[metric](b.ranks)
    rng = np.random.default_rng(seed)
    n = diff.shape[0]
    means = [float(diff[rng.integers(0, n, size=n)].mean()) for _ in range(n_resamples)]
    return Interval(
        float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    )
