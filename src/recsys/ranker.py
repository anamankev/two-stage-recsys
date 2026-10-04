"""Second stage: LightGBM LambdaRank over the retrieved candidates.

Training rows are the validation-stage candidates of each user (label 1 for the validation
target), evaluation rows are the test-stage candidates. Every feature of a (user, candidate) row
uses only events before that stage's target time:

==================  ==================================================================
feature             definition
==================  ==================================================================
tt_score            two-tower dot product (user vector built from the history)
tt_rank             position in the retrieved candidate list (0 = best)
pop_before          training positives of the item strictly before the target time
pop_recent_30d      the same, restricted to the 30 days before the target time
year                release year of the movie
genre_affinity      cosine(user's history genre profile, movie genres)
user_activity       number of positives in the user's history
gender, age,        user demographics (categorical)
occupation
==================  ==================================================================

Optional extra features (selected through ``RankerConfig.features``):

==================  ==================================================================
sim_last1           cosine(candidate item vector, vector of the user's latest history item)
sim_recent5_mean    mean cosine to the user's 5 latest history items
sim_recent5_max     max cosine to the user's 5 latest history items
item_age_years      calendar year of the target time minus the movie's release year
==================  ==================================================================

The similarity features add *recency*, which the two-tower user vector (a mean over up to 50
history items) cannot express.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data import MovieLens, Split, Stage
from .features import SECONDS_PER_DAY, PopularityIndex, genre_affinity, genre_profiles_for

I64 = NDArray[np.int64]
F32 = NDArray[np.float32]

FEATURES = (
    "tt_score",
    "tt_rank",
    "pop_before",
    "pop_recent_30d",
    "year",
    "genre_affinity",
    "user_activity",
    "gender",
    "age",
    "occupation",
)
EXTRA_FEATURES = ("sim_last1", "sim_recent5_mean", "sim_recent5_max", "item_age_years")
ALL_FEATURES = FEATURES + EXTRA_FEATURES
CATEGORICAL = ("gender", "age", "occupation")
DEMOGRAPHIC_FEATURES = frozenset(CATEGORICAL)  # unavailable in MovieLens-10M
FEATURE_BLOCK_USERS = 4096  # users per compute_features call in build_features


@dataclass(frozen=True)
class RankerConfig:
    """LightGBM LambdaRank hyperparameters."""

    num_leaves: int = 31
    learning_rate: float = 0.05
    max_rounds: int = 1000
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    early_stopping_rounds: int = 50
    holdout_fraction: float = 0.1
    seed: int = 0
    num_threads: int = 1
    features: tuple[str, ...] = FEATURES

    def __post_init__(self) -> None:
        unknown = set(self.features) - set(ALL_FEATURES)
        if unknown:
            raise ValueError(f"unknown ranker features: {sorted(unknown)}")
        object.__setattr__(self, "features", tuple(self.features))


@dataclass(frozen=True)
class FeatureBlock:
    """Features for ``n_users x n_candidates`` rows, row-major by user."""

    matrix: F32  # (n_users * n_cand, n_features)
    labels: NDArray[np.int32]
    n_candidates: int
    feature_names: tuple[str, ...] = FEATURES

    def select(self, names: tuple[str, ...]) -> FeatureBlock:
        """A block restricted to ``names`` (in that order)."""
        idx = [self.feature_names.index(n) for n in names]
        return FeatureBlock(self.matrix[:, idx], self.labels, self.n_candidates, names)

    def group_sizes(self, n_users: int) -> I64:
        """LightGBM query sizes (one group per user)."""
        return np.full(n_users, self.n_candidates, dtype=np.int64)


def build_features(
    split: Split,
    ml: MovieLens,
    stage: Stage,
    users: I64,
    candidates: I64,
    user_vectors: F32,
    item_vectors: F32,
    popularity: PopularityIndex,
    features: tuple[str, ...] = FEATURES,
) -> FeatureBlock:
    """Feature matrix for every (user, candidate) pair of ``stage`` (offline training/eval).

    ``candidates`` must not contain padding (``-1``); targets and target times come from
    ``split``; histories are everything before the ``stage`` target.
    """
    targets = split.val_item[users] if stage == "val" else split.test_item[users]
    times = split.val_time[users] if stage == "val" else split.test_time[users]
    indptr, history = split.history_csr(stage)
    # Users in blocks: every row depends only on its own user, so the result is identical to one
    # call, without (users x candidates x dim) temporaries on large datasets.
    parts = [
        compute_features(
            ml,
            users[start : start + FEATURE_BLOCK_USERS],
            candidates[start : start + FEATURE_BLOCK_USERS],
            times[start : start + FEATURE_BLOCK_USERS],
            indptr,
            history,
            user_vectors[users[start : start + FEATURE_BLOCK_USERS]],
            item_vectors,
            popularity,
            features,
        )
        for start in range(0, users.shape[0], FEATURE_BLOCK_USERS)
    ]
    matrix = np.concatenate(parts) if parts else np.zeros((0, len(features)), dtype=np.float32)
    labels = (candidates == targets[:, None]).ravel().astype(np.int32)
    return FeatureBlock(matrix, labels, candidates.shape[1], tuple(features))


def compute_features(
    ml: MovieLens,
    users: I64,
    candidates: I64,
    times: I64,
    history_indptr: I64,
    history_items: I64,
    user_vectors: F32,
    item_vectors: F32,
    popularity: PopularityIndex,
    features: tuple[str, ...] = FEATURES,
) -> F32:
    """The single feature computation shared by offline training and online serving.

    ``times`` is the prediction moment of each user (target time offline, request time online);
    ``history_*`` is the CSR (indexed by user id) of each user's history before that moment;
    ``user_vectors`` row ``i`` is the user vector of ``users[i]``.
    """
    if (candidates < 0).any():
        raise ValueError("candidates contain padding; retrieve enough items first")
    if not ml.demographics and set(features) & DEMOGRAPHIC_FEATURES:
        raise ValueError(
            f"features {sorted(set(features) & DEMOGRAPHIC_FEATURES)} need user demographics, "
            "which this dataset does not have"
        )
    n_users, n_cand = candidates.shape
    indptr, history = history_indptr, history_items
    profiles = genre_profiles_for(indptr, history, users, ml.genres)  # row i = users[i]
    flat_items = candidates.ravel()
    flat_users = np.repeat(users, n_cand)
    flat_times = np.repeat(times, n_cand)
    tt_score = np.einsum("ud,ucd->uc", user_vectors, item_vectors[candidates]).ravel()
    columns = {
        "tt_score": tt_score,
        "tt_rank": np.tile(np.arange(n_cand), n_users),
        "pop_before": popularity.count_before(flat_items, flat_times),
        "pop_recent_30d": popularity.count_between(
            flat_items, flat_times - 30 * SECONDS_PER_DAY, flat_times
        ),
        "year": ml.years[flat_items],
        "genre_affinity": genre_affinity(
            profiles, np.repeat(np.arange(n_users), n_cand), flat_items, ml.genres
        ),
        "user_activity": np.diff(indptr)[flat_users],
        "gender": ml.gender[flat_users],
        "age": ml.age[flat_users],
        "occupation": ml.occupation[flat_users],
    }
    if set(features) & set(EXTRA_FEATURES):
        sims = _recent_similarities(indptr, history, users, candidates, item_vectors, 5)
        columns["sim_last1"] = sims[0].ravel()
        columns["sim_recent5_mean"] = sims[1].ravel()
        columns["sim_recent5_max"] = sims[2].ravel()
        target_year = (1970 + flat_times / (365.2425 * SECONDS_PER_DAY)).astype(np.int64)
        columns["item_age_years"] = target_year - ml.years[flat_items]
    return np.column_stack([np.asarray(columns[name], dtype=np.float32) for name in features])


def _recent_similarities(
    indptr: I64, history: I64, users: I64, candidates: I64, item_vectors: F32, recent: int
) -> tuple[F32, F32, F32]:
    """Cosine of each candidate to the user's latest ``recent`` history items.

    Returns ``(to_latest, mean_over_recent, max_over_recent)``, each ``(n_users, n_cand)``;
    zeros for users with an empty history.
    """
    unit = item_vectors / np.maximum(np.linalg.norm(item_vectors, axis=1, keepdims=True), 1e-12)
    stop = indptr[users + 1]
    start = np.maximum(indptr[users], stop - recent)
    offsets = start[:, None] + np.arange(recent)[None, :]
    valid = offsets < stop[:, None]
    hist = np.where(valid, history[np.minimum(offsets, max(history.shape[0] - 1, 0))], 0)
    sims = np.einsum("ucd,uhd->uch", unit[candidates], unit[hist])  # (users, cand, recent)
    count = valid.sum(axis=1)
    has = count > 0
    last_idx = np.maximum(count - 1, 0)
    to_latest = np.take_along_axis(sims, last_idx[:, None, None].repeat(sims.shape[1], 1), 2)[
        ..., 0
    ]
    masked = np.where(valid[:, None, :], sims, np.nan)
    with np.errstate(invalid="ignore"), warnings_suppressed():
        mean = np.nan_to_num(np.nanmean(masked, axis=2))
        best = np.nan_to_num(np.nanmax(masked, axis=2))
    zero = ~has
    to_latest[zero], mean[zero], best[zero] = 0.0, 0.0, 0.0
    return to_latest.astype(np.float32), mean.astype(np.float32), best.astype(np.float32)


def holdout_split(
    labels: NDArray[np.int32], n_candidates: int, fraction: float, seed: int
) -> tuple[I64, I64]:
    """Users usable for training (target among the candidates), split into fit and hold-out."""
    per_user = labels.reshape(-1, n_candidates)
    usable = np.flatnonzero(per_user.any(axis=1))
    mask = np.random.default_rng(seed).random(usable.shape[0]) < fraction
    return usable[~mask].astype(np.int64), usable[mask].astype(np.int64)


def ndcg_of_ordering(
    scores: NDArray[np.floating[Any]], labels: NDArray[np.int32], k: int = 10
) -> float:
    """Mean NDCG@k when each user's candidates are sorted by ``scores`` (one relevant item).

    Ties keep the original candidate order (stable sort).
    """
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked = np.take_along_axis(labels, order, axis=1)
    position = ranked.argmax(axis=1) + 1
    hit = ranked.max(axis=1) > 0
    gains = np.where(hit & (position <= k), 1.0 / np.log2(position + 1.0), 0.0)
    return float(gains.mean())


class LambdaRanker:
    """LightGBM LambdaRank with early stopping on a held-out share of the training users."""

    def __init__(self, config: RankerConfig) -> None:
        self.config = config
        self.booster: Any = None
        self.best_iteration = 0

    def params(self) -> dict[str, Any]:
        """LightGBM parameters (deterministic, single-threaded)."""
        c = self.config
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [10],
            "num_leaves": c.num_leaves,
            "learning_rate": c.learning_rate,
            "min_data_in_leaf": c.min_data_in_leaf,
            "feature_fraction": c.feature_fraction,
            "bagging_fraction": c.bagging_fraction,
            "bagging_freq": c.bagging_freq,
            "lambda_l2": c.lambda_l2,
            "seed": c.seed,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": c.num_threads,
            "verbosity": -1,
        }

    def fit(self, block: FeatureBlock, n_users: int) -> dict[str, Any]:
        """Train on users whose group contains the target; returns a training summary."""
        import lightgbm as lgb

        block = block.select(self.config.features)
        matrix = block.matrix.reshape(n_users, block.n_candidates, -1)
        labels = block.labels.reshape(n_users, block.n_candidates)
        # Groups without the target carry no LambdaRank gradient, so only usable users train.
        fit_rows, holdout_rows = holdout_split(
            block.labels, block.n_candidates, self.config.holdout_fraction, self.config.seed
        )
        parts = {"train": fit_rows, "holdout": holdout_rows}
        usable = np.concatenate([fit_rows, holdout_rows])
        sets = {}
        for name, rows in parts.items():
            sets[name] = lgb.Dataset(
                matrix[rows].reshape(-1, matrix.shape[2]),
                labels[rows].ravel(),
                group=np.full(rows.shape[0], block.n_candidates),
                feature_name=list(block.feature_names),
                categorical_feature=[c for c in CATEGORICAL if c in block.feature_names],
                free_raw_data=False,
            )
        evals: dict[str, Any] = {}
        self.booster = lgb.train(
            self.params(),
            sets["train"],
            num_boost_round=self.config.max_rounds,
            valid_sets=[sets["holdout"]],
            valid_names=["holdout"],
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                lgb.record_evaluation(evals),
            ],
        )
        self.best_iteration = int(self.booster.best_iteration)
        gain = self.booster.feature_importance(importance_type="gain")
        return {
            "users_total": n_users,
            "users_with_target_in_candidates": int(usable.shape[0]),
            "train_users": int(parts["train"].shape[0]),
            "holdout_users": int(parts["holdout"].shape[0]),
            "best_iteration": self.best_iteration,
            "holdout_ndcg@10_best": float(max(evals["holdout"]["ndcg@10"])),
            "feature_importance_gain": dict(
                zip(block.feature_names, map(float, gain), strict=True)
            ),
        }

    def scores(self, block: FeatureBlock) -> NDArray[np.float64]:
        """Predicted relevance of every row (best iteration)."""
        block = block.select(self.config.features)
        out: NDArray[np.float64] = self.booster.predict(
            block.matrix, num_iteration=self.best_iteration
        )
        return out

    def rerank(self, block: FeatureBlock, candidates: I64) -> I64:
        """Candidates of each user reordered by predicted score (stable for ties)."""
        block = block.select(self.config.features)
        scores = self.booster.predict(block.matrix, num_iteration=self.best_iteration)
        scores = scores.reshape(candidates.shape)
        order = np.argsort(-scores, axis=1, kind="stable")
        return np.take_along_axis(candidates, order, axis=1)


@contextmanager
def warnings_suppressed() -> Iterator[None]:
    """Silence NumPy's all-NaN slice warnings (empty histories are handled explicitly)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yield
