"""Baselines: popularity, item-item co-occurrence and implicit ALS matrix factorisation.

Each model is fit on training positives only and exposes ``score(users, stage)``, returning a
``(len(users), n_items)`` matrix (higher is better) for use with
:func:`recsys.evaluation.rank_targets`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data import Split, Stage

I64 = NDArray[np.int64]
F32 = NDArray[np.float32]


BLOCK_USERS = 4096  # users per dense block when building co-occurrence counts


def _history_rows(indptr: I64, items: I64, users: I64, n_items: int) -> F32:
    """Dense binary rows (one per user in ``users``) of a history CSR."""
    rows = np.zeros((users.shape[0], n_items), dtype=np.float32)
    counts = indptr[users + 1] - indptr[users]
    owners = np.repeat(np.arange(users.shape[0]), counts)
    if owners.size:
        cols = np.concatenate([items[indptr[u] : indptr[u + 1]] for u in users])
        rows[owners, cols] = 1.0
    return rows


class Popularity:
    """Score every item by its number of training positives (the same list for every user)."""

    def __init__(self, split: Split) -> None:
        counts = np.bincount(split.train_items, minlength=split.n_items)
        self.counts = counts.astype(np.float32)

    def score(self, users: I64, stage: Stage) -> F32:
        """Identical popularity scores for each user."""
        return np.broadcast_to(self.counts, (users.shape[0], self.counts.shape[0]))


class ItemCooccurrence:
    """Item-based collaborative filtering with cosine-normalised co-occurrence counts.

    ``sim(i, j) = C_ij / sqrt(n_i * n_j)`` where ``C`` counts users with both items among their
    training positives and ``n_i`` is item ``i``'s count; ``score(u, j) = sum over history items
    i of sim(i, j)``.
    """

    def __init__(self, split: Split) -> None:
        self.split = split
        # Training positives only (the history before the validation target). Counts are summed
        # over blocks of users; float32 holds these integer counts exactly, so the result does
        # not depend on the block size, and no users-by-items matrix is ever materialised.
        indptr, items = split.history_csr("val")
        co = np.zeros((split.n_items, split.n_items), dtype=np.float32)
        everyone = np.arange(split.n_users)
        for start in range(0, split.n_users, BLOCK_USERS):
            block = _history_rows(
                indptr, items, everyone[start : start + BLOCK_USERS], split.n_items
            )
            co += block.T @ block
        diag = np.sqrt(np.maximum(np.diag(co), 1.0))
        sim = co / diag[:, None] / diag[None, :]
        np.fill_diagonal(sim, 0.0)
        self.similarity = sim.astype(np.float32)
        self._histories = {stage: split.history_csr(stage) for stage in ("val", "test")}

    def score(self, users: I64, stage: Stage) -> F32:
        """Sum of similarities to the user's history items."""
        indptr, items = self._histories[stage]
        return _history_rows(indptr, items, users, self.split.n_items) @ self.similarity


@dataclass(frozen=True)
class ALSConfig:
    """Hyperparameters of implicit ALS."""

    factors: int = 64
    regularization: float = 1.0
    alpha: float = 10.0
    iterations: int = 15
    seed: int = 0


class ImplicitALS:
    """Implicit-feedback matrix factorisation (Hu, Koren and Volinsky, ICDM 2008).

    Minimises ``sum_ui c_ui (p_ui - x_u.y_i)^2 + lambda (|X|^2 + |Y|^2)`` with preference
    ``p_ui = 1`` for training positives (else 0) and confidence ``c_ui = 1 + alpha * p_ui``.
    Alternates exact least-squares solves for users and items using the standard identity
    ``Y^T C_u Y = Y^T Y + alpha * Y_u^T Y_u``, so each solve costs ``O(|I_u| f^2 + f^3)``.
    """

    def __init__(self, split: Split, config: ALSConfig) -> None:
        self.split = split
        self.config = config
        rng = np.random.default_rng(config.seed)
        f = config.factors
        self.user_factors = (rng.standard_normal((split.n_users, f)) * 0.01).astype(np.float64)
        self.item_factors = (rng.standard_normal((split.n_items, f)) * 0.01).astype(np.float64)
        self._user_items = (split.train_indptr, split.train_items)
        self._item_users = _transpose_csr(split.train_indptr, split.train_items, split.n_items)
        self._test_history = split.history_csr("test")

    def fit(self) -> ImplicitALS:
        """Run the configured number of alternating iterations."""
        for _ in range(self.config.iterations):
            self.user_factors = self._solve(self.item_factors, *self._user_items)
            self.item_factors = self._solve(self.user_factors, *self._item_users)
        return self

    def _solve(self, fixed: NDArray[np.float64], indptr: I64, cols: I64) -> NDArray[np.float64]:
        f = fixed.shape[1]
        alpha, reg = self.config.alpha, self.config.regularization
        gram = fixed.T @ fixed + reg * np.eye(f)
        out = np.zeros((indptr.shape[0] - 1, f), dtype=np.float64)
        for row in range(indptr.shape[0] - 1):
            idx = cols[indptr[row] : indptr[row + 1]]
            if idx.size == 0:
                continue
            factors = fixed[idx]
            a = gram + alpha * (factors.T @ factors)
            b = (1.0 + alpha) * factors.sum(axis=0)
            out[row] = np.linalg.solve(a, b)
        return out

    def score(self, users: I64, stage: Stage) -> F32:
        """Dot products of user and item factors.

        For the test stage the user vector is re-solved with the validation item added to the
        history (the item factors stay fixed), so the test prediction uses everything known
        before the test target, exactly like the other methods.
        """
        if stage == "val":
            vectors = self.user_factors[users]
        else:
            indptr, items = self._test_history
            sub_indptr = np.zeros(users.shape[0] + 1, dtype=np.int64)
            np.cumsum(indptr[users + 1] - indptr[users], out=sub_indptr[1:])
            sub_items = np.concatenate([items[indptr[u] : indptr[u + 1]] for u in users])
            vectors = self._solve(self.item_factors, sub_indptr, sub_items)
        return (vectors @ self.item_factors.T).astype(np.float32)


def _transpose_csr(indptr: I64, cols: I64, n_cols: int) -> tuple[I64, I64]:
    rows = np.repeat(np.arange(indptr.shape[0] - 1), np.diff(indptr))
    order = np.argsort(cols, kind="stable")
    new_indptr = np.zeros(n_cols + 1, dtype=np.int64)
    np.cumsum(np.bincount(cols, minlength=n_cols), out=new_indptr[1:])
    return new_indptr, rows[order].astype(np.int64)
