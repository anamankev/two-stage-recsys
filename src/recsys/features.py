"""Time-aware features: everything computed for a (user, target) pair uses only earlier events.

* Item popularity at time ``t`` counts *training* positives the item received strictly before
  ``t`` (an exact count via binary search over per-item sorted timestamps), so no future event
  of any user leaks into the feature.
* User-side features (genre profile, activity) are computed from the user's history before the
  target, as defined by :meth:`recsys.data.Split.history`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

I64 = NDArray[np.int64]
F32 = NDArray[np.float32]
SECONDS_PER_DAY = 86_400


class PopularityIndex:
    """Answers "how many training positives did item i get before time t?" exactly."""

    def __init__(self, items: I64, times: I64, n_items: int) -> None:
        if items.shape != times.shape:
            raise ValueError("items and times must have the same shape")
        self.n_items = n_items
        # Keys item * span + (time - base) are unique per (item, time) and sort by item, then time.
        self._base = int(times.min()) if times.size else 0
        self._time_span = (int(times.max()) - self._base + 2) if times.size else 2
        order = np.lexsort((times, items))
        self._keys = items[order] * self._time_span + (times[order] - self._base)
        self._start = np.zeros(n_items + 1, dtype=np.int64)
        np.cumsum(np.bincount(items, minlength=n_items), out=self._start[1:])

    def count_before(self, items: I64, t: I64 | int) -> I64:
        """Positives of each item with timestamp strictly before ``t`` (broadcast)."""
        relative = np.asarray(t, dtype=np.int64) - self._base
        # Clamping keeps every query key inside its own item's block of keys.
        keys = items * self._time_span + np.clip(relative, 0, self._time_span - 1)
        return (np.searchsorted(self._keys, keys, side="left") - self._start[items]).astype(
            np.int64
        )

    def count_between(self, items: I64, start: I64 | int, end: I64 | int) -> I64:
        """Positives of each item with ``start <= timestamp < end``."""
        return self.count_before(items, end) - self.count_before(items, start)

    def totals(self) -> I64:
        """All training positives per item (used by the plain popularity baseline)."""
        return np.diff(self._start).astype(np.int64)


def genre_profiles(history_indptr: I64, history_items: I64, genres: F32) -> F32:
    """Each user's mean genre vector over their history (zero vector for an empty history)."""
    n_users = history_indptr.shape[0] - 1
    counts = np.diff(history_indptr)
    owners = np.repeat(np.arange(n_users), counts)
    sums = np.zeros((n_users, genres.shape[1]), dtype=np.float64)
    np.add.at(sums, owners, genres[history_items])
    with np.errstate(invalid="ignore", divide="ignore"):
        profiles = np.where(counts[:, None] > 0, sums / counts[:, None], 0.0)
    return profiles.astype(np.float32)


def genre_profiles_for(history_indptr: I64, history_items: I64, users: I64, genres: F32) -> F32:
    """Mean genre vector over the history of each listed user only (row ``i`` is ``users[i]``).

    Same values as :func:`genre_profiles` for those users (genre vectors are 0/1, so the sums are
    exact integers in any order), at a cost proportional to their histories rather than to the
    whole dataset. This is what makes per-request feature computation cheap when serving.
    """
    out = np.zeros((users.shape[0], genres.shape[1]), dtype=np.float64)
    for row, user in enumerate(users):
        items = history_items[history_indptr[user] : history_indptr[user + 1]]
        if items.size:
            out[row] = genres[items].astype(np.float64).sum(axis=0) / items.size
    return out.astype(np.float32)


def genre_affinity(profiles: F32, users: I64, items: I64, genres: F32) -> F32:
    """Cosine similarity between each user's genre profile and each item's genre vector."""
    p = profiles[users]
    g = genres[items]
    num = (p * g).sum(axis=1)
    den = np.linalg.norm(p, axis=1) * np.linalg.norm(g, axis=1)
    affinity: F32 = np.divide(num, den, out=np.zeros_like(num), where=den > 0).astype(np.float32)
    return affinity


def year_bucket(years: I64, *, start: int = 1920, width: int = 5) -> I64:
    """Release year in ``width``-year buckets; everything before ``start`` shares bucket 0."""
    return (np.maximum(years - start, -1) // width + 1).astype(np.int64)
