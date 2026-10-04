"""Time-aware features (checked against brute force) and baseline models."""

from pathlib import Path

import numpy as np
import pytest

from recsys.baselines import ALSConfig, ImplicitALS, ItemCooccurrence, Popularity
from recsys.data import load, make_split
from recsys.features import (
    PopularityIndex,
    genre_affinity,
    genre_profiles,
    genre_profiles_for,
    year_bucket,
)


def test_popularity_before_t_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    items = rng.integers(0, 20, size=2000)
    times = rng.integers(0, 100, size=2000)  # many exact timestamp ties
    index = PopularityIndex(items, times, n_items=20)
    q_items = rng.integers(0, 20, size=500)
    q_times = rng.integers(-5, 110, size=500)
    expected = [
        int(((items == i) & (times < t)).sum()) for i, t in zip(q_items, q_times, strict=True)
    ]
    np.testing.assert_array_equal(index.count_before(q_items, q_times), expected)
    window = [
        int(((items == i) & (times >= t - 30) & (times < t)).sum())
        for i, t in zip(q_items, q_times, strict=True)
    ]
    np.testing.assert_array_equal(index.count_between(q_items, q_times - 30, q_times), window)
    np.testing.assert_array_equal(index.totals(), np.bincount(items, minlength=20))


def test_popularity_never_counts_the_target_moment() -> None:
    index = PopularityIndex(np.array([0, 0, 0]), np.array([5, 10, 10]), n_items=1)
    assert index.count_before(np.array([0]), 10).tolist() == [1]  # events at t=10 excluded
    assert index.count_before(np.array([0]), 11).tolist() == [3]


def test_genre_profiles_and_affinity() -> None:
    genres = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32)
    indptr = np.array([0, 2, 2])  # user 0 history [0, 1]; user 1 empty
    profiles = genre_profiles(indptr, np.array([0, 1]), genres)
    np.testing.assert_allclose(profiles, [[1.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
    affinity = genre_affinity(profiles, np.array([0, 0, 1]), np.array([0, 2, 0]), genres)
    np.testing.assert_allclose(affinity, [1 / np.sqrt(1.25), 0.0, 0.0], rtol=1e-6)


def test_year_bucket() -> None:
    np.testing.assert_array_equal(
        year_bucket(np.array([1919, 1920, 1924, 1925, 2000])), [0, 1, 1, 2, 17]
    )


def test_popularity_baseline(tiny_zip: Path) -> None:
    split = make_split(load(tiny_zip))
    scores = Popularity(split).score(np.array([0, 1]), "val")
    # training positives: user0 [m1, m2], user1 [m1, m6], user2 [m2] -> m1: 2, m2: 2, m6: 1
    np.testing.assert_array_equal(scores[0], [2, 2, 0, 0, 1, 0])
    np.testing.assert_array_equal(scores[0], scores[1])


def test_item_cooccurrence(tiny_zip: Path) -> None:
    split = make_split(load(tiny_zip))
    model = ItemCooccurrence(split)
    sim = model.similarity
    np.testing.assert_allclose(sim, sim.T)
    assert np.all(np.diag(sim) == 0)
    # m1 (id 0) and m2 (id 1) co-occur once (user 0); each has 2 positives -> 1 / sqrt(2 * 2).
    assert sim[0, 1] == pytest.approx(0.5)
    # User 2's history is [m2]; for the test stage it also contains the validation item m3.
    np.testing.assert_allclose(model.score(np.array([2]), "val")[0], sim[1])
    np.testing.assert_allclose(model.score(np.array([2]), "test")[0], sim[1] + sim[2])


def test_implicit_als_learns_block_structure() -> None:
    # Two communities: users 0-19 like items 0-9, users 20-39 like items 10-19.
    from recsys.data import Split

    rng = np.random.default_rng(0)
    n_users, n_items = 40, 20
    rows, cols = [], []
    for u in range(n_users):
        block = np.arange(10) if u < 20 else np.arange(10, 20)
        chosen = np.sort(rng.choice(block, size=7, replace=False))
        rows.append(chosen[:5])  # training items; 6th and 7th serve as val and test
        cols.append(chosen[5:])
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum([len(r) for r in rows], out=indptr[1:])
    split = Split(
        n_users=n_users,
        n_items=n_items,
        users=np.arange(n_users),
        train_indptr=indptr,
        train_items=np.concatenate(rows).astype(np.int64),
        train_times=np.zeros(int(indptr[-1]), dtype=np.int64),
        val_item=np.array([c[0] for c in cols], dtype=np.int64),
        val_time=np.ones(n_users, dtype=np.int64),
        test_item=np.array([c[1] for c in cols], dtype=np.int64),
        test_time=np.full(n_users, 2, dtype=np.int64),
    )
    model = ImplicitALS(split, ALSConfig(factors=4, regularization=0.1, alpha=5.0, iterations=10))
    scores = model.fit().score(np.arange(n_users), "val")
    in_block = np.concatenate([scores[:20, :10].ravel(), scores[20:, 10:].ravel()])
    out_block = np.concatenate([scores[:20, 10:].ravel(), scores[20:, :10].ravel()])
    assert in_block.mean() > out_block.mean() + 0.3
    # The test-stage user vector is re-solved with the validation item in the history.
    assert not np.allclose(model.score(np.array([0]), "test"), model.score(np.array([0]), "val"))


def test_profiles_for_selected_users_match_full_profiles() -> None:
    rng = np.random.default_rng(1)
    genres = (rng.random((30, 5)) < 0.3).astype(np.float32)
    lengths = rng.integers(0, 6, size=12)
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    items = rng.integers(0, 30, size=int(indptr[-1])).astype(np.int64)
    users = np.array([7, 0, 3, 3, 11])
    full = genre_profiles(indptr, items, genres)
    np.testing.assert_array_equal(genre_profiles_for(indptr, items, users, genres), full[users])
