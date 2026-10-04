"""Metrics against hand-computed values, ranking rules, and bootstrap intervals."""

import math

import numpy as np
import pytest

from recsys.evaluation import (
    RankingResult,
    bootstrap,
    coverage,
    ndcg_at_k,
    paired_difference,
    rank_targets,
    ranks_from_lists,
    recall_at_k,
    reciprocal_rank,
    summarize,
    top_k_ids,
)

RANKS = np.array([1, 2, 11, 60])


def test_recall_ndcg_mrr_hand_computed() -> None:
    np.testing.assert_array_equal(recall_at_k(RANKS, 10), [1, 1, 0, 0])
    np.testing.assert_allclose(ndcg_at_k(RANKS, 10), [1.0, 1 / math.log2(3), 0.0, 0.0])
    np.testing.assert_allclose(reciprocal_rank(RANKS), [1.0, 0.5, 1 / 11, 1 / 60])
    metrics = summarize(RANKS, np.array([[0, 1], [1, 2], [2, 3], [3, 0]]), n_items=8)
    assert metrics["recall@10"] == pytest.approx(0.5)
    assert metrics["recall@50"] == pytest.approx(0.75)
    assert metrics["ndcg@10"] == pytest.approx((1 + 1 / math.log2(3)) / 4)
    assert metrics["mrr"] == pytest.approx((1 + 0.5 + 1 / 11 + 1 / 60) / 4)


def test_coverage() -> None:
    assert coverage(np.array([[0, 1, 5], [1, 2, 6]]), n_items=4, k=2) == pytest.approx(0.75)


def test_rank_targets_excludes_history_and_breaks_ties_by_id() -> None:
    #            item:   0    1    2    3    4
    scores = np.array([[0.9, 0.5, 0.5, 0.8, 0.1]])
    indptr, history = np.array([0, 1]), np.array([0])  # item 0 is in the user's history

    def score_fn(users: np.ndarray) -> np.ndarray:
        return np.repeat(scores, users.shape[0], axis=0)

    # Target item 2 ties with item 1 (lower id wins the tie): ranking is 3, 1, 2, 4.
    result = rank_targets(score_fn, np.array([0]), np.array([2]), indptr, history, 5, k_max=4)
    assert result.ranks.tolist() == [3]
    assert result.top_k.tolist() == [[3, 1, 2, 4]]
    result = rank_targets(score_fn, np.array([0]), np.array([1]), indptr, history, 5, k_max=4)
    assert result.ranks.tolist() == [2]


def test_rank_targets_rejects_target_in_history() -> None:
    with pytest.raises(ValueError, match="history"):
        rank_targets(
            lambda u: np.ones((u.shape[0], 3)),
            np.array([0]),
            np.array([1]),
            np.array([0, 1]),
            np.array([1]),
            3,
            k_max=2,
        )


def test_rank_targets_checks_score_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        rank_targets(
            lambda u: np.ones((u.shape[0], 2)),
            np.array([0]),
            np.array([1]),
            np.array([0, 0]),
            np.array([], dtype=np.int64),
            3,
            k_max=2,
        )


def test_top_k_ids_tie_breaking() -> None:
    assert top_k_ids(np.array([0.1, 0.7, 0.7, 0.7, 0.2]), 2).tolist() == [1, 2]
    assert top_k_ids(np.array([3.0, -np.inf, 1.0]), 3).tolist() == [0, 2, 1]


def test_ranks_from_lists() -> None:
    ranked = np.array([[2, 0, 1], [1, 2, 0]])
    assert ranks_from_lists(ranked, np.array([0, 1])).tolist() == [2, 1]
    with pytest.raises(ValueError, match="missing"):
        ranks_from_lists(np.array([[2, 1]]), np.array([0]))


def test_bootstrap_interval_contains_point_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    ranks = rng.integers(1, 100, size=500)
    result = RankingResult(ranks, rng.integers(0, 50, size=(500, 50)), n_items=50)
    a = bootstrap(result, n_resamples=200, seed=1)
    b = bootstrap(result, n_resamples=200, seed=1)
    assert a.as_dict() == b.as_dict()
    for interval in a.intervals.values():
        assert interval.low <= interval.value <= interval.high


def test_bootstrap_degenerate_case() -> None:
    result = RankingResult(np.ones(50, dtype=np.int64), np.zeros((50, 50), dtype=np.int64), 10)
    recall = bootstrap(result, n_resamples=50).intervals["recall@10"]
    assert (recall.value, recall.low, recall.high) == (1.0, 1.0, 1.0)


def test_paired_difference() -> None:
    top = np.zeros((4, 50), dtype=np.int64)
    a = RankingResult(np.array([1, 1, 20, 20]), top, 10)
    b = RankingResult(np.array([20, 20, 20, 20]), top, 10)
    diff = paired_difference(a, b, "recall@10", n_resamples=200)
    assert diff.value == pytest.approx(0.5)
    assert diff.low <= 0.5 <= diff.high
    same = paired_difference(a, a, "mrr", n_resamples=50)
    assert (same.value, same.low, same.high) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="same users"):
        paired_difference(a, RankingResult(np.array([1]), top[:1], 10))


def test_coverage_is_not_bootstrapped_because_resampling_biases_it_low() -> None:
    # 200 users each recommended a distinct item: coverage is 200 / 400 = 0.5. Resampling users
    # with replacement keeps only ~63% distinct users, so resampled coverage is ~0.32: a
    # percentile interval would not even contain the true value.
    top = np.arange(200)[:, None].repeat(50, axis=1)
    result = RankingResult(np.ones(200, dtype=np.int64), top, n_items=400)
    assert result.metrics()["coverage@10"] == 0.5
    rng = np.random.default_rng(0)
    resampled = [coverage(top[rng.integers(0, 200, 200)], 400, 10) for _ in range(200)]
    assert max(resampled) < 0.5
    assert "coverage@10" not in bootstrap(result, n_resamples=20).intervals
