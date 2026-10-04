"""Two-tower model, retrieval and ranker: correctness and leakage tests on small data."""

from pathlib import Path

import numpy as np
import pytest
import torch

from recsys.data import load, make_split
from recsys.evaluation import rank_with_candidates
from recsys.features import PopularityIndex
from recsys.ranker import FEATURES, LambdaRanker, RankerConfig, build_features
from recsys.retrieval import (
    Candidates,
    ExactRetriever,
    HnswEngineRetriever,
    candidate_recall,
    overlap,
    retrieve,
)
from recsys.two_tower import Trainer, TwoTowerConfig, embed, evaluate, padded_histories

# ------------------------------------------------------------------ two-tower


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="negatives"):
        TwoTowerConfig(negatives="hard")
    with pytest.raises(ValueError, match="temperature"):
        TwoTowerConfig(temperature=0.0)
    with pytest.raises(ValueError, match="positive"):
        TwoTowerConfig(batch_size=0)


def test_padded_histories_only_see_earlier_items() -> None:
    indptr = np.array([0, 4, 4])
    items = np.array([10, 11, 12, 13])
    users = np.array([0, 0, 0, 1])
    ends = np.array([0, 2, 4, 0])
    out = padded_histories(indptr, items, users, ends, length=3, pad=99)
    np.testing.assert_array_equal(out, [[99, 99, 99], [10, 11, 99], [11, 12, 13], [99, 99, 99]])


def test_training_examples_never_contain_their_target(tiny_zip: Path) -> None:
    ml = load(tiny_zip)
    split = make_split(ml)
    trainer = Trainer(split, ml, TwoTowerConfig(history_length=5, batch_size=4, max_epochs=1))
    hist = padded_histories(
        split.train_indptr, split.train_items, trainer.ex_user, trainer.ex_pos, 5, split.n_items
    )
    for row, (user, pos) in enumerate(zip(trainer.ex_user, trainer.ex_pos, strict=True)):
        earlier = split.train_history(user)[:pos].tolist()
        assert [h for h in hist[row] if h != split.n_items] == earlier[-5:]
        assert trainer.ex_item[row] == split.train_history(user)[pos]


@pytest.mark.parametrize("negatives", ["in_batch", "in_batch_logq", "uniform", "mixed"])
def test_training_runs_for_every_negative_mode(tiny_zip: Path, negatives: str) -> None:
    ml = load(tiny_zip)
    split = make_split(ml)
    config = TwoTowerConfig(
        embedding_dim=8,
        hidden_dim=8,
        history_length=3,
        batch_size=4,
        max_epochs=2,
        negatives=negatives,
        n_uniform_negatives=4,
        num_threads=1,
        eval_k=3,
    )
    trainer = Trainer(split, ml, config)
    model = trainer.fit(verbose=False)
    assert all(np.isfinite(e["loss"]) for e in trainer.log)
    users, items = embed(model, split, "test")
    assert users.shape == (split.n_users, 8)
    assert items.shape == (split.n_items, 8)
    np.testing.assert_allclose(np.linalg.norm(items, axis=1), 1.0, rtol=1e-5)
    ranks = evaluate(model, split, "val", k_max=3).result.ranks
    assert ((ranks >= 1) & (ranks <= split.n_items)).all()


def test_duplicate_items_in_a_batch_are_masked(tiny_zip: Path) -> None:
    ml = load(tiny_zip)
    split = make_split(ml)
    trainer = Trainer(split, ml, TwoTowerConfig(embedding_dim=8, hidden_dim=8, num_threads=1))
    # Users 0 and 1 both have item 0 among their training positives.
    loss = trainer._loss(np.array([0, 1]), np.array([0, 0]), np.array([0, 0]))
    assert torch.isfinite(loss)


# ------------------------------------------------------------------ retrieval


@pytest.fixture
def unit_vectors() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    items = rng.normal(size=(300, 16)).astype(np.float32)
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    users = rng.normal(size=(20, 16)).astype(np.float32)
    users /= np.linalg.norm(users, axis=1, keepdims=True)
    return users, items


def test_exact_retriever_matches_brute_force(unit_vectors) -> None:
    users, items = unit_vectors
    retriever = ExactRetriever()
    retriever.build(items)
    for u in users:
        expected = np.argsort(-(items @ u), kind="stable")[:10]
        np.testing.assert_array_equal(retriever.search(u, 10), expected)


def _approximate_recall(retriever, users: np.ndarray, items: np.ndarray) -> float:
    retriever.build(items)
    exact = ExactRetriever()
    exact.build(items)
    hits = [
        len(set(retriever.search(u, 10).tolist()) & set(exact.search(u, 10).tolist()))
        for u in users
    ]
    return float(np.mean(hits) / 10)


def test_hnsw_engine_retriever_finds_true_neighbours(unit_vectors) -> None:
    users, items = unit_vectors
    retriever = HnswEngineRetriever(ef_search=100)
    assert _approximate_recall(retriever, users, items) >= 0.95
    assert retriever.describe()["name"] == "hnsw-engine"


FAISS_CHECK = """
import numpy as np
from recsys.retrieval import ExactRetriever, FaissHnswRetriever
rng = np.random.default_rng(0)
items = rng.normal(size=(300, 16)).astype(np.float32)
items /= np.linalg.norm(items, axis=1, keepdims=True)
users = rng.normal(size=(20, 16)).astype(np.float32)
users /= np.linalg.norm(users, axis=1, keepdims=True)
faiss_r, exact = FaissHnswRetriever(), ExactRetriever()
faiss_r.build(items); exact.build(items)
hits = [len(set(faiss_r.search(u, 10).tolist()) & set(exact.search(u, 10).tolist())) for u in users]
assert np.mean(hits) / 10 >= 0.95, hits
assert faiss_r.describe()["name"] == "faiss-hnsw"
print("ok")
"""


def test_faiss_retriever_finds_true_neighbours_in_own_process() -> None:
    # PyTorch and FAISS each bundle an OpenMP runtime; running both in one process aborts on
    # macOS (OMP error 15). This test process has PyTorch loaded, so FAISS runs in a child.
    import subprocess
    import sys

    pytest.importorskip("faiss")
    out = subprocess.run(
        [sys.executable, "-c", FAISS_CHECK],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "ok"


def test_retrieve_excludes_history_and_measures_latency(unit_vectors) -> None:
    users, items = unit_vectors
    retriever = ExactRetriever()
    retriever.build(items)
    top = retriever.search(users[0], 5)
    indptr, history = np.array([0, 2]), top[:2]  # the user's two best items are history
    cands = retrieve(retriever, users[:1], np.array([0]), indptr, history, n=3)
    np.testing.assert_array_equal(cands.items[0], top[2:5])
    assert (cands.latency_ns > 0).all()
    assert set(cands.latency_summary()) == {"mean_ms", "p50_ms", "p95_ms", "p99_ms"}


def test_candidate_recall_and_overlap() -> None:
    a = Candidates(np.array([0, 1]), np.array([[5, 6, 7], [1, 2, 3]]), np.array([1, 1]))
    b = Candidates(np.array([0, 1]), np.array([[6, 5, 9], [4, 5, 6]]), np.array([1, 1]))
    assert candidate_recall(a, np.array([6, 9]), (1, 2, 3)) == {
        "recall@1": 0.0,
        "recall@2": 0.5,
        "recall@3": 0.5,
    }
    assert overlap(a, b, 3) == pytest.approx((2 / 3 + 0) / 2)


def test_rank_with_candidates() -> None:
    scores = np.array([[0.1, 0.9, 0.8, 0.7, 0.6, 0.5]])

    def fallback(users: np.ndarray) -> np.ndarray:
        return np.repeat(scores, users.shape[0], axis=0)

    candidates = np.array([[3, 1]])  # a re-ordered two-item candidate list
    indptr, history = np.array([0, 1]), np.array([2])  # item 2 is history

    def rank(target: int) -> int:
        return int(
            rank_with_candidates(
                candidates, fallback, np.array([0]), np.array([target]), indptr, history, 6, k_max=2
            ).ranks[0]
        )

    assert rank(3) == 1
    assert rank(1) == 2
    # Outside the list: 2 candidates, then the rest by score (item 4: 0.6, item 5: 0.5, item 0).
    assert rank(4) == 3
    assert rank(5) == 4
    assert rank(0) == 5
    with pytest.raises(ValueError, match="history"):
        rank(2)


# --------------------------------------------------------------------- ranker


def test_features_shapes_labels_and_time_rule(tiny_zip: Path) -> None:
    ml = load(tiny_zip)
    split = make_split(ml)
    users = split.users
    candidates = np.array([[3, 5, 2], [5, 2, 3], [3, 2, 0]])  # val targets 3, 5, 2
    rng = np.random.default_rng(0)
    u_vec = rng.normal(size=(split.n_users, 4)).astype(np.float32)
    i_vec = rng.normal(size=(split.n_items, 4)).astype(np.float32)
    pop = PopularityIndex(split.train_items, split.train_times, split.n_items)
    block = build_features(split, ml, "val", users, candidates, u_vec, i_vec, pop)
    assert block.matrix.shape == (9, len(FEATURES))
    np.testing.assert_array_equal(block.labels.reshape(3, 3).argmax(1), [0, 0, 1])
    col = {name: block.matrix[:, i].reshape(3, 3) for i, name in enumerate(FEATURES)}
    np.testing.assert_array_equal(col["tt_rank"][0], [0, 1, 2])
    np.testing.assert_array_equal(col["user_activity"][:, 0], [2, 2, 1])
    np.testing.assert_allclose(col["tt_score"][0, 0], u_vec[0] @ i_vec[3], rtol=1e-6)
    # No future events: each user's popularity equals a count over events before *their* time.
    for row, user in enumerate(users):
        t = split.val_time[user]
        before = split.train_times < t
        expected = [int(((split.train_items == c) & before).sum()) for c in candidates[row]]
        np.testing.assert_array_equal(col["pop_before"][row], expected)
    with pytest.raises(ValueError, match="padding"):
        build_features(split, ml, "val", users, np.full((3, 3), -1), u_vec, i_vec, pop)


def test_lambda_ranker_learns_an_informative_feature() -> None:
    from recsys.ranker import FeatureBlock

    rng = np.random.default_rng(0)
    n_users, n_cand = 300, 20
    matrix = rng.normal(size=(n_users * n_cand, len(FEATURES))).astype(np.float32)
    labels = np.zeros((n_users, n_cand), dtype=np.int32)
    labels[np.arange(n_users), rng.integers(0, n_cand, n_users)] = 1
    matrix[:, 2] += 3.0 * labels.ravel()  # pop_before reveals the positive
    for c in (7, 8, 9):  # categorical columns must hold small non-negative integers
        matrix[:, c] = rng.integers(0, 3, size=matrix.shape[0])
    block = FeatureBlock(matrix, labels.ravel(), n_cand)
    ranker = LambdaRanker(RankerConfig(max_rounds=50, early_stopping_rounds=10, min_data_in_leaf=5))
    summary = ranker.fit(block, n_users)
    assert summary["users_with_target_in_candidates"] == n_users
    candidates = np.tile(np.arange(n_cand), (n_users, 1))
    reranked = ranker.rerank(block, candidates)
    top1 = reranked[:, 0] == labels.argmax(axis=1)
    assert top1.mean() > 0.8


def test_demographic_inputs_are_refused_without_demographics(tiny_zip_10m: Path) -> None:
    # MovieLens-10M has no user demographics: models must not silently train on zeros.
    ml = load(tiny_zip_10m)
    split = make_split(ml)
    with pytest.raises(ValueError, match="demographics"):
        Trainer(split, ml, TwoTowerConfig(use_side_features=True))
    Trainer(split, ml, TwoTowerConfig(use_side_features=False, num_threads=1))  # fine
    candidates = np.array([[1, 2], [1, 2]])
    vectors = np.ones((split.n_users, 4), dtype=np.float32)
    items = np.ones((split.n_items, 4), dtype=np.float32)
    pop = PopularityIndex(split.train_items, split.train_times, split.n_items)
    with pytest.raises(ValueError, match="demographics"):
        build_features(split, ml, "test", split.users, candidates, vectors, items, pop)
    no_demographics = tuple(f for f in FEATURES if f not in ("gender", "age", "occupation"))
    block = build_features(
        split, ml, "test", split.users, candidates, vectors, items, pop, no_demographics
    )
    assert block.matrix.shape == (4, len(no_demographics))


def test_feature_blocks_equal_one_call(tiny_zip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # build_features processes users in blocks; any block size must give the same matrix.
    from recsys import ranker

    ml = load(tiny_zip)
    split = make_split(ml)
    candidates = np.array([[3, 5, 2], [5, 2, 3], [3, 2, 0]])
    rng = np.random.default_rng(1)
    u_vec = rng.normal(size=(split.n_users, 4)).astype(np.float32)
    i_vec = rng.normal(size=(split.n_items, 4)).astype(np.float32)
    pop = PopularityIndex(split.train_items, split.train_times, split.n_items)
    args = (split, ml, "val", split.users, candidates, u_vec, i_vec, pop, ranker.ALL_FEATURES)
    whole = build_features(*args)  # type: ignore[arg-type]
    monkeypatch.setattr(ranker, "FEATURE_BLOCK_USERS", 1)
    blocked = build_features(*args)  # type: ignore[arg-type]
    np.testing.assert_array_equal(whole.matrix, blocked.matrix)
    np.testing.assert_array_equal(whole.labels, blocked.labels)
