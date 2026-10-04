"""Milestone B7: cold-start buckets, popularity bias and qualitative sanity checks.

Reads the saved test rankings of the main runs and writes ``results/analysis/analysis.json``:

* **cold start**: Recall@10 and NDCG@10 by the number of positives a user had before the test
  target;
* **popularity bias**: Recall@10 for test targets among the most popular 20% of movies ("head")
  versus the rest ("tail"), catalog coverage, the Gini coefficient of how often each movie is
  recommended, and the mean popularity percentile of recommended movies;
* **sanity check**: real titles of a few users' latest liked movies next to their top-10.

Usage: ``python experiments/analyze.py``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from recsys.data import download, load, make_split
from recsys.evaluation import RankingResult, ndcg_at_k, recall_at_k
from recsys.runs import load_ranking, save_json

SOURCES = {
    "Popularity": "results/baselines/main/popularity_test.npz",
    "Implicit ALS": "results/baselines/main/implicit_als_test.npz",
    "Two-tower (exact)": "results/two_tower/main/two_tower_test.npz",
    "Two-tower + ranker": "results/ranker/main/ranker_test.npz",
}
BUCKETS = ((3, 20), (21, 50), (51, 100), (101, 250), (251, 10_000))


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of a non-negative count vector (0 = perfectly even, 1 = one item)."""
    values = np.sort(counts.astype(np.float64))
    n = values.shape[0]
    if values.sum() == 0:
        return 0.0
    cumulative = np.cumsum(values)
    return float((n + 1 - 2 * (cumulative / cumulative[-1]).sum()) / n)


def main() -> None:
    ml = load(download("data"))
    split = make_split(ml)
    users = split.users
    targets, _ = split.targets("test")
    indptr, _ = split.history_csr("test")
    history_len = np.diff(indptr)[users]
    train_counts = np.bincount(split.train_items, minlength=split.n_items)
    order = np.argsort(-train_counts, kind="stable")
    head = np.zeros(split.n_items, dtype=bool)
    head[order[: int(0.2 * split.n_items)]] = True
    percentile = np.empty(split.n_items)
    percentile[order] = 1.0 - np.arange(split.n_items) / split.n_items  # 1.0 = most popular
    target_is_head = head[targets]

    report: dict[str, Any] = {
        "head_definition": "top 20% of movies by training positives",
        "head_share_of_training_positives": float(train_counts[head].sum() / train_counts.sum()),
        "test_targets_in_head": float(target_is_head.mean()),
        "buckets": [list(b) for b in BUCKETS],
        "bucket_sizes": [
            int(((history_len >= lo) & (history_len <= hi)).sum()) for lo, hi in BUCKETS
        ],
        "methods": {},
    }
    rankings: dict[str, RankingResult] = {}
    for name, path in SOURCES.items():
        ranked_users, result = load_ranking(path)
        assert (ranked_users == users).all()
        rankings[name] = result
        rec, ndcg = recall_at_k(result.ranks, 10), ndcg_at_k(result.ranks, 10)
        rec_counts = np.bincount(result.top_k[:, :10].ravel(), minlength=split.n_items)
        report["methods"][name] = {
            "cold_start": [
                {"recall@10": float(rec[m].mean()), "ndcg@10": float(ndcg[m].mean())}
                for m in ((history_len >= lo) & (history_len <= hi) for lo, hi in BUCKETS)
            ],
            "recall@10_head_targets": float(rec[target_is_head].mean()),
            "recall@10_tail_targets": float(rec[~target_is_head].mean()),
            "coverage@10": float((rec_counts > 0).mean()),
            "gini_of_recommendation_counts": gini(rec_counts),
            "mean_popularity_percentile_of_recommendations": float(
                percentile[result.top_k[:, :10]].mean()
            ),
        }

    final = rankings["Two-tower + ranker"]
    examples = []
    for user in (users[100], users[2000], users[4000]):
        row = int(np.flatnonzero(users == user)[0])
        history = split.history(int(user), "test")
        examples.append(
            {
                "user_id": int(ml.user_ids[user]),
                "history_size": int(history.shape[0]),
                "latest_liked": [ml.titles[i] for i in history[-10:][::-1]],
                "held_out_test_movie": ml.titles[targets[row]],
                "test_movie_rank": int(final.ranks[row]),
                "top10": [ml.titles[i] for i in final.top_k[row, :10]],
            }
        )
    report["sanity_examples"] = examples
    out = Path("results/analysis")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "analysis.json", report)
    print({k: v for k, v in report.items() if k != "sanity_examples"})


if __name__ == "__main__":
    main()
