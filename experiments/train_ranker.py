"""Milestone B5: train the LightGBM LambdaRank ranker and evaluate the full two-stage system.

Training groups are the validation-stage hnsw-engine candidates (label: the validation target);
evaluation re-ranks the test-stage candidates. Items outside the 200 candidates keep their
two-tower order behind them, so every metric is defined over the full catalog.

Usage: ``python experiments/train_ranker.py --config configs/ranker.yaml --name main``
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from recsys.config import build, load_config
from recsys.data import load_dataset, make_split
from recsys.evaluation import bootstrap, rank_with_candidates
from recsys.features import PopularityIndex
from recsys.ranker import LambdaRanker, RankerConfig, build_features
from recsys.runs import save_json, save_ranking, start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/ranker.yaml")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ranker_config = build(RankerConfig, config["ranker"])
    run_dir = start_run("ranker", config, name=args.name)
    ml = load_dataset(config["data"])
    split = make_split(ml, min_positives=config["data"]["min_positives"])
    tt_dir, ret_dir = Path(config["two_tower_run"]), Path(config["retrieval_run"])
    items = np.load(tt_dir / "item_vectors.npy")
    source = config["candidate_source"]
    popularity = PopularityIndex(split.train_items, split.train_times, split.n_items)

    def block(stage: str) -> tuple[np.ndarray, np.ndarray, object]:
        users_vec = np.load(tt_dir / f"user_vectors_{stage}.npy")
        cands = np.load(ret_dir / f"candidates_{source}_{stage}.npy")
        feats = build_features(
            split,
            ml,
            stage,  # type: ignore[arg-type]
            split.users,
            cands,
            users_vec,
            items,
            popularity,
            features=ranker_config.features,
        )
        return users_vec, cands, feats

    t0 = time.perf_counter()
    _, _, val_block = block("val")
    ranker = LambdaRanker(ranker_config)
    summary = ranker.fit(val_block, split.users.shape[0])  # type: ignore[arg-type]
    train_seconds = time.perf_counter() - t0
    ranker.booster.save_model(str(run_dir / "ranker.txt"))

    test_vec, test_cands, test_block = block("test")
    reranked = ranker.rerank(test_block, test_cands)  # type: ignore[arg-type]
    np.save(run_dir / "reranked_test.npy", reranked)
    targets, _ = split.targets("test")
    indptr, history = split.history_csr("test")
    ev = config["evaluation"]
    ranking = rank_with_candidates(
        reranked,
        lambda u: test_vec[u] @ items.T,
        split.users,
        targets,
        indptr,
        history,
        split.n_items,
        k_max=ev["k_max"],
    )
    save_ranking(run_dir / "ranker_test.npz", ranking, split.users)
    metrics = {
        "train_seconds": train_seconds,
        "training": summary,
        "test": bootstrap(
            ranking, n_resamples=ev["bootstrap_resamples"], seed=ev["seed"]
        ).as_dict(),
    }
    save_json(run_dir / "metrics.json", metrics)
    print({k: round(v, 4) for k, v in ranking.metrics().items()})
    print("best iteration", summary["best_iteration"], "results in", run_dir)


if __name__ == "__main__":
    main()
