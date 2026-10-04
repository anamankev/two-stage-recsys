"""Choose the ranker's features and LightGBM settings using validation data only.

Validation users whose target is among their candidates are split three ways (seeded): a
*selection* set (10%) used only to compare variants, and the rest, from which the ranker carves
its own early-stopping hold-out. Every variant is scored by NDCG@10 of its candidate ordering on
the selection set, next to simple orderings (two-tower rank, popularity) as baselines. Test data
is never touched.

Usage: ``python experiments/tune_ranker.py --config configs/ranker.yaml --name main``
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from recsys.config import build, load_config
from recsys.data import load_dataset, make_split
from recsys.features import PopularityIndex
from recsys.ranker import (
    ALL_FEATURES,
    FEATURES,
    FeatureBlock,
    LambdaRanker,
    RankerConfig,
    build_features,
    holdout_split,
    ndcg_of_ordering,
)
from recsys.runs import save_json, start_run

# Explicit settings, independent of configs/ranker.yaml (which holds the *selected* result, so
# basing variants on it would make a re-run compare the winner against itself).
PARAM_SETS: dict[str, dict[str, Any]] = {
    "default": {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 50,
        "lambda_l2": 1.0,
        "feature_fraction": 0.9,
    },
    "conservative": {
        "num_leaves": 15,
        "learning_rate": 0.03,
        "min_data_in_leaf": 200,
        "lambda_l2": 10.0,
        "feature_fraction": 0.8,
    },
    "very_conservative": {
        "num_leaves": 7,
        "learning_rate": 0.03,
        "min_data_in_leaf": 400,
        "lambda_l2": 10.0,
        "feature_fraction": 0.8,
    },
}
SHARED_KEYS = (
    "max_rounds",
    "bagging_fraction",
    "bagging_freq",
    "early_stopping_rounds",
    "holdout_fraction",
    "seed",
    "num_threads",
)
FEATURE_SETS: dict[str, tuple[str, ...]] = {"base": FEATURES, "base+recency": ALL_FEATURES}


def subset(block: FeatureBlock, rows: np.ndarray) -> FeatureBlock:
    """Rows (users) of a block."""
    n = block.n_candidates
    idx = (rows[:, None] * n + np.arange(n)[None, :]).ravel()
    return FeatureBlock(block.matrix[idx], block.labels[idx], n, block.feature_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/ranker.yaml")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    base_config = build(RankerConfig, {k: config["ranker"][k] for k in SHARED_KEYS})
    run_dir = start_run("ranker_tuning", config, name=args.name)
    ml = load_dataset(config["data"])
    split = make_split(ml, min_positives=config["data"]["min_positives"])
    tt_dir, ret_dir = Path(config["two_tower_run"]), Path(config["retrieval_run"])
    items = np.load(tt_dir / "item_vectors.npy")
    users_vec = np.load(tt_dir / "user_vectors_val.npy")
    cands = np.load(ret_dir / f"candidates_{config['candidate_source']}_val.npy")
    popularity = PopularityIndex(split.train_items, split.train_times, split.n_items)
    block = build_features(
        split, ml, "val", split.users, cands, users_vec, items, popularity, features=ALL_FEATURES
    )

    rest, selection = holdout_split(block.labels, block.n_candidates, 0.1, seed=12345)
    sel_block, rest_block = subset(block, selection), subset(block, rest)
    sel_labels = sel_block.labels.reshape(-1, block.n_candidates)
    col = {
        name: sel_block.matrix[:, i].reshape(-1, block.n_candidates)
        for i, name in enumerate(block.feature_names)
    }
    results: dict[str, Any] = {
        "selection_users": int(selection.shape[0]),
        "training_pool_users": int(rest.shape[0]),
        "baselines": {
            "two_tower_order": ndcg_of_ordering(-col["tt_rank"], sel_labels),
            "popularity_before_t": ndcg_of_ordering(col["pop_before"], sel_labels),
            "genre_affinity": ndcg_of_ordering(col["genre_affinity"], sel_labels),
            "sim_recent5_max": ndcg_of_ordering(col["sim_recent5_max"], sel_labels),
        },
        "variants": [],
    }
    print("baselines", {k: round(v, 4) for k, v in results["baselines"].items()}, flush=True)
    for (fname, features), (pname, params) in itertools.product(
        FEATURE_SETS.items(), PARAM_SETS.items()
    ):
        cfg = replace(base_config, features=features, **params)
        ranker = LambdaRanker(cfg)
        summary = ranker.fit(rest_block, rest.shape[0])
        scores = ranker.scores(sel_block).reshape(-1, block.n_candidates)
        ndcg = ndcg_of_ordering(scores, sel_labels)
        results["variants"].append(
            {
                "features": fname,
                "params": pname,
                "config": cfg.__dict__,
                "selection_ndcg@10": ndcg,
                "best_iteration": summary["best_iteration"],
                "feature_importance_gain": summary["feature_importance_gain"],
            }
        )
        print(
            fname,
            pname,
            "selection ndcg@10",
            round(ndcg, 4),
            "iters",
            summary["best_iteration"],
            flush=True,
        )
    best = max(results["variants"], key=lambda v: v["selection_ndcg@10"])
    results["selected"] = {
        "features": best["features"],
        "params": best["params"],
        "selection_ndcg@10": best["selection_ndcg@10"],
    }
    save_json(run_dir / "tuning.json", results)
    print("selected", results["selected"], "results in", run_dir)


if __name__ == "__main__":
    main()
