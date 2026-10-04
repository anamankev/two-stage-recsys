"""Milestone B7: two-tower ablations, three seeds each.

Variants change one setting of ``configs/two_tower.yaml`` at a time: negative sampling (in-batch,
in-batch with log-Q correction (the default), uniform sampled, mixed), side features off, and the
embedding size (32 and 128 against the default 64). Each variant is trained with seeds 0, 1, 2 and
evaluated by exact full-catalog retrieval on validation and test. Ablations are analysis: they do
not feed back into the main model choice. Writes ``results/ablations/ablations.json``.

Usage: ``python experiments/run_ablations.py``
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from recsys.config import build, load_config
from recsys.data import load_dataset, make_split
from recsys.runs import save_json, start_run
from recsys.two_tower import Trainer, TwoTowerConfig, evaluate

VARIANTS: dict[str, dict[str, Any]] = {
    "in_batch_logq (default)": {},
    "in_batch (no log-Q)": {"negatives": "in_batch"},
    "uniform sampled": {"negatives": "uniform"},
    "mixed (in-batch + uniform)": {"negatives": "mixed"},
    "no side features": {"use_side_features": False},
    "embedding 32": {"embedding_dim": 32},
    "embedding 128": {"embedding_dim": 128},
}
METRICS = ("recall@10", "recall@50", "ndcg@10", "coverage@10")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/two_tower.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--name", default="main")
    args = parser.parse_args()
    config = load_config(args.config)
    base = build(TwoTowerConfig, config["model"])
    run_dir = start_run(
        "ablations", {**config, "variants": VARIANTS, "seeds": args.seeds}, name=args.name
    )
    ml = load_dataset(config["data"])
    split = make_split(ml, min_positives=config["data"]["min_positives"])
    runs = []
    for variant, overrides in VARIANTS.items():
        for seed in args.seeds:
            cfg = replace(base, **overrides, seed=seed)
            t0 = time.perf_counter()
            trainer = Trainer(split, ml, cfg)
            model = trainer.fit(verbose=False)
            seconds = time.perf_counter() - t0
            val = evaluate(model, split, "val", k_max=100).result
            test = evaluate(model, split, "test", k_max=100).result
            runs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "train_seconds": seconds,
                    "epochs": len(trainer.log),
                    "val": val.metrics(),
                    "test": test.metrics(),
                    "val_recall@100": float((val.ranks <= 100).mean()),
                    "test_recall@100": float((test.ranks <= 100).mean()),
                }
            )
            print(
                variant,
                seed,
                {k: round(v, 4) for k, v in test.metrics().items()},
                round(seconds),
                "s",
                flush=True,
            )
    summary = []
    for variant in VARIANTS:
        rows = [r for r in runs if r["variant"] == variant]
        entry: dict[str, Any] = {"variant": variant, "seeds": len(rows)}
        for stage in ("val", "test"):
            for metric in METRICS:
                values = np.array([r[stage][metric] for r in rows])
                entry[f"{stage}_{metric}_mean"] = float(values.mean())
                entry[f"{stage}_{metric}_std"] = float(values.std(ddof=1)) if len(rows) > 1 else 0.0
        summary.append(entry)
    save_json(Path(run_dir) / "ablations.json", {"runs": runs, "summary": summary})
    print("results in", run_dir)


if __name__ == "__main__":
    main()
