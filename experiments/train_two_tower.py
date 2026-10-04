"""Milestone B3: train the two-tower retrieval model and evaluate exact full-catalog retrieval.

Saves under ``results/two_tower/<run>/``: the model weights, the training log, user and item
embeddings for both stages (used by retrieval, the ranker and the service), per-user rankings and
metrics with bootstrap intervals.

Usage: ``python experiments/train_two_tower.py --config configs/two_tower.yaml --name main``
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from recsys.config import build, load_config
from recsys.data import load_dataset, make_split
from recsys.evaluation import bootstrap
from recsys.runs import save_json, save_ranking, start_run
from recsys.two_tower import Trainer, TwoTowerConfig, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/two_tower.yaml")
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override model config values, e.g. negatives=uniform seed=1",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    for item in args.set:
        key, value = item.split("=", 1)
        config["model"][key] = (
            type(config["model"][key])(value)
            if not isinstance(config["model"][key], bool)
            else value.lower() == "true"
        )
    model_config = build(TwoTowerConfig, config["model"])
    run_dir = start_run("two_tower", config, name=args.name)
    ml = load_dataset(config["data"])
    split = make_split(ml, min_positives=config["data"]["min_positives"])

    t0 = time.perf_counter()
    trainer = Trainer(split, ml, model_config)
    model = trainer.fit()
    train_seconds = time.perf_counter() - t0

    ev = config["evaluation"]
    outputs = {}
    for stage in ("val", "test"):
        result = evaluate(model, split, stage, k_max=ev["k_max"])
        save_ranking(run_dir / f"two_tower_{stage}.npz", result.result, split.users)
        np.save(run_dir / f"user_vectors_{stage}.npy", result.user_vectors)
        outputs[stage] = result
    np.save(run_dir / "item_vectors.npy", outputs["test"].item_vectors)
    torch.save(model.state_dict(), run_dir / "model.pt")
    test = outputs["test"].result
    metrics = {
        "train_seconds": train_seconds,
        "epochs_run": len(trainer.log),
        "best_epoch": int(np.argmax([e[f"recall@{model_config.eval_k}"] for e in trainer.log]) + 1),
        "final_temperature": model.temperature().item(),
        "val": outputs["val"].result.metrics(),
        "val_recall@100": float((outputs["val"].result.ranks <= 100).mean()),
        "test": bootstrap(test, n_resamples=ev["bootstrap_resamples"], seed=ev["seed"]).as_dict(),
        "test_recall@100": float((test.ranks <= 100).mean()),
        "test_recall@200": float((test.ranks <= 200).mean()),
    }
    save_json(run_dir / "training_log.json", trainer.log)
    save_json(run_dir / "metrics.json", metrics)
    print({k: round(v, 4) for k, v in test.metrics().items()}, "train s", round(train_seconds))
    print("results in", run_dir)


if __name__ == "__main__":
    main()
