"""Milestone B2: evaluate popularity, item-item co-occurrence and implicit ALS.

ALS hyperparameters are chosen by validation NDCG@10; the chosen model is evaluated once on the
test targets. Saves metrics, bootstrap intervals and per-user rankings under
``results/baselines/<run>/``.

Usage: ``python experiments/run_baselines.py --config configs/baselines.yaml``
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Any

import numpy as np

from recsys.baselines import ALSConfig, ImplicitALS, ItemCooccurrence, Popularity
from recsys.config import load_config
from recsys.data import Split, load_dataset, make_split
from recsys.evaluation import RankingResult, bootstrap, rank_targets
from recsys.runs import save_json, save_ranking, start_run


def evaluate(model: Any, split: Split, stage: str, k_max: int) -> RankingResult:
    """Rank every evaluated user's ``stage`` target with ``model.score``."""
    items, _ = split.targets(stage)  # type: ignore[arg-type]
    indptr, history = split.history_csr(stage)  # type: ignore[arg-type]
    return rank_targets(
        lambda users: model.score(users, stage),
        split.users,
        items,
        indptr,
        history,
        split.n_items,
        k_max=k_max,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--skip-simple",
        action="store_true",
        help="only the ALS grid (for later tuning passes; popularity and co-occurrence unchanged)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = start_run("baselines", config, name=args.name)
    ev = config["evaluation"]
    split = make_split(load_dataset(config["data"]), min_positives=config["data"]["min_positives"])

    summary: dict[str, Any] = {}

    def record(name: str, model: Any, seconds: float) -> None:
        val = evaluate(model, split, "val", ev["k_max"])
        test = evaluate(model, split, "test", ev["k_max"])
        save_ranking(run_dir / f"{name}_val.npz", val, split.users)
        save_ranking(run_dir / f"{name}_test.npz", test, split.users)
        summary[name] = {
            "fit_seconds": seconds,
            "val": val.metrics(),
            "test": bootstrap(
                test, n_resamples=ev["bootstrap_resamples"], seed=ev["seed"]
            ).as_dict(),
        }
        print(name, {k: round(v, 4) for k, v in test.metrics().items()}, flush=True)

    if not args.skip_simple:
        t0 = time.perf_counter()
        record("popularity", Popularity(split), time.perf_counter() - t0)
        t0 = time.perf_counter()
        record("item_cooccurrence", ItemCooccurrence(split), time.perf_counter() - t0)

    grid = config["als_grid"]
    trials: list[dict[str, Any]] = []
    for factors, reg, alpha in itertools.product(
        grid["factors"], grid["regularization"], grid["alpha"]
    ):
        cfg = ALSConfig(
            factors=factors,
            regularization=reg,
            alpha=alpha,
            iterations=grid["iterations"],
            seed=grid["seed"],
        )
        t0 = time.perf_counter()
        model = ImplicitALS(split, cfg).fit()
        seconds = time.perf_counter() - t0
        val = evaluate(model, split, "val", ev["k_max"]).metrics()
        trials.append({"config": cfg.__dict__, "val": val, "fit_seconds": seconds})
        print("als", cfg.__dict__, "val ndcg@10", round(val["ndcg@10"], 4), flush=True)
    best = max(trials, key=lambda t: t["val"]["ndcg@10"])
    save_json(run_dir / "als_grid.json", trials)
    best_cfg = ALSConfig(**best["config"])
    t0 = time.perf_counter()
    record("implicit_als", ImplicitALS(split, best_cfg).fit(), time.perf_counter() - t0)
    summary["implicit_als"]["selected_config"] = best["config"]
    summary["n_users"] = int(split.users.shape[0])
    summary["n_items"] = split.n_items
    save_json(run_dir / "metrics.json", summary)
    print("results in", run_dir)


if __name__ == "__main__":
    np.seterr(all="raise", under="ignore")
    main()
