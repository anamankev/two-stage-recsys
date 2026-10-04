"""Milestone B4: serve two-tower retrieval from hnsw-engine and measure what approximation costs.

Loads the item and user embeddings saved by ``train_two_tower.py`` (no PyTorch import: PyTorch
and FAISS must not share a process on macOS, see ``docs/DECISIONS.md``), indexes the items with
exact search, hnsw-engine and FAISS HNSW, retrieves 200 fresh candidates per user for both stages,
and reports candidate recall, overlap with exact retrieval, end-to-end ranking metrics and
per-query latency (one thread). Candidates are saved for the ranker.

Usage: ``python experiments/run_retrieval.py --config configs/retrieval.yaml --name main``
"""

from __future__ import annotations

import argparse
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")  # single-threaded latency measurements

import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from recsys.config import load_config  # noqa: E402
from recsys.data import load_dataset, make_split  # noqa: E402
from recsys.evaluation import bootstrap, rank_with_candidates  # noqa: E402
from recsys.retrieval import (  # noqa: E402
    ExactRetriever,
    FaissHnswRetriever,
    HnswEngineRetriever,
    Retriever,
    candidate_recall,
    overlap,
    retrieve,
)
from recsys.runs import save_json, save_ranking, start_run  # noqa: E402

FACTORIES: dict[str, type[Retriever]] = {
    "exact": ExactRetriever,
    "hnsw-engine": HnswEngineRetriever,
    "hnsw-engine-numba": HnswEngineRetriever,
    "faiss-hnsw": FaissHnswRetriever,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/retrieval.yaml")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = start_run("retrieval", config, name=args.name)
    split = make_split(load_dataset(config["data"]), min_positives=config["data"]["min_positives"])
    tt_dir = Path(config["two_tower_run"])
    items = np.load(tt_dir / "item_vectors.npy")
    n = config["n_candidates"]
    ev = config["evaluation"]

    report: dict[str, Any] = {"n_candidates": n, "two_tower_run": str(tt_dir)}
    exact_candidates = {}
    for name, params in config["retrievers"].items():
        retriever = FACTORIES[name](**params)
        t0 = time.perf_counter()
        retriever.build(items)
        build_seconds = time.perf_counter() - t0
        entry: dict[str, Any] = {"retriever": retriever.describe(), "build_seconds": build_seconds}
        for stage in ("val", "test"):
            users_vec = np.load(tt_dir / f"user_vectors_{stage}.npy")
            indptr, history = split.history_csr(stage)
            targets, _ = split.targets(stage)
            cands = retrieve(retriever, users_vec, split.users, indptr, history, n)
            np.save(run_dir / f"candidates_{name}_{stage}.npy", cands.items)
            if name == "exact":
                exact_candidates[stage] = cands

            def exact_scores(users: np.ndarray, vectors: np.ndarray = users_vec) -> np.ndarray:
                scores: np.ndarray = vectors[users] @ items.T
                return scores

            ranking = rank_with_candidates(
                cands.items,
                exact_scores,
                split.users,
                targets,
                indptr,
                history,
                split.n_items,
                k_max=ev["k_max"],
            )
            save_ranking(run_dir / f"{name}_{stage}.npz", ranking, split.users)
            entry[stage] = {
                "candidate_recall": candidate_recall(cands, targets, tuple(config["recall_ks"])),
                "overlap_with_exact@200": overlap(exact_candidates[stage], cands, n)
                if stage in exact_candidates
                else None,
                "latency": cands.latency_summary(),
                "metrics": ranking.metrics(),
            }
            if stage == "test":
                entry["test_bootstrap"] = bootstrap(
                    ranking, n_resamples=ev["bootstrap_resamples"], seed=ev["seed"]
                ).as_dict()
        report[name] = entry
        print(
            name,
            {k: entry["test"][k] for k in ("candidate_recall", "overlap_with_exact@200")},
            entry["test"]["latency"],
            flush=True,
        )
    save_json(run_dir / "metrics.json", report)
    print("results in", run_dir)


if __name__ == "__main__":
    main()
