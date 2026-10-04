"""Milestone B6: assemble the final results table from saved per-user rankings.

All intervals are recomputed here from the saved rankings with one seed, so every row is
comparable; paired bootstrap differences use the same resampled users for both methods.

Usage::

    python experiments/make_results.py --baselines results/baselines/main \
        --two-tower results/two_tower/main --retrieval results/retrieval/main \
        --ranker results/ranker/main --inject docs/RESULTS.md --inject README.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from recsys.evaluation import RankingResult, bootstrap, paired_difference
from recsys.runs import load_ranking

METRICS = ("recall@10", "recall@50", "ndcg@10", "mrr", "coverage@10")
LABELS = {
    "recall@10": "Recall@10",
    "recall@50": "Recall@50",
    "ndcg@10": "NDCG@10",
    "mrr": "MRR",
    "coverage@10": "Coverage@10 (point)",
}


def fmt(interval: dict[str, Any]) -> str:
    """``value [low, high]`` with four decimals (just ``value`` when there is no interval)."""
    if interval["low"] is None:
        return f"{interval['value']:.4f}"
    return f"{interval['value']:.4f} [{interval['low']:.4f}, {interval['high']:.4f}]"


def inject(document: Path, name: str, content: str) -> None:
    """Replace the text between ``<!-- table:name -->`` and ``<!-- /table:name -->``."""
    text = document.read_text()
    marker = re.escape(name)
    pattern = re.compile(rf"(<!-- table:{marker} -->).*?(<!-- /table:{marker} -->)", re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"markers for table {name!r} not found in {document}")
    document.write_text(pattern.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}", text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--two-tower", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument(
        "--als-run",
        type=Path,
        default=None,
        help="baselines run holding the selected ALS model, if a later tuning pass (default: "
        "--baselines)",
    )
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/tables"))
    parser.add_argument("--inject", type=Path, action="append", default=[])
    parser.add_argument(
        "--candidate-source",
        default="hnsw-engine",
        help="retriever whose 200 candidates the two-stage rows use (file <name>_test.npz)",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="prefix for table markers and output files, e.g. ml10m (default: none, ML-1M)",
    )
    args = parser.parse_args()
    prefix = f"{args.tag}_" if args.tag else ""

    sources = {
        "Popularity": args.baselines / "popularity_test.npz",
        "Item-item co-occurrence": args.baselines / "item_cooccurrence_test.npz",
        "Implicit ALS": (args.als_run or args.baselines) / "implicit_als_test.npz",
        "Two-tower, exact retrieval": args.two_tower / "two_tower_test.npz",
        "Two-tower via hnsw-engine (200 candidates)": args.retrieval
        / f"{args.candidate_source}_test.npz",
        "Two-tower + LightGBM ranker": args.ranker / "ranker_test.npz",
    }
    rankings: dict[str, RankingResult] = {}
    users_ref = None
    for name, path in sources.items():
        users, result = load_ranking(path)
        if users_ref is not None and not (users == users_ref).all():
            raise ValueError(f"{path} evaluates different users")
        users_ref = users
        rankings[name] = result

    rows: list[dict[str, Any]] = []
    for name, result in rankings.items():
        report: dict[str, Any] = bootstrap(
            result, n_resamples=args.resamples, seed=args.seed
        ).as_dict()
        # Coverage is a whole-population statistic: point value only (see evaluation.bootstrap).
        report["coverage@10"] = {
            "value": result.metrics()["coverage@10"],
            "low": None,
            "high": None,
        }
        rows.append({"method": name, **{m: report[m] for m in METRICS}})

    pairs = [
        ("Two-tower + LightGBM ranker", "Two-tower via hnsw-engine (200 candidates)"),
        ("Two-tower via hnsw-engine (200 candidates)", "Two-tower, exact retrieval"),
        ("Two-tower, exact retrieval", "Implicit ALS"),
        ("Two-tower + LightGBM ranker", "Implicit ALS"),
    ]
    paired = []
    for a, b in pairs:
        for metric in ("ndcg@10", "recall@10"):
            iv = paired_difference(
                rankings[a], rankings[b], metric, n_resamples=args.resamples, seed=args.seed
            )
            paired.append({"comparison": f"{a} minus {b}", "metric": metric, **iv.as_dict()})

    retrieval = json.loads((args.retrieval / "metrics.json").read_text())
    ceilings = []
    for name in ("exact", "hnsw-engine", "hnsw-engine-numba", "faiss-hnsw"):
        if name in retrieval:
            test = retrieval[name]["test"]
            ceilings.append(
                {
                    "retriever": name,
                    "recall@200": test["candidate_recall"]["recall@200"],
                    "overlap_with_exact@200": test["overlap_with_exact@200"],
                    "mean_ms": test["latency"]["mean_ms"],
                    "p99_ms": test["latency"]["p99_ms"],
                }
            )

    header = "| Method | " + " | ".join(LABELS[m] for m in METRICS) + " |"
    rule = "|---|" + "---:|" * len(METRICS)
    lines = [header, rule] + [
        f"| {r['method']} | " + " | ".join(fmt(r[m]) for m in METRICS) + " |" for r in rows
    ]
    main_table = "\n".join(lines)
    paired_lines = ["| Comparison | Metric | Difference [95% CI] |", "|---|---|---:|"] + [
        f"| {p['comparison']} | {p['metric']} | {p['value']:+.4f} [{p['low']:+.4f}, "
        f"{p['high']:+.4f}] |"
        for p in paired
    ]
    ceiling_lines = [
        "| Retriever | Recall@200 (ranking ceiling) | Overlap with exact top-200 | "
        "Mean latency (ms) | p99 latency (ms) |",
        "|---|---:|---:|---:|---:|",
    ] + [
        f"| {c['retriever']} | {c['recall@200']:.4f} | "
        + ("-" if c["overlap_with_exact@200"] is None else f"{c['overlap_with_exact@200']:.4f}")
        + f" | {c['mean_ms']:.4f} | {c['p99_ms']:.4f} |"
        for c in ceilings
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    final = Path("results/final")
    final.mkdir(parents=True, exist_ok=True)
    (final / f"{prefix}results.json").write_text(
        json.dumps(
            {
                "sources": {k: str(v) for k, v in sources.items()},
                "rows": rows,
                "paired": paired,
                "retrieval": ceilings,
                "resamples": args.resamples,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    (args.out / f"{prefix}results_main.md").write_text(main_table + "\n")
    (args.out / f"{prefix}results_paired.md").write_text("\n".join(paired_lines) + "\n")
    (args.out / f"{prefix}results_retrieval.md").write_text("\n".join(ceiling_lines) + "\n")
    with open(args.out / f"{prefix}results_main.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", *[f"{m}_{part}" for m in METRICS for part in ("value", "low", "high")]]
        )
        for r in rows:
            writer.writerow(
                [r["method"], *[r[m][part] for m in METRICS for part in ("value", "low", "high")]]
            )
    for document in args.inject:
        inject(document, f"{prefix}results_main", main_table)
        if "RESULTS" in document.name:
            inject(document, f"{prefix}results_paired", "\n".join(paired_lines))
            inject(document, f"{prefix}results_retrieval", "\n".join(ceiling_lines))
    print(main_table)
    print("\n".join(paired_lines))
    print("\n".join(ceiling_lines))


if __name__ == "__main__":
    main()
