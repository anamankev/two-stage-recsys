"""Render ablation and analysis results as Markdown tables in docs/ANALYSIS.md.

Reads ``results/ablations/main/ablations.json`` and ``results/analysis/analysis.json``; each table
replaces the matching ``<!-- table:<name> -->`` section.

Usage: ``python experiments/report_analysis.py --inject docs/ANALYSIS.md``
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def inject(document: Path, name: str, content: str) -> None:
    """Replace the text between ``<!-- table:name -->`` and ``<!-- /table:name -->``."""
    text = document.read_text()
    marker = re.escape(name)
    pattern = re.compile(rf"(<!-- table:{marker} -->).*?(<!-- /table:{marker} -->)", re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"markers for table {name!r} not found in {document}")
    document.write_text(pattern.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}", text))


def table(header: list[str], body: list[list[str]]) -> str:
    """Markdown table with a left-aligned first column."""
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in body])


def pm(entry: dict[str, Any], stage: str, metric: str) -> str:
    """``mean +- std`` over seeds."""
    return f"{entry[f'{stage}_{metric}_mean']:.4f} ± {entry[f'{stage}_{metric}_std']:.4f}"


def ablations(data: dict[str, Any]) -> str:
    body = [
        [
            e["variant"],
            str(e["seeds"]),
            pm(e, "test", "recall@10"),
            pm(e, "test", "ndcg@10"),
            pm(e, "test", "recall@50"),
            pm(e, "test", "coverage@10"),
            pm(e, "val", "ndcg@10"),
        ]
        for e in data["summary"]
    ]
    return table(
        [
            "Variant",
            "Seeds",
            "Test Recall@10",
            "Test NDCG@10",
            "Test Recall@50",
            "Test coverage@10",
            "Validation NDCG@10",
        ],
        body,
    )


def cold_start(data: dict[str, Any]) -> str:
    methods = list(data["methods"])
    header = ["History length (positives before test)", "Users"] + [
        f"Recall@10: {m}" for m in methods
    ]
    body = []
    for i, (lo, hi) in enumerate(data["buckets"]):
        label = f"{lo}+" if hi >= 10_000 else f"{lo}-{hi}"
        body.append(
            [label, f"{data['bucket_sizes'][i]:,}"]
            + [f"{data['methods'][m]['cold_start'][i]['recall@10']:.4f}" for m in methods]
        )
    return table(header, body)


def popularity(data: dict[str, Any]) -> str:
    body = []
    for name, m in data["methods"].items():
        body.append(
            [
                name,
                f"{m['recall@10_head_targets']:.4f}",
                f"{m['recall@10_tail_targets']:.4f}",
                f"{m['coverage@10']:.4f}",
                f"{m['gini_of_recommendation_counts']:.4f}",
                f"{m['mean_popularity_percentile_of_recommendations']:.4f}",
            ]
        )
    return table(
        [
            "Method",
            "Recall@10, head targets",
            "Recall@10, tail targets",
            "Coverage@10",
            "Gini of recommendation counts",
            "Mean popularity percentile of top-10",
        ],
        body,
    )


def sanity(data: dict[str, Any]) -> str:
    return "\n\n".join(
        f"**User {ex['user_id']}** ({ex['history_size']} liked movies; held-out next like: "
        f"*{ex['held_out_test_movie']}*, ranked {ex['test_movie_rank']} by the full system)\n\n"
        + table(
            ["Latest liked (newest first)", "Recommended top 10"],
            [[a, b] for a, b in zip(ex["latest_liked"], ex["top10"], strict=False)],
        )
        for ex in data["sanity_examples"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ablations", type=Path, default=Path("results/ablations/main/ablations.json")
    )
    parser.add_argument("--analysis", type=Path, default=Path("results/analysis/analysis.json"))
    parser.add_argument("--inject", type=Path, action="append", default=[])
    args = parser.parse_args()
    abl = json.loads(args.ablations.read_text())
    ana = json.loads(args.analysis.read_text())
    tables = {
        "ablations": ablations(abl),
        "cold_start": cold_start(ana),
        "popularity": popularity(ana),
        "sanity": sanity(ana),
    }
    for name, markdown in tables.items():
        print(f"## {name}\n{markdown}\n")
        for document in args.inject:
            inject(document, name, markdown)


if __name__ == "__main__":
    main()
