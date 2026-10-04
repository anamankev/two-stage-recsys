"""Side-by-side table of two load tests (``load_test.py`` outputs), e.g. native versus Docker.

Reads two JSON files from ``results/serving/``, prints a Markdown table, and with ``--inject``
replaces the ``<!-- table:serving_docker -->`` section of a document. The power state and load
recorded with each run are printed under the table, because laptop timings depend on them.

Usage::

    python experiments/serving_compare.py load_test_native_compare.json load_test_docker.json \
        --inject docs/MODEL_CARD.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SERVING = Path("results/serving")
MARKER = "serving_docker"


def row(label: str, values: list[str]) -> str:
    return f"| {label} | " + " | ".join(values) + " |"


def conditions(report: dict[str, Any]) -> str:
    env = report["environment"]
    power = env.get("power") or {}
    activity = env.get("activity") or {}
    load = activity.get("load_average_1_5_15_min") or [float("nan")]
    low_power = {"1": "on", "0": "off"}.get(str(power.get("low_power_mode")), "unknown")
    return f"{power.get('source', 'unknown')}, Low Power Mode {low_power}, load {load[0]:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("native", help="load test of the service run natively")
    parser.add_argument("docker", help="load test of the service in the container")
    parser.add_argument("--inject", type=Path, action="append", default=[])
    args = parser.parse_args()
    native = json.loads((SERVING / args.native).read_text())
    docker = json.loads((SERVING / args.docker).read_text())
    runs = (native, docker)
    clients = docker["concurrent"]["clients"]
    lines = [
        "| Measure | Native (host) | Docker container |",
        "|---|---:|---:|",
        row(
            "Sequential throughput (requests/s)",
            [f"{r['sequential']['throughput_rps']:.1f}" for r in runs],
        ),
    ]
    lines.extend(
        row(
            f"Sequential {p} latency (ms)",
            [f"{r['sequential']['client_latency'][f'{p}_ms']:.2f}" for r in runs],
        )
        for p in ("p50", "p95", "p99")
    )
    lines.append(
        row(
            f"Concurrent throughput, {clients} clients (requests/s)",
            [f"{r['concurrent']['throughput_rps']:.1f}" for r in runs],
        )
    )
    lines.append(
        row(
            f"Concurrent p99 latency, {clients} clients (ms)",
            [f"{r['concurrent']['client_latency']['p99_ms']:.2f}" for r in runs],
        )
    )
    lines.append(
        row(
            "Server-side total per request, mean (ms)",
            [f"{r['sequential']['server_breakdown_mean_ms']['total']:.2f}" for r in runs],
        )
    )
    note = (
        f"\nBoth runs back to back on the same machine, {docker['sequential']['requests']:,} "
        f"requests per mode, client on the host. Native: {conditions(native)} "
        f"(`results/serving/{args.native}`). Docker: {conditions(docker)} "
        f"(`results/serving/{args.docker}`)."
    )
    table = "\n".join(lines) + "\n" + note
    for document in args.inject:
        text = document.read_text()
        open_tag, close_tag = f"<!-- table:{MARKER} -->", f"<!-- /table:{MARKER} -->"
        if open_tag not in text or close_tag not in text:
            raise SystemExit(f"{MARKER} markers missing in {document}")
        head, rest = text.split(open_tag, 1)
        _, tail = rest.split(close_tag, 1)
        document.write_text(f"{head}{open_tag}\n{table}\n{close_tag}{tail}")
    print(table)


if __name__ == "__main__":
    main()
