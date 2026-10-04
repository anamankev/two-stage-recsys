"""Milestone B8: load-test the recommendation service over HTTP.

Starts the FastAPI service (one uvicorn worker, single-threaded numerics) in a child process, or
targets a running one with ``--url`` (for example the Docker container), waits until it is
healthy, then measures client-side latency for sequential requests and throughput with concurrent
clients. Writes ``results/serving/load_test.json`` (``--out`` for another name).

Usage: ``python experiments/load_test.py --requests 2000 --concurrency 4``
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from hnsw_engine.bench.env import capture_environment

from recsys.runs import EXTRA_PACKAGES, save_json

SINGLE_THREAD = dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"), "1"
)


def percentiles(ms: np.ndarray) -> dict[str, float]:
    """Mean and tail latencies in milliseconds."""
    return {
        "mean_ms": float(ms.mean()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": float(ms.max()),
    }


def wait_healthy(client: httpx.Client, timeout: float = 180.0) -> dict[str, Any]:
    """Poll /health until the service answers."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = client.get("/health")
            if response.status_code == 200:
                return dict(response.json())
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    raise RuntimeError("service did not become healthy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--url", default=None, help="test this running service instead of starting one"
    )
    parser.add_argument("--out", default="load_test.json", help="file name in results/serving/")
    parser.add_argument("--marker", default="serving", help="table marker in --inject documents")
    parser.add_argument(
        "--setting",
        default="one uvicorn worker, single-threaded numerics, client on the same machine",
        help="how the service ran; recorded and printed under the table",
    )
    parser.add_argument(
        "--inject",
        type=Path,
        action="append",
        default=[],
        help="documents whose <!-- table:serving --> section receives the table",
    )
    args = parser.parse_args()
    base_url = args.url or f"http://127.0.0.1:{args.port}"
    server = None
    if args.url is None:
        server = subprocess.Popen(
            [sys.executable, "-m", "recsys.serving.app", "--port", str(args.port)],
            env={**os.environ, **SINGLE_THREAD},
        )
    try:
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            health = wait_healthy(client)
            rng = np.random.default_rng(args.seed)
            user_ids = rng.integers(1, health["users"] + 1, size=args.requests + args.warmup)
            for uid in user_ids[: args.warmup]:
                client.get(f"/recommend/{uid}")
            client_ms: list[float] = []
            breakdown_ms: dict[str, list[float]] = {
                "embedding": [],
                "retrieval": [],
                "ranking": [],
                "total": [],
            }
            start = time.perf_counter()
            for uid in user_ids[args.warmup :]:
                t0 = time.perf_counter()
                response = client.get(f"/recommend/{uid}")
                client_ms.append(1000 * (time.perf_counter() - t0))
                response.raise_for_status()
                for key, value in response.json()["latency_ms"].items():
                    breakdown_ms[key].append(value)
            sequential_seconds = time.perf_counter() - start

        def worker(ids: np.ndarray) -> list[float]:
            out = []
            with httpx.Client(base_url=base_url, timeout=30.0) as c:
                for uid in ids:
                    t0 = time.perf_counter()
                    c.get(f"/recommend/{uid}").raise_for_status()
                    out.append(1000 * (time.perf_counter() - t0))
            return out

        chunks = np.array_split(user_ids[args.warmup :], args.concurrency)
        start = time.perf_counter()
        with ThreadPoolExecutor(args.concurrency) as pool:
            concurrent_ms = [ms for part in pool.map(worker, chunks) for ms in part]
        concurrent_seconds = time.perf_counter() - start
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=30)

    report = {
        "config": vars(args),
        "sequential": {
            "requests": args.requests,
            "throughput_rps": args.requests / sequential_seconds,
            "client_latency": percentiles(np.array(client_ms)),
            "server_breakdown_mean_ms": {k: float(np.mean(v)) for k, v in breakdown_ms.items()},
            "server_breakdown_p99_ms": {
                k: float(np.percentile(v, 99)) for k, v in breakdown_ms.items()
            },
        },
        "concurrent": {
            "clients": args.concurrency,
            "requests": args.requests,
            "throughput_rps": args.requests / concurrent_seconds,
            "client_latency": percentiles(np.array(concurrent_ms)),
        },
        "environment": capture_environment(EXTRA_PACKAGES),
    }
    out = Path("results/serving")
    out.mkdir(parents=True, exist_ok=True)
    save_json(
        out / args.out,
        {**report, "config": {**vars(args), "inject": [str(d) for d in args.inject]}},
    )
    seq, conc = report["sequential"], report["concurrent"]
    lat, clat = seq["client_latency"], conc["client_latency"]
    brk = seq["server_breakdown_mean_ms"]
    table = "\n".join(
        [
            "| Measure | Sequential (1 client) | Concurrent ("
            + str(conc["clients"])
            + " clients) |",
            "|---|---:|---:|",
            f"| Throughput (requests/s) | {seq['throughput_rps']:.1f} "
            f"| {conc['throughput_rps']:.1f} |",
            f"| p50 latency (ms) | {lat['p50_ms']:.2f} | {clat['p50_ms']:.2f} |",
            f"| p95 latency (ms) | {lat['p95_ms']:.2f} | {clat['p95_ms']:.2f} |",
            f"| p99 latency (ms) | {lat['p99_ms']:.2f} | {clat['p99_ms']:.2f} |",
        ]
    )
    breakdown = (
        f"\nServer-side mean per request (sequential): user embedding {brk['embedding']:.2f} ms, "
        f"retrieval {brk['retrieval']:.2f} ms, ranking {brk['ranking']:.2f} ms, total "
        f"{brk['total']:.2f} ms. {args.requests:,} requests per mode for random users, "
        f"{args.setting} (`results/serving/{args.out}`)."
    )
    for document in args.inject:
        text = document.read_text()
        open_tag, close_tag = f"<!-- table:{args.marker} -->", f"<!-- /table:{args.marker} -->"
        if open_tag not in text or close_tag not in text:
            raise SystemExit(f"serving markers missing in {document}")
        head, rest = text.split(open_tag, 1)
        _, tail = rest.split(close_tag, 1)
        document.write_text(f"{head}{open_tag}\n{table}\n{breakdown}\n{close_tag}{tail}")
    print(table)
    print(breakdown)


if __name__ == "__main__":
    main()
