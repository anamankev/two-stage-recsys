"""Milestone B8 follow-up: build the service image, run it, and check what it answers.

Steps, all recorded in ``results/serving/docker_smoke.json`` with the Docker versions:

1. ``docker build`` from the workspace root (timed; full log in ``docker_build.log``), then the
   image size;
2. ``docker run`` with ``data/`` and ``results/`` mounted read-only, timed until ``/health``
   answers, and the container's own health check;
3. ``/recommend`` for a few users, compared item by item with the same artifacts served natively
   in this process (``load_recommender``), and the error cases (unknown user 404, ``k`` outside
   1-100 422);
4. the package versions installed in the image (``pip freeze``).

The container keeps running (named ``two-stage-recsys``) so that
``load_test.py --url http://127.0.0.1:8000`` can measure it; stop it with
``docker stop two-stage-recsys``.

Usage: ``python experiments/docker_smoke.py [--skip-build]``
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from hnsw_engine.bench.env import capture_environment

from recsys.runs import EXTRA_PACKAGES, save_json

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
TAG = "two-stage-recsys:0.1.0"
NAME = "two-stage-recsys"
URL = "http://127.0.0.1:8000"
USERS = (1, 42, 1000, 3000, 6040)
OUT = PROJECT / "results" / "serving"


def docker_binary() -> str:
    """The Docker CLI: on PATH, or where Docker Desktop installs it."""
    for candidate in (
        shutil.which("docker"),
        str(Path.home() / ".docker" / "bin" / "docker"),
        "/Applications/Docker.app/Contents/Resources/bin/docker",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("docker CLI not found; is Docker Desktop installed and running?")


def docker_env(docker: str) -> dict[str, str]:
    """The environment with the CLI's folder on PATH: Docker Desktop's credential helper
    (``docker-credential-desktop``) lives next to the ``docker`` binary."""
    return {**os.environ, "PATH": f"{Path(docker).parent}{os.pathsep}{os.environ['PATH']}"}


def run(*args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    env = docker_env(args[0]) if Path(args[0]).name == "docker" else None
    return subprocess.run(args, capture_output=True, text=True, check=True, env=env, **kwargs)


def wait_healthy(timeout: float = 300.0) -> float:
    """Seconds until ``/health`` answers 200."""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        try:
            if httpx.get(f"{URL}/health", timeout=2.0).status_code == 200:
                return time.perf_counter() - start
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    raise SystemExit("the container did not become healthy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-build", action="store_true", help="reuse the existing image")
    args = parser.parse_args()
    docker = docker_binary()
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "image": TAG,
        "docker_version": json.loads(run(docker, "version", "--format", "{{json .}}").stdout),
    }

    if not args.skip_build:
        start = time.perf_counter()
        build = subprocess.run(
            [docker, "build", "-f", "two-stage-recsys/Dockerfile", "-t", TAG, "."],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            env=docker_env(docker),
        )
        seconds = time.perf_counter() - start
        (OUT / "docker_build.log").write_text(build.stdout + build.stderr)
        if build.returncode != 0:
            save_json(OUT / "docker_smoke.json", {**report, "build_failed": True})
            raise SystemExit("docker build failed; see results/serving/docker_build.log")
        save_json(
            OUT / "docker_build.json",
            {"image": TAG, "build_seconds": seconds, "environment": capture_environment()},
        )
    if (OUT / "docker_build.json").exists():  # kept across --skip-build runs
        report["build"] = json.loads((OUT / "docker_build.json").read_text())
    report["image_size_bytes"] = int(
        run(docker, "image", "inspect", TAG, "--format", "{{.Size}}").stdout.strip()
    )

    subprocess.run(
        [docker, "rm", "-f", NAME], capture_output=True, text=True, env=docker_env(docker)
    )
    run(
        docker,
        "run",
        "-d",
        "--name",
        NAME,
        "-p",
        "8000:8000",
        "-v",
        f"{PROJECT / 'data'}:/app/two-stage-recsys/data:ro",
        "-v",
        f"{PROJECT / 'results'}:/app/two-stage-recsys/results:ro",
        TAG,
    )
    report["startup_seconds"] = wait_healthy()
    health = httpx.get(f"{URL}/health", timeout=5.0).json()
    report["health"] = health

    from recsys.serving.app import load_recommender  # PyTorch is imported only here

    os.chdir(PROJECT)
    native = load_recommender()
    checks = []
    for user in USERS:
        response = httpx.get(f"{URL}/recommend/{user}", params={"k": 10}, timeout=30.0)
        response.raise_for_status()
        served = response.json()["items"]
        local = native.recommend(user, 10)["items"]
        checks.append(
            {
                "user_id": user,
                "status": response.status_code,
                "movie_ids_container": [item["movie_id"] for item in served],
                "movie_ids_native": [item["movie_id"] for item in local],
                "same_movies_same_order": [i["movie_id"] for i in served]
                == [i["movie_id"] for i in local],
                "max_abs_score_difference": max(
                    abs(a["score"] - b["score"]) for a, b in zip(served, local, strict=True)
                ),
                "first_title": served[0]["title"],
            }
        )
    report["recommendations"] = checks
    report["errors"] = {
        "unknown_user_status": httpx.get(f"{URL}/recommend/999999", timeout=10.0).status_code,
        "k_0_status": httpx.get(f"{URL}/recommend/1", params={"k": 0}, timeout=10.0).status_code,
        "k_101_status": httpx.get(
            f"{URL}/recommend/1", params={"k": 101}, timeout=10.0
        ).status_code,
    }
    time.sleep(12)  # let Docker's own health check (every 10 s) run at least once
    report["container_health"] = run(
        docker, "inspect", "--format", "{{.State.Health.Status}}", NAME
    ).stdout.strip()
    report["image_packages"] = run(
        docker, "run", "--rm", "--entrypoint", "pip", TAG, "freeze"
    ).stdout.splitlines()
    report["environment"] = capture_environment(EXTRA_PACKAGES)
    save_json(OUT / "docker_smoke.json", report)
    summary = {k: v for k, v in report.items() if k not in ("docker_version", "image_packages")}
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
