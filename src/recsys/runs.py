"""Run bookkeeping: every experiment run gets a folder with its config, environment and outputs."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from hnsw_engine.bench.env import capture_environment

from .evaluation import RankingResult

EXTRA_PACKAGES = ("torch", "lightgbm", "pandas", "pyyaml", "two-stage-recsys")


def start_run(
    experiment: str,
    config: dict[str, Any],
    *,
    results_dir: str | Path = "results",
    name: str | None = None,
) -> Path:
    """Create ``results/<experiment>/<name>/`` and save the resolved config and environment."""
    run_dir = Path(results_dir) / experiment / (name or dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    save_json(run_dir / "environment.json", capture_environment(EXTRA_PACKAGES))
    return run_dir


def save_json(path: str | Path, obj: Any) -> None:
    """Write JSON with stable formatting; NumPy scalars are converted."""
    Path(path).write_text(json.dumps(obj, indent=2, default=_jsonable))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def save_ranking(path: str | Path, result: RankingResult, users: np.ndarray) -> None:
    """Persist per-user ranks and top-K lists for later paired comparisons."""
    np.savez_compressed(
        path, users=users, ranks=result.ranks, top_k=result.top_k, n_items=np.array(result.n_items)
    )


def load_ranking(path: str | Path) -> tuple[np.ndarray, RankingResult]:
    """Inverse of :func:`save_ranking`; returns ``(users, result)``."""
    with np.load(path) as data:
        return data["users"], RankingResult(data["ranks"], data["top_k"], int(data["n_items"]))
