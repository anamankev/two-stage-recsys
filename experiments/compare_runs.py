"""Check that two run directories hold bit-identical arrays (used after refactors).

Every ``.npy`` and ``.npz`` file present in both directories is compared exactly (dtype, shape
and every value). The report lists each file with its verdict and is written as JSON.

Usage::

    python experiments/compare_runs.py results/baselines/main results/baselines/check \
        --out results/verification/baselines.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np


def arrays(path: Path) -> dict[str, np.ndarray]:
    """Named arrays of one ``.npy`` or ``.npz`` file."""
    if path.suffix == ".npy":
        return {"": np.load(path, allow_pickle=False)}
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def same(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    """Exact equality of every named array, including dtype and shape."""
    return a.keys() == b.keys() and all(
        a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and np.array_equal(a[k], b[k])
        for k in a
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    names = sorted(
        p.name
        for p in args.first.iterdir()
        if p.suffix in (".npy", ".npz") and (args.second / p.name).exists()
    )
    if not names:
        raise SystemExit("no array files in common")
    files = {name: same(arrays(args.first / name), arrays(args.second / name)) for name in names}
    report: dict[str, Any] = {
        "first": str(args.first),
        "second": str(args.second),
        "checked_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "files": files,
        "all_identical": all(files.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["all_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
