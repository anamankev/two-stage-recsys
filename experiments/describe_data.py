"""Milestone B1: record dataset provenance and split statistics in ``results/data/``.

MovieLens-1M goes to ``summary.json``; ``--dataset ml-10m`` (milestone B9) writes
``summary_ml10m.json``, without the demographic fields ML-10M does not have.

Usage: ``python experiments/describe_data.py [--dataset ml-10m]``
"""

from __future__ import annotations

import argparse
import datetime as dt
import zipfile
from pathlib import Path

import numpy as np

from recsys.data import (
    AGE_BUCKETS,
    DATASETS,
    POSITIVE_THRESHOLD,
    download,
    get_spec,
    load,
    make_split,
    sha256_file,
)
from recsys.runs import save_json


def star(value: float) -> str:
    """Rating label: ``"4"`` for whole stars, ``"4.5"`` for half stars."""
    return str(int(value)) if float(value).is_integer() else str(float(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="ml-1m", choices=sorted(DATASETS))
    args = parser.parse_args()
    spec = get_spec(args.dataset)
    path = download("data", dataset=args.dataset)
    ml = load(path)
    split = make_split(ml)
    with zipfile.ZipFile(path) as archive:
        n_movies_file = sum(1 for _ in archive.read(f"{spec.folder}/movies.dat").splitlines() if _)
    train_counts = np.diff(split.train_indptr)[split.users]
    positive = ml.rating_value >= POSITIVE_THRESHOLD

    def date(ts: int) -> str:
        return dt.datetime.fromtimestamp(int(ts), dt.UTC).date().isoformat()

    summary = {
        "source": {
            "url": spec.url,
            "size_bytes": Path(path).stat().st_size,
            "sha256": spec.sha256 or sha256_file(path),
        },
        "users": ml.n_users,
        "movies_in_movies_dat": n_movies_file,
        "catalog_items_with_a_rating": ml.n_items,
        "ratings": int(ml.rating_user.shape[0]),
        "rating_distribution": {
            star(r): int((ml.rating_value == r).sum()) for r in np.unique(ml.rating_value)
        },
        "positives_rating_ge_4": int(positive.sum()),
        "positive_share": float(positive.mean()),
        "density_positives": float(positive.sum() / (ml.n_users * ml.n_items)),
    }
    if ml.demographics:  # ML-1M only; keeps the original key order of summary.json
        summary["users_male_share"] = float(ml.gender.mean())
        summary["users_by_age_bucket"] = {
            str(code): int((ml.age == i).sum()) for i, code in enumerate(AGE_BUCKETS)
        }
    summary |= {
        "first_rating_date": date(int(ml.rating_time.min())),
        "last_rating_date": date(int(ml.rating_time.max())),
        "split": {
            "kept_users": int(split.users.shape[0]),
            "dropped_users_fewer_than_3_positives": int(ml.n_users - split.users.shape[0]),
            "train_positives": int(split.train_items.shape[0]),
            "validation_targets": int(split.users.shape[0]),
            "test_targets": int(split.users.shape[0]),
            "train_positives_per_user": {
                "min": int(train_counts.min()),
                "median": float(np.median(train_counts)),
                "mean": float(train_counts.mean()),
                "max": int(train_counts.max()),
            },
            "distinct_test_items": int(np.unique(split.test_item[split.users]).shape[0]),
            "test_items_never_seen_in_training": int(
                (~np.isin(split.test_item[split.users], split.train_items)).sum()
            ),
        },
    }
    out = Path("results/data")
    out.mkdir(parents=True, exist_ok=True)
    name = (
        "summary.json" if args.dataset == "ml-1m" else f"summary_{spec.name.replace('-', '')}.json"
    )
    save_json(out / name, summary)
    print(summary)


if __name__ == "__main__":
    main()
