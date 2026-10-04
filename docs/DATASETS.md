# Dataset: MovieLens-1M

All numbers below come from `results/data/summary.json`, produced by
`experiments/describe_data.py`.

## Provenance

| | |
|---|---|
| Source | GroupLens Research, https://files.grouplens.org/datasets/movielens/ml-1m.zip |
| Downloaded | 2026-10-02 |
| Size | 5,917,549 bytes |
| SHA-256 | `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20` |
| Licence | GroupLens research licence (non-commercial research use; see the archive's README) |

`recsys.data.download` refuses a file with a different checksum.

## Contents

| Quantity | Value |
|---|---:|
| Users (all with gender, age bucket, occupation) | 6,040 |
| Movies listed in `movies.dat` | 3,883 |
| Movies with at least one rating (the catalog) | 3,706 |
| Ratings | 1,000,209 |
| Ratings 1 / 2 / 3 / 4 / 5 | 56,174 / 107,557 / 261,197 / 348,971 / 226,310 |
| Positives (rating 4 or 5) | 575,281 (57.5% of ratings) |
| Density of positives in the user x catalog matrix | 2.57% |
| Rating dates | 2000-04-25 to 2003-02-28 |

## Split (per-user, leave-last-out, by `(timestamp, movie id)`)

| Quantity | Value |
|---|---:|
| Users kept (at least 3 positives) | 6,035 |
| Users dropped (fewer than 3 positives) | 5 |
| Training positives | 563,206 |
| Validation targets / test targets | 6,035 / 6,035 |
| Training positives per user: min / median / mean / max | 2 / 56 / 93.3 / 1,433 |
| Distinct test items | 1,825 |
| Test items that never appear in training | 4 |

Side information used by the models: gender (2 values), age bucket (7), occupation (21), movie
genres (18, multi-hot), release year (from the title, bucketed in 5-year bins).

## Reproduction

```bash
python experiments/describe_data.py
```

## Scale check: MovieLens-10M (Tier 3, milestone B9)

From `results/data/summary_ml10m.json` (`python experiments/describe_data.py --dataset ml-10m`).

| | |
|---|---|
| Source | https://files.grouplens.org/datasets/movielens/ml-10m.zip (65,566,137 bytes) |
| SHA-256 (observed on first download, 2026-10-03; pinned in `data.py`) | `813c411ccb6122564edfe752e7f80c4dcc5aa25fa94c93622f6877a7ba252862` |
| Licence | GroupLens research licence (non-commercial use, citation required) |
| Users / movies with a rating / ratings | 69,878 / 10,677 / 10,000,054 |
| Ratings | half stars, 0.5 to 5.0 |
| Positives (4.0 stars or more) | 5,005,684 (50.1% of ratings) |
| Positive density | 0.67% (2.57% for ML-1M) |
| Rating dates | 1995-01-09 to 2009-01-05 |
| Users kept / dropped (fewer than 3 positives) | 69,613 / 265 |
| Training positives | 4,866,172 |
| Training positives per user: min / median / mean / max | 1 / 36 / 69.9 / 2,791 |
| Distinct test items / test items never seen in training | 5,849 / 24 |

ML-10M has no `users.dat`, so there is no gender, age or occupation. Genres follow ML-1M's 18 after
mapping "Children" to "Children's"; "IMAX" and "(no genres listed)" are ignored.

