# Model card: two-stage MovieLens recommender

Format after Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019). Numbers come from
`results/final/results.json`, `results/analysis/analysis.json` and `results/serving/load_test.json`.

## Model details

* **What it is.** A two-stage top-K recommender: a two-tower retrieval model (PyTorch; user tower:
  user id, mean of the latest 50 liked movies, gender, age bucket, occupation; item tower: movie id,
  genres, release-year bucket; 64-dimensional L2-normalised outputs) retrieves 200 candidates
  through an HNSW index (hnsw-engine); a LightGBM LambdaRank model with 14 features reorders them.
* **Training.** In-batch softmax with log-Q correction and accidental-hit masking, early stopping on
  validation Recall@100; ranker trained on validation targets. Deterministic given the seed.
* **Version.** 0.1.0, October 2026. Code and configs in this repository; every run is saved with
  its configuration and environment.

## Intended use

* Demonstrating and studying a production-style two-stage recommender architecture: offline
  evaluation methodology, candidate retrieval with approximate nearest neighbour search, learning
  to rank, and serving latency.
* **Not intended** for recommending to real people: the data are 20 years old, small, and
  collected under a research licence (non-commercial use).

## Data

MovieLens-1M (GroupLens): 1,000,209 ratings from 6,040 users on 3,706 movies, collected
2000-2003 (`docs/DATASETS.md`). Ratings of 4-5 are treated as positive implicit feedback. Per-user
temporal split: last positive for test, second-to-last for validation, rest for training; 6,035
users kept.

## Evaluation

Full ranking over all 3,706 movies with each user's history excluded (no sampled negatives),
6,035 test users, 95% bootstrap intervals over users.

| Metric (test) | Two-tower + ranker | Tuned implicit ALS | Popularity |
|---|---:|---:|---:|
| Recall@10 | 0.1692 [0.1597, 0.1781] | 0.0886 [0.0817, 0.0961] | 0.0393 [0.0345, 0.0441] |
| NDCG@10 | 0.0906 [0.0848, 0.0959] | 0.0445 [0.0409, 0.0489] | 0.0193 [0.0167, 0.0222] |
| MRR | 0.0806 [0.0757, 0.0852] | 0.0439 [0.0409, 0.0475] | 0.0205 [0.0184, 0.0229] |
| Coverage@10 (point) | 0.5774 | 0.2178 | 0.0310 |

Subgroup results (history length, popular versus long-tail targets) are in `docs/ANALYSIS.md`.

**Scale check.** The same configuration trained on MovieLens-10M (no demographic inputs, which
that release lacks) reaches NDCG@10 0.1047 [0.1030, 0.1066] on 69,613 test users and 10,677 movies,
against 0.0524 for tuned implicit ALS (`docs/RESULTS.md`). The service in this repository serves
the MovieLens-1M model.

## Serving

<!-- table:serving -->
| Measure | Sequential (1 client) | Concurrent (4 clients) |
|---|---:|---:|
| Throughput (requests/s) | 969.5 | 1220.4 |
| p50 latency (ms) | 1.01 | 3.25 |
| p95 latency (ms) | 1.11 | 3.98 |
| p99 latency (ms) | 1.17 | 4.33 |

Server-side mean per request (sequential): user embedding 0.06 ms, retrieval 0.13 ms, ranking 0.37 ms, total 0.56 ms. 2,000 requests per mode for random users, one uvicorn worker, single-threaded numerics, client on the same machine (`results/serving/load_test.json`).
<!-- /table:serving -->

**In Docker.** The image (`Dockerfile`, built from the workspace root) runs the same service with
the data and trained artifacts mounted read-only. `experiments/docker_smoke.py` builds and starts
it and checks its answers (`results/serving/docker_smoke.json`); `load_test.py --url` then
measures it against the native service under the same conditions:

<!-- table:serving_docker -->
| Measure | Native (host) | Docker container |
|---|---:|---:|
| Sequential throughput (requests/s) | 520.5 | 399.2 |
| Sequential p50 latency (ms) | 1.88 | 2.39 |
| Sequential p95 latency (ms) | 2.09 | 2.84 |
| Sequential p99 latency (ms) | 2.21 | 3.53 |
| Concurrent throughput, 4 clients (requests/s) | 656.5 | 385.1 |
| Concurrent p99 latency, 4 clients (ms) | 7.86 | 22.86 |
| Server-side total per request, mean (ms) | 1.12 | 1.37 |

Both runs back to back on the same machine, 2,000 requests per mode, client on the host. Native: Battery Power, Low Power Mode on, load 2.1 (`results/serving/load_test_native_compare.json`). Docker: Battery Power, Low Power Mode on, load 2.0 (`results/serving/load_test_docker.json`).
<!-- /table:serving_docker -->

The container's answers are the native service's answers: for five users (1, 42, 1,000, 3,000 and
6,040) it returned the same ten movies in the same order with identical scores (largest
difference 0.0); an unknown user gets 404, `k` outside 1-100 gets 422, and Docker's own health
check reports the container healthy. The first build took 399.5 s including downloads
(`results/serving/docker_smoke_first_build.json`); the image is 1,997.8 MB and the service answers
8.3-8.4 s after `docker run`. Inside Docker Desktop's virtual machine the service is slower, most of
all with concurrent clients. Server-side time per request rises from 1.12 to 1.37 ms, less than
the client-side p50 (1.88 to 2.39 ms), so part of the extra cost is moving requests into and out of
the virtual machine; with four clients the gap widens (p99 7.86 to 22.86 ms). Both columns ran
with macOS Low Power Mode on, so both are slower than the table above (Low Power Mode off):
compare them with each other.

## Limitations

* **Protocol.** Per-user leave-last-out is not a global time split: models train on other users'
  interactions that happened after a given user's test moment, which can make absolute numbers
  optimistic. Validation targets are used both for the two-tower model's early stopping and as
  the ranker's training labels.
* **Scale.** 3,706 movies; at this size exact retrieval is faster than the ANN index, which is used
  to demonstrate the serving path. Results on one dataset, offline only; no online test.
* **Cold start.** Users need at least three positives to be evaluated; brand-new users and movies
  rely only on side features and were not evaluated separately.

## Known biases and ethical considerations

* **Demographic features.** Gender, age bucket and occupation are model inputs (as the brief
  requires). In the trained ranker, gender and age received zero split gain, occupation some; the
  two-tower model's use of them was not audited per group. Using protected attributes in
  recommendations can reproduce stereotypes and should be justified and audited in any real use.
* **Popularity bias.** Recommenders tend to over-recommend popular items. Coverage and the Gini
  coefficient of recommendation counts per method are reported in `docs/ANALYSIS.md`; the two-tower
  model's log-Q correction is designed to reduce this bias.
* **Population.** MovieLens users of 2000-2003 who chose to rate movies, about 72% of them men; the
  data do not represent today's audiences.
