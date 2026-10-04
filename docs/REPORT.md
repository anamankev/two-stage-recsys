# Technical report: a two-stage recommender served from a from-scratch vector index

## Abstract

We build a two-stage movie recommender in the architecture used by large production systems and
evaluate it on MovieLens-1M with full-catalog ranking, a leakage-tested per-user temporal split and
bootstrap confidence intervals. A two-tower model (PyTorch) retrieves 200 candidates per user
through hnsw-engine, our own HNSW implementation; a LightGBM LambdaRank model reorders them. The
full system reaches NDCG@10 0.0906 [0.0848, 0.0959] on 6,035 test users against 0.0445 for a
carefully tuned implicit-ALS baseline (paired difference +0.0461 [+0.0407, +0.0513]). Approximate
retrieval costs 0.4 points of candidate recall (Recall@200 0.6131 vs 0.6171). A first ranker built
on the brief's feature list scored *below* plain retrieval order; diagnosing it with validation
data only showed that the two-tower user vector lacks recency, and four recency features turned the
ranker into a +0.0287 NDCG@10 [+0.0243, +0.0332] improvement. On MovieLens-10M, ten times larger,
the same settings reach 0.1047 against 0.0524 for re-tuned ALS.

## 1. Problem

Given a user's history of liked movies, rank the whole catalog so that the next liked movie
appears near the top. Production systems split this into *retrieval* (cheap, recall-oriented,
millions of items to hundreds) and *ranking* (expensive, precision-oriented, hundreds to ten). We
reproduce that design at laboratory scale, with the evaluation discipline of a research paper.

## 2. Data and protocol

MovieLens-1M (1,000,209 ratings, 6,040 users, 3,706 rated movies, 2000-2003). Ratings of 4-5 are
positives. Each user's positives are ordered by `(timestamp, movie id)`; the last is the test
target, the second-to-last the validation target, the rest training data; users with fewer than
three positives are dropped (5 of 6,040). Every method ranks all 3,706 movies, excluding the
user's earlier positives, and is scored with Recall@10/@50, NDCG@10, MRR and catalog coverage@10,
with 95% percentile bootstrap intervals over users (1,000 resamples) and paired bootstrap
differences. Sampled-negative evaluation is avoided because it distorts comparisons (Krichene and
Rendle, KDD 2020). Details: `DESIGN.md`, `DATASETS.md`.

**Leakage controls, each enforced by tests.** No target in its user's training data; a model's
user-history input contains only earlier positives (also for every training example); popularity
features count only training events strictly before the target's timestamp (checked against a
brute-force count for every value). Hyperparameters and early stopping use validation targets
only; the final configuration of each method is evaluated on test once.

## 3. Methods

* **Baselines.** Popularity; item-item co-occurrence with cosine normalisation; implicit ALS (Hu,
  Koren and Volinsky, 2008) written in NumPy and tuned in three grid passes on validation NDCG@10
  (128 factors, regularisation 30, alpha 1).
* **Two-tower retrieval.** User tower: user id, mean of the latest 50 liked movies, gender, age
  bucket, occupation. Item tower: movie id, genres, release-year bucket. Two-layer MLPs to 64
  L2-normalised dimensions, learned temperature. In-batch softmax with log-Q correction (Yi et al.,
  RecSys 2019) and accidental-hit masking; Adam, batch 1,024; early stopping on validation
  Recall@100 (best epoch 6 of 9; 46.8 s on a laptop CPU).
* **Serving the retrieval.** Item vectors indexed by hnsw-engine (inner product, M 16,
  efConstruction 200, efSearch max(200, k)); each user receives 200 fresh candidates.
* **Ranker.** LightGBM LambdaRank over the 200 candidates with 14 features: two-tower score and
  rank, popularity before the request (all time and last 30 days), release year, movie age, genre
  affinity with the history, user activity, demographics, and cosine similarity of the candidate to
  the user's latest movie and five latest movies. Trained on validation targets (lists that contain
  the target), early stopping on a 10% user hold-out.

## 4. Results

<!-- table:results_main -->
| Method | Recall@10 | Recall@50 | NDCG@10 | MRR | Coverage@10 (point) |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.0393 [0.0345, 0.0441] | 0.1287 [0.1208, 0.1375] | 0.0193 [0.0167, 0.0222] | 0.0205 [0.0184, 0.0229] | 0.0310 |
| Item-item co-occurrence | 0.0547 [0.0489, 0.0603] | 0.1814 [0.1723, 0.1914] | 0.0276 [0.0244, 0.0311] | 0.0283 [0.0256, 0.0311] | 0.0977 |
| Implicit ALS | 0.0886 [0.0817, 0.0961] | 0.2848 [0.2741, 0.2956] | 0.0445 [0.0409, 0.0489] | 0.0439 [0.0409, 0.0475] | 0.2178 |
| Two-tower, exact retrieval | 0.1210 [0.1128, 0.1294] | 0.3319 [0.3200, 0.3433] | 0.0615 [0.0570, 0.0664] | 0.0568 [0.0528, 0.0609] | 0.5013 |
| Two-tower via hnsw-engine (200 candidates) | 0.1215 [0.1133, 0.1301] | 0.3317 [0.3196, 0.3430] | 0.0619 [0.0573, 0.0668] | 0.0571 [0.0532, 0.0612] | 0.5024 |
| Two-tower + LightGBM ranker | 0.1692 [0.1597, 0.1781] | 0.3992 [0.3872, 0.4118] | 0.0906 [0.0848, 0.0959] | 0.0806 [0.0757, 0.0852] | 0.5774 |
<!-- /table:results_main -->

Paired differences and retrieval costs: `RESULTS.md`. Ablations and subgroup analyses:
`ANALYSIS.md`. Serving latency: `MODEL_CARD.md`.

**Scale check on MovieLens-10M (Tier 3).** With every ML-1M setting carried over (no demographic
inputs, which ML-10M lacks; ALS re-tuned on validation in three passes), on 69,613 test users and
all 10,677 movies: tuned ALS 0.0524 [0.0511, 0.0536] NDCG@10, two-tower 0.0808 [0.0792, 0.0824],
two-tower via hnsw-engine 0.0805, full system 0.1047 [0.1030, 0.1066]. Paired: two-tower minus ALS
+0.0285 [+0.0269, +0.0300]; ranker over retrieval order +0.0243 [+0.0230, +0.0255]. The index
keeps 97.7% of the exact top-200 (`RESULTS.md`, MovieLens-10M section).

## 5. Analysis

1. **Two-tower retrieval beats every baseline** (+0.0170 NDCG@10 over tuned ALS, paired interval
   [+0.0122, +0.0217]) and recommends a much wider part of the catalog (coverage 0.5013 vs 0.2178).
2. **Approximate retrieval is almost free here** (98.4% of the exact top-200; Recall@200 0.6131 vs
   0.6171) but not needed for speed at 3,706 items: exact scoring takes 0.046 ms per user against
   0.094 ms for hnsw-engine's compiled engine (identical candidates to the pure-Python engine, which
   takes 1.07 ms) and 0.13 ms for FAISS HNSW.
3. **A ranker needs information retrieval lacks.** The brief's base features lost to retrieval
   order (validation selection set: 0.094-0.104 vs 0.109 NDCG@10; test: 0.057 vs 0.062). Recency
   features supplied what the mean-pooled user vector cannot express, and the ranker then gained
   +0.0287 NDCG@10 on test.
4. **The ceiling is retrieval.** The target is among the 200 candidates for 61.3% of users; no
   ranker can do better than that in its top 200.

5. **Ablations** (three seeds each, `ANALYSIS.md`): removing the log-Q correction costs the most
   (test Recall@10 0.0914 ± 0.0036 versus 0.1218 ± 0.0018) while pushing coverage to 0.9552, the
   popularity over-correction the method exists to prevent; uniform sampled negatives win on this
   small catalog (0.1284 ± 0.0039, also best on validation), which we report rather than adopt after
   the fact; side features change nothing measurable; 32-dimensional embeddings are worse.
6. **Who benefits.** Accuracy falls with history length for every method, but the ranker's relative
   gain grows with it (+19% for users with 3-20 positives, +73% beyond 250), consistent with recency
   features compensating for the diluted mean-pooled user vector. The full system recalls long-tail
   targets about 40 times as often as ALS (0.0759 versus 0.0018) while still favouring popular
   movies.
7. **Serving.** A FastAPI service with one worker answers in 0.56 ms server-side on average (user
   embedding 0.06 ms, retrieval 0.13 ms, ranking 0.37 ms) and sustains 969.5 requests/s from one
   client with p99 latency 1.17 ms over HTTP (`MODEL_CARD.md`). The first load test exposed
   features computed for every user on every request; the fix was verified bit-identical.

## 6. Limitations

* Per-user leave-last-out lets models see other users' later interactions; validation targets
  serve both two-tower early stopping and ranker training (`DESIGN.md`).
* One dataset family: MovieLens-1M (3,706 movies) for every analysis, MovieLens-10M (10,677
  movies) as a scale check of the main table only; offline evaluation only, CPU only.
* The ranker was evaluated on test twice (first attempt and final configuration); both are
  reported.

## 7. Future work

Sequential user encoders (self-attention over the history) to capture recency inside retrieval;
more candidates or a second retrieval source to raise the 61% ceiling; a global temporal split; a
catalog large enough to make approximate retrieval necessary rather than illustrative (at
MovieLens-10M's 10,677 movies exact search is still as fast).

## References

* Hu, Koren, Volinsky. Collaborative Filtering for Implicit Feedback Datasets. ICDM 2008.
* Yi et al. Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations. RecSys 2019.
* Krichene, Rendle. On Sampled Metrics for Item Recommendation. KDD 2020.
* Burges. From RankNet to LambdaRank to LambdaMART: An Overview. Microsoft Research, 2010.
* Ji et al. A Critical Study on Data Leakage in Recommender System Offline Evaluation. TOIS 2023.
* Malkov, Yashunin. Efficient and robust approximate nearest neighbor search using Hierarchical
  Navigable Small World graphs. arXiv:1603.09320.
