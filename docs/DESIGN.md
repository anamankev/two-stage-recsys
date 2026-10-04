# two-stage-recsys: design

Status: living document; sections marked *Revised* record where reality differed from the plan.

## 1. Goals

1. A two-stage recommender on MovieLens-1M in the shape used by large production systems: a
   **two-tower retrieval model** proposes 200 candidates from the full catalog via approximate
   nearest neighbour search served by **hnsw-engine** (Project A), then a **learned ranker**
   (LightGBM LambdaRank) orders them.
2. Evaluation a reviewer can trust: full-catalog ranking (no sampled negatives), a leakage-free
   split with tests that prove it, strong baselines, 95% bootstrap confidence intervals, ablations.
3. Engineering standard: typed, linted, tested, configured by versioned files, every run seeded
   and saved with its config and environment, reproducible with one command.

## 2. Non-goals

* State-of-the-art accuracy on MovieLens (sequential transformers would score higher; the point is
  a sound, explainable two-stage system).
* GPU training, distributed training, online learning, A/B testing infrastructure.
* Explicit rating prediction (RMSE). The task is implicit-feedback top-K recommendation.

## 3. Data and split

* **Source:** MovieLens-1M (GroupLens): 1,000,209 ratings by 6,040 users of 3,706 movies; user
  gender, age bucket and occupation; movie title (with year) and genres.
* **Implicit feedback:** a rating of 4 or 5 is a positive interaction. Ratings 1-3 are ignored
  (neither positive nor used as features).
* **Per-user temporal split** (leave-last-out): each user's positives are sorted by
  `(timestamp, movie_id)` (MovieLens timestamps have one-second resolution and many ties; the
  movie id makes the order deterministic). Last positive -> test target; second-to-last ->
  validation target; the rest -> training. Users with fewer than 3 positives are dropped.
* **History at prediction time:** everything the user did *before* the target: training positives
  for the validation target; training positives plus the validation item for the test target.
  Items in that history are excluded from the ranked list.
* **Feature rule:** every feature computed for a (user, target) pair uses only events with a
  timestamp strictly before the target's timestamp. Item popularity, for example, is the number of
  *training* positives the item received before that moment (binary search over per-item sorted
  timestamps), not a global count. Training examples for the two-tower model follow the same rule:
  the user-history input of a training example contains only that user's earlier positives.
* **Known limitation (stated, not hidden):** a per-user split is not a global time split. Models
  are fit on all training interactions, some of which happened after other users' validation or
  test moments. This is the standard leave-last-out protocol the brief prescribes; it can flatter
  absolute numbers slightly (Ji et al., "A Critical Study on Data Leakage in Recommender System
  Offline Evaluation", 2023) but affects every method equally.
* **Second known limitation:** validation targets are used twice, for early stopping of the
  two-tower model and as training labels for the ranker. The two-tower score is therefore slightly
  more reliable on validation than on test, which can make the ranker over-trust it. Test targets
  are never used for training or selection; splitting the validation users between the two uses
  would remove the overlap at the cost of less ranker training data.

## 4. Evaluation

* **Full ranking** over the entire catalog (every movie with at least one rating), excluding the
  user's history items; never sampled negatives (Krichene and Rendle, KDD 2020).
* **Metrics** (one target per user): Recall@10 and @50 (hit rate), NDCG@10, MRR (reciprocal rank
  over the full ranking), catalog coverage@10 (share of the catalog appearing in at least one
  user's top 10). Each metric has a hand-computed unit test.
* **Uncertainty:** 95% percentile bootstrap confidence intervals over users (1,000 resamples, fixed
  seed); paired bootstrap for the key differences (same resampled users for both methods).
* **Two-stage ranking:** the ranker reorders the 200 candidates; the full ranking is the reordered
  candidates followed by the remaining catalog in retrieval order, so every metric (including MRR)
  is defined over the full catalog for every method.

## 5. Models

| Method | Description |
|---|---|
| Popularity | Items by number of training positives (the classic baseline) |
| Item-item co-occurrence | Score = sum over history items of cosine-normalised co-occurrence counts |
| Implicit ALS | Matrix factorisation for implicit feedback (Hu, Koren, Volinsky 2008), NumPy, exact solves |
| Two-tower | User tower: id embedding, mean of history item embeddings, gender/age/occupation embeddings -> MLP. Item tower: id embedding, genre multi-hot, year bucket -> MLP. L2-normalised outputs, learned temperature. In-batch softmax with log-Q sampling-bias correction (Yi et al., RecSys 2019) and accidental-hit masking. Early stopping on validation Recall@100. |
| Two-tower + ranker | LightGBM LambdaRank over the 200 retrieved candidates. Features: two-tower score and rank, time-aware item popularity, movie year, genre affinity with the history, user activity, demographics. Trained on validation targets, evaluated on test targets. |

Hyperparameters of every method are chosen on the validation targets only; test targets are used
once, for the final table.

## 6. Architecture

```
src/recsys/
  config.py      typed configs loaded from configs/*.yaml
  data.py        download (SHA-256), parsing, id mapping, implicit positives, per-user split
  features.py    time-aware features (popularity before t, genre affinity, activity)
  evaluation.py  full-catalog ranking with exclusions, metrics, bootstrap intervals
  baselines.py   popularity, item-item co-occurrence, implicit ALS
  two_tower.py   PyTorch towers, training loop, inference
  retrieval.py   candidate generation: hnsw-engine, FAISS HNSW, exact
  ranker.py      LightGBM LambdaRank training and re-ranking
  serving/       FastAPI app (milestone B8)
experiments/     one script per milestone; each writes results/<run>/ with config and environment
configs/         versioned YAML
```

## 7. Serving (milestone B8)

`GET /recommend/{user_id}` loads the trained towers, the hnsw-engine index of item embeddings and
the ranker once at start-up; per request: user embedding -> ANN retrieval (200) -> features ->
LightGBM scores -> top 10, with a latency breakdown (retrieval, ranking, total).

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Leakage through features or histories | Feature rule above, enforced by tests (no target in history, no feature uses events at or after the target time) |
| Leakage through model selection | Hyperparameters and early stopping use validation targets only; test used once |
| Sampled-metric illusions | Full-catalog ranking for every method |
| Overstated differences | Bootstrap confidence intervals, paired comparisons, three seeds in ablations |
| LightGBM and PyTorch OpenMP conflicts on macOS | LightGBM built without OpenMP (see DECISIONS) |
| CPU training time | Small towers (64-d), batch 1,024, early stopping; time-boxed to about 20 minutes |

## 9. Scale check on MovieLens-10M (Tier 3, milestone B9; designed before implementation)

*Question.* Do the ML-1M conclusions (two-tower beats tuned ALS; the ranker adds a large gain; the
ANN index loses nothing against exact retrieval) hold on a dataset ten times larger?

* **Data.** MovieLens-10M (GroupLens `ml-10m.zip`, 65.6 MB, checksum pinned on first download):
  10,000,054 ratings by 69,878 users of 10,677 movies. Ratings are in half stars; positives are
  ratings of 4.0 or more, the same threshold as ML-1M. Same split protocol and minimum of three
  positives.
* **What differs from ML-1M.** ML-10M has no user demographics, so the user tower runs without
  side features (the ML-1M ablation found they add nothing measurable to retrieval) and the ranker
  drops gender, age and occupation. Genres: ML-10M writes "Children" for ML-1M's "Children's";
  "IMAX" and "(no genres listed)" are not ML-1M genres and are ignored, so the 18 genre features
  mean the same in both datasets.
* **Transfer, not re-tuning.** The two-tower model and the ranker keep their ML-1M settings: the
  check is whether those choices carry over. ALS is the exception: it gets a small grid around its
  ML-1M optimum, chosen on validation data (its regularisation interacts with the number of
  ratings per user). A random subset of validation users for the ranker was planned in case
  training exceeded the 20-minute time box; it was not needed (all 42,808 eligible validation
  users, 409.5 s including feature building).
* **Evaluation.** Unchanged: full-catalog ranking over all 10,677 movies with history exclusion,
  bootstrap intervals over the test users, paired differences.
* **Code.** `data.py` gains a dataset registry (`ml-1m`, `ml-10m`) with URL, checksum, archive
  layout and parser; configs name their dataset (default `ml-1m`, so every ML-1M result is
  unchanged); ML-10M runs write to `results/<stage>/ml10m`.
