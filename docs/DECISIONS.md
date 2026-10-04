# Decision log (two-stage-recsys)

Each entry: context, decision, alternatives considered, consequences. Newest last.

## R1. LightGBM built without OpenMP on macOS

* **Context.** The LightGBM 4.7.0 macOS wheel links against Homebrew's `libomp`, which is not
  installed; installing it would be a system-wide change outside the workspace.
* **Decision.** Build LightGBM 4.7.0 from source with its documented `USE_OPENMP=OFF` option:
  `uv pip install --no-binary lightgbm "lightgbm==4.7.0" -C cmake.define.USE_OPENMP=OFF`.
* **Alternatives.** `brew install libomp` (system change); patching the wheel's library path
  (fragile); XGBoost (the brief asks for LightGBM).
* **Consequences.** LightGBM is single-threaded, which is what the benchmarks require anyway. On
  Linux (CI) the standard wheel works.

## R2. PyTorch and FAISS never run in the same process

* **Context.** Both wheels bundle their own OpenMP runtime. Executing both in one process aborts
  with "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized"
  (reproduced in either import order). The suggested `KMP_DUPLICATE_LIB_OK=TRUE` is documented as
  unsafe ("may cause crashes or silently produce incorrect results").
* **Decision.** Training (PyTorch) writes user and item embeddings to `.npy` files; the retrieval
  comparison (FAISS, hnsw-engine, exact) and the ranker run in separate processes that never import
  PyTorch. The serving app uses PyTorch with hnsw-engine and LightGBM, never FAISS. The one test
  that exercises FAISS runs in a child process.
* **Consequences.** Clean separation of the training and serving stages, which mirrors
  production systems where embeddings are exported to the retrieval service.

## R3. Per-user temporal split with a deterministic tie-break

* **Context.** MovieLens timestamps have one-second resolution; users often rated several movies
  in the same second, so "last" is ambiguous.
* **Decision.** Order each user's positives by `(timestamp, movie id)`; last is test,
  second-to-last validation. "Before the target" means earlier in that order for the user's own
  history; aggregates over other users' events (item popularity) use timestamps strictly before
  the target's.
* **Alternatives.** Global temporal split (stronger against cross-user leakage, but not the
  protocol the brief prescribes); random tie-breaking (not reproducible).
* **Consequences.** Deterministic, testable split. The known limitation of per-user splits
  (models see other users' later interactions) is stated in `DESIGN.md` and the model card.

## R4. History is excluded from rankings; validation joins the history for test

* **Decision.** When ranking a target, the user's earlier positives are excluded from the
  candidate list. For the test target this includes the validation item, which happened earlier.
  Low ratings (1-3) are neither positives nor excluded.
* **Consequences.** Every method is evaluated on exactly the same exclusion sets.

## R5. Ties in scores are broken by item id, in metrics and in top-K lists alike

* **Decision.** A target's rank counts strictly better items plus equally scored items with a
  lower id; top-K lists use the same order.
* **Consequences.** Metrics describe the lists a user would actually see. Popularity, which has
  many ties, is evaluated deterministically.

## R6. ALS tuned in three passes, each centred on the previous optimum

* **Context.** A first grid (factors 32-128, regularization 0.1-10, alpha 5-20) selected
  (128, 10, 5): an edge of all three ranges, so better settings could lie outside it.
* **Decision.** Re-centre the grid on each optimum until it lands inside or the gains become
  noise: pass 2 (factors 64-256, regularization 1-100, alpha 1-10) selected (64, 10, 1); pass 3
  (factors 32-128, regularization 3-30, alpha 0.3-3) selected (128, 30, 1) with validation
  NDCG@10 0.0523. The best settings of pass 3 differ by about 0.001 NDCG@10, inside the ~0.004
  95% sampling half-width of the validation metric, so tuning stopped. Selection always uses
  validation NDCG@10; test targets are evaluated once.
* **Consequences.** A fairly tuned baseline. Pass 2's trials are kept in
  `results/baselines/grid-pass2/als_grid.json`, pass 3's in `results/baselines/main/als_grid.json`;
  pass 1's grid is recorded in `configs/baselines.yaml`.

## R7. Two-stage metrics over the full catalog

* **Decision.** The ranker reorders the 200 candidates; items outside the candidate list follow in
  exact two-tower order. The same construction scores the hnsw-engine retrieval row.
* **Consequences.** MRR and all cut-off metrics are defined over the full catalog for every row of
  the results table, so rows are directly comparable.

## R8. Ranker trained on validation targets with a 10% user hold-out for early stopping

* **Decision.** Training groups are validation-stage candidate lists that contain the validation
  target (lists without it carry no LambdaRank gradient); 10% of those users (seeded) are held out
  for early stopping on NDCG@10. The early-stopped model is used directly on the test stage.
* **Consequences.** The test targets are never seen during training or model selection.

## R9. Ranker features: the base set lost to retrieval order; recency features added

* **Context.** The first ranker used only the brief's feature list (two-tower score and rank,
  popularity before the target time, year, genre affinity, activity, demographics) with default
  LightGBM settings. Its single test evaluation scored *below* the plain two-tower order (NDCG@10
  0.057 vs 0.062; kept in `results/ranker/first-attempt-base-features/`).
* **Decision.** Diagnose and choose using validation data only (`experiments/tune_ranker.py`): a
  10% *selection* subset of the validation users (362 users) compares variants against simple
  orderings. The base feature set lost to the two-tower order under every LightGBM setting
  (selection NDCG@10 0.094-0.104 vs 0.109). Adding four features that the mean-pooled user vector
  cannot express (cosine of each candidate to the user's latest history item and to the five
  latest, mean and max; movie age at request time) reached 0.127-0.148; the most regularised
  setting (7 leaves, min 400 rows per leaf, L2 10, learning rate 0.03) was selected. Gains between
  the top settings are within the selection set's noise (about +-0.015).
* **Alternatives.** Reporting the failed ranker only (honest but leaves the two-stage design
  untested); tuning on test (invalid).
* **Consequences.** The final configuration was evaluated on test exactly once more. Both test
  evaluations are reported; the brief's features are all still in the model.

## R10. Coverage is reported without a confidence interval

* **Context.** The first results table showed coverage intervals that excluded their own point
  estimate (for example 0.5013 [0.4498, 0.4652]).
* **Decision.** Bootstrap only the per-user mean metrics. Coverage counts distinct items across all
  users; resampling users with replacement repeats about 37% of them and mechanically shrinks
  that set, so its bootstrap distribution is biased low. A unit test demonstrates the effect.
* **Consequences.** Coverage appears as a point value, labelled "(point)" in the tables.

## R11. Training is deterministic

* **Decision.** Fixed seeds for NumPy and PyTorch and a fixed thread count (4). Two training runs
  with the same config produced bit-identical user and item embeddings
  (`results/two_tower/determinism-check/`), so the whole pipeline reruns to the same numbers.

## R12. The main model keeps in-batch negatives with log-Q, although uniform sampling won the ablation

* **Context.** The ablations (`ANALYSIS.md`) show 1,024 uniformly sampled negatives per batch
  beating the brief's in-batch + log-Q configuration on validation (NDCG@10 0.0842 vs 0.0780) and
  test, on this small catalog.
* **Decision.** Keep the main configuration, which was fixed before the ablations ran; adopting the
  winner after seeing its test numbers would turn the test set into a selection set. Report the
  finding instead, with its likely cause (1,024 negatives are 28% of a 3,706-movie catalog, close to
  a full softmax).
* **Consequences.** The headline numbers are slightly conservative; a future version should select
  the negative-sampling scheme on validation data before any test evaluation.

## R13. Serving computes features for the requested user only

* **Context.** The first load test ran at about 0.44 s per request: the feature code rebuilt the
  genre profile of all 6,040 users on every call (harmless offline, where every user is scored).
* **Decision.** `genre_profiles_for` computes profiles only for the users being scored. Genre
  vectors are 0/1, so the sums are exact; the full offline feature matrices (1,207,000 x 14 for
  each stage) were verified bit-identical before and after, so the trained ranker is unaffected.
* **Consequences.** About 0.56 ms server-side per request; the load-test script's own bug (a
  variable shadowing the server process, which left the server running) was also fixed, and the
  experiment scripts are now type-checked in CI.

## R14. MovieLens-10M scale check: transfer the ML-1M settings, do not re-tune

* **Context.** Tier 3 asks whether the results hold on a larger MovieLens release. Re-tuning every
  model on ML-10M would answer a different question (how good can each model get) at several times
  the compute; ML-10M also lacks the user demographics the ML-1M models use.
* **Decision.** MovieLens-10M (10,000,054 ratings, 69,878 users, 10,677 movies; checksum pinned on
  first download) with the unchanged protocol. The two-tower model and the ranker keep their ML-1M
  settings; the user tower runs without side features and the ranker drops gender, age and
  occupation, and both refuse demographic inputs on a dataset without them (tested) instead of
  silently training on zeros. ALS gets a three-point grid over regularisation around its ML-1M
  optimum, selected on validation NDCG@10; the optimum sat on the edge of the first grid, so two
  more passes extended and refined it (regularisation 100 selected from seven values; each pass
  also scores its own best on test, but no choice used test numbers). ML-10M's "Children" genre
  maps to ML-1M's "Children's"; "IMAX" and "(no genres listed)" are ignored, so the 18 genre
  features mean the same thing.
* **Alternatives.** ML-20M (twice the size again; expected, not measured, to push two-tower
  training past the 20-minute time box on this laptop); re-tuning each model (a fairer contest at
  ML-10M scale, but not a transfer test).
* **Consequences.** Every ML-1M conclusion held (`RESULTS.md`, MovieLens-10M section): full system
  0.1047 NDCG@10 against 0.0524 for tuned ALS. Two memory problems had to be fixed first, both
  verified not to change any ML-1M number: item co-occurrence built dense users-by-items matrices (about 6 GB at ML-10M) and
  now sums counts over blocks of users (float32 holds the integer counts exactly), and the ranker's
  feature builder processes users in blocks of 4,096 instead of materialising
  users x candidates x dimension arrays. Re-running ML-1M after the changes gave bit-identical
  rankings for all three baselines and the ranker (`results/verification/tier3_*_ml1m.json`) and a
  byte-identical dataset summary.

## R15. The service image: root build context, allowlisted inputs, artifacts mounted read-only

* **Context.** The service depends on the sibling hnsw-engine project, so the image must be built
  from the workspace root; the trained artifacts and data live in git-ignored folders. The first
  real build (2026-10-04, Docker Desktop 4.93) exposed two problems in the Dockerfile written in
  Tier 2: its `.dockerignore` sat in `two-stage-recsys/`, where Docker never reads it for a
  root-context build (the context would have included `.venv`, every `data/` and `results/`
  folder), and the smoke script could not reach Docker Desktop's credential helper.
* **Decision.** `two-stage-recsys/Dockerfile.dockerignore` (the per-Dockerfile ignore file BuildKit
  reads) is an allowlist of the two packages' `pyproject.toml`, `README.md` and `src/`. Data and
  artifacts are mounted read-only at run time instead of baked in. CPU-only PyTorch at the
  development version (2.14.1), an unprivileged user, a writable Numba cache directory,
  single-threaded numerics and a health check. `experiments/docker_smoke.py` builds, runs and
  checks the container against the native service; `serving_compare.py` puts a native and a
  container load test side by side.
* **Alternatives.** Baking the artifacts into the image (self-contained, but rebuilt for every
  retrained model and much larger); vendoring hnsw-engine into the recommender project (no root
  context needed, but two copies of the engine).
* **Consequences.** The container returns exactly the native service's recommendations (same
  movies, order and scores for five users) and passes its health check; the image is 1,997.8 MB.
  Requests pay for Docker Desktop's virtual machine, most visibly with concurrent clients
  (`MODEL_CARD.md`).
