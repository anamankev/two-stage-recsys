# Reproduction targets for two-stage-recsys. PY defaults to the shared workspace environment.
PY ?= ../.venv/bin/python
RESULTS_ARGS = --baselines results/baselines/main --two-tower results/two_tower/main \
	--retrieval results/retrieval/main --ranker results/ranker/main

.PHONY: test lint typecheck check data baselines two-tower retrieval tune-ranker ranker results \
	ablations analysis serve load-test docker load-test-docker reproduce reproduce-all ml10m

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests experiments
	$(PY) -m ruff format --check src tests experiments

typecheck:
	$(PY) -m mypy src/recsys experiments

check: lint typecheck test

data:
	$(PY) experiments/describe_data.py

baselines:
	$(PY) experiments/run_baselines.py --name main

two-tower:
	$(PY) experiments/train_two_tower.py --name main

retrieval:
	$(PY) experiments/run_retrieval.py --name main

tune-ranker:
	$(PY) experiments/tune_ranker.py --name main

ranker:
	$(PY) experiments/train_ranker.py --name main

results:
	$(PY) experiments/make_results.py $(RESULTS_ARGS) --inject docs/RESULTS.md --inject README.md \
		--inject docs/REPORT.md

ablations:
	$(PY) experiments/run_ablations.py

analysis:
	$(PY) experiments/analyze.py
	$(PY) experiments/report_analysis.py --inject docs/ANALYSIS.md

serve:
	$(PY) -m recsys.serving.app --port 8000

load-test:
	$(PY) experiments/load_test.py --requests 2000 --concurrency 4 --inject docs/MODEL_CARD.md

# Build the image, start the container (left running on port 8000) and check its answers.
docker:
	$(PY) experiments/docker_smoke.py

# The running container next to the native service, back to back, then the comparison table.
DOCKER ?= $(shell command -v docker 2>/dev/null || echo $(HOME)/.docker/bin/docker)
DOCKER_SETTING = one uvicorn worker in the Docker container (Docker Desktop, linux/arm64), \
	single-threaded numerics, client on the host
load-test-docker:
	$(PY) experiments/load_test.py --url http://127.0.0.1:8000 --requests 2000 --concurrency 4 \
		--out load_test_docker.json --setting "$(DOCKER_SETTING)"
	$(DOCKER) stop two-stage-recsys
	$(PY) experiments/load_test.py --requests 2000 --concurrency 4 --out load_test_native_compare.json
	$(PY) experiments/serving_compare.py load_test_native_compare.json load_test_docker.json \
		--inject docs/MODEL_CARD.md

# Main pipeline, about 6 minutes on an Apple M5 (baselines 3 min, two-tower 1 min).
reproduce: data baselines two-tower retrieval tune-ranker ranker results

# Everything including ablations (about 20 minutes) and the load test.
reproduce-all: reproduce ablations analysis load-test

# Tier 3 (B9): the same pipeline on MovieLens-10M (65.6 MB download), about an hour.
ML10M_RESULTS = --baselines results/baselines/ml10m --two-tower results/two_tower/ml10m \
	--retrieval results/retrieval/ml10m --ranker results/ranker/ml10m \
	--candidate-source hnsw-engine-numba --tag ml10m

ml10m:
	$(PY) experiments/describe_data.py --dataset ml-10m
	$(PY) experiments/run_baselines.py --config configs/ml10m/baselines.yaml --name ml10m
	$(PY) experiments/run_baselines.py --config configs/ml10m/als_pass2.yaml \
		--name ml10m-als-pass2 --skip-simple
	$(PY) experiments/run_baselines.py --config configs/ml10m/als_pass3.yaml \
		--name ml10m-als-pass3 --skip-simple
	$(PY) experiments/train_two_tower.py --config configs/ml10m/two_tower.yaml --name ml10m
	$(PY) experiments/run_retrieval.py --config configs/ml10m/retrieval.yaml --name ml10m
	$(PY) experiments/train_ranker.py --config configs/ml10m/ranker.yaml --name ml10m
	$(PY) experiments/make_results.py $(ML10M_RESULTS) --inject docs/RESULTS.md
