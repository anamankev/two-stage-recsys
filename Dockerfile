# Container for the recommendation service (milestone B8).
# Build from the workspace root, because the service depends on the sibling hnsw-engine project;
# Dockerfile.dockerignore (next to this file) limits the build context to the two packages:
#   docker build -f two-stage-recsys/Dockerfile -t two-stage-recsys:0.1.0 .
# Data and trained artifacts are not baked in; mount them read-only:
#   docker run --rm -p 8000:8000 \
#     -v "$PWD/two-stage-recsys/data:/app/two-stage-recsys/data:ro" \
#     -v "$PWD/two-stage-recsys/results:/app/two-stage-recsys/results:ro" two-stage-recsys:0.1.0
# Built and tested with Docker Desktop 4.93 on linux/arm64 (experiments/docker_smoke.py;
# numbers in docs/MODEL_CARD.md).
FROM python:3.12-slim

# LightGBM's Linux wheel needs the GNU OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-only PyTorch first (the default Linux wheels can bundle CUDA and weigh several GB), at the
# version used in development.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY hnsw-engine /app/hnsw-engine
COPY two-stage-recsys /app/two-stage-recsys
RUN pip install --no-cache-dir /app/hnsw-engine "/app/two-stage-recsys[serve]"

# Run unprivileged; Numba caches compiled code in a writable directory.
RUN useradd --create-home --uid 10001 app
USER app
WORKDIR /app/two-stage-recsys
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    NUMBA_CACHE_DIR=/tmp/numba-cache PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["python", "-m", "recsys.serving.app", "--host", "0.0.0.0", "--port", "8000"]
