"""Candidate generation: nearest item embeddings to a user embedding.

Three interchangeable retrievers over the same L2-normalised item vectors (inner product):

* :class:`ExactRetriever` -- brute-force dot products (the reference);
* :class:`HnswEngineRetriever` -- Project A's ``HNSWIndex`` (``metric="ip"``);
* :class:`FaissHnswRetriever` -- FAISS ``IndexHNSWFlat`` with inner product.

History items are removed after retrieval, so each search asks for ``n + len(history)`` items to
guarantee ``n`` fresh candidates. Every search is timed individually (one thread).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from hnsw_engine import HNSWIndex
from numpy.typing import NDArray

from .evaluation import top_k_ids

I64 = NDArray[np.int64]
F32 = NDArray[np.float32]


class Retriever(ABC):
    """Maximum-inner-product search over item vectors."""

    name: str

    @abstractmethod
    def build(self, item_vectors: F32) -> None:
        """Index the item vectors."""

    @abstractmethod
    def search(self, vector: F32, k: int) -> I64:
        """Ids of the ``k`` items with the highest inner product (best first)."""

    def describe(self) -> dict[str, Any]:
        """Configuration recorded with results."""
        return {"name": self.name}


class ExactRetriever(Retriever):
    """Brute force: score every item, take the top ``k`` (ties by lower id)."""

    name = "exact"

    def build(self, item_vectors: F32) -> None:
        """Keep a contiguous copy of the item vectors."""
        self.items = np.ascontiguousarray(item_vectors, dtype=np.float32)

    def search(self, vector: F32, k: int) -> I64:
        """Exact top-``k`` by inner product."""
        return top_k_ids(self.items @ vector, k)


class HnswEngineRetriever(Retriever):
    """Project A's HNSW index with inner-product distance."""

    name = "hnsw-engine"

    def __init__(
        self,
        *,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 200,
        engine: str = "reference",
        seed: int = 0,
    ) -> None:
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.engine = engine
        self.seed = seed

    def build(self, item_vectors: F32) -> None:
        """Build the graph over the item vectors."""
        self.index = HNSWIndex(
            item_vectors.shape[1],
            metric="ip",
            M=self.M,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            seed=self.seed,
            engine=self.engine,
        )
        self.index.add(item_vectors)
        self.index.search(item_vectors[0], k=1)  # warm-up: load compiled code before timing

    def search(self, vector: F32, k: int) -> I64:
        """Approximate top-``k`` (beam width ``max(ef_search, k)``)."""
        return self.index.search(vector, k=min(k, len(self.index)))[0]

    def describe(self) -> dict[str, Any]:
        """Name and index parameters."""
        return {
            "name": self.name,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "engine": self.engine,
            "seed": self.seed,
        }


class FaissHnswRetriever(Retriever):
    """FAISS ``IndexHNSWFlat`` with inner product, one thread."""

    name = "faiss-hnsw"

    def __init__(self, *, M: int = 16, ef_construction: int = 200, ef_search: int = 200) -> None:
        self.params = {"M": M, "ef_construction": ef_construction, "ef_search": ef_search}

    def build(self, item_vectors: F32) -> None:
        """Build the FAISS graph."""
        import faiss

        faiss.omp_set_num_threads(1)
        self.faiss = faiss
        self.index = faiss.IndexHNSWFlat(
            item_vectors.shape[1], self.params["M"], faiss.METRIC_INNER_PRODUCT
        )
        self.index.hnsw.efConstruction = self.params["ef_construction"]
        self.index.add(np.ascontiguousarray(item_vectors, dtype=np.float32))

    def search(self, vector: F32, k: int) -> I64:
        """Approximate top-``k`` (efSearch raised to ``k`` when needed)."""
        self.index.hnsw.efSearch = max(self.params["ef_search"], k)
        _, ids = self.index.search(vector[None, :].astype(np.float32), min(k, self.index.ntotal))
        result: I64 = np.asarray(ids[0], dtype=np.int64)
        return result

    def describe(self) -> dict[str, Any]:
        """Name and index parameters."""
        return {"name": self.name, **self.params}


@dataclass(frozen=True)
class Candidates:
    """``n`` fresh candidates per user (history removed), best first, plus search latencies."""

    users: I64
    items: I64  # (n_users, n)
    latency_ns: I64  # per-user search time

    def latency_summary(self) -> dict[str, float]:
        """Mean and percentile search latency in milliseconds."""
        ms = self.latency_ns / 1e6
        return {
            "mean_ms": float(ms.mean()),
            "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)),
            "p99_ms": float(np.percentile(ms, 99)),
        }


def retrieve(
    retriever: Retriever,
    user_vectors: F32,
    users: I64,
    history_indptr: I64,
    history_items: I64,
    n: int = 200,
) -> Candidates:
    """Top ``n`` non-history items for each user; pads with ``-1`` if fewer are returned."""
    out = np.full((users.shape[0], n), -1, dtype=np.int64)
    latency = np.empty(users.shape[0], dtype=np.int64)
    for row, user in enumerate(users):
        history = history_items[history_indptr[user] : history_indptr[user + 1]]
        vector = np.ascontiguousarray(user_vectors[user], dtype=np.float32)
        start = time.perf_counter_ns()
        found = retriever.search(vector, n + history.shape[0])
        latency[row] = time.perf_counter_ns() - start
        fresh = found[(found >= 0) & ~np.isin(found, history)][:n]
        out[row, : fresh.shape[0]] = fresh
    return Candidates(users, out, latency)


def candidate_recall(candidates: Candidates, targets: I64, ks: tuple[int, ...]) -> dict[str, float]:
    """Share of users whose target is among their first ``k`` candidates, for each ``k``."""
    hits = candidates.items == targets[:, None]
    position = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, np.iinfo(np.int64).max)
    return {f"recall@{k}": float((position <= k).mean()) for k in ks}


def overlap(a: Candidates, b: Candidates, k: int) -> float:
    """Mean fraction of ``b``'s first ``k`` candidates that also appear in ``a``'s first ``k``."""
    shared = [
        len(set(x[:k].tolist()) & set(y[:k].tolist())) / k
        for x, y in zip(a.items, b.items, strict=True)
    ]
    return float(np.mean(shared))
