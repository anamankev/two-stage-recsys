"""Online serving: ``GET /recommend/{user_id}`` returns the top movies with a latency breakdown.

At start-up the service loads the trained two-tower model, indexes the item vectors with
**hnsw-engine** (the compiled engine), and loads the LightGBM ranker. Per request:

1. **embedding**: the user tower turns the user's history (all known likes) into a vector;
2. **retrieval**: one approximate nearest-neighbour query returns 200 fresh candidates;
3. **ranking**: the same feature code as offline training (:func:`recsys.ranker.compute_features`)
   builds the candidate features, LightGBM scores them, and the top ``k`` are returned.

The service clock is frozen one second after the last MovieLens rating, so "popularity before
now" is computed exactly as at training time. PyTorch, Numba and LightGBM (built without OpenMP)
share the process; FAISS never does (see ``docs/DECISIONS.md``, R2).

Run: ``python -m recsys.serving.app --port 8000``
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Query
from hnsw_engine import HNSWIndex
from numpy.typing import NDArray

from ..config import build
from ..data import POSITIVE_THRESHOLD, MovieLens, download, load
from ..features import PopularityIndex
from ..ranker import compute_features
from ..two_tower import SideInfo, TwoTower, TwoTowerConfig, padded_histories

I64 = NDArray[np.int64]


@dataclass
class Recommender:
    """Everything a request needs, loaded once."""

    ml: MovieLens
    model: TwoTower
    index: HNSWIndex
    item_vectors: NDArray[np.float32]
    booster: Any
    features: tuple[str, ...]
    history_indptr: I64
    history_items: I64
    popularity: PopularityIndex
    now: int
    n_candidates: int = 200

    def user_index(self, user_id: int) -> int:
        """Internal index of a MovieLens user id (``KeyError`` if unknown)."""
        position = int(np.searchsorted(self.ml.user_ids, user_id))
        if position >= self.ml.n_users or self.ml.user_ids[position] != user_id:
            raise KeyError(user_id)
        return position

    def history(self, user: int) -> I64:
        """All movies the user liked, in time order."""
        return self.history_items[self.history_indptr[user] : self.history_indptr[user + 1]]

    @torch.no_grad()
    def recommend(self, user_id: int, k: int = 10) -> dict[str, Any]:
        """Top ``k`` movies for ``user_id`` with per-stage latencies in milliseconds."""
        t0 = time.perf_counter()
        user = self.user_index(user_id)
        history = self.history(user)
        hist = padded_histories(
            self.history_indptr,
            self.history_items,
            np.array([user]),
            np.array([history.shape[0]]),
            self.model.config.history_length,
            self.ml.n_items,
        )
        vector = (
            self.model.user_tower(torch.tensor([user]), torch.from_numpy(hist), self.model.side)
            .numpy()[0]
            .astype(np.float32)
        )
        t1 = time.perf_counter()
        want = min(self.n_candidates + history.shape[0], len(self.index))
        found = self.index.search(vector, k=want)[0]
        fresh = found[(found >= 0) & ~np.isin(found, history)][: self.n_candidates]
        t2 = time.perf_counter()
        matrix = compute_features(
            self.ml,
            np.array([user]),
            fresh[None, :],
            np.array([self.now]),
            self.history_indptr,
            self.history_items,
            vector[None, :],
            self.item_vectors,
            self.popularity,
            self.features,
        )
        scores = self.booster.predict(matrix)
        order = np.argsort(-scores, kind="stable")[:k]
        t3 = time.perf_counter()
        items = [
            {
                "movie_id": int(self.ml.movie_ids[fresh[i]]),
                "title": self.ml.titles[fresh[i]],
                "score": float(scores[i]),
            }
            for i in order
        ]
        return {
            "user_id": user_id,
            "history_size": int(history.shape[0]),
            "items": items,
            "latency_ms": {
                "embedding": 1000 * (t1 - t0),
                "retrieval": 1000 * (t2 - t1),
                "ranking": 1000 * (t3 - t2),
                "total": 1000 * (t3 - t0),
            },
        }


def load_recommender(
    data_dir: str | Path = "data",
    two_tower_run: str | Path = "results/two_tower/main",
    ranker_run: str | Path = "results/ranker/main",
    *,
    engine: str = "numba",
) -> Recommender:
    """Load trained artifacts and build the serving index."""
    import lightgbm as lgb

    torch.set_num_threads(1)
    ml = load(download(data_dir))
    tt_dir, rk_dir = Path(two_tower_run), Path(ranker_run)
    tt_config = yaml.safe_load((tt_dir / "config.yaml").read_text())
    rk_config = yaml.safe_load((rk_dir / "config.yaml").read_text())
    model = TwoTower(
        ml.n_users,
        ml.n_items,
        SideInfo.from_movielens(ml),
        build(TwoTowerConfig, tt_config["model"]),
    )
    model.load_state_dict(torch.load(tt_dir / "model.pt", weights_only=True))
    model.eval()
    item_vectors = np.load(tt_dir / "item_vectors.npy")
    index = HNSWIndex(
        item_vectors.shape[1],
        metric="ip",
        M=16,
        ef_construction=200,
        ef_search=200,
        seed=0,
        engine=engine,
    )
    index.add(item_vectors)
    positive = ml.rating_value >= POSITIVE_THRESHOLD
    users, items, times = (
        ml.rating_user[positive],
        ml.rating_item[positive],
        ml.rating_time[positive],
    )
    order = np.lexsort((items, times, users))
    indptr = np.zeros(ml.n_users + 1, dtype=np.int64)
    np.cumsum(np.bincount(users, minlength=ml.n_users), out=indptr[1:])
    features = tuple(rk_config["ranker"]["features"])
    return Recommender(
        ml=ml,
        model=model,
        index=index,
        item_vectors=item_vectors,
        booster=lgb.Booster(model_file=str(rk_dir / "ranker.txt")),
        features=features,
        history_indptr=indptr,
        history_items=items[order],
        popularity=PopularityIndex(items, times, ml.n_items),
        now=int(ml.rating_time.max()) + 1,
    )


def create_app(recommender: Recommender) -> FastAPI:
    """FastAPI application around a loaded :class:`Recommender`."""
    app = FastAPI(title="two-stage-recsys", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "users": recommender.ml.n_users,
            "items": recommender.ml.n_items,
            "index_size": len(recommender.index),
        }

    @app.get("/recommend/{user_id}")
    def recommend(user_id: int, k: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
        try:
            return recommender.recommend(user_id, k)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown user {user_id}") from None

    return app


def main() -> None:
    """Serve with uvicorn (single worker)."""
    import uvicorn

    parser = argparse.ArgumentParser(description="two-stage-recsys service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--two-tower-run", default="results/two_tower/main")
    parser.add_argument("--ranker-run", default="results/ranker/main")
    args = parser.parse_args()
    recommender = load_recommender(args.data_dir, args.two_tower_run, args.ranker_run)
    uvicorn.run(create_app(recommender), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
