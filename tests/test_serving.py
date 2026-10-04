"""The FastAPI service, built from tiny real artifacts (no network, no saved models)."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient
from hnsw_engine import HNSWIndex

from recsys.data import POSITIVE_THRESHOLD, load, make_split
from recsys.features import PopularityIndex
from recsys.ranker import ALL_FEATURES
from recsys.serving.app import Recommender, create_app
from recsys.two_tower import Trainer, TwoTowerConfig, embed


@pytest.fixture
def recommender(tiny_zip: Path) -> Recommender:
    ml = load(tiny_zip)
    split = make_split(ml)
    config = TwoTowerConfig(
        embedding_dim=8,
        hidden_dim=8,
        history_length=3,
        batch_size=4,
        max_epochs=1,
        num_threads=1,
        eval_k=3,
    )
    model = Trainer(split, ml, config).fit(verbose=False)
    _, items = embed(model, split, "test")
    index = HNSWIndex(8, metric="ip", M=4, ef_construction=8, seed=0)
    index.add(items)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, len(ALL_FEATURES)))
    features[:, 7:10] = rng.integers(0, 3, size=(60, 3))  # categorical columns
    booster = lgb.train(
        {"objective": "lambdarank", "verbosity": -1, "num_threads": 1, "min_data_in_leaf": 1},
        lgb.Dataset(features, rng.integers(0, 2, 60), group=[6] * 10),
        num_boost_round=3,
    )
    positive = ml.rating_value >= POSITIVE_THRESHOLD
    users, item_ids, times = (
        ml.rating_user[positive],
        ml.rating_item[positive],
        ml.rating_time[positive],
    )
    order = np.lexsort((item_ids, times, users))
    indptr = np.zeros(ml.n_users + 1, dtype=np.int64)
    np.cumsum(np.bincount(users, minlength=ml.n_users), out=indptr[1:])
    return Recommender(
        ml=ml,
        model=model,
        index=index,
        item_vectors=items,
        booster=booster,
        features=ALL_FEATURES,
        history_indptr=indptr,
        history_items=item_ids[order],
        popularity=PopularityIndex(item_ids, times, ml.n_items),
        now=int(ml.rating_time.max()) + 1,
    )


def test_recommendations_exclude_history_and_report_latency(recommender: Recommender) -> None:
    out = recommender.recommend(user_id=1, k=3)
    liked = {int(recommender.ml.movie_ids[i]) for i in recommender.history(0)}
    returned = [item["movie_id"] for item in out["items"]]
    assert 0 < len(returned) <= 3
    assert not set(returned) & liked
    assert len(set(returned)) == len(returned)
    assert all(isinstance(item["title"], str) for item in out["items"])
    lat = out["latency_ms"]
    assert set(lat) == {"embedding", "retrieval", "ranking", "total"}
    assert lat["total"] >= max(lat["embedding"], lat["retrieval"], lat["ranking"])
    assert out["history_size"] == len(liked)


def test_unknown_user(recommender: Recommender) -> None:
    with pytest.raises(KeyError):
        recommender.recommend(user_id=999)


def test_http_endpoints(recommender: Recommender) -> None:
    client = TestClient(create_app(recommender))
    health = client.get("/health").json()
    assert health == {"status": "ok", "users": 4, "items": 6, "index_size": 6}
    response = client.get("/recommend/2", params={"k": 2})
    assert response.status_code == 200
    assert len(response.json()["items"]) <= 2
    assert client.get("/recommend/999").status_code == 404
    assert client.get("/recommend/1", params={"k": 0}).status_code == 422
    assert client.get("/recommend/1", params={"k": 101}).status_code == 422
