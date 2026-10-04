"""Parsing and the per-user temporal split, including the leakage guarantees."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from recsys import data
from recsys.data import GENRES, load, make_split

REAL_DATA = Path(__file__).resolve().parents[1] / "data" / "ml-1m.zip"


def test_parsing(tiny_zip: Path) -> None:
    ml = load(tiny_zip)
    assert (ml.n_users, ml.n_items) == (4, 6)
    np.testing.assert_array_equal(ml.movie_ids, [1, 2, 3, 5, 6, 10])  # movie 7 has no rating
    assert ml.titles[2] == "Café au Lait (1993)"  # latin-1 decoded
    np.testing.assert_array_equal(ml.years, [1995, 1995, 1993, 1995, 1995, 1995])
    toy_story = {GENRES[i] for i in np.flatnonzero(ml.genres[0])}
    assert toy_story == {"Animation", "Children's", "Comedy"}
    np.testing.assert_array_equal(ml.gender, [0, 1, 1, 0])
    np.testing.assert_array_equal(ml.age, [0, 6, 2, 4])  # codes 1, 56, 25, 45
    np.testing.assert_array_equal(ml.occupation, [10, 16, 15, 7])
    assert ml.rating_user.shape[0] == 16


def test_split_matches_hand_computation(tiny_zip: Path) -> None:
    split = make_split(load(tiny_zip))
    np.testing.assert_array_equal(split.users, [0, 1, 2])  # user 4 (index 3) dropped
    # internal ids: m1->0, m2->1, m3->2, m5->3, m6->4, m10->5
    assert split.train_history(0).tolist() == [0, 1]
    assert split.train_history(1).tolist() == [0, 4]
    assert split.train_history(2).tolist() == [1]
    assert split.train_history(3).tolist() == []
    np.testing.assert_array_equal(split.val_item, [3, 5, 2, -1])
    np.testing.assert_array_equal(split.test_item, [4, 2, 3, -1])
    np.testing.assert_array_equal(split.val_time, [200, 70, 10, -1])
    assert split.history(0, "test").tolist() == [0, 1, 3]
    items, times = split.targets("test")
    np.testing.assert_array_equal(items, [4, 2, 3])
    np.testing.assert_array_equal(times, [300, 90, 10])


def test_history_csr_matches_per_user_histories(tiny_zip: Path) -> None:
    split = make_split(load(tiny_zip))
    for stage in ("val", "test"):
        indptr, items = split.history_csr(stage)
        for user in split.users:
            assert (
                items[indptr[user] : indptr[user + 1]].tolist()
                == split.history(user, stage).tolist()
            )


def test_min_positives_validation(tiny_zip: Path) -> None:
    with pytest.raises(ValueError, match=">= 3"):
        make_split(load(tiny_zip), min_positives=2)


def _assert_no_leakage(split: data.Split, ml: data.MovieLens) -> None:
    for user in split.users:
        train = split.train_history(user)
        start, stop = split.train_indptr[user], split.train_indptr[user + 1]
        times = split.train_times[start:stop]
        val, test = split.val_item[user], split.test_item[user]
        # No target item among the user's training positives; targets differ.
        assert val not in train
        assert test not in train
        assert val != test
        # Every training event precedes the validation target, which precedes the test target,
        # in the (timestamp, movie id) order.
        keys = list(zip(times.tolist(), train.tolist(), strict=True))
        assert keys == sorted(keys)
        val_key = (split.val_time[user], val)
        test_key = (split.test_time[user], test)
        assert all(k < val_key for k in keys)
        assert val_key < test_key
    # The split uses exactly the positives of the kept users.
    positive = ml.rating_value >= data.POSITIVE_THRESHOLD
    kept = np.isin(ml.rating_user, split.users) & positive
    assert split.train_items.shape[0] + 2 * split.users.shape[0] == int(kept.sum())


def test_no_leakage_on_tiny_data(tiny_zip: Path) -> None:
    ml = load(tiny_zip)
    _assert_no_leakage(make_split(ml), ml)


@pytest.mark.skipif(not REAL_DATA.exists(), reason="MovieLens-1M not downloaded (no network in CI)")
def test_no_leakage_on_movielens_1m() -> None:
    ml = load(REAL_DATA)
    split = make_split(ml)
    assert (ml.n_users, ml.n_items) == (6040, 3706)
    _assert_no_leakage(split, ml)


def test_download_from_file_url_and_checksum(tiny_zip: Path, tmp_path: Path) -> None:
    target_dir = tmp_path / "downloads"
    path = data.download(target_dir, url=tiny_zip.as_uri(), sha256=data.sha256_file(tiny_zip))
    assert path.read_bytes() == tiny_zip.read_bytes()
    with pytest.raises(ValueError, match="checksum"):
        data.download(target_dir, url=tiny_zip.as_uri(), sha256="0" * 64)


def test_parsing_movielens_10m_layout(tiny_zip_10m: Path) -> None:
    ml = load(tiny_zip_10m)
    assert not ml.demographics
    assert (ml.n_users, ml.n_items) == (3, 4)  # movie 8 has no rating
    np.testing.assert_array_equal(ml.user_ids, [10, 20, 30])
    np.testing.assert_array_equal(ml.movie_ids, [1, 2, 3, 5])
    assert ml.titles[2] == "Amélie (2001)"  # UTF-8 decoded
    toy_story = {GENRES[i] for i in np.flatnonzero(ml.genres[0])}
    assert toy_story == {"Adventure", "Animation", "Children's", "Comedy", "Fantasy"}
    avatar = {GENRES[i] for i in np.flatnonzero(ml.genres[3])}
    assert avatar == {"Action", "Sci-Fi"}  # IMAX is not an ML-1M genre
    for unknown in (ml.gender, ml.age, ml.occupation):
        assert (unknown == 0).all()
    np.testing.assert_array_equal(ml.rating_value[:4], [4.5, 4.0, 5.0, 3.5])


def test_split_on_movielens_10m_layout(tiny_zip_10m: Path) -> None:
    ml = load(tiny_zip_10m)
    split = make_split(ml)
    np.testing.assert_array_equal(split.users, [0, 1])  # user 30 has one positive
    assert split.train_history(0).tolist() == [0]  # m1; 3.5 stars is not a positive
    np.testing.assert_array_equal(split.val_item[:2], [1, 1])
    np.testing.assert_array_equal(split.test_item[:2], [2, 2])
    assert split.train_history(1).tolist() == [0, 3]  # m1@1, m5@2
    _assert_no_leakage(split, ml)


def test_dataset_registry_and_config_loading(
    tiny_zip_10m: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "downloads"
    digest = data.sha256_file(tiny_zip_10m)
    path = data.download(target, dataset="ml-10m", url=tiny_zip_10m.as_uri(), sha256=digest)
    assert path.name == "ml-10m.zip"
    config = {"data_dir": str(target), "dataset": "ml-10m"}
    # The registry pins the real archive's checksum, so the miniature archive is rejected ...
    with pytest.raises(ValueError, match="checksum"):
        data.load_dataset(config)
    # ... and loads once the registry entry names its checksum.
    spec = dataclasses.replace(data.DATASETS["ml-10m"], sha256=digest)
    monkeypatch.setitem(data.DATASETS, "ml-10m", spec)
    assert data.load_dataset(config).n_users == 3
    with pytest.raises(ValueError, match="unknown dataset"):
        data.download(target, dataset="ml-100m")


def test_unsupported_archive_is_rejected(tmp_path: Path) -> None:
    import zipfile

    path = tmp_path / "other.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("something/ratings.csv", "a,b\n")
    with pytest.raises(ValueError, match="not a supported MovieLens archive"):
        load(path)
