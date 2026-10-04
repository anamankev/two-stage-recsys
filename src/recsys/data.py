"""MovieLens: download, parsing, implicit positives and the per-user temporal split.

Two releases are supported (``DATASETS``): MovieLens-1M, the main dataset, and MovieLens-10M for
the Tier 3 scale check (``docs/DESIGN.md`` section 9). Split protocol (section 3): ratings of 4
stars or more are positives. Each user's positives are ordered by ``(timestamp, movie id)``; the
last one is the test target, the second-to-last the validation target, the rest is training data.
Users with fewer than three positives are dropped. "Before" a target means earlier in that order.
"""

from __future__ import annotations

import hashlib
import io
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
ML10M_URL = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
# Observed on first download (2026-10-03); GroupLens publishes no checksums.
ML10M_SHA256 = "813c411ccb6122564edfe752e7f80c4dcc5aa25fa94c93622f6877a7ba252862"
GENRES = (
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
    "War", "Western",
)  # fmt: skip
AGE_BUCKETS = (1, 18, 25, 35, 45, 50, 56)  # MovieLens age codes, mapped to indices 0..6
N_OCCUPATIONS = 21
POSITIVE_THRESHOLD = 4

I64 = NDArray[np.int64]
Stage = Literal["val", "test"]
# ML-10M spelling differences; genres outside the 18 ML-1M genres are ignored.
GENRE_ALIASES = {"Children": "Children's"}
IGNORED_GENRES = frozenset({"IMAX", "(no genres listed)"})


@dataclass(frozen=True)
class DatasetSpec:
    """Where a MovieLens release comes from and how its archive is laid out."""

    name: str
    url: str
    sha256: str | None
    folder: str  # top-level folder inside the zip archive
    encoding: str
    demographics: bool


DATASETS = {
    "ml-1m": DatasetSpec("ml-1m", ML1M_URL, ML1M_SHA256, "ml-1m", "latin-1", True),
    "ml-10m": DatasetSpec("ml-10m", ML10M_URL, ML10M_SHA256, "ml-10M100K", "utf-8", False),
}


@dataclass(frozen=True)
class MovieLens:
    """A parsed MovieLens release with contiguous internal ids.

    Users ``0..n_users-1`` follow ascending original UserID (ML-1M: every user in ``users.dat``;
    ML-10M: every user with a rating); items ``0..n_items-1`` follow ascending MovieID over the
    movies that have at least one rating (the catalog).
    """

    user_ids: I64
    movie_ids: I64
    titles: tuple[str, ...]
    years: I64
    genres: NDArray[np.float32]
    gender: I64
    age: I64
    occupation: I64
    rating_user: I64
    rating_item: I64
    rating_value: NDArray[Any]  # stars: integers for ML-1M, half stars (float) for ML-10M
    rating_time: I64
    demographics: bool = True  # False when gender, age and occupation are unknown (all zero)

    @property
    def n_users(self) -> int:
        """Number of users."""
        return int(self.user_ids.shape[0])

    @property
    def n_items(self) -> int:
        """Number of catalog items (movies with at least one rating)."""
        return int(self.movie_ids.shape[0])


@dataclass(frozen=True)
class Split:
    """Per-user leave-last-out split of the positive interactions.

    Training positives are stored in compressed sparse row form over *all* users (dropped users
    have empty rows), ordered by ``(timestamp, item)`` within each user. ``val_item`` and
    ``test_item`` are ``-1`` for dropped users.
    """

    n_users: int
    n_items: int
    users: I64
    train_indptr: I64
    train_items: I64
    train_times: I64
    val_item: I64
    val_time: I64
    test_item: I64
    test_time: I64

    def train_history(self, user: int) -> I64:
        """Training positives of ``user`` in time order."""
        return self.train_items[self.train_indptr[user] : self.train_indptr[user + 1]]

    def history(self, user: int, stage: Stage) -> I64:
        """Everything ``user`` did before the ``stage`` target, in time order."""
        items = self.train_history(user)
        if stage == "test":
            items = np.append(items, self.val_item[user])
        return items

    def targets(self, stage: Stage) -> tuple[I64, I64]:
        """``(items, timestamps)`` of the ``stage`` targets for ``self.users``."""
        if stage == "val":
            return self.val_item[self.users], self.val_time[self.users]
        return self.test_item[self.users], self.test_time[self.users]

    def history_csr(self, stage: Stage) -> tuple[I64, I64]:
        """``(indptr, items)`` of every user's history before the ``stage`` target."""
        if stage == "val":
            return self.train_indptr, self.train_items
        counts = np.diff(self.train_indptr) + (self.val_item >= 0)
        indptr = np.zeros(self.n_users + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        items = np.empty(int(indptr[-1]), dtype=np.int64)
        for user in range(self.n_users):
            items[indptr[user] : indptr[user + 1]] = (
                self.history(user, "test")
                if (self.val_item[user] >= 0)
                else self.train_history(user)
            )
        return indptr, items


def sha256_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    data_dir: str | Path,
    *,
    dataset: str = "ml-1m",
    url: str | None = None,
    sha256: str | None = None,
) -> Path:
    """Download ``<dataset>.zip`` into ``data_dir`` unless present, then verify its checksum.

    ``url`` and ``sha256`` default to the registry entry (``DATASETS``).
    """
    spec = get_spec(dataset)
    target = Path(data_dir) / f"{spec.name}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_name(target.name + ".part")
        with urllib.request.urlopen(url or spec.url) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(target)
    expected = sha256 or spec.sha256
    if expected is not None and sha256_file(target) != expected:
        raise ValueError(f"checksum mismatch for {target}")
    return target


def get_spec(dataset: str) -> DatasetSpec:
    """Registry lookup with a helpful error."""
    try:
        return DATASETS[dataset]
    except KeyError:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}"
        ) from None


def load_dataset(data_config: dict[str, Any]) -> MovieLens:
    """Download (if needed) and parse the dataset a config's ``data`` section names."""
    dataset = data_config.get("dataset", "ml-1m")
    return load(download(data_config["data_dir"], dataset=dataset))


def _read_dat(
    archive: zipfile.ZipFile,
    spec: DatasetSpec,
    name: str,
    columns: list[str],
    dtype: Any = str,
) -> pd.DataFrame:
    text = archive.read(f"{spec.folder}/{name}").decode(spec.encoding).replace("::", "\t")
    return pd.read_csv(
        io.StringIO(text), sep="\t", header=None, names=columns, quoting=3, dtype=dtype
    )


def _genre_matrix(values: pd.Series) -> NDArray[np.float32]:
    genre_index = {g: i for i, g in enumerate(GENRES)}
    genres = np.zeros((len(values), len(GENRES)), dtype=np.float32)
    for row, value in enumerate(values):
        for genre in value.split("|"):
            if genre in IGNORED_GENRES:
                continue
            genres[row, genre_index[GENRE_ALIASES.get(genre, genre)]] = 1.0
    return genres


def load(path: str | Path) -> MovieLens:
    """Parse a MovieLens zip archive (no extraction needed); the release is detected."""
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        spec = next((s for s in DATASETS.values() if f"{s.folder}/ratings.dat" in names), None)
        if spec is None:
            raise ValueError(f"{path} is not a supported MovieLens archive")
        if not spec.demographics:
            return _load_without_users(archive, spec)
        ratings = _read_dat(archive, spec, "ratings.dat", ["user", "movie", "rating", "time"])
        users = _read_dat(
            archive, spec, "users.dat", ["user", "gender", "age", "occupation", "zip"]
        )
        movies = _read_dat(archive, spec, "movies.dat", ["movie", "title", "genres"])
    ratings = ratings.astype(
        {"user": np.int64, "movie": np.int64, "rating": np.int64, "time": np.int64}
    )
    user_ids = np.sort(users["user"].astype(np.int64).to_numpy())
    movie_ids = np.unique(ratings["movie"].to_numpy())

    users = users.assign(user=users["user"].astype(np.int64)).set_index("user").loc[user_ids]
    age_index = {code: i for i, code in enumerate(AGE_BUCKETS)}
    movies = movies.assign(movie=movies["movie"].astype(np.int64)).set_index("movie")
    movies = movies.loc[movie_ids]
    genres = _genre_matrix(movies["genres"])
    years = np.array([_year(t) for t in movies["title"]], dtype=np.int64)

    return MovieLens(
        user_ids=user_ids,
        movie_ids=movie_ids,
        titles=tuple(movies["title"]),
        years=years,
        genres=genres,
        gender=(users["gender"] == "M").astype(np.int64).to_numpy(),
        age=users["age"].astype(np.int64).map(age_index).to_numpy(dtype=np.int64),
        occupation=users["occupation"].astype(np.int64).to_numpy(),
        rating_user=np.searchsorted(user_ids, ratings["user"].to_numpy()).astype(np.int64),
        rating_item=np.searchsorted(movie_ids, ratings["movie"].to_numpy()).astype(np.int64),
        rating_value=ratings["rating"].to_numpy(),
        rating_time=ratings["time"].to_numpy(),
    )


def _load_without_users(archive: zipfile.ZipFile, spec: DatasetSpec) -> MovieLens:
    """MovieLens-10M layout: half-star ratings, no ``users.dat``. Users are those who rated;
    demographics are unknown and stored as zeros (``demographics=False``)."""
    numeric = {"user": np.int64, "movie": np.int64, "rating": np.float64, "time": np.int64}
    ratings = _read_dat(
        archive, spec, "ratings.dat", ["user", "movie", "rating", "time"], dtype=numeric
    )
    movies = _read_dat(archive, spec, "movies.dat", ["movie", "title", "genres"])
    user_ids = np.unique(ratings["user"].to_numpy())
    movie_ids = np.unique(ratings["movie"].to_numpy())
    movies = movies.assign(movie=movies["movie"].astype(np.int64)).set_index("movie")
    movies = movies.loc[movie_ids]
    unknown = np.zeros(user_ids.shape[0], dtype=np.int64)
    return MovieLens(
        user_ids=user_ids,
        movie_ids=movie_ids,
        titles=tuple(movies["title"]),
        years=np.array([_year(t) for t in movies["title"]], dtype=np.int64),
        genres=_genre_matrix(movies["genres"]),
        gender=unknown,
        age=unknown.copy(),
        occupation=unknown.copy(),
        rating_user=np.searchsorted(user_ids, ratings["user"].to_numpy()).astype(np.int64),
        rating_item=np.searchsorted(movie_ids, ratings["movie"].to_numpy()).astype(np.int64),
        rating_value=ratings["rating"].to_numpy(),
        rating_time=ratings["time"].to_numpy(),
        demographics=False,
    )


def _year(title: str) -> int:
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match is None:
        raise ValueError(f"no release year in title {title!r}")
    return int(match.group(1))


def make_split(ml: MovieLens, *, min_positives: int = 3) -> Split:
    """Build the per-user leave-last-out split of positive interactions."""
    if min_positives < 3:
        raise ValueError("min_positives must be >= 3 (train, validation and test)")
    positive = ml.rating_value >= POSITIVE_THRESHOLD
    user, item, time = ml.rating_user[positive], ml.rating_item[positive], ml.rating_time[positive]
    order = np.lexsort((item, time, user))  # by user, then timestamp, then movie id
    user, item, time = user[order], item[order], time[order]
    counts = np.bincount(user, minlength=ml.n_users)
    starts = np.zeros(ml.n_users + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])

    kept = np.flatnonzero(counts >= min_positives).astype(np.int64)
    val_item = np.full(ml.n_users, -1, dtype=np.int64)
    test_item = np.full(ml.n_users, -1, dtype=np.int64)
    val_time = np.full(ml.n_users, -1, dtype=np.int64)
    test_time = np.full(ml.n_users, -1, dtype=np.int64)
    last = starts[kept + 1] - 1
    test_item[kept], test_time[kept] = item[last], time[last]
    val_item[kept], val_time[kept] = item[last - 1], time[last - 1]

    train_counts = np.zeros(ml.n_users, dtype=np.int64)
    train_counts[kept] = counts[kept] - 2
    train_indptr = np.zeros(ml.n_users + 1, dtype=np.int64)
    np.cumsum(train_counts, out=train_indptr[1:])
    is_train = np.zeros(user.shape[0], dtype=bool)
    for u in kept:
        is_train[starts[u] : starts[u + 1] - 2] = True
    return Split(
        n_users=ml.n_users,
        n_items=ml.n_items,
        users=kept,
        train_indptr=train_indptr,
        train_items=item[is_train],
        train_times=time[is_train],
        val_item=val_item,
        val_time=val_time,
        test_item=test_item,
        test_time=test_time,
    )
