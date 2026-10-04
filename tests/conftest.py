"""A hand-made miniature MovieLens archive whose split can be worked out by hand.

Positives (rating >= 4) per user, ordered by (timestamp, movie id):

* user 1: m1@100, m2@200, m5@200, m6@300   -> train [m1, m2], val m5, test m6
  (m2 and m5 share a timestamp: the movie id breaks the tie; m3@150 is rated 2, ignored)
* user 2: m1@50, m6@60, m10@70, m3@90      -> train [m1, m6], val m10, test m3
* user 3: m2@10, m3@10, m5@10              -> train [m2], val m3, test m5 (all tied)
* user 4: m1@10, m3@30                     -> only two positives: dropped

Catalog (movies with a rating): 1, 2, 3, 5, 6, 10 -> internal ids 0..5. Movie 7 has no rating.
"""

import zipfile
from pathlib import Path

import pytest

USERS = """1::F::1::10::48067
2::M::56::16::70072
3::M::25::15::55117
4::F::45::7::02460
"""
MOVIES = """1::Toy Story (1995)::Animation|Children's|Comedy
2::Jumanji (1995)::Adventure|Children's|Fantasy
3::Café au Lait (1993)::Comedy|Romance
5::Father of the Bride Part II (1995)::Comedy
6::Heat (1995)::Action|Crime|Thriller
7::Sabrina (1995)::Comedy|Romance
10::GoldenEye (1995)::Action|Adventure|Thriller
"""
RATINGS = """1::1::5::100
1::2::4::200
1::3::2::150
1::5::5::200
1::6::4::300
2::1::4::50
2::6::5::60
2::10::4::70
2::2::3::80
2::3::5::90
3::2::5::10
3::3::4::10
3::5::4::10
4::1::5::10
4::2::1::20
4::3::4::30
"""


@pytest.fixture
def tiny_zip(tmp_path: Path) -> Path:
    path = tmp_path / "ml-1m.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ml-1m/users.dat", USERS.encode("latin-1"))
        archive.writestr("ml-1m/movies.dat", MOVIES.encode("latin-1"))
        archive.writestr("ml-1m/ratings.dat", RATINGS.encode("latin-1"))
    return path


# MovieLens-10M layout: half-star ratings, no users.dat, UTF-8, "Children" and extra genres.
# Positives (rating >= 4.0): user 10: m1@5, m2@6, m3@7 (m5 rated 3.5 is not positive);
# user 20: m1@1, m5@2, m2@3, m3@9; user 30: m2@4 only (dropped).
MOVIES_10M = """1::Toy Story (1995)::Adventure|Animation|Children|Comedy|Fantasy
2::Jumanji (1995)::Adventure|Children|Fantasy
3::Amélie (2001)::Comedy|Romance
5::Avatar (2009)::Action|Sci-Fi|IMAX
8::Unrated (1990)::(no genres listed)
"""
RATINGS_10M = """10::1::4.5::5
10::2::4::6
10::3::5::7
10::5::3.5::8
20::1::4::1
20::5::5::2
20::2::4.5::3
20::3::4::9
30::2::4::4
30::3::0.5::5
"""


@pytest.fixture
def tiny_zip_10m(tmp_path: Path) -> Path:
    path = tmp_path / "ml-10m.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ml-10M100K/movies.dat", MOVIES_10M.encode("utf-8"))
        archive.writestr("ml-10M100K/ratings.dat", RATINGS_10M.encode("utf-8"))
    return path
