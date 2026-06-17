#!/usr/bin/env python3
"""Tests for the Lichess game importer (task 0011).

No network: ``fetch_pgn`` is exercised with a fake opener and the cache, and
``build_game`` runs on an embedded PGN. Needs python-chess + the chess_equity
package (run from the repo with ``PYTHONPATH=src``, like the other web tests).
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # import sibling modules (game_json, import_game)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import game_json  # noqa: E402
import import_game  # noqa: E402
from chess_equity.cli import build_model  # noqa: E402

# Légal-style miniature with Lichess-shaped [%eval] annotations and a mate.
SAMPLE_PGN = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1480"]

1. e4 { [%eval 0.2] } e5 { [%eval 0.15] } 2. Bc4 { [%eval 0.25] } Nc6 { [%eval 0.2] }
3. Qh5 { [%eval 0.1] } Nf6 { [%eval #1] } 4. Qxf7# 1-0
"""


# --- pure parsing -----------------------------------------------------------

@pytest.mark.parametrize(
    "source,expected",
    [
        ("abcd1234", "abcd1234"),
        ("https://lichess.org/abcd1234", "abcd1234"),
        ("https://lichess.org/abcd1234/white", "abcd1234"),
        ("https://lichess.org/abcd1234#33", "abcd1234"),
        ("  lichess.org/abcd1234?foo=1 ", "abcd1234"),
    ],
)
def test_extract_game_id(source, expected):
    assert import_game.extract_game_id(source) == expected


def test_extract_game_id_rejects_garbage():
    with pytest.raises(import_game.ImportError_):
        import_game.extract_game_id("not-an-id")


@pytest.mark.parametrize(
    "token,cp",
    [("0.24", 24.0), ("-1.5", -150.0), ("0", 0.0), ("#3", 10000.0), ("#-2", -10000.0)],
)
def test_eval_to_cp_white(token, cp):
    assert game_json.eval_to_cp_white(token) == cp


def test_eval_to_cp_white_bad_token():
    assert game_json.eval_to_cp_white("") is None
    assert game_json.eval_to_cp_white("oops") is None


# --- fetch + cache ----------------------------------------------------------

def test_fetch_caches_and_avoids_second_request(tmp_path):
    calls = {"n": 0}

    def fake_opener(url, headers, timeout):
        calls["n"] += 1
        assert "game/export/abcd1234" in url
        assert headers["Accept"] == "application/x-chess-pgn"
        return SAMPLE_PGN

    pgn1 = import_game.fetch_pgn("abcd1234", cache_dir=str(tmp_path), opener=fake_opener)
    pgn2 = import_game.fetch_pgn("abcd1234", cache_dir=str(tmp_path), opener=fake_opener)
    assert pgn1 == pgn2 == SAMPLE_PGN
    assert calls["n"] == 1  # second call served from cache, no network
    assert (tmp_path / "abcd1234.pgn").exists()


def test_fetch_sends_token_when_given(tmp_path):
    seen = {}

    def fake_opener(url, headers, timeout):
        seen.update(headers)
        return SAMPLE_PGN

    import_game.fetch_pgn("abcd1234", cache_dir=str(tmp_path), token="abc", opener=fake_opener)
    assert seen.get("Authorization") == "Bearer abc"


def test_fetch_empty_pgn_raises(tmp_path):
    with pytest.raises(import_game.ImportError_):
        import_game.fetch_pgn("abcd1234", cache_dir=str(tmp_path), opener=lambda *a: "  ")


# --- build_game -------------------------------------------------------------

def test_build_game_schema_and_ratings_autofill():
    data = game_json.build_game(SAMPLE_PGN, model=build_model("baseline"))
    g = data["game"]
    # Real ratings auto-filled and present in the band grid (the acceptance criterion).
    assert g["white_elo_default"] == 1500 and g["black_elo_default"] == 1480
    assert 1500 in data["rating_bands"] and 1480 in data["rating_bands"]
    assert g["white"] == "alice" and g["black"] == "bob"
    # Schema the web page consumes.
    assert len(data["moves"]) == 8  # start + 7 plies
    for m in data["moves"]:
        assert m["fen"].count("/") == 7
        key = "%d-%d" % (g["white_elo_default"], g["black_elo_default"])
        assert key in m["equity"] and 0.0 <= m["equity"][key] <= 100.0


def test_build_game_uses_pgn_eval_for_classic_bar():
    data = game_json.build_game(SAMPLE_PGN, model=build_model("baseline"))
    # ply 1 is after 1.e4, annotated [%eval 0.2] -> 20cp White.
    assert data["moves"][1]["cp"] == 20.0
    # The mate (#1 after ...Nf6, then Qxf7#) is decisive for White.
    assert data["moves"][-1]["cp"] >= 9000.0


def test_build_game_grades_present_after_first_ply():
    data = game_json.build_game(SAMPLE_PGN, model=build_model("baseline"))
    assert data["moves"][0]["grade"] is None
    assert all(m["grade"] and m["grade"]["label"] for m in data["moves"][1:])


def test_build_game_rejects_empty_pgn():
    with pytest.raises(ValueError):
        game_json.build_game("", model=build_model("baseline"))


# --- end to end via main() --------------------------------------------------

def test_main_writes_renderable_json(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "abcd1234.pgn").write_text(SAMPLE_PGN, encoding="utf-8")  # pre-seed: no network
    out = tmp_path / "imported-game.json"
    rc = import_game.main(
        ["https://lichess.org/abcd1234", "--out", str(out), "--cache-dir", str(cache)]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["game"]["white"] == "alice"
    assert data["moves"] and data["rating_bands"]
