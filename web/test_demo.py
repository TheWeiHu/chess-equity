#!/usr/bin/env python3
"""Schema + content tests for the web demo's precomputed game (task 0010).

Runs with pytest *or* plain ``python3 test_demo.py`` (stdlib only) so it gates the
data contract the static page (app.js) consumes — independent of whether the JSON was
generated with the illustrative ``demo`` model or real ``maia2``. Asserts the two
acceptance criteria the demo exists to show: the rating slider moves the equity bar,
and there is a move that is green on equity but red on centipawns.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GAME = os.path.join(HERE, "demo-game.json")


def load():
    with open(GAME, "r", encoding="utf-8") as fh:
        return json.load(fh)


def cp_to_white(cp):
    """Mirror app.js cpToWhite — classic Lichess logistic, White win fraction [0,1]."""
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


def test_top_level_schema():
    d = load()
    assert d["rating_bands"], "need rating bands for the sliders"
    assert all(isinstance(b, int) for b in d["rating_bands"])
    assert d["game"]["white_elo_default"] in d["rating_bands"]
    assert d["game"]["black_elo_default"] in d["rating_bands"]
    assert len(d["moves"]) >= 6, "need a few moves to scrub through"


def test_moves_conform_to_schema():
    d = load()
    bands = d["rating_bands"]
    for m in d["moves"]:
        assert isinstance(m["ply"], int)
        assert m["san"]
        assert m["fen"].count("/") == 7, "fen must have 8 ranks"
        assert isinstance(m["cp"], (int, float))
        # The equity grid must cover every (white, black) band combination.
        for we in bands:
            for be in bands:
                key = "%d-%d" % (we, be)
                assert key in m["equity"], "missing equity for " + key
                assert 0.0 <= m["equity"][key] <= 100.0
        assert m["grade"] is None or m["grade"]["label"]


def test_rating_slider_moves_the_equity_bar():
    """Acceptance: dragging a rating slider visibly changes the equity bar.

    Somewhere in the game, holding the position fixed, the White-POV equity must
    differ by a clear margin across the rating grid.
    """
    d = load()
    bands = d["rating_bands"]
    max_spread = 0.0
    for m in d["moves"]:
        vals = [m["equity"]["%d-%d" % (we, be)] for we in bands for be in bands]
        max_spread = max(max_spread, max(vals) - min(vals))
    assert max_spread >= 10.0, (
        "equity should swing >=10pts across the rating grid somewhere; "
        "max spread was %.1f" % max_spread
    )


def test_flagship_green_on_equity_red_on_centipawns():
    """Acceptance: a move that is winning on equity while the centipawn bar says lost.

    Légal's queen sac: White is material-down (centipawn bar collapses) yet mating
    (equity bar stays high). This contradiction is the whole pitch.
    """
    d = load()
    wd, bd = d["game"]["white_elo_default"], d["game"]["black_elo_default"]
    key = "%d-%d" % (wd, bd)
    found = False
    for m in d["moves"]:
        cp_white_pct = cp_to_white(m["cp"]) * 100.0
        eq_white = m["equity"][key]
        if cp_white_pct <= 35.0 and eq_white >= 65.0:
            found = True
    assert found, "expected a position where centipawns say White is lost but equity says winning"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except AssertionError as exc:
            failures += 1
            print("FAIL", t.__name__, "-", exc)
    raise SystemExit(1 if failures else 0)
