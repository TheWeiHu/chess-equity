"""Tests for the failure-mode engine verifier (task 0028).

Exercises the pure checking logic with crafted :class:`Analysis` objects and a fake
engine, so the cp-agreement and only-move rules are covered without a real binary.
"""
import importlib.util
import os

from chess_equity.adapters import ObjectiveEval
from chess_equity.stockfish import Analysis

# verify_engine.py lives in baseline/, not the package — load it by path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_HERE, "..", "baseline", "verify_engine.py")
_spec = importlib.util.spec_from_file_location("verify_engine", _PATH)
verify_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_engine)


def draw_pos(**kw):
    base = {"id": "d", "name": "draw", "engine_cp": 0}
    base.update(kw)
    return base


def decisive_pos(cp=1000, **kw):
    base = {"id": "w", "name": "win", "engine_cp": cp}
    base.update(kw)
    return base


# --- cp agreement -----------------------------------------------------------

def test_draw_agrees_when_near_zero():
    r = verify_engine.check_position(draw_pos(), Analysis(ObjectiveEval(cp=20.0), "a1a2"))
    assert r["ok"]


def test_draw_disagrees_when_decisive_cp():
    r = verify_engine.check_position(draw_pos(), Analysis(ObjectiveEval(cp=900.0), "a1a2"))
    assert not r["ok"]


def test_draw_disagrees_on_forced_mate():
    r = verify_engine.check_position(draw_pos(), Analysis(ObjectiveEval(mate=5), "a1a2"))
    assert not r["ok"]


def test_decisive_agrees_on_matching_sign_and_magnitude():
    r = verify_engine.check_position(decisive_pos(1000), Analysis(ObjectiveEval(cp=850.0), "a1a2"))
    assert r["ok"]


def test_decisive_disagrees_on_wrong_sign():
    r = verify_engine.check_position(decisive_pos(1000), Analysis(ObjectiveEval(cp=-850.0), "a1a2"))
    assert not r["ok"]


def test_decisive_disagrees_when_engine_sees_draw():
    r = verify_engine.check_position(decisive_pos(800), Analysis(ObjectiveEval(cp=10.0), "a1a2"))
    assert not r["ok"]


def test_decisive_accepts_forced_mate():
    r = verify_engine.check_position(decisive_pos(1000), Analysis(ObjectiveEval(mate=4), "a1a2"))
    assert r["ok"]


# --- only-move --------------------------------------------------------------

def test_only_move_match_passes():
    pos = decisive_pos(800, only_move_uci="e7e8n")
    r = verify_engine.check_position(pos, Analysis(ObjectiveEval(cp=850.0), "e7e8n"))
    assert r["ok"]
    assert any(c["name"] == "only-move" for c in r["checks"])


def test_only_move_mismatch_fails():
    pos = decisive_pos(800, only_move_uci="e7e8n")
    r = verify_engine.check_position(pos, Analysis(ObjectiveEval(cp=850.0), "e7e8q"))
    assert not r["ok"]


def test_no_only_move_skips_that_check():
    r = verify_engine.check_position(decisive_pos(800), Analysis(ObjectiveEval(cp=850.0), "x"))
    assert [c["name"] for c in r["checks"]] == ["cp"]


# --- end-to-end over a fake engine + the committed set ----------------------

class FakeEngine:
    """Returns whatever the position says it should, so every check passes."""

    def analyse(self, fen):
        # cp matches the curated engine_cp; best move matches any only_move_uci.
        pos = _BY_FEN[fen]
        cp = float(pos["engine_cp"])
        ev = ObjectiveEval(cp=cp if cp != 0 else 0.0)
        return Analysis(eval=ev, best_move=pos.get("only_move_uci", "a1a2"))


_POSITIONS = verify_engine.load_positions(
    os.path.join(_HERE, "..", "baseline", "failure_modes.json")
)
_BY_FEN = {p["fen"]: p for p in _POSITIONS}


def test_committed_set_passes_against_a_matching_engine():
    results = verify_engine.verify(_POSITIONS, FakeEngine())
    assert all(r["ok"] for r in results)
    assert len(results) == len(_POSITIONS)


def test_render_is_readable():
    out = verify_engine.render(verify_engine.verify(_POSITIONS, FakeEngine()))
    assert "PASS" in out and "agree with the engine" in out
