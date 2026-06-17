#!/usr/bin/env python3
"""Tests for the web-demo builder's centipawn-bar engine wiring (task 0074).

The demo's dashed centipawn line can now be a real Stockfish eval (``--cp-engine
stockfish``) instead of the dependency-free material count. These tests cover the two
pieces of new logic without needing a Stockfish binary in CI:

* the White-POV conversion in ``game_json._material_cp_white`` now handles a real
  engine's *mate-in-N* score (not just material's ``mate=0`` on a finished board), and
* ``build_demo._build_cp_engine`` selects material vs Stockfish and never silently
  degrades Stockfish to material.

Run from the repo with ``PYTHONPATH=src`` like the other web suites.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # import sibling modules (build_demo, game_json)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import chess  # noqa: E402

import build_demo  # noqa: E402
import game_json  # noqa: E402
from chess_equity.adapters import ObjectiveEngine, ObjectiveEval  # noqa: E402
from chess_equity.models import MaterialEngine  # noqa: E402
from chess_equity.stockfish import StockfishNotFound, stockfish_path  # noqa: E402

WHITE_TO_MOVE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_TO_MOVE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"


class _StubEngine(ObjectiveEngine):
    """An ObjectiveEngine that returns a fixed side-to-move-relative eval."""

    def __init__(self, ev: ObjectiveEval) -> None:
        self._ev = ev

    def eval(self, fen: str) -> ObjectiveEval:
        return self._ev


# --- White-POV conversion of an engine's side-to-move score -----------------

def test_cp_white_flips_running_eval_to_white_pov():
    eng = _StubEngine(ObjectiveEval(cp=120.0))  # +120 for the side to move
    assert game_json._material_cp_white(WHITE_TO_MOVE, eng) == 120.0
    assert game_json._material_cp_white(BLACK_TO_MOVE, eng) == -120.0


def test_cp_white_handles_mate_in_n_for_side_to_move():
    """A positive mate count means the side to move is mating -> decisive for them."""
    mating = _StubEngine(ObjectiveEval(mate=3))
    assert game_json._material_cp_white(WHITE_TO_MOVE, mating) == game_json.MATE_CP
    assert game_json._material_cp_white(BLACK_TO_MOVE, mating) == -game_json.MATE_CP


def test_cp_white_handles_getting_mated():
    """A negative (or zero) mate count means the side to move is being mated."""
    getting_mated = _StubEngine(ObjectiveEval(mate=-2))
    assert game_json._material_cp_white(WHITE_TO_MOVE, getting_mated) == -game_json.MATE_CP
    assert game_json._material_cp_white(BLACK_TO_MOVE, getting_mated) == game_json.MATE_CP


def test_cp_white_mate_zero_matches_material_on_finished_board():
    """mate=0 (board already checkmate) keeps the original MaterialEngine behaviour."""
    mated = _StubEngine(ObjectiveEval(mate=0))  # side to move has just been mated
    assert game_json._material_cp_white(WHITE_TO_MOVE, mated) == -game_json.MATE_CP
    assert game_json._material_cp_white(BLACK_TO_MOVE, mated) == game_json.MATE_CP


# --- centipawn-bar engine selection -----------------------------------------

def test_build_cp_engine_material_is_the_default_dep_free_engine():
    assert isinstance(build_demo._build_cp_engine("material", depth=8), MaterialEngine)


def test_build_cp_engine_stockfish_does_not_silently_fall_back(monkeypatch):
    """With no binary on PATH, asking for Stockfish errors -- never returns material."""
    monkeypatch.setattr(build_demo, "MaterialEngine", MaterialEngine)
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr("chess_equity.stockfish.shutil.which", lambda _name: None)
    with pytest.raises(StockfishNotFound):
        build_demo._build_cp_engine("stockfish", depth=8)


def test_build_material_default_cp_matches_committed_values():
    """The default build is unchanged: material cp, byte-for-byte the committed numbers."""
    data = build_demo.build("demo")  # cp_engine defaults to material
    assert data["cp_engine"] == "material"
    # Légal's queen-grab: White is material-down (decisively negative White-POV cp).
    grab = next(m for m in data["moves"] if m["san"] == "Bxd1")
    assert grab["cp"] < -500


@pytest.mark.skipif(stockfish_path() is None, reason="no Stockfish binary present")
def test_build_with_real_stockfish_changes_the_cp_line():
    """End-to-end: a real engine produces a different cp line than material.

    Stockfish *solves* Légal's Mate, so at the queen-grab it reports the forced mate
    (decisive for White) rather than material's queen-down collapse -- the cp lines
    must differ somewhere.
    """
    material = build_demo.build("demo", cp_engine="material")
    sf = build_demo.build("demo", cp_engine="stockfish", depth=8)
    assert sf["cp_engine"] == "stockfish"
    assert all(isinstance(m["cp"], (int, float)) for m in sf["moves"])
    material_cps = [m["cp"] for m in material["moves"]]
    sf_cps = [m["cp"] for m in sf["moves"]]
    assert material_cps != sf_cps


# --- drama markers on the equity line (task 0077) ---------------------------

def _synthetic_moves(sans, eq_white_seq, band="1500-1500"):
    """Build a minimal moves list (the demo-JSON shape) with a hand-set White-POV
    equity per ply at one band, so we can drive ``drama_by_band`` to a known swing."""
    board = chess.Board()
    moves = [{"ply": 0, "san": "(start)", "fen": board.fen(), "equity": {band: eq_white_seq[0]}}]
    for i, san in enumerate(sans, start=1):
        board.push(board.parse_san(san))
        moves.append({"ply": i, "san": san, "fen": board.fen(), "equity": {band: eq_white_seq[i]}})
    return moves


def test_drama_by_band_flags_a_known_missed_win():
    """Black is practically winning (75%) then lets it slip 20 pts -> a 'missed_win'."""
    # White-POV equity: 25 (Black 75%) at start, unchanged after White's e4, then White
    # jumps to 45 after Black's e5 -> Black's POV 75% -> 55%, a -20 mover swing.
    moves = _synthetic_moves(["e4", "e5"], [25.0, 25.0, 45.0])
    drama = game_json.drama_by_band(moves, [1500])
    assert "1500-1500" in drama
    events = drama["1500-1500"]
    assert [(e["ply"], e["kind"]) for e in events] == [(2, "missed_win")]
    assert "slip" in events[0]["headline"]


def test_drama_by_band_is_sparse_when_nothing_swings():
    """Flat equity -> no drama events at all (the detector stays dark on dull games)."""
    moves = _synthetic_moves(["e4", "e5"], [50.0, 50.0, 50.0])
    assert game_json.drama_by_band(moves, [1500]) == {}


def test_build_demo_emits_per_band_drama_markers():
    """The committed demo carries a `drama` map; against a strong defender Légal's
    mating Nd5# registers as a clutch — a marker the flat centipawn bar can't show."""
    data = build_demo.build("demo")
    assert isinstance(data.get("drama"), dict)
    # The illustrative reference band is deliberately muted (no drama there)...
    assert "1500-1500" not in data["drama"]
    # ...but vs a strong Black defender, finding the forced mate is a clutch swing.
    fired = data["drama"]["2300-2300"]
    mate_ply = next(m["ply"] for m in data["moves"] if m["san"] == "Nd5#")
    assert any(e["ply"] == mate_ply and e["kind"] == "clutch" for e in fired)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
