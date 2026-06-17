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


# --- multi-game catalog (task 0084) -----------------------------------------

def test_build_accepts_every_catalog_game():
    """Each bundled game builds with ref_equity sized to its move list."""
    for key, spec in build_demo.GAMES.items():
        data = build_demo.build("demo", game=key)
        assert data["game"]["key"] == key
        assert data["game"]["name"] == spec["name"]
        # one node per ply plus the start position
        assert len(data["moves"]) == len(spec["moves"]) + 1


def test_build_default_game_is_legals_unchanged():
    """Omitting --game still builds the flagship Légal's Mate."""
    assert build_demo.build("demo")["game"]["key"] == build_demo.DEFAULT_GAME
    assert build_demo.DEFAULT_GAME == "legals"


def test_manifest_lists_every_game_with_its_file():
    manifest = build_demo.build_manifest()
    assert manifest["default"] == build_demo.DEFAULT_GAME
    keys = {g["key"] for g in manifest["games"]}
    assert keys == set(build_demo.GAMES)
    for g in manifest["games"]:
        assert g["file"] == build_demo.GAMES[g["key"]]["file"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
