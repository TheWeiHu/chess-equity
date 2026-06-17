"""Tests for the Stockfish ObjectiveEngine adapter (task 0028).

All run against a *fake* backend, so no Stockfish binary is needed — the real UCI
path is exercised only when a binary is present (the verifier's job), not in CI.
"""
import pytest

from chess_equity.adapters import ObjectiveEngine, ObjectiveEval
from chess_equity.models import MaterialEngine
from chess_equity.stockfish import (
    Analysis,
    StockfishEngine,
    StockfishNotFound,
    resolve_objective_engine,
    stockfish_path,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def fake_backend(eval_, best_move):
    """A backend that always returns the same verdict, recording the depth seen."""
    calls = {}

    def backend(fen, depth):
        calls["fen"] = fen
        calls["depth"] = depth
        return Analysis(eval=eval_, best_move=best_move)

    backend.calls = calls
    return backend


def test_is_an_objective_engine():
    eng = StockfishEngine(fake_backend(ObjectiveEval(cp=12.0), "e2e4"))
    assert isinstance(eng, ObjectiveEngine)


def test_eval_returns_cp_from_backend():
    eng = StockfishEngine(fake_backend(ObjectiveEval(cp=34.0), "e2e4"))
    assert eng.eval(START_FEN) == ObjectiveEval(cp=34.0)


def test_eval_passes_mate_through():
    eng = StockfishEngine(fake_backend(ObjectiveEval(mate=3), "d1h5"))
    out = eng.eval(START_FEN)
    assert out.mate == 3 and out.cp is None


def test_best_move_and_analyse():
    eng = StockfishEngine(fake_backend(ObjectiveEval(cp=5.0), "g1f3"))
    assert eng.best_move(START_FEN) == "g1f3"
    a = eng.analyse(START_FEN)
    assert a.best_move == "g1f3" and a.eval.cp == 5.0


def test_uses_configured_depth():
    be = fake_backend(ObjectiveEval(cp=0.0), None)
    StockfishEngine(be, depth=7).eval(START_FEN)
    assert be.calls["depth"] == 7


def test_real_backend_raises_without_binary(monkeypatch):
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(StockfishNotFound):
        StockfishEngine()  # no backend injected -> tries to locate a real binary


def test_stockfish_path_prefers_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    binary = tmp_path / "sf"
    binary.write_text("#!/bin/sh\n")
    assert stockfish_path(str(binary)) == str(binary)
    # A non-existent explicit path does not count.
    assert stockfish_path(str(tmp_path / "missing")) is None


def test_stockfish_path_reads_env(tmp_path, monkeypatch):
    binary = tmp_path / "sf"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv("STOCKFISH_PATH", str(binary))
    assert stockfish_path() == str(binary)


def test_stockfish_path_none_when_absent(monkeypatch):
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert stockfish_path() is None


# --- resolve_objective_engine: the shipped centipawn bar (task 0043) --------

def test_resolver_falls_back_to_material_without_binary(monkeypatch, recwarn):
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    eng = resolve_objective_engine(depth=12)
    assert isinstance(eng, MaterialEngine)
    assert any(issubclass(w.category, RuntimeWarning) for w in recwarn.list)


def test_resolver_is_silent_when_warn_false(monkeypatch, recwarn):
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    resolve_objective_engine(warn=False)
    assert not recwarn.list


def test_resolver_uses_stockfish_when_binary_present(tmp_path, monkeypatch):
    binary = tmp_path / "sf"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv("STOCKFISH_PATH", str(binary))
    eng = resolve_objective_engine(depth=14)
    assert isinstance(eng, StockfishEngine)
    assert eng.depth == 14  # depth threads through


def test_build_model_baseline_uses_resolved_engine(tmp_path, monkeypatch):
    # No binary -> baseline's centipawn bar still renders, via material.
    from chess_equity.cli import build_model

    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    model = build_model("baseline")
    assert isinstance(model.engine, MaterialEngine)

    # Binary present -> the shipped baseline uses Stockfish at the requested depth.
    binary = tmp_path / "sf"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv("STOCKFISH_PATH", str(binary))
    model2 = build_model("baseline", depth=9)
    assert isinstance(model2.engine, StockfishEngine)
    assert model2.engine.depth == 9
