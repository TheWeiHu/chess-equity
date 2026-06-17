"""Tests for the Maia-2 adapter (task 0005).

Everything runs against a *fake* inference backend — no torch, no checkpoint — so the
suite stays light. The fake mimics ``maia2.inference.inference_each``: it returns a
move distribution plus a side-to-move ``win_prob`` that responds to the ratings, which
is exactly what the rating-conditioned acceptance criteria need.
"""

import sys

import chess
import pytest

from chess_equity.cli import build_model, main
from chess_equity.maia2 import (
    CachedBackend,
    Maia2Equity,
    Maia2NotInstalled,
    Maia2Policy,
    RealMaia2Backend,
    wdl_from_equity,
)
from chess_equity.models import LichessBaselineModel

START = chess.STARTING_FEN


def make_backend(calls=None):
    """A fake Maia-2 backend: win_prob rises with the mover's rating edge.

    move_probs deliberately includes one *illegal* uci ("a1a1") to prove the policy
    filters to legal moves before normalising.
    """

    def backend(fen, elo_self, elo_oppo):
        if calls is not None:
            calls.append((fen, elo_self, elo_oppo))
        # Equity in [0,1] that moves with the rating gap — a stronger side to move
        # gets a higher win_prob for the *same* position.
        edge = (elo_self - elo_oppo) / 4000.0
        win_prob = min(max(0.5 + edge, 0.01), 0.99)
        board = chess.Board(fen)
        legal = [m.uci() for m in board.legal_moves]
        # Skill-sensitive-ish: weight the first legal move more for stronger players.
        probs = {"a1a1": 0.1}  # illegal sentinel that must be dropped
        for i, uci in enumerate(legal):
            probs[uci] = 1.0 + (1.0 if (i == 0 and elo_self >= 2000) else 0.0)
        return probs, win_prob

    return backend


def test_wdl_from_equity_is_valid_and_faithful():
    for e in (0.0, 0.2, 0.5, 0.73, 1.0):
        wdl = wdl_from_equity(e)
        total = wdl.p_win + wdl.p_draw + wdl.p_loss
        assert total == pytest.approx(1.0)
        assert wdl.p_win >= 0 and wdl.p_draw >= 0 and wdl.p_loss >= 0
        # The scalar bar is faithful to Maia-2's win_prob; only the draw split is modelled.
        assert wdl.equity == pytest.approx(e, abs=1e-9)


def test_wdl_draw_mass_peaks_at_equality():
    assert wdl_from_equity(0.5).p_draw > wdl_from_equity(0.9).p_draw
    assert wdl_from_equity(0.5).p_draw > wdl_from_equity(0.1).p_draw


def test_policy_normalizes_over_legal_moves_only():
    policy = Maia2Policy(make_backend())
    probs = policy.move_probs(START, 1500)
    legal = {m.uci() for m in chess.Board(START).legal_moves}
    assert set(probs) <= legal           # the illegal "a1a1" sentinel is gone
    assert "a1a1" not in probs
    assert sum(probs.values()) == pytest.approx(1.0)


def test_policy_shifts_with_rating():
    policy = Maia2Policy(make_backend())
    weak = policy.move_probs(START, 1100)
    strong = policy.move_probs(START, 2200)
    assert weak != strong


def test_equity_is_rating_conditioned():
    """The whole point: same position, different ratings -> different equity."""
    model = Maia2Equity(make_backend())
    a = model.evaluate(START, white_elo=2600, black_elo=800)
    b = model.evaluate(START, white_elo=800, black_elo=2600)
    assert a.equity_white > b.equity_white
    assert a.source == "maia2"
    # Contrast with the rating-blind baseline, which can't tell these apart.
    base = LichessBaselineModel()
    assert base.evaluate(START, 2600, 800).equity_white == pytest.approx(
        base.evaluate(START, 800, 2600).equity_white
    )


def test_equity_white_pov_in_range():
    model = Maia2Equity(make_backend())
    eq = model.evaluate(START, 1500, 1500)
    assert 0.0 <= eq.equity_white <= 100.0
    assert eq.equity_white == pytest.approx(50.0, abs=1.0)


def test_equity_white_pov_stable_across_turn():
    """A White rating edge should read the same side regardless of whose move it is."""
    model = Maia2Equity(make_backend())
    white_turn = "4k3/8/8/8/8/8/8/4K2R w - - 0 1"
    black_turn = "4k3/8/8/8/8/8/8/4K2R b - - 0 1"
    # Strong White vs weak Black -> White-POV equity should be > 50 on either turn.
    assert model.evaluate(white_turn, 2600, 1000).equity_white > 50.0
    assert model.evaluate(black_turn, 2600, 1000).equity_white > 50.0


def test_terminal_positions_bypass_backend():
    """Checkmate/stalemate have no legal moves, which Maia-2's preprocessing can't encode.
    The outcome is settled, so evaluate() must read it off the rules without touching the
    backend: checkmated side-to-move = loss (equity 0), stalemate = draw (equity 50)."""

    def exploding_backend(fen, elo_self, elo_oppo):  # must never be called for terminals
        raise AssertionError(f"backend called for terminal position {fen}")

    model = Maia2Equity(exploding_backend)

    # Fool's mate: White to move and checkmated -> White has lost.
    mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert model.evaluate(mate, 1500, 1500).equity_white == pytest.approx(0.0)

    # Stalemate (White to move, no legal moves, not in check) -> draw.
    stale = "5k2/8/1p1n2p1/2p2n2/8/3K4/1q6/2b5 w - - 6 52"
    assert model.evaluate(stale, 1500, 1500).equity_white == pytest.approx(50.0)


def test_cache_hits_avoid_recompute():
    calls = []
    cached = CachedBackend(make_backend(calls))
    cached(START, 1500, 1500)
    cached(START, 1500, 1500)
    assert len(calls) == 1            # second call served from cache
    assert cached.hits == 1 and cached.misses == 1


def test_cache_persists_across_restart(tmp_path):
    path = str(tmp_path / "maia2.pkl")
    calls = []
    first = CachedBackend(make_backend(calls), path=path)
    first(START, 1500, 1500)
    assert len(calls) == 1

    # A fresh process/instance reads the on-disk cache — no new backend call.
    fresh_calls = []
    second = CachedBackend(make_backend(fresh_calls), path=path)
    second(START, 1500, 1500)
    assert fresh_calls == []
    assert second.hits == 1


def test_real_backend_raises_clean_error_without_maia2(monkeypatch):
    # Force `import maia2` to fail deterministically regardless of the environment.
    monkeypatch.setitem(sys.modules, "maia2", None)
    with pytest.raises(Maia2NotInstalled):
        RealMaia2Backend()._ensure_loaded()


def test_build_model_selects_maia2():
    assert isinstance(build_model("maia2"), Maia2Equity)
    assert not isinstance(build_model("baseline"), Maia2Equity)
    with pytest.raises(ValueError):
        build_model("nonsense")


def test_cli_eval_maia2_reports_missing_install(monkeypatch, capsys, tmp_path):
    # Without maia2 installed, `eval --model maia2` must fail cleanly (exit 1), not crash.
    monkeypatch.setitem(sys.modules, "maia2", None)
    # Isolate the on-disk cache to a fresh, nonexistent path so a populated host cache
    # (~/.cache/chess-equity/maia2.pkl) can't satisfy the lookup and mask the missing
    # backend -> the real backend is invoked and raises Maia2NotInstalled (rc 1).
    monkeypatch.setattr(
        "chess_equity.maia2.DEFAULT_CACHE_PATH", str(tmp_path / "none.pkl")
    )
    rc = main(["eval", START, "--model", "maia2"])
    assert rc == 1
    assert "error" in capsys.readouterr().err.lower()


def test_real_backend_converts_white_pov_to_side_to_move(monkeypatch):
    """maia2's ``inference_each`` returns ``win_prob`` from WHITE's POV, but our Backend
    contract is the side-to-move's equity. The real backend must convert, or every
    black-to-move bar inverts. The fake backend can't catch this (it already speaks
    side-to-move), so we inject a stand-in maia2 whose value head is fixed White-POV.
    """
    import types as pytypes

    fake = pytypes.ModuleType("maia2")
    fake.inference = pytypes.SimpleNamespace(
        prepare=lambda: object(),
        # 0.80 = White's win prob, the same value maia2 reports for either side to move.
        inference_each=lambda model, prepared, fen, elo_self, elo_oppo: ({}, 0.80),
    )
    fake.model = pytypes.SimpleNamespace(from_pretrained=lambda type, device: object())
    monkeypatch.setitem(sys.modules, "maia2", fake)

    backend = RealMaia2Backend()
    white_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    black_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    # White to move: side-to-move == White, so win_prob passes through unchanged.
    assert backend(white_fen, 1500, 1500)[1] == pytest.approx(0.80)
    # Black to move: side-to-move == Black, so it must become 1 - 0.80.
    assert backend(black_fen, 1500, 1500)[1] == pytest.approx(0.20)
