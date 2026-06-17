"""Tests for the Maia self-play Monte Carlo rollout oracle (task 0007).

These run with NO Maia/torch installed: the model takes a ``HumanPolicy`` and a leaf
``EquityModel`` by injection, so we drive it with the uniform policy + the material
baseline. A fixed seed makes every assertion deterministic.
"""

from __future__ import annotations

from chess_equity.grading import UniformPolicy
from chess_equity.models import LichessBaselineModel
from chess_equity.rollout import (
    MaiaRolloutModel,
    RolloutEstimate,
    build_maia_rollout,
    estimate_to_equity,
)
from chess_equity.types import Equity

# Fool's mate: White to move but checkmated by Qh4# -> Black won, White-POV equity 0.
FOOLS_MATE = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
# Stalemate: Black to move, not in check, no legal move -> draw.
STALEMATE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
# White is up a queen with the move; rollouts to the material leaf should favour White.
WHITE_WINNING = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"


def _model(**kw) -> MaiaRolloutModel:
    return MaiaRolloutModel(UniformPolicy(), LichessBaselineModel(), seed=7, **kw)


def test_checkmate_scores_terminal_equity() -> None:
    # Already terminal (White is mated): every rollout ends at ply 0, White-POV equity 0.
    est = _model(n=16).estimate(FOOLS_MATE, 1500, 1500)
    assert est.equity_white == 0.0
    assert est.n_terminal == 16
    assert est.mean_plies == 0.0
    assert est.wdl.p_loss == 1.0


def test_stalemate_scores_draw() -> None:
    est = _model(n=16).estimate(STALEMATE, 1500, 1500)
    assert est.equity_white == 50.0
    assert est.n_terminal == 16
    assert est.wdl.p_draw == 1.0


def test_winning_position_favours_white() -> None:
    # KQ vs k, White to move: the material leaf at the cutoff keeps White well ahead.
    est = _model(n=64, max_plies=4).estimate(WHITE_WINNING, 1500, 1500)
    assert est.equity_white > 50.0


def test_estimate_is_deterministic_with_seed() -> None:
    a = _model(n=32, max_plies=6).estimate(WHITE_WINNING, 1500, 1500)
    b = _model(n=32, max_plies=6).estimate(WHITE_WINNING, 1500, 1500)
    assert a == b


def test_confidence_interval_brackets_estimate_and_is_in_range() -> None:
    est = _model(n=64, max_plies=6).estimate(WHITE_WINNING, 1500, 1500)
    assert 0.0 <= est.ci_low <= est.equity_white <= est.ci_high <= 100.0
    assert est.n == 64


def test_more_rollouts_tighten_the_interval() -> None:
    narrow = _model(n=400, max_plies=6).estimate(WHITE_WINNING, 1500, 1500)
    wide = _model(n=25, max_plies=6).estimate(WHITE_WINNING, 1500, 1500)
    width = lambda e: e.ci_high - e.ci_low
    assert width(narrow) < width(wide)


def test_evaluate_returns_white_pov_equity() -> None:
    model = _model(n=16, max_plies=4)
    equity = model.evaluate(WHITE_WINNING, 1500, 1500)
    assert isinstance(equity, Equity)
    assert equity.source == "maia-rollout"
    assert 0.0 <= equity.equity_white <= 100.0


def test_evaluate_pov_consistent_for_black_to_move() -> None:
    # equity_white must be White-POV regardless of whose move it is.
    black_to_move = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1"  # Black to move, down a queen
    est = _model(n=16, max_plies=4).estimate(black_to_move, 1500, 1500)
    equity = estimate_to_equity(est, black_to_move, "maia-rollout")
    assert abs(equity.equity_white - est.equity_white) < 1e-9


def test_invalid_n_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        MaiaRolloutModel(UniformPolicy(), LichessBaselineModel(), n=0)


def test_build_maia_rollout_constructs_without_loading_maia() -> None:
    # The factory must not eagerly import torch/weights — it only wires the lazy backend.
    model = build_maia_rollout(n=10, seed=1)
    assert isinstance(model, MaiaRolloutModel)
    assert model.n == 10


def test_rollout_estimate_is_frozen() -> None:
    est = RolloutEstimate(50.0, est_wdl(), 40.0, 60.0, 10, 0, 5.0)
    import pytest

    with pytest.raises(Exception):
        est.equity_white = 1.0  # type: ignore[misc]


def est_wdl():
    from chess_equity.types import WDL

    return WDL(0.5, 0.0, 0.5)
