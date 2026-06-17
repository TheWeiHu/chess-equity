import chess
import pytest

from chess_equity.models import (
    LichessBaselineModel,
    MaterialEngine,
    placeholder_equity_warning,
)


def test_material_engine_scores_start_even():
    assert MaterialEngine().eval(chess.STARTING_FEN).cp == pytest.approx(0.0)


def test_material_engine_side_to_move_pov():
    # White up a queen, but Black to move -> negative from side-to-move POV.
    fen = "4k3/8/8/8/8/8/8/Q3K3 b - - 0 1"
    assert MaterialEngine().eval(fen).cp == pytest.approx(-900.0)


def test_material_engine_reports_mate():
    # Fool's mate position, White is checkmated.
    board = chess.Board()
    for mv in ["f3", "e5", "g4", "Qh4"]:
        board.push_san(mv)
    assert MaterialEngine().eval(board.fen()).mate == 0


def test_baseline_model_startpos_is_fifty_fifty():
    eq = LichessBaselineModel().evaluate(chess.STARTING_FEN, 1500, 1500)
    assert eq.equity_white == pytest.approx(50.0, abs=0.5)
    assert eq.source == "lichess-baseline"
    total = eq.wdl.p_win + eq.wdl.p_draw + eq.wdl.p_loss
    assert total == pytest.approx(1.0)


def test_baseline_model_is_rating_blind():
    """Placeholder ignores ratings — that's the baseline 0009 must beat."""
    model = LichessBaselineModel()
    a = model.evaluate(chess.STARTING_FEN, 800, 2600)
    b = model.evaluate(chess.STARTING_FEN, 2600, 800)
    assert a.equity_white == pytest.approx(b.equity_white)


def test_baseline_model_white_pov_stable():
    # White up a rook should read > 50% regardless of whose turn it is.
    white_turn = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
    black_turn = "4k3/8/8/8/8/8/8/R3K3 b - - 0 1"
    model = LichessBaselineModel()
    assert model.evaluate(white_turn, 1500, 1500).equity_white > 50.0
    assert model.evaluate(black_turn, 1500, 1500).equity_white > 50.0


def test_placeholder_warning_flags_material_baseline():
    # The default baseline (MaterialEngine, no Stockfish) is the rating-blind placeholder.
    msg = placeholder_equity_warning(LichessBaselineModel())
    assert msg is not None
    assert "maia2" in msg and "material-only" in msg


def test_placeholder_warning_none_for_non_placeholder():
    # A non-baseline model (any other object) gets no warning.
    assert placeholder_equity_warning(object()) is None
