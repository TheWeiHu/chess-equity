"""Tests for Δequity move grading (task 0008)."""

import io

import chess
import chess.pgn
import pytest

from chess_equity.adapters import EquityModel
from chess_equity.grading import (
    BASE_BANDS,
    EquityGrader,
    UniformPolicy,
    grade_label,
    scaled_bands,
)
from chess_equity.models import LichessBaselineModel
from chess_equity.types import WDL, Equity


# --------------------------------------------------------------------------- #
# UniformPolicy
# --------------------------------------------------------------------------- #


def test_uniform_policy_is_uniform_over_legal_moves():
    probs = UniformPolicy().move_probs(chess.STARTING_FEN, 1500)
    assert len(probs) == 20  # 20 legal opening moves
    assert all(p == pytest.approx(1 / 20) for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0)


def test_uniform_policy_empty_on_terminal_position():
    # Stalemate: no legal moves.
    stalemate = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    assert UniformPolicy().move_probs(stalemate, 1500) == {}


# --------------------------------------------------------------------------- #
# Bands / labels — rating-aware
# --------------------------------------------------------------------------- #


def test_scaled_bands_widen_at_lower_ratings():
    strong = scaled_bands(2000)
    weak = scaled_bands(800)
    assert strong == BASE_BANDS  # base bands at/above 2000
    # Lower rating widens the magnitude of every threshold.
    assert abs(weak[0][0]) > abs(strong[0][0])
    assert abs(weak[-1][0]) > abs(strong[-1][0])


def test_grade_label_thresholds():
    assert grade_label(20.0, 2000) == "brilliant"
    assert grade_label(5.0, 2000) == "good"
    assert grade_label(0.0, 2000) == "ok"
    assert grade_label(-5.0, 2000) == "inaccuracy"
    assert grade_label(-10.0, 2000) == "mistake"
    assert grade_label(-50.0, 2000) == "blunder"


# --------------------------------------------------------------------------- #
# EquityGrader with the real placeholder model
# --------------------------------------------------------------------------- #


def test_capturing_a_hanging_queen_grades_positive():
    # White rook on d1, Black queen hanging on d4. Rxd4 is far above the average
    # legal move, so it must grade POSITIVE vs peers — the whole point of 0008.
    fen = "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"
    grader = EquityGrader(LichessBaselineModel())
    grade = grader.grade_move(fen, chess.Move.from_uci("d1d4"), 1500, 1500)
    assert grade.grade_peer > 0  # beats the rating-typical mix
    assert grade.grade_best == pytest.approx(0.0)  # it IS the best move
    assert grade.label in ("good", "brilliant")


def test_best_move_grade_best_is_zero_others_negative():
    fen = "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"
    grader = EquityGrader(LichessBaselineModel())
    # A move that ignores the free queen leaves equity below the best -> grade_best < 0.
    weak = grader.grade_move(fen, chess.Move.from_uci("e1e2"), 1500, 1500)
    assert weak.grade_best < 0


def test_grade_move_rejects_illegal():
    grader = EquityGrader(LichessBaselineModel())
    with pytest.raises(ValueError):
        grader.grade_move(chess.STARTING_FEN, chess.Move.from_uci("e2e5"), 1500, 1500)


# --------------------------------------------------------------------------- #
# Flagship demo: a centipawn-LOSING move with a POSITIVE equity grade
# --------------------------------------------------------------------------- #


class _MockModel(EquityModel):
    """Equity decoupled from centipawns, to stage the trap case.

    One target position (after the 'trap' move) is given high equity but a *losing*
    centipawn score; every other resulting position is even. This is exactly the
    shape Maia-2 (0005) produces on real traps — a move a rating-peer opponent likely
    refutes wrongly. Here we hand-set the numbers to prove the grader surfaces it.
    """

    def __init__(self, trap_fen: str) -> None:
        self.trap_fen = trap_fen

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        if fen == self.trap_fen:
            # mover-POV equity 80%, but mover-POV cp = -100 (material lost).
            # Equity.cp is opponent-POV after the move, so store +100.
            wdl = WDL.from_unnormalized(0.75, 0.1, 0.15)
            return Equity(wdl=wdl, equity_white=80.0, source="mock", cp=100.0)
        wdl = WDL.from_unnormalized(0.45, 0.1, 0.45)
        return Equity(wdl=wdl, equity_white=50.0, source="mock", cp=0.0)


def test_cp_losing_move_can_have_positive_equity_grade():
    # Pick a real position and treat one legal move as the trap.
    fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    board = chess.Board(fen)
    trap_move = chess.Move.from_uci("e2e4")
    board.push(trap_move)
    trap_fen = board.fen()

    grader = EquityGrader(_MockModel(trap_fen))
    grade = grader.grade_move(fen, trap_move, 1500, 1500)

    # The headline: positive peer-relative grade despite a centipawn LOSS.
    assert grade.grade_peer > 0, "a move stronger than peers must score positive"
    assert grade.cp_loss is not None and grade.cp_loss > 0, "and it lost centipawns"
    assert grade.equity_after == pytest.approx(80.0)


# --------------------------------------------------------------------------- #
# grade_game over a PGN
# --------------------------------------------------------------------------- #


def test_grade_game_grades_every_move():
    pgn = "1. e4 e5 2. Nf3 Nc6 *"
    game = chess.pgn.read_game(io.StringIO(pgn))
    grades = EquityGrader(LichessBaselineModel()).grade_game(game, 1500, 1500)
    assert [g.ply for g in grades] == [1, 2, 3, 4]
    assert [g.san for g in grades] == ["e4", "e5", "Nf3", "Nc6"]
    # Every grade is JSON-friendly and labelled.
    for g in grades:
        assert g.label
        assert "grade_peer" in g.to_dict()


def test_grader_mover_pov_alternates():
    # After 1.e4, it's Black to move; grading 1...e5 must be from Black's POV.
    pgn = "1. e4 e5 *"
    game = chess.pgn.read_game(io.StringIO(pgn))
    grades = EquityGrader(LichessBaselineModel()).grade_game(game, 1500, 1500)
    assert grades[0].mover_white is True
    assert grades[1].mover_white is False
