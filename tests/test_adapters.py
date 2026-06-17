"""Smoke tests for the adapter contracts: they are abstract and importable."""

import chess
import pytest

from chess_equity.adapters import (
    EquityModel,
    HumanPolicy,
    ObjectiveEngine,
    ObjectiveEval,
    white_to_move,
)


@pytest.mark.parametrize("cls", [ObjectiveEngine, EquityModel, HumanPolicy])
def test_interfaces_are_abstract(cls):
    with pytest.raises(TypeError):
        cls()  # abstract — cannot instantiate without implementing the method


def test_objective_eval_holds_cp_or_mate():
    assert ObjectiveEval(cp=120.0).cp == 120.0
    assert ObjectiveEval(mate=3).mate == 3


def test_white_to_move_helper():
    assert white_to_move(chess.STARTING_FEN) is True
    board = chess.Board()
    board.push_san("e4")
    assert white_to_move(board.fen()) is False
