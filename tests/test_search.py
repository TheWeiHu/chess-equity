"""Tests for the Maia-weighted expectimax search (task 0006).

These run with NO Maia/torch installed: the model takes a ``HumanPolicy`` and a leaf
``EquityModel`` by injection. Most tests drive it with a *scripted* policy + a *stub*
leaf so the expectimax arithmetic is exact and the two headline behaviors — an
unlikely refutation barely moving the bar, and probability mass on losing moves
bleeding equity — are crisp and deterministic. One integration test uses the uniform
policy + the real material baseline.
"""

from __future__ import annotations

from typing import Dict

import chess
import pytest

from chess_equity.adapters import EquityModel, HumanPolicy, white_to_move
from chess_equity.grading import UniformPolicy
from chess_equity.models import LichessBaselineModel
from chess_equity.search import (
    MaiaSearchModel,
    SearchEstimate,
    build_maia_search,
    estimate_to_equity,
)
from chess_equity.types import WDL, Equity

# White Qd1 faces Black Qd8/Ke8: Qxd8+ wins the queen; e1f1 is a quiet equal move.
ROOT = "3qk3/8/8/8/8/8/8/3QK3 w - - 0 1"
AFTER_QXD8 = "3Qk3/8/8/8/8/8/8/4K3 b - - 0 1"  # White up a queen
AFTER_KF1 = "3qk3/8/8/8/8/8/8/3Q1K2 b - - 1 1"  # equal material
AFTER_QD2 = "3qk3/8/8/8/8/8/3Q4/4K3 b - - 1 1"  # equal material (a different quiet move)


class ScriptedPolicy(HumanPolicy):
    """Return hand-set move probabilities per FEN, uniform over legal moves elsewhere."""

    def __init__(self, by_fen: Dict[str, Dict[str, float]]) -> None:
        self.by_fen = by_fen

    def move_probs(self, fen: str, elo: int) -> Dict[str, float]:
        if fen in self.by_fen:
            return dict(self.by_fen[fen])
        moves = [m.uci() for m in chess.Board(fen).legal_moves]
        if not moves:
            return {}
        p = 1.0 / len(moves)
        return {u: p for u in moves}


class StubLeaf(EquityModel):
    """A leaf model returning a fixed *White-POV* equity per FEN (default 0.5).

    Stores values from White's POV and rebuilds the side-to-move WDL the
    :class:`EquityModel` contract expects, so the search's POV handling is exercised
    while the test controls the numbers exactly.
    """

    def __init__(self, white_equity_by_fen: Dict[str, float], default: float = 0.5) -> None:
        self.map = white_equity_by_fen
        self.default = default

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        e = self.map.get(fen, self.default)
        white_wdl = WDL(e, 0.0, 1.0 - e)
        stm = white_wdl if white_to_move(fen) else white_wdl.flipped()
        return Equity(wdl=stm, equity_white=100.0 * e, source="stub")


def test_unlikely_refutation_barely_moves_the_bar() -> None:
    # Same crushing move (Qxd8 -> White ~95%) under two priors: near-zero vs near-one.
    leaf = StubLeaf({AFTER_QXD8: 0.95, AFTER_KF1: 0.50})
    low = MaiaSearchModel(
        ScriptedPolicy({ROOT: {"d1d8": 0.02, "e1f1": 0.98}}), leaf, depth=1, k=8
    )
    high = MaiaSearchModel(
        ScriptedPolicy({ROOT: {"d1d8": 0.98, "e1f1": 0.02}}), leaf, depth=1, k=8
    )
    eq_low = low.evaluate(ROOT, 1500, 1500).equity_white
    eq_high = high.evaluate(ROOT, 1500, 1500).equity_white
    # When no human finds the refutation, the bar stays near the quiet move's 50%.
    assert eq_low < 60.0
    # When the refutation is the likely move, the bar swings to it.
    assert eq_high > 85.0
    assert eq_high > eq_low


def test_mass_on_losing_moves_bleeds_equity() -> None:
    # Objectively equal root, but half the human mass goes to a move that loses.
    leaf = StubLeaf({AFTER_QD2: 0.10, AFTER_KF1: 0.50})
    model = MaiaSearchModel(
        ScriptedPolicy({ROOT: {"d1d2": 0.5, "e1f1": 0.5}}), leaf, depth=1, k=8
    )
    eq = model.evaluate(ROOT, 1500, 1500).equity_white
    assert eq < 45.0  # 0.5*10 + 0.5*50 = 30


def test_renormalization_preserves_constant_leaf_value() -> None:
    # If every reachable leaf is worth the same, the weighted average is that value
    # regardless of k — proving truncated mass is renormalized, not leaked.
    leaf = StubLeaf({}, default=0.7)
    model = MaiaSearchModel(UniformPolicy(), leaf, depth=1, k=3)
    assert model.evaluate(ROOT, 1500, 1500).equity_white == pytest.approx(70.0)


def test_depth_zero_equals_leaf() -> None:
    leaf = StubLeaf({}, default=0.42)
    model = MaiaSearchModel(UniformPolicy(), leaf, depth=0, k=4)
    est = model.estimate(ROOT, 1500, 1500)
    assert est.equity_white == pytest.approx(42.0)
    assert est.n_leaves == 1
    assert est.n_terminal == 0


def test_topk_limits_branching_and_reports_truncation() -> None:
    # 20 legal moves at the startpos, uniform; k=2 keeps two and drops the rest.
    model = MaiaSearchModel(UniformPolicy(), StubLeaf({}), depth=1, k=2)
    est = model.estimate(chess.STARTING_FEN, 1500, 1500)
    assert est.n_leaves == 2  # only the top-k children were expanded
    assert est.truncated_mass > 0.8  # ~18/20 of the mass was dropped at the root


def test_checkmate_short_circuits_to_terminal() -> None:
    # Fool's mate: White is already mated -> White-POV equity 0, counted terminal.
    fools = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    est = MaiaSearchModel(UniformPolicy(), StubLeaf({}), depth=3, k=4).estimate(
        fools, 1500, 1500
    )
    assert est.equity_white == 0.0
    assert est.n_terminal == 1
    assert est.n_leaves == 0


def test_stalemate_is_a_draw() -> None:
    stalemate = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    est = MaiaSearchModel(UniformPolicy(), StubLeaf({}), depth=3, k=4).estimate(
        stalemate, 1500, 1500
    )
    assert est.equity_white == 50.0
    assert est.wdl.p_draw == 1.0


def test_winning_position_favours_white_with_material_leaf() -> None:
    # KQ vs k under the real material baseline: White stays well ahead.
    winning = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    model = MaiaSearchModel(UniformPolicy(), LichessBaselineModel(), depth=2, k=3)
    assert model.evaluate(winning, 1500, 1500).equity_white > 50.0


def test_is_deterministic() -> None:
    model_a = MaiaSearchModel(UniformPolicy(), LichessBaselineModel(), depth=2, k=3)
    model_b = MaiaSearchModel(UniformPolicy(), LichessBaselineModel(), depth=2, k=3)
    winning = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    assert model_a.estimate(winning, 1500, 1500) == model_b.estimate(winning, 1500, 1500)


def test_evaluate_returns_white_pov_equity() -> None:
    model = MaiaSearchModel(UniformPolicy(), StubLeaf({}, default=0.6), depth=1, k=4)
    eq = model.evaluate(ROOT, 1500, 1500)
    assert isinstance(eq, Equity)
    assert eq.source == "maia-search"
    assert eq.equity_white == pytest.approx(60.0)


def test_pov_consistent_for_black_to_move() -> None:
    black_to_move = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1"  # Black to move, down a queen
    model = MaiaSearchModel(UniformPolicy(), LichessBaselineModel(), depth=1, k=3)
    est = model.estimate(black_to_move, 1500, 1500)
    eq = estimate_to_equity(est, black_to_move, "maia-search")
    assert abs(eq.equity_white - est.equity_white) < 1e-9


def test_invalid_depth_rejected() -> None:
    with pytest.raises(ValueError):
        MaiaSearchModel(UniformPolicy(), StubLeaf({}), depth=-1)


def test_invalid_k_rejected() -> None:
    with pytest.raises(ValueError):
        MaiaSearchModel(UniformPolicy(), StubLeaf({}), k=0)


def test_build_maia_search_constructs_without_loading_maia() -> None:
    model = build_maia_search(depth=3, k=5)
    assert isinstance(model, MaiaSearchModel)
    assert model.depth == 3
    assert model.k == 5


def test_search_estimate_is_frozen() -> None:
    est = SearchEstimate(50.0, WDL(0.5, 0.0, 0.5), 2, 4, 3, 0, 0.0)
    with pytest.raises(Exception):
        est.equity_white = 1.0  # type: ignore[misc]
