"""0007 acceptance: the rollout oracle agrees with 0006 expectimax (within CI).

Both `MaiaRolloutModel` (Monte Carlo self-play, 0007) and `MaiaSearchModel`
(Maia-weighted expectimax, 0006) estimate the *same* quantity — the policy-weighted
expected outcome of a position — by different means. So they must agree, and this
pins exactly when:

- **Matched horizon.** At `search(depth=1, k≥branching)` vs `rollout(max_plies=1)`,
  the search computes the exact expectation `E_{move~policy}[value(child)]` that the
  rollout Monte-Carlo-estimates. The search value must therefore land inside the
  rollout's 95% CI. (Same per-child scoring: a terminal child uses its game result in
  both; a non-terminal child uses the shared leaf model in both.)
- **Forced line.** A deterministic policy collapses both to one line, so they agree
  *exactly* (the rollout CI degenerates to a point).
- **Mismatched horizon is expected to diverge** — and is the "or divergence
  explained" half of the criterion: a depth-2 expectimax and a rollout played to the
  game's end measure different horizons, so they need not agree. The matched-horizon
  tests above are the apples-to-apples comparison.

All runs use the uniform/scripted policy + the material baseline, so no Maia/torch is
needed; a fixed seed makes every assertion deterministic.
"""
from __future__ import annotations

from typing import Dict

import chess

from chess_equity.adapters import HumanPolicy
from chess_equity.grading import UniformPolicy
from chess_equity.models import LichessBaselineModel
from chess_equity.rollout import MaiaRolloutModel
from chess_equity.search import MaiaSearchModel

# Already-terminal positions (rollout's existing fixtures): both engines must read the
# game result straight off the board.
FOOLS_MATE = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"  # White mated
STALEMATE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"  # draw
# A back-rank mate-in-1 for White (Ra8#), used for the forced-line case.
BACKRANK_M1 = "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1"
# Middlegame-ish positions for the matched-horizon expectation check. Both have a
# capturing move (material variety across first moves), so the rollout has genuine
# variance and a non-degenerate CI — unlike, say, KP-vs-K where every first move keeps
# equal material and the CI collapses to a (float-fuzzy) point.
MIDGAME = [
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Nxe5 grabs a pawn
    "3qk3/8/8/8/8/8/8/3QK3 w - - 0 1",  # Qxd8 wins the queen
]

# Absorbs float-rounding between search's weighted sum and rollout's mean·100.
_TOL = 1e-6


class ScriptedPolicy(HumanPolicy):
    """Hand-set move probabilities per FEN; uniform over legal moves elsewhere."""

    def __init__(self, by_fen: Dict[str, Dict[str, float]]) -> None:
        self.by_fen = by_fen

    def move_probs(self, fen: str, elo: int) -> Dict[str, float]:
        if fen in self.by_fen:
            return dict(self.by_fen[fen])
        moves = [m.uci() for m in chess.Board(fen).legal_moves]
        return {u: 1.0 / len(moves) for u in moves} if moves else {}


def _mating_move(fen: str) -> str:
    """The UCI of a move that delivers checkmate from ``fen`` (asserts one exists)."""
    board = chess.Board(fen)
    for move in board.legal_moves:
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return move.uci()
    raise AssertionError(f"no mate-in-1 in {fen}")


# --- matched horizon: search expectation lands inside the rollout CI ---------

def test_matched_horizon_search_value_within_rollout_ci():
    policy, leaf = UniformPolicy(), LichessBaselineModel()
    # depth=1 with a large k = the full uniform expectation (no top-k pruning), which
    # is exactly what a 1-ply rollout samples.
    search = MaiaSearchModel(policy, leaf, depth=1, k=10_000)
    rollout = MaiaRolloutModel(policy, leaf, n=4000, max_plies=1, seed=0)
    for fen in MIDGAME:
        s = search.estimate(fen, 1500, 1500).equity_white
        r = rollout.estimate(fen, 1500, 1500)
        assert r.ci_low - _TOL <= s <= r.ci_high + _TOL, (
            f"{fen}: expectimax {s:.2f} outside rollout CI "
            f"[{r.ci_low:.2f}, {r.ci_high:.2f}]"
        )


def test_matched_horizon_means_are_close():
    # Beyond "within CI", the point estimates should be near each other at n=4000.
    policy, leaf = UniformPolicy(), LichessBaselineModel()
    fen = MIDGAME[0]
    s = MaiaSearchModel(policy, leaf, depth=1, k=10_000).estimate(fen, 1500, 1500)
    r = MaiaRolloutModel(policy, leaf, n=4000, max_plies=1, seed=0).estimate(fen, 1500, 1500)
    assert abs(s.equity_white - r.equity_white) < 2.0


# --- forced line: deterministic policy => exact agreement --------------------

def test_forced_mate_in_one_agrees_exactly():
    mate = _mating_move(BACKRANK_M1)
    policy = ScriptedPolicy({BACKRANK_M1: {mate: 1.0}})
    leaf = LichessBaselineModel()
    s = MaiaSearchModel(policy, leaf, depth=2, k=4).estimate(BACKRANK_M1, 1500, 1500)
    r = MaiaRolloutModel(policy, leaf, n=8, max_plies=4, seed=0).estimate(BACKRANK_M1, 1500, 1500)
    assert s.equity_white == 100.0
    assert r.equity_white == 100.0
    assert r.ci_low == r.ci_high == 100.0  # zero variance: a forced line
    assert r.ci_low <= s.equity_white <= r.ci_high


# --- already terminal: both read the result off the board --------------------

def test_terminal_positions_agree():
    policy, leaf = UniformPolicy(), LichessBaselineModel()
    search = MaiaSearchModel(policy, leaf, depth=2, k=4)
    rollout = MaiaRolloutModel(policy, leaf, n=8, max_plies=6, seed=0)
    for fen, expected in ((FOOLS_MATE, 0.0), (STALEMATE, 50.0)):
        assert search.estimate(fen, 1500, 1500).equity_white == expected
        r = rollout.estimate(fen, 1500, 1500)
        assert r.equity_white == expected
        assert r.ci_low <= search.estimate(fen, 1500, 1500).equity_white <= r.ci_high


# --- mismatched horizon: divergence is allowed (the "or explained" clause) ---

def test_mismatched_horizon_may_diverge():
    """A deep rollout and a shallow expectimax measure different horizons.

    This documents the criterion's escape hatch: we do NOT require agreement when the
    horizons differ. Asserted weakly (both produce a valid bar) so the test states the
    contract without depending on a brittle gap.
    """
    policy, leaf = UniformPolicy(), LichessBaselineModel()
    fen = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"  # KQ vs k
    deep = MaiaRolloutModel(policy, leaf, n=200, max_plies=40, seed=0).estimate(fen, 1500, 1500)
    shallow = MaiaSearchModel(policy, leaf, depth=2, k=4).estimate(fen, 1500, 1500)
    assert 0.0 <= deep.equity_white <= 100.0
    assert 0.0 <= shallow.equity_white <= 100.0
