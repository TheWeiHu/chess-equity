"""Maia self-play Monte Carlo rollouts — the slow, conceptually clean equity oracle.

Where :class:`~chess_equity.maia2.Maia2Equity` reads equity straight off Maia-2's
value head in one forward pass, this estimates it the honest-but-expensive way:
**play the position out.** From a FEN we repeatedly sample White moves from a human
policy at ``white_elo`` and Black moves at ``black_elo`` (so both sides err like
players of their rating), to checkmate / a draw rule / a ply cutoff. Each rollout
yields a White-POV result; averaging many gives the equity plus a confidence
interval on it. At the cutoff (no natural terminal) a leaf :class:`EquityModel`
scores the position so a rollout always returns a number.

This is the **reference oracle** for task 0009 validation and for sanity-checking the
expectimax search (0006) — *not* the interactive bar. It is O(n · plies) policy
queries per position; with real Maia that is hundreds of forward passes, so it is
gated behind explicit invocation (``--model maia-rollout``) and clearly non-interactive.
Throughput is dominated entirely by the policy: with the uniform fallback it is
~instant; with real Maia-2 it is roughly ``n * mean_plies`` inferences per position,
so n=500 to a 80-ply cutoff is up to ~40k inferences — minutes, not milliseconds.

The model is deliberately decoupled from Maia: it takes a :class:`HumanPolicy` (the
move sampler) and a leaf :class:`EquityModel` (the cutoff scorer), so it is fully
testable with the uniform policy + material baseline and no torch/weights installed.
:func:`build_maia_rollout` wires the real Maia-2 policy + value head.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import sqrt
from typing import Dict, Optional, Tuple

import chess

from chess_equity.adapters import EquityModel, HumanPolicy, white_to_move
from chess_equity.types import Equity, WDL

# 95% normal-approximation interval. Rollouts are i.i.d. Bernoulli-ish draws of a
# bounded [0, 1] score, so the sample mean is asymptotically normal and 1.96 SEs is
# a fine CI for the n we run (hundreds+).
_Z_95 = 1.96


@dataclass(frozen=True)
class RolloutEstimate:
    """The outcome of an N-rollout equity estimate, all from White's POV.

    ``equity_white`` and the CI bounds are on the [0, 100]% bar scale. ``ci_low`` /
    ``ci_high`` are the 95% interval for the *mean* (they shrink ~1/sqrt(n)), not the
    spread of individual rollouts. ``n_terminal`` is how many rollouts reached a real
    game end (vs hitting the ply cutoff and being scored by the leaf model).
    """

    equity_white: float
    wdl: WDL  # White's POV
    ci_low: float
    ci_high: float
    n: int
    n_terminal: int
    mean_plies: float


def estimate_to_equity(est: RolloutEstimate, fen: str, source: str) -> Equity:
    """Render a White-POV :class:`RolloutEstimate` as a side-to-move :class:`Equity`."""
    is_white = white_to_move(fen)
    stm_wdl = est.wdl if is_white else est.wdl.flipped()
    return Equity.from_side_to_move(stm_wdl, white_to_move=is_white, source=source)


def _sample_move(probs: Dict[str, float], rng: random.Random) -> Optional[str]:
    """Sample one UCI move ~ ``probs`` (already normalised over legal moves)."""
    if not probs:
        return None
    moves = list(probs.keys())
    weights = list(probs.values())
    return rng.choices(moves, weights=weights, k=1)[0]


class MaiaRolloutModel(EquityModel):
    """Equity by Monte Carlo self-play: sample both sides from a rating policy.

    ``policy`` supplies P(move | fen, elo) for the side to move; ``leaf`` scores a
    position when a rollout hits ``max_plies`` without a natural result. A fixed
    ``seed`` makes the estimate reproducible.
    """

    SOURCE = "maia-rollout"

    def __init__(
        self,
        policy: HumanPolicy,
        leaf: EquityModel,
        *,
        n: int = 500,
        max_plies: int = 80,
        seed: Optional[int] = None,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if max_plies < 0:
            raise ValueError(f"max_plies must be >= 0, got {max_plies}")
        self.policy = policy
        self.leaf = leaf
        self.n = n
        self.max_plies = max_plies
        self.seed = seed

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        est = self.estimate(fen, white_elo, black_elo)
        return estimate_to_equity(est, fen, self.SOURCE)

    def estimate(self, fen: str, white_elo: int, black_elo: int) -> RolloutEstimate:
        """Run ``n`` rollouts and aggregate into a White-POV equity + 95% CI."""
        rng = random.Random(self.seed)
        sum_score = 0.0
        sum_sq = 0.0
        sum_w = sum_d = sum_l = 0.0
        n_terminal = 0
        sum_plies = 0
        for _ in range(self.n):
            score, wdl, plies, terminal = self._rollout_once(
                fen, white_elo, black_elo, rng
            )
            sum_score += score
            sum_sq += score * score
            sum_w += wdl.p_win
            sum_d += wdl.p_draw
            sum_l += wdl.p_loss
            sum_plies += plies
            if terminal:
                n_terminal += 1

        n = self.n
        mean = sum_score / n
        # Population variance of the per-rollout score, then SE of the mean.
        var = max(sum_sq / n - mean * mean, 0.0)
        se = sqrt(var / n)
        half = _Z_95 * se
        wdl_white = WDL.from_unnormalized(sum_w / n, sum_d / n, sum_l / n)
        return RolloutEstimate(
            equity_white=100.0 * mean,
            wdl=wdl_white,
            ci_low=100.0 * max(0.0, mean - half),
            ci_high=100.0 * min(1.0, mean + half),
            n=n,
            n_terminal=n_terminal,
            mean_plies=sum_plies / n,
        )

    def _rollout_once(
        self, fen: str, white_elo: int, black_elo: int, rng: random.Random
    ) -> Tuple[float, WDL, int, bool]:
        """Play one game out; return (White score, White WDL, plies, reached_terminal)."""
        board = chess.Board(fen)
        plies = 0
        while plies < self.max_plies and not board.is_game_over(claim_draw=True):
            elo = white_elo if board.turn == chess.WHITE else black_elo
            uci = _sample_move(self.policy.move_probs(board.fen(), elo), rng)
            if uci is None:  # degenerate policy with no moves — stop and score the leaf
                break
            board.push(chess.Move.from_uci(uci))
            plies += 1
        score, wdl = self._score_leaf(board, white_elo, black_elo)
        return score, wdl, plies, board.is_game_over(claim_draw=True)

    def _score_leaf(
        self, board: chess.Board, white_elo: int, black_elo: int
    ) -> Tuple[float, WDL]:
        """White-POV (scalar, WDL) for a leaf: the true result, or the leaf model."""
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                return 0.5, WDL(0.0, 1.0, 0.0)
            if outcome.winner == chess.WHITE:
                return 1.0, WDL(1.0, 0.0, 0.0)
            return 0.0, WDL(0.0, 0.0, 1.0)
        # Cutoff: defer to the leaf model and convert its side-to-move WDL to White-POV.
        eq = self.leaf.evaluate(board.fen(), white_elo, black_elo)
        white_wdl = eq.wdl if board.turn == chess.WHITE else eq.wdl.flipped()
        return eq.equity_white / 100.0, white_wdl


def build_maia_rollout(
    *,
    n: int = 500,
    max_plies: int = 80,
    seed: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> MaiaRolloutModel:
    """Wire a production rollout model: Maia-2 as both the move policy and the leaf.

    Lazy-imports :mod:`chess_equity.maia2` so the common paths never load it. Both
    sides sample from Maia-2 at their rating, and a position that survives to the ply
    cutoff is scored by Maia-2's value head — a self-consistent Maia oracle.
    """
    from chess_equity.maia2 import (
        CachedBackend,
        Maia2Equity,
        Maia2Policy,
        RealMaia2Backend,
    )

    backend = RealMaia2Backend()
    if cache_path:
        backend = CachedBackend(backend, path=cache_path)
    return MaiaRolloutModel(
        Maia2Policy(backend),
        Maia2Equity(backend),
        n=n,
        max_plies=max_plies,
        seed=seed,
    )
