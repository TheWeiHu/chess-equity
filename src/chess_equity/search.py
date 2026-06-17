"""Maia-weighted expectimax — equity as an explicit search over human move priors.

Where :class:`~chess_equity.maia2.Maia2Equity` reads equity straight off Maia-2's
value head in one forward pass, and :class:`~chess_equity.rollout.MaiaRolloutModel`
estimates it by sampling whole games to the end, this computes it the *deterministic*
middle way: a small **expectimax** tree where every node averages its children
weighted by how likely Maia thinks each move is at the mover's rating::

    equity(node) = Σ_move  P_maia(move | side_to_move_elo) · equity(child)

Both sides are **expectation** nodes (not min/max): each plays to its own rating's
move distribution, so the whole tree is one expectation of the White-POV leaf equity.
That is exactly Wei's mechanism for the two hard cases:

- an "absurd refutation" Maia gives ~0% probability contributes ~0 to the sum, so the
  bar stays near equal instead of swinging on a move no human will find;
- a position that is objectively holdable but *hard* bleeds equity, because Maia
  assigns real probability mass to the losing human continuations.

Like the rollout oracle, this is deliberately decoupled from Maia: it takes a
:class:`HumanPolicy` (the move priors) and a leaf :class:`EquityModel` (the
depth-cutoff scorer), so it is fully testable with a hand-written policy + the
material baseline and no torch/weights installed. :func:`build_maia_search` wires the
real Maia-2 policy + value head.

**Truncation.** Each node keeps only the top-``k`` moves by Maia probability and
**renormalises** their mass to sum to 1 — the long tail is near-zero probability and
exploring it costs exponentially. The dropped probability mass is reported on the
estimate (``truncated_mass``) rather than silently capped. The search is
``O(k**depth)`` leaf scorings per position, so depth/k are the cost knobs that feed
the 0012 perf work; it is non-interactive at real depths with real Maia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess

from chess_equity.adapters import EquityModel, HumanPolicy, white_to_move
from chess_equity.types import WDL, Equity


@dataclass(frozen=True)
class SearchEstimate:
    """The outcome of an expectimax equity search, all from White's POV.

    ``equity_white`` is on the [0, 100]% bar scale; ``wdl`` is the White-POV triple it
    is derived from. ``n_leaves`` is how many positions were scored by the leaf model
    (cutoff nodes), ``n_terminal`` how many recursion paths bottomed out at a real
    game end, and ``truncated_mass`` is the average per-node Maia probability dropped
    by the top-``k`` cut (0 means k never bit; near 1 would mean k is far too small).
    """

    equity_white: float
    wdl: WDL  # White's POV
    depth: int
    k: int
    n_leaves: int
    n_terminal: int
    truncated_mass: float


def estimate_to_equity(est: SearchEstimate, fen: str, source: str) -> Equity:
    """Render a White-POV :class:`SearchEstimate` as a side-to-move :class:`Equity`."""
    is_white = white_to_move(fen)
    stm_wdl = est.wdl if is_white else est.wdl.flipped()
    return Equity.from_side_to_move(stm_wdl, white_to_move=is_white, source=source)


def _top_k(probs: Dict[str, float], k: int) -> Tuple[List[Tuple[str, float]], float]:
    """Top-``k`` (uci, prob) by probability, renormalised, plus the dropped mass.

    Ties break on the UCI string so the search is deterministic. Returns the kept
    moves with weights summing to 1 and the total probability mass that was dropped
    (for transparency on the estimate).
    """
    if not probs:
        return [], 0.0
    ordered = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = ordered[:k]
    kept_mass = sum(p for _, p in kept)
    total = sum(probs.values())
    dropped = max(total - kept_mass, 0.0)
    if kept_mass <= 0:
        return [], dropped
    return [(uci, p / kept_mass) for uci, p in kept], dropped


class MaiaSearchModel(EquityModel):
    """Equity by Maia-weighted expectimax to a fixed depth.

    ``policy`` supplies P(move | fen, elo) for the side to move; ``leaf`` scores a
    position at the depth cutoff. ``depth`` is the ply budget and ``k`` the top-moves
    kept per node. The result is deterministic given ``depth``/``k`` (no sampling).
    """

    SOURCE = "maia-search"

    def __init__(
        self,
        policy: HumanPolicy,
        leaf: EquityModel,
        *,
        depth: int = 2,
        k: int = 4,
    ) -> None:
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.policy = policy
        self.leaf = leaf
        self.depth = depth
        self.k = k

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        est = self.estimate(fen, white_elo, black_elo)
        return estimate_to_equity(est, fen, self.SOURCE)

    def estimate(self, fen: str, white_elo: int, black_elo: int) -> SearchEstimate:
        """Run the expectimax and aggregate into a White-POV equity estimate."""
        board = chess.Board(fen)
        # Mutable counters threaded through the recursion (kept off the hot return path).
        stats = {"leaves": 0, "terminal": 0, "dropped": 0.0, "nodes": 0}
        white_wdl = self._search(board, white_elo, black_elo, self.depth, stats)
        nodes = max(stats["nodes"], 1)
        return SearchEstimate(
            equity_white=100.0 * white_wdl.equity,
            wdl=white_wdl,
            depth=self.depth,
            k=self.k,
            n_leaves=int(stats["leaves"]),
            n_terminal=int(stats["terminal"]),
            truncated_mass=stats["dropped"] / nodes,
        )

    def _search(
        self,
        board: chess.Board,
        white_elo: int,
        black_elo: int,
        depth: int,
        stats: Dict[str, float],
    ) -> WDL:
        """Return the White-POV WDL of ``board`` under expectimax to ``depth``."""
        if board.is_game_over(claim_draw=True):
            stats["terminal"] += 1
            return self._terminal_wdl(board)
        if depth <= 0:
            stats["leaves"] += 1
            return self._leaf_wdl(board, white_elo, black_elo)

        elo = white_elo if board.turn == chess.WHITE else black_elo
        moves, dropped = _top_k(self.policy.move_probs(board.fen(), elo), self.k)
        stats["nodes"] += 1
        stats["dropped"] += dropped
        if not moves:  # degenerate policy (no probs) — fall back to the leaf model
            stats["leaves"] += 1
            return self._leaf_wdl(board, white_elo, black_elo)

        acc_w = acc_d = acc_l = 0.0
        for uci, weight in moves:
            board.push(chess.Move.from_uci(uci))
            child = self._search(board, white_elo, black_elo, depth - 1, stats)
            board.pop()
            acc_w += weight * child.p_win
            acc_d += weight * child.p_draw
            acc_l += weight * child.p_loss
        # Weights sum to 1 and each child WDL sums to 1, so this already sums to 1;
        # route through from_unnormalized to absorb float drift and keep invariants.
        return WDL.from_unnormalized(acc_w, acc_d, acc_l)

    @staticmethod
    def _terminal_wdl(board: chess.Board) -> WDL:
        """White-POV WDL for a finished game (checkmate / draw rule)."""
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return WDL(0.0, 1.0, 0.0)
        if outcome.winner == chess.WHITE:
            return WDL(1.0, 0.0, 0.0)
        return WDL(0.0, 0.0, 1.0)

    def _leaf_wdl(
        self, board: chess.Board, white_elo: int, black_elo: int
    ) -> WDL:
        """White-POV WDL from the leaf model at a depth cutoff."""
        stats_eq = self.leaf.evaluate(board.fen(), white_elo, black_elo)
        stm_wdl = stats_eq.wdl
        return stm_wdl if board.turn == chess.WHITE else stm_wdl.flipped()


def build_maia_search(
    *,
    depth: int = 2,
    k: int = 4,
    cache_path: Optional[str] = None,
) -> MaiaSearchModel:
    """Wire a production search model: Maia-2 as both the move policy and the leaf.

    Lazy-imports :mod:`chess_equity.maia2` so the common paths never load it. Each
    node's priors come from Maia-2 at the mover's rating; positions at the depth
    cutoff are scored by Maia-2's value head — a self-consistent Maia search whose
    only difference from :class:`~chess_equity.maia2.Maia2Equity` is the explicit
    look-ahead over likely human replies.
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
    return MaiaSearchModel(Maia2Policy(backend), Maia2Equity(backend), depth=depth, k=k)
