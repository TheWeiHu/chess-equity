"""Precompute a whole game's equity in one pass → a UI-ready JSON (task 0012).

The web demo (0010) and overlay (0011) shouldn't need a live model backend to scrub
through a finished game: this evaluates every ply once, through the cache, and emits a
self-describing JSON the UI loads instantly. Re-running over the same game (or sharing
the on-disk cache) turns every position into a cache hit, which is how the "warm
per-move latency" target is met in practice.

The per-ply record mirrors the White-POV bar the rest of the project renders:
``equity_white`` in [0, 100]% plus the side-to-move WDL split and the objective cp.
"""

from __future__ import annotations

import io
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import chess
import chess.pgn

from chess_equity.adapters import EquityModel
from chess_equity.cache import CachingEquityModel


@dataclass(frozen=True)
class PlyEquity:
    """Equity at one ply of the game (ply 0 = the starting position, before any move)."""

    ply: int
    san: Optional[str]  # the move that led here; None at the start position
    fen: str
    white_to_move: bool
    equity_white: float
    p_win: float
    p_draw: float
    p_loss: float
    cp: Optional[float]


@dataclass(frozen=True)
class PrecomputedGame:
    """A full game's per-ply equity plus the metadata a UI needs to render it."""

    source: str
    white_elo: int
    black_elo: int
    plies: List[PlyEquity]
    compute_ms: float
    cache_hits: int
    cache_misses: int

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["plies"] = [asdict(p) for p in self.plies]
        return d


def _ply_equity(model: EquityModel, ply: int, san: Optional[str], board: chess.Board,
                white_elo: int, black_elo: int) -> PlyEquity:
    eq = model.evaluate(board.fen(), white_elo, black_elo)
    return PlyEquity(
        ply=ply,
        san=san,
        fen=board.fen(),
        white_to_move=board.turn == chess.WHITE,
        equity_white=eq.equity_white,
        p_win=eq.wdl.p_win,
        p_draw=eq.wdl.p_draw,
        p_loss=eq.wdl.p_loss,
        cp=eq.cp,
    )


def precompute_game(
    model: EquityModel,
    pgn_text: str,
    *,
    white_elo: int = 1500,
    black_elo: int = 1500,
) -> PrecomputedGame:
    """Evaluate every position of the first game in ``pgn_text`` and bundle the result.

    Wraps ``model`` in a :class:`CachingEquityModel` if it isn't one already, so the
    pass is cache-backed and the returned hit/miss counts are meaningful. Timing is the
    wall-clock of the whole pass (cold on first run; near-zero once cached).
    """
    cached: CachingEquityModel = (
        model if isinstance(model, CachingEquityModel) else CachingEquityModel(model)
    )
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("no game found in PGN")

    board = game.board()
    started = time.perf_counter()
    plies = [_ply_equity(cached, 0, None, board, white_elo, black_elo)]
    for ply, move in enumerate(game.mainline_moves(), start=1):
        san = board.san(move)
        board.push(move)
        plies.append(_ply_equity(cached, ply, san, board, white_elo, black_elo))
    compute_ms = (time.perf_counter() - started) * 1000.0

    return PrecomputedGame(
        source=cached.model_key,
        white_elo=white_elo,
        black_elo=black_elo,
        plies=plies,
        compute_ms=compute_ms,
        cache_hits=cached.hits,
        cache_misses=cached.misses,
    )
