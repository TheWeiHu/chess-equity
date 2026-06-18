"""Placeholder implementations so the package runs end-to-end today.

NONE of this is the real product — it exists so the CLI returns *something* and the
interfaces are exercised before the real models land:

- :class:`MaterialEngine` — a trivial :class:`ObjectiveEngine` scoring only material.
  A real engine (Stockfish/Lc0) replaces it; we use it so we don't depend on an
  external binary just to scaffold.
- :class:`LichessBaselineModel` — the rating-BLIND logistic, wrapped as an
  :class:`EquityModel`. It deliberately IGNORES ratings: it is the baseline our
  thesis aims to beat (task 0009), reproduced here as the placeholder. Task 0004
  (regression) and 0005 (Maia-2 value head) supply the real rating-conditioned ones.
"""

from __future__ import annotations

from math import exp
from typing import Optional

import chess

from chess_equity.adapters import EquityModel, ObjectiveEngine, ObjectiveEval, white_to_move
from chess_equity.types import Equity, WDL, lichess_win_percent

# Standard centipawn material values, White - Black, side-to-move adjusted in eval().
_PIECE_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


class MaterialEngine(ObjectiveEngine):
    """A trivial objective engine: centipawns = material balance only.

    No search, no positional terms — just enough to drive the placeholder equity
    model end-to-end. Replace with Stockfish/Lc0 behind the same interface.
    """

    def eval(self, fen: str) -> ObjectiveEval:
        board = chess.Board(fen)
        if board.is_checkmate():
            # Side to move is mated.
            return ObjectiveEval(mate=0)
        score = 0
        for piece_type, value in _PIECE_CP.items():
            score += value * len(board.pieces(piece_type, chess.WHITE))
            score -= value * len(board.pieces(piece_type, chess.BLACK))
        # Report from the side-to-move's POV.
        cp = score if board.turn == chess.WHITE else -score
        return ObjectiveEval(cp=float(cp))


def _wdl_from_cp(cp: float) -> WDL:
    """Map a centipawn eval to a placeholder WDL (side-to-move POV).

    The SCALAR equity here is Lichess's exact published Win% (:func:`lichess_win_percent`),
    not an approximation. Only the *decomposition* into win/draw/loss is a stand-in shape,
    not a fit: draw mass is highest near equality and decays with |cp|, and win/loss are
    split so the scalar equity is preserved. (The validation gate scores the scalar
    directly, so its baseline is the real Lichess curve regardless of this draw split.)
    """
    equity = lichess_win_percent(cp) / 100.0  # P(win) + 0.5*P(draw), in [0, 1]
    p_draw = 0.5 * exp(-abs(cp) / 300.0)
    p_win = equity - 0.5 * p_draw
    p_loss = 1.0 - p_win - p_draw
    return WDL.from_unnormalized(p_win=p_win, p_draw=p_draw, p_loss=p_loss)


class LichessBaselineModel(EquityModel):
    """Placeholder equity model: rating-blind Lichess Win% over a material eval.

    Ignores ``white_elo``/``black_elo`` on purpose — it is the rating-blind baseline.
    Threading the ratings through the signature keeps the contract honest for the
    real models that will use them.
    """

    SOURCE = "lichess-baseline"

    def __init__(self, engine: ObjectiveEngine | None = None) -> None:
        self.engine = engine or MaterialEngine()

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        obj = self.engine.eval(fen)
        if obj.mate is not None:
            # Side to move is mated (mate=0) or being mated: treat as a loss.
            wdl = WDL(p_win=0.0, p_draw=0.0, p_loss=1.0) if obj.mate <= 0 else WDL(1.0, 0.0, 0.0)
            cp = None
        else:
            cp = obj.cp if obj.cp is not None else 0.0
            wdl = _wdl_from_cp(cp)
        return Equity.from_side_to_move(
            wdl,
            white_to_move=white_to_move(fen),
            source=self.SOURCE,
            cp=cp,
        )


def placeholder_equity_warning(model: object) -> Optional[str]:
    """An advisory string if ``model`` is the rating-blind placeholder, else ``None``.

    The shipped demo defaults to :class:`LichessBaselineModel`, which IGNORES player
    ratings and — when no Stockfish is found — scores material only. So the equity bar
    can look like a naive material count, which surprises anyone expecting Maia-2 (task
    0081). This lets callers that feed a UI (e.g. ``precompute``) say so out loud rather
    than emit a placeholder bar silently. The White-POV direction itself is correct.
    """
    if not isinstance(model, LichessBaselineModel):
        return None
    detail = (
        "material-only, no Stockfish found"
        if isinstance(model.engine, MaterialEngine)
        else "objective-engine centipawns"
    )
    return (
        f"note: '--model baseline' is the rating-blind placeholder ({detail}); the "
        "equity bar is NOT Maia-2 and ignores player ratings. Pass '--model maia2' for "
        "the real rating-conditioned equity."
    )
