"""Core value types for chess equity.

These are deliberately model-agnostic: every :class:`~chess_equity.adapters.EquityModel`
returns an :class:`Equity`, regardless of whether it came from a logistic fit, a Maia
value head, or a search. Downstream consumers (the bar, move grading) only ever touch
these types — never a model's internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Optional

# Lichess's published rating-blind Win% logistic, fit on ~2300-rated games.
# Win% = 50 + 50 * (2 / (1 + exp(-LICHESS_K * cp)) - 1)
# This is the rating-BLIND baseline our whole thesis aims to beat (see task 0009).
LICHESS_K = 0.00368208


def lichess_win_percent(cp: float) -> float:
    """Lichess's rating-blind win percentage for a centipawn eval, in [0, 100].

    Rating-blind by construction — it ignores who is playing. We reproduce it as the
    baseline to beat, not as the product. ``cp`` is from the side-to-move's POV.
    """
    return 50.0 + 50.0 * (2.0 / (1.0 + exp(-LICHESS_K * cp)) - 1.0)


@dataclass(frozen=True)
class WDL:
    """A win/draw/loss probability triple from one side's point of view.

    Probabilities are expected to be non-negative and sum to ~1. Use
    :meth:`normalized` to enforce that after constructing from a noisy source.
    """

    p_win: float
    p_draw: float
    p_loss: float

    def __post_init__(self) -> None:
        for name, value in (
            ("p_win", self.p_win),
            ("p_draw", self.p_draw),
            ("p_loss", self.p_loss),
        ):
            if value < -1e-6:
                raise ValueError(f"{name} must be non-negative, got {value}")

    @classmethod
    def from_unnormalized(cls, p_win: float, p_draw: float, p_loss: float) -> "WDL":
        """Build a valid triple from noisy values: clamp negatives, rescale to sum 1.

        A constructed :class:`WDL` always satisfies the non-negative invariant, so a
        model whose arithmetic can drift slightly out of range should come through
        here rather than the bare constructor.
        """
        win = max(p_win, 0.0)
        draw = max(p_draw, 0.0)
        loss = max(p_loss, 0.0)
        total = win + draw + loss
        if total <= 0:
            return cls(0.0, 1.0, 0.0)
        return cls(win / total, draw / total, loss / total)

    @property
    def equity(self) -> float:
        """Scalar equity = P(win) + 0.5 * P(draw), in [0, 1]."""
        return self.p_win + 0.5 * self.p_draw

    def flipped(self) -> "WDL":
        """The same triple from the opponent's point of view (win <-> loss)."""
        return WDL(p_win=self.p_loss, p_draw=self.p_draw, p_loss=self.p_win)


@dataclass(frozen=True)
class Equity:
    """A complete equity evaluation of a position.

    ``wdl`` is from the side-to-move's POV; ``equity_white`` is the bar value in
    [0, 100]% always rendered from White's POV (so the bar is stable as turns
    alternate). ``cp`` is the optional objective centipawn eval, kept for
    side-by-side display against the classic bar.
    """

    wdl: WDL
    equity_white: float
    source: str
    cp: Optional[float] = None

    @classmethod
    def from_side_to_move(
        cls,
        wdl: WDL,
        *,
        white_to_move: bool,
        source: str,
        cp: Optional[float] = None,
    ) -> "Equity":
        """Build an ``Equity`` from a side-to-move WDL, deriving the White-POV bar."""
        white_wdl = wdl if white_to_move else wdl.flipped()
        return cls(
            wdl=wdl,
            equity_white=100.0 * white_wdl.equity,
            source=source,
            cp=cp,
        )
