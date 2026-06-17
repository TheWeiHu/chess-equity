"""Pluggable adapter interfaces — the contract later tasks implement.

Three roles, kept separate so models compose:

- :class:`ObjectiveEngine` — the classic perfect-play view (centipawns / mate).
  Stockfish or Lc0 plug in here later.
- :class:`EquityModel` — the thing this project is about: rating-conditioned
  win-probability. The placeholder lives in :mod:`chess_equity.models`; Maia-2's
  value head becomes the real one in task 0005; the regression fit is 0004.
- :class:`HumanPolicy` — P(move | position, rating), i.e. how a human of a given
  strength actually plays. Maia wires up here (0005); it's what lets equity know
  whether an "absurd refutation" will actually be found.

Only :class:`EquityModel` is needed to render a bar. The other two are inputs that
some equity models use and others don't.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import chess

from chess_equity.types import Equity


@dataclass(frozen=True)
class ObjectiveEval:
    """An objective engine's view of a position, from the side-to-move's POV.

    Exactly one of ``cp`` / ``mate`` is meaningful: ``mate`` is set (signed plies to
    mate) for forced mates, otherwise ``cp`` holds the centipawn score.
    """

    cp: Optional[float] = None
    mate: Optional[int] = None


class ObjectiveEngine(ABC):
    """Classic perfect-play evaluation: FEN -> centipawns / mate."""

    @abstractmethod
    def eval(self, fen: str) -> ObjectiveEval:
        """Evaluate ``fen`` from the side-to-move's point of view."""
        raise NotImplementedError


class EquityModel(ABC):
    """Rating-conditioned equity: (FEN, white_elo, black_elo) -> :class:`Equity`.

    Implementations MUST render ``equity_white`` from White's POV so the bar is
    stable as turns alternate (use :meth:`Equity.from_side_to_move`). The CLI and
    bar depend only on this interface — swapping models needs no changes there.
    """

    @abstractmethod
    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        """Evaluate ``fen`` given both players' ratings."""
        raise NotImplementedError


class HumanPolicy(ABC):
    """How a human of a given rating plays: (FEN, elo) -> P(move).

    Returned moves are UCI strings mapping to probabilities that sum to ~1. Maia
    implements this in task 0005.
    """

    @abstractmethod
    def move_probs(self, fen: str, elo: int) -> Dict[str, float]:
        """Probability distribution over legal moves for a player rated ``elo``."""
        raise NotImplementedError


def white_to_move(fen: str) -> bool:
    """True if it is White's turn in ``fen``. Shared helper for equity models."""
    return chess.Board(fen).turn == chess.WHITE
