"""chess_equity — a rating-conditioned win-probability ("equity") chess eval.

The classic centipawn bar assumes perfect play, so a move can only ever be *bad*
(it moves the bar against you) — never *good*. This package reframes evaluation as
**equity**: the probability *you* win, given your rating and your opponent's, so
finding a strong move can move the bar in your favour.

    equity = P(win) + 0.5 * P(draw)   (rendered from White's POV in [0, 100]%)

The public contract everything else implements:

- :class:`~chess_equity.types.WDL` / :class:`~chess_equity.types.Equity` — the
  values an evaluation produces (full win/draw/loss, plus the scalar bar).
- :class:`~chess_equity.adapters.EquityModel` — fen + ratings -> ``Equity``.
- :class:`~chess_equity.adapters.ObjectiveEngine` — fen -> centipawns/mate.
- :class:`~chess_equity.adapters.HumanPolicy` — fen + rating -> move distribution
  (Maia, wired up in task 0005).

Swap in a new ``EquityModel`` and the CLI / bar keep working unchanged.
"""

from chess_equity.adapters import EquityModel, HumanPolicy, ObjectiveEngine
from chess_equity.models import LichessBaselineModel, MaterialEngine
from chess_equity.types import Equity, WDL, lichess_win_percent

__all__ = [
    "WDL",
    "Equity",
    "lichess_win_percent",
    "EquityModel",
    "ObjectiveEngine",
    "HumanPolicy",
    "LichessBaselineModel",
    "MaterialEngine",
]

__version__ = "0.0.1"
