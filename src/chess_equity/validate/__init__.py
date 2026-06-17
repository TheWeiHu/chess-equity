"""The validation gate (task 0009): does rating-conditioned equity predict real game
outcomes better than the rating-blind centipawn baseline?

Public surface:

- :data:`~chess_equity.validate.harness.PREDICTORS` — name -> row-predictor registry
  (ships ``baseline``; Approach A from 0004 registers here).
- :func:`~chess_equity.validate.harness.evaluate` — score predictors over rows,
  overall and sliced by rating band / phase.
- :func:`~chess_equity.validate.harness.format_report` — render reports as Markdown.
- :func:`~chess_equity.validate.split.game_level_split` — held-out train/test split at
  the game level so no game's positions leak across the split (task 0030).
- :mod:`chess_equity.validate.metrics` — log-loss / Brier / ECE (soft-label aware).

Front door: ``chess-equity validate --data <dataset> --models baseline``.
"""

from chess_equity.validate.harness import (
    PREDICTORS,
    SLICERS,
    PredictorReport,
    Scores,
    evaluate,
    format_report,
)
from chess_equity.validate.split import game_level_split

__all__ = [
    "PREDICTORS",
    "SLICERS",
    "Scores",
    "PredictorReport",
    "evaluate",
    "format_report",
    "game_level_split",
]
