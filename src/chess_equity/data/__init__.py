"""The data pipeline: Lichess PGN dumps -> a tabular (eval, ratings, outcome) dataset.

This is the training + validation substrate for the rating-conditioned model (task
0004) and the validation harness (task 0009). The public surface:

- :class:`~chess_equity.data.schema.PositionRow` — one evaluated position as a row.
- :func:`~chess_equity.data.build.build_dataset` — PGN -> CSV/Parquet.
- :func:`~chess_equity.data.build.load_rows` / ``load_dataframe`` — read it back.

See ``chess-equity data build --help`` for the CLI front door.
"""

from chess_equity.data.build import (
    build_dataset,
    load_dataframe,
    load_rows,
    month_url,
    open_pgn,
)
from chess_equity.data.schema import PositionRow

__all__ = [
    "PositionRow",
    "build_dataset",
    "load_rows",
    "load_dataframe",
    "open_pgn",
    "month_url",
]
