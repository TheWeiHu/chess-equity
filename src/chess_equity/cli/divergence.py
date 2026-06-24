"""``chess-equity divergence`` parser builder."""

from __future__ import annotations

import argparse


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    dv = sub.add_parser(
        "divergence",
        help="measure how far the equity bar DIVERGES from the Stockfish bar (task 0171)",
    )
    dv.add_argument("--data", required=True, help="path to a built dataset (csv/parquet)")
    dv.add_argument(
        "--equity",
        default="wdl-a",
        help="the rating-aware equity predictor to compare (default wdl-a; any "
        "validate --models name that reads cp_eval, e.g. baseline+clock)",
    )
    dv.add_argument(
        "--stockfish",
        default="baseline",
        help="the classic Stockfish-bar predictor to diverge from (default baseline: "
        "Lichess Win%% of cp_eval)",
    )
    dv.add_argument("--out", help="write the Markdown report here (default: stdout)")
    return dv
