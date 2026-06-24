"""``chess-equity highlights`` parser builder."""

from __future__ import annotations

import argparse

from chess_equity.cli._common import add_model_arg


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    hl = sub.add_parser(
        "highlights",
        help="detect drama/clutch moments in a game (task 0020)",
    )
    hl.add_argument("--pgn", required=True, help="PGN file to scan for drama")
    hl.add_argument("--white-elo", type=int, default=None, help="override White rating")
    hl.add_argument("--black-elo", type=int, default=None, help="override Black rating")
    hl.add_argument("--top", type=int, default=5, help="size of the highlight reel (default 5)")
    hl.add_argument("--json", action="store_true", help="emit the reel as JSON")
    hl.add_argument(
        "--depth", type=int, default=2,
        help="Stockfish baseline search depth (also the maia-search ply budget)",
    )
    add_model_arg(hl)
    return hl
