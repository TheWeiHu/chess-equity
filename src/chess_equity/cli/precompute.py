"""``chess-equity precompute`` parser builder."""

from __future__ import annotations

import argparse

from chess_equity.cli._common import add_model_arg, add_profile_args


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    pc = sub.add_parser(
        "precompute",
        help="evaluate a whole game's equity into a UI-ready JSON (task 0012)",
    )
    pc.add_argument("--pgn", required=True, help="PGN file to precompute")
    pc.add_argument("--white-elo", type=int, default=1500)
    pc.add_argument("--black-elo", type=int, default=1500)
    pc.add_argument("--out", help="write the JSON here (default: stdout)")
    pc.add_argument(
        "--cache", help="persistent cache path for warm restarts (omit = in-memory only)"
    )
    pc.add_argument(
        "--depth", type=int, default=2,
        help="Stockfish baseline search depth (also the maia-search ply budget)",
    )
    add_profile_args(pc)
    add_model_arg(pc)
    return pc
