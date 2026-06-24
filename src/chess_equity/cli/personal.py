"""``chess-equity personal`` parser builder."""

from __future__ import annotations

import argparse

from chess_equity.cli._common import START_FEN


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    pp = sub.add_parser(
        "personal",
        help="mine a player's per-phase quality profile and personalize the bar (task 0014)",
    )
    pp.add_argument("--user", help="Lichess username to mine over the network")
    pp.add_argument("--pgn", help="local PGN file to profile from instead (offline)")
    pp.add_argument("--name", help="which player in --pgn to profile (defaults to --user)")
    pp.add_argument("--max-games", type=int, default=50, help="cap mined games (default 50)")
    pp.add_argument("--token", default=None, help="Lichess API token (optional, raises rate limit)")
    pp.add_argument("--json", action="store_true", help="emit the profile as JSON")
    pp.add_argument(
        "--demo",
        action="store_true",
        help="also show band-average vs personalized equity for --fen",
    )
    pp.add_argument("--fen", default=START_FEN, help="position for --demo (default: startpos)")
    return pp
