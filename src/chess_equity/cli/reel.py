"""``chess-equity reel`` parser builder."""

from __future__ import annotations

import argparse

from chess_equity.cli._common import add_model_arg


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    rl = sub.add_parser(
        "reel",
        help="export a ranked highlight reel (JSON + markdown) from a replayed game (task 0168)",
    )
    rl.add_argument(
        "--pgn",
        default="data/sample/sample_games.pgn",
        help="PGN file to replay (default: committed sample fixture)",
    )
    rl.add_argument("--white-elo", type=int, default=None, help="override White rating")
    rl.add_argument("--black-elo", type=int, default=None, help="override Black rating")
    rl.add_argument("--top", type=int, default=None, help="cap the reel to the top N moments")
    rl.add_argument(
        "--min-magnitude",
        type=float,
        default=None,
        metavar="FLOOR",
        help=(
            "drop moments whose 0..1 drama magnitude is below FLOOR before "
            "ranking/rendering (a quiet-game noise filter; logs how many were dropped)"
        ),
    )
    rl.add_argument(
        "--round",
        dest="round_recap",
        action="store_true",
        help=(
            "cross-game ROUND recap: pool the drama across every game in a multi-game "
            "PGN and label each moment with its source board # + pairing (task 0198)"
        ),
    )
    rl.add_argument(
        "--out-dir",
        default=None,
        help="write reel.json + reel.md here (otherwise print markdown to stdout)",
    )
    rl.add_argument(
        "--html",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help=(
            "emit a self-contained HTML clip player (no deps/CDN, opens offline) to "
            "PATH (or stdout if PATH omitted; reel.html in --out-dir if both given)"
        ),
    )
    rl.add_argument(
        "--srt",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help=(
            "emit the narration as an SRT subtitle file (for Premiere/Resolve/CapCut) "
            "to PATH (or stdout if PATH omitted; reel.srt in --out-dir if both given)"
        ),
    )
    rl.add_argument(
        "--chapters",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help=(
            "emit VOD chapter markers (HH:MM:SS Title lines for a YouTube/Twitch "
            "description) to PATH (or stdout if PATH omitted; reel.chapters.txt in "
            "--out-dir if both given)"
        ),
    )
    rl.add_argument(
        "--posters",
        default=None,
        metavar="DIR",
        help=(
            "write one static SVG poster per ranked moment (board + White-POV equity "
            "bar + social caption) into DIR — shareable social cards"
        ),
    )
    rl.add_argument("--title", default="Highlight reel", help="reel title")
    rl.add_argument(
        "--depth", type=int, default=2,
        help="Stockfish baseline search depth (also the maia-search ply budget)",
    )
    add_model_arg(rl)
    return rl
