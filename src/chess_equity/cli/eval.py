"""``chess-equity eval`` parser builder."""

from __future__ import annotations

import argparse

from chess_equity.cli._common import START_FEN, add_model_arg, add_profile_args


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    ev = sub.add_parser("eval", help="evaluate a position or a whole game")
    ev.add_argument("fen", nargs="?", default=START_FEN, help="FEN (default: startpos)")
    ev.add_argument("--pgn", help="annotate every move of a PGN file instead")
    ev.add_argument(
        "--fens", metavar="FILE",
        help="batch-score many FENs: one FEN per line from FILE (- for stdin; blank/# lines "
             "skipped). With --json emits a JSON array of {fen, white_equity, label}; else one "
             "'<fen>\\t<pct> (side)' line per FEN. Reuses the single-eval scoring path.",
    )
    ev.add_argument(
        "--json", action="store_true",
        help="with --fens, emit the batch results as a JSON array instead of text lines",
    )
    ev.add_argument(
        "--svg", metavar="OUT",
        help="write a self-contained White-POV equity-bar SVG snapshot for FEN to OUT "
             "(shareable still image; ignored with --pgn)",
    )
    ev.add_argument("--white-elo", type=int, default=1500)
    ev.add_argument("--black-elo", type=int, default=1500)
    ev.add_argument(
        "--n", type=int, default=500, help="rollout count for --model maia-rollout"
    )
    ev.add_argument(
        "--seed", type=int, default=None, help="RNG seed for --model maia-rollout (reproducible)"
    )
    ev.add_argument(
        "--depth", type=int, default=2, help="ply budget for --model maia-search"
    )
    ev.add_argument(
        "--k", type=int, default=4, help="top Maia moves kept per node for --model maia-search"
    )
    add_profile_args(ev)
    add_model_arg(ev)
    return ev
