"""``chess-equity score`` parser builder."""

from __future__ import annotations

import argparse

from chess_equity.cli._common import add_model_arg, add_profile_args


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    sc = sub.add_parser(
        "score",
        help="scorecard one game: the score, the real result, and what equity predicts",
    )
    sc.add_argument("--pgn", required=True, help="PGN file (uses its [%%eval] + result)")
    sc.add_argument(
        "--svg", metavar="OUT", default=None,
        help="also write a self-contained shareable SVG scorecard (Twitter/Discord/recap) "
        "to OUT — the visual sibling of the text card; no external fonts/JS",
    )
    sc.add_argument(
        "--white-elo", type=int, default=None,
        help="override White rating (default: the PGN's WhiteElo, else 1500)",
    )
    sc.add_argument(
        "--black-elo", type=int, default=None,
        help="override Black rating (default: the PGN's BlackElo, else 1500)",
    )
    sc.add_argument(
        "--n", type=int, default=500, help="rollout count for --model maia-rollout"
    )
    sc.add_argument(
        "--seed", type=int, default=None, help="RNG seed for --model maia-rollout"
    )
    sc.add_argument(
        "--depth", type=int, default=2,
        help="Stockfish baseline search depth (also the maia-search ply budget)",
    )
    sc.add_argument(
        "--k", type=int, default=4, help="top Maia moves kept per node for --model maia-search"
    )
    add_profile_args(sc)
    add_model_arg(sc)
    return sc
