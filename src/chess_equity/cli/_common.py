"""Shared building blocks for the ``chess-equity`` CLI subcommand modules.

Holds the constants and reusable argument groups that several subcommand parser
builders under :mod:`chess_equity.cli` reference, so each ``cli/<command>.py``
module is self-contained and the package's ``__init__`` only has to assemble the
parser from them. Pure argparse/data — no model or handler logic lives here.
"""

from __future__ import annotations

import argparse

import chess

START_FEN = chess.STARTING_FEN


def add_model_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        choices=("baseline", "maia2", "wdl-a", "maia-rollout", "maia-search"),
        default="baseline",
        help=(
            "equity model: rating-blind baseline (default), rating-conditioned "
            "maia2, the wdl-a regression, the maia-rollout Monte Carlo oracle, or "
            "the maia-search expectimax (last two slow, non-interactive)"
        ),
    )


def add_profile_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--white-profile",
        help="personalize White: a Lichess username (mined live), or 'player@game.pgn' "
        "to profile from a local PGN offline (task 0086)",
    )
    p.add_argument(
        "--black-profile",
        help="personalize Black: same forms as --white-profile",
    )
