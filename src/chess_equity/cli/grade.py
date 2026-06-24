"""``chess-equity grade`` parser builder."""

from __future__ import annotations

import argparse

from chess_equity.cli._common import add_model_arg


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    gr = sub.add_parser("grade", help="grade every move of a PGN by Δequity vs rating peers")
    gr.add_argument("--pgn", required=True, help="PGN file to grade")
    gr.add_argument("--white-elo", type=int, default=1500)
    gr.add_argument("--black-elo", type=int, default=1500)
    gr.add_argument(
        "--depth", type=int, default=2,
        help="Stockfish baseline search depth (also the maia-search ply budget)",
    )
    gr.add_argument(
        "--annotate-pgn", metavar="OUT",
        help="instead of printing, write an equity-annotated PGN to OUT "
             "({[%%equity 0..1]} White-POV + grade label/NAG, preserving [%%eval]/[%%clk])",
    )
    gr.add_argument(
        "--round", action="store_true",
        help="pool a multi-game broadcast PGN and print an accuracy leaderboard ranking "
             "every player (accuracy %%, blunder/mistake counts, mean Δequity) across all "
             "boards; with --summary-json emits per-player rows",
    )
    gr.add_argument(
        "--sort", choices=("accuracy", "lead", "blunders"), default="accuracy",
        help="with --round, pick the PRIMARY leaderboard key: 'accuracy' (default, "
             "accuracy %% desc), 'lead' (mean Δpeer desc — rewards beating peers over "
             "avoiding mistakes), or 'blunders' (fewest first); the other metrics then "
             "name remain as deterministic tie-breaks",
    )
    gr.add_argument(
        "--summary-json", metavar="OUT",
        help="also write the per-side scoreline (grade-label counts, mean Δpeer, "
             "worst move per color) as machine-readable JSON to OUT — or, with --round, "
             "the per-player leaderboard rows",
    )
    gr.add_argument(
        "--json", action="store_true",
        help="with --round, print the leaderboard to stdout as a JSON array of "
             "{rank, player, rating, n_moves, accuracy, avg_delta} (suppresses the text "
             "table; pipeable into broadcast lower-third graphics)",
    )
    gr.add_argument(
        "--csv", action="store_true",
        help="with --round, print the leaderboard to stdout as CSV with the same columns "
             "as --json (suppresses the text table)",
    )
    gr.add_argument(
        "--leaderboard-md", metavar="OUT",
        help="with --round, also write a GitHub/Discord-flavored markdown leaderboard "
             "table (rank, player, accuracy, lead, blunders, worst move; ordered by "
             "--sort) to OUT — a paste-ready caster recap, independent of the stdout "
             "format",
    )
    gr.add_argument(
        "--sparkline", action="store_true",
        help="also print a one-line eighth-block sparkline of the per-ply White-POV "
             "equity series (one block per graded ply) — the swing shape at a glance",
    )
    gr.add_argument(
        "--trajectory-svg", metavar="OUT",
        help="write a standalone SVG win-equity trajectory chart (per-ply White-POV "
             "area/line with a 50%% midline) to OUT — an overlay/VOD asset, the "
             "graphical sibling of --sparkline",
    )
    add_model_arg(gr)
    return gr
