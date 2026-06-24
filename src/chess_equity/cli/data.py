"""``chess-equity data`` parser builder (build / stamp subcommands)."""

from __future__ import annotations

import argparse


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    data = sub.add_parser("data", help="build / manage the training+validation dataset")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    build = data_sub.add_parser("build", help="parse a Lichess PGN dump into a dataset")
    build.add_argument("--pgn", help="path to a PGN file (plain or .zst)")
    build.add_argument(
        "--month", help="YYYY-MM Lichess month — streams + caches the dump, then builds"
    )
    build.add_argument("--sample", type=int, default=None, help="cap the number of rows")
    build.add_argument("--out", default="data", help="output directory (default: data/)")
    build.add_argument("--format", choices=("csv", "parquet"), default="csv")
    build.add_argument(
        "--with-fen",
        action="store_true",
        help="record each position's FEN (needed to validate board models like Maia; ~3x size)",
    )
    build.add_argument(
        "--partition",
        action="store_true",
        help="write a hive-partitioned dir (tc_bucket=…/rating_bucket=…) for efficient slicing",
    )
    build.add_argument(
        "--dump-dir",
        default=None,
        help="cache dir for downloaded --month dumps (default: ~/.cache/chess-equity/dumps)",
    )

    stamp = data_sub.add_parser(
        "stamp",
        help="backfill the source-month sidecar on an existing dataset (task 0127)",
    )
    stamp.add_argument("path", help="path to a built dataset (csv/parquet file or partitioned dir)")
    stamp.add_argument("month", help="the YYYY-MM Lichess month the dataset was drawn from")
    return data
