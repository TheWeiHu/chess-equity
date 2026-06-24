"""``chess-equity doctor`` parser builder."""

from __future__ import annotations

import argparse


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    dr = sub.add_parser(
        "doctor",
        help="check the optional engines (Stockfish, Maia-2) are installed and working",
    )
    dr.add_argument(
        "--engine",
        action="append",
        choices=["stockfish", "maia2"],
        help="check only this engine (repeatable); default checks all. Use "
        "`--engine stockfish` on a binary-only runner with no torch/Maia-2.",
    )
    dr.add_argument(
        "--evidence",
        action="store_true",
        help="also verify the committed real-data gate reports listed in "
        "reports/SUMMARY.md exist on disk and still state their expected verdict "
        "(the deliberate wdl_net_real FAIL is allowlisted). Reads no datasets.",
    )
    dr.add_argument(
        "--model",
        metavar="NAME",
        default=None,
        help="also preflight the ACTIVE equity model before air: --model wdl-a checks "
        "its committed artifact loads, carries fit provenance (n_train/fit_month), and "
        "produces a finite 0..1 bar; --model baseline checks the objective-engine bar. "
        "Torch-free for baseline/wdl-a; absent provenance WARNs (doesn't fail).",
    )
    return dr
