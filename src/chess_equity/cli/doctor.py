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
        "--broadcast",
        "--feed",
        dest="broadcast",
        metavar="SPEC",
        default=None,
        help="also run a go-live preflight on a broadcast feed before air: a Lichess "
        "round id, a PGN URL, or a local .pgn file. Verifies the feed is reachable "
        "and emitting at least one parseable move (no torch/engine needed).",
    )
    dr.add_argument(
        "--token", default=None, help="Lichess API token for --broadcast round feeds (optional)"
    )
    dr.add_argument(
        "--overlay",
        action="store_true",
        help="also run a go-live preflight on the streaming overlay bundle: its "
        "HTML/JS assets parse and the bundled replay + live overlay events conform to "
        "the documented event schema (no torch/engine/network needed).",
    )
    dr.add_argument(
        "--serve-sse",
        action="store_true",
        help="also run a go-live preflight on the live SSE wiring OBS points at: bind "
        "`broadcast --serve-sse` on an ephemeral port over a local PGN replay and confirm "
        "`/sse` is reachable and emits >=1 overlay event (no torch/engine/network needed).",
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
