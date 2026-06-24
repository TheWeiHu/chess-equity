"""``chess-equity headline`` parser builder."""

from __future__ import annotations

import argparse


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    from chess_equity.validate.headline import HEADLINE_OUT, SMOKE_DATA

    hd = sub.add_parser(
        "headline",
        help="run the pinned headline thesis comparison (baseline,wdl-a,maia2 -> "
        f"{HEADLINE_OUT}; needs a --with-fen dataset for the maia2 leg)",
    )
    hd.add_argument(
        "--data",
        default=SMOKE_DATA,
        help=f"path to a --with-fen dataset to score (default: {SMOKE_DATA}, the "
        "committed dry-run sample; the real run points this at a full built dump)",
    )
    hd.add_argument("--out", default=HEADLINE_OUT, help=f"report path (default: {HEADLINE_OUT})")
    hd.add_argument(
        "--bootstrap", type=int, default=2000, metavar="N",
        help="paired-bootstrap resamples for the significance CIs (0 disables)",
    )
    hd.add_argument("--seed", type=int, default=0, help="RNG seed for the bootstrap")
    return hd
