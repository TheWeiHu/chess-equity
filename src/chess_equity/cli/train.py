"""``chess-equity train`` parser builder."""

from __future__ import annotations

import argparse


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    tr = sub.add_parser("train", help="fit the wdl-a rating-conditioned WDL model (task 0004)")
    tr.add_argument("--data", required=True, help="path to a built dataset (csv/parquet)")
    tr.add_argument("--out", help="artifact path (default: the packaged wdl_a.json)")
    tr.add_argument("--iters", type=int, default=3000, help="gradient-descent iterations")
    tr.add_argument("--lr", type=float, default=0.5, help="learning rate")
    tr.add_argument("--l2", type=float, default=1e-4, help="L2 regularisation strength")
    tr.add_argument(
        "--train-month",
        default=None,
        help="YYYY-MM the dataset came from, stamped into meta['fit_month'] for the "
        "leakage guard (task 0112) — set it so a held-out eval on a different month "
        "isn't mistaken for in-distribution",
    )
    return tr
