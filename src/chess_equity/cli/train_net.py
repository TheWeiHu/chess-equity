"""``chess-equity train-net`` parser builder."""

from __future__ import annotations

import argparse


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    tn = sub.add_parser(
        "train-net",
        help="fit the end-to-end board → WDL net (Approach D, task 0013; needs torch + a --with-fen dataset)",
    )
    tn.add_argument("--data", required=True, help="path to a --with-fen dataset (csv/parquet)")
    tn.add_argument("--out", help="artifact path (default: the packaged wdl_net.pt)")
    tn.add_argument("--epochs", type=int, default=8, help="training epochs (default 8)")
    tn.add_argument("--batch-size", type=int, default=512, help="minibatch size (default 512)")
    tn.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (default 1e-3)")
    tn.add_argument("--seed", type=int, default=0, help="RNG seed for shuffling + init")
    tn.add_argument(
        "--train-month",
        default=None,
        help="YYYY-MM the dataset came from, stamped into the artifact for the leakage guard",
    )
    return tn
