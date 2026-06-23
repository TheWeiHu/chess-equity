#!/usr/bin/env python
"""Build a high-rating-ONLY (2000+) eval dataset from a cached Lichess dump (task 0165).

The gate's worst slice is the 2000-2399 band, but on the small 2013-01 dump that band is
only n=415 — far too few to tell a real failure from sampling noise (task 0161 documents
the caveat). This script FIXES the power: it samples a big mixed slice of a cached dump
and keeps only the 2000+ rating bands (mean-Elo >= 2000), so the whole eval set is
high-rating and n_high is in the tens of thousands instead of a few hundred.

It reuses the existing build + load filter — `build_dataset` to parse the dump, then
`load_rows(..., rating_bucket=[...])` (the `rb_sel` pushdown) to keep only the high bands,
then `_write_csv` to re-serialize a single dataset `validate` can read. The dump month is
stamped via `write_source_month` so the leakage guard stays honest (wdl-a's fit_month is
2016-05, so a 2016-05 eval is IN-DISTRIBUTION — this measures power/calibration at proper
n, not clean held-out skill; the clean held-out is the sibling cross-dump-refit task 0164).

Usage:
    uv run python scripts/build_highrating_eval.py \
        --pgn ~/.cache/chess-equity/dumps/lichess_db_standard_rated_2016-05.pgn.zst \
        --month 2016-05 --sample 300000 --out data/highrating_2016-05.csv

The output lands under data/ (gitignored) — the committed deliverable is the report the
`validate` run produces, not this multi-MB CSV.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

from chess_equity.data.build import _write_csv, build_dataset, load_rows
from chess_equity.data.schema import columns as schema_columns, rating_bucket
from chess_equity.data.source_month import write_source_month

# Bands whose floored-mean Elo is >= this are "high rating" (master-ish). rating_bucket()
# labels by the floored mean of both Elos in 200-wide bands, so 2000+ = these labels.
HIGH_ELO_FLOOR = 2000


def _is_high(label: str) -> bool:
    try:
        return int(label) >= HIGH_ELO_FLOOR
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgn", required=True, help="path to the cached Lichess dump (.zst ok)")
    ap.add_argument("--month", help="YYYY-MM the dump is from (stamps the source-month sidecar)")
    ap.add_argument(
        "--sample",
        type=int,
        default=300000,
        help="mixed rows to parse before filtering to 2000+ (~20%% survive)",
    )
    ap.add_argument("--out", required=True, help="output CSV path for the high-rating dataset")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1) Parse a big mixed sample of the dump into a temp CSV.
    with tempfile.TemporaryDirectory() as tmp:
        print(
            f"parsing {args.sample} mixed rows from {args.pgn} ...", file=sys.stderr
        )
        mixed = build_dataset(args.pgn, tmp, sample=args.sample, name="mixed")
        # 2) Keep only the 2000+ rating bands (rb_sel pushdown).
        high_labels = sorted(
            {
                lbl
                for r in load_rows(str(mixed))
                if _is_high(lbl := rating_bucket(r.white_elo, r.black_elo))
            },
            key=int,
        )
        rows = load_rows(str(mixed), rating_bucket=high_labels)

    # 3) Re-serialize a single high-rating dataset.
    n = _write_csv(rows, out, schema_columns())
    if args.month:
        write_source_month(out, args.month)

    bands = Counter(rating_bucket(r.white_elo, r.black_elo) for r in rows)
    print(f"wrote {out}  n_high={n}", file=sys.stderr)
    for lbl in sorted(bands, key=int):
        print(f"  band {lbl}+: {bands[lbl]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
