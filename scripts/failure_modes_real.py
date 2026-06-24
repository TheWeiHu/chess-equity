#!/usr/bin/env python3
"""Build the REAL binned-outcomes failure-modes report (task 0151).

Feeds a real Lichess dataset to :mod:`chess_equity.validate.binned_outcomes` and writes
``reports/failure_modes_real.md``. The data policy (CLAUDE.md) is absolute: this artifact
must come from a real dump — never a fixture, never synthetic rows. Point it at a dataset
built via ``chess-equity data build --month YYYY-MM`` (cached under
``~/.cache/chess-equity/dumps``); the header records the dump label + n so the provenance
travels with the numbers.

    uv run --extra data python scripts/failure_modes_real.py \
        --data ~/.cache/chess-equity/build_0128/dataset.parquet \
        --dump lichess_db_standard_rated_2013-01 \
        --out reports/failure_modes_real.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chess_equity.data.build import load_rows
from chess_equity.validate.binned_outcomes import bin_outcomes, format_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="real dataset parquet/csv (or partition dir)")
    ap.add_argument(
        "--dump",
        default="",
        help="dump label for the header, e.g. lichess_db_standard_rated_2013-01",
    )
    ap.add_argument("--out", default="reports/failure_modes_real.md", help="report path")
    ap.add_argument(
        "--min-n",
        type=int,
        default=30,
        help="drop cells below this row count (under-powered); 1 keeps all",
    )
    ap.add_argument("--seed", type=int, default=0, help="recorded in the header (no sampling here)")
    args = ap.parse_args(argv)

    rows = load_rows(args.data)
    if not rows:
        print(f"no rows loaded from {args.data}", file=sys.stderr)
        return 1

    dump = args.dump or Path(args.data).stem
    cells = bin_outcomes(rows, min_n=args.min_n)
    report = format_report(cells, dump=dump, n=len(rows), seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out} ({len(rows)} rows, {len(cells)} cells, min_n={args.min_n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
