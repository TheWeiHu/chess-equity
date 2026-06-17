#!/usr/bin/env python3
"""Replace the *hypothesised* practical numbers in ``failure_modes.json`` with
**measured** rating-sliced outcomes from the 0002 dataset (task 0027).

For each curated position, its ``hypothesized_practical_white_<band>`` is a guess at
how the position *class* (its engine eval, at that rating band) actually scores for
White. This script measures that class on a real (eval, ratings, outcome) dataset —
rows whose centipawn eval is within ``--cp-window`` of the position's ``engine_cp`` and
whose rating band matches — and writes back ``measured_practical_white_<band>`` plus
``measured_n`` (the sample size). ``null`` with ``measured_n: 0`` means the dataset had
no row in that class (expected on the tiny committed sample; a full dump from task 0024
fills it in).

Usage:
    python3 baseline/measure_practical.py --data data/sample/dataset.csv [--write]
"""
from __future__ import annotations

import argparse
import json
import os

from chess_equity.data.build import load_rows
from chess_equity.validate.calibration import measure_position_classes
from chess_equity.validate.harness import band_for_avg

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, "failure_modes.json")


def _band_of(pos: dict):
    """Return (hypothesis_value, band) from a position's hypothesized_* field."""
    for key, val in pos.items():
        if key.startswith("hypothesized_practical_white_"):
            return float(val), key.rsplit("_", 1)[-1]
    return None, None


def measure(data_path: str, json_path: str, cp_window: float):
    rows = load_rows(data_path)
    with open(json_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    positions = doc["positions"]
    results = []
    for pos in positions:
        hyp, rating = _band_of(pos)
        if rating is None:
            continue
        # The json key suffix is a raw rating (e.g. "1500"); measure within its band.
        band = band_for_avg(float(rating))
        m = measure_position_classes(
            rows, float(pos["engine_cp"]), band, cp_window=cp_window
        )
        pos[f"measured_practical_white_{rating}"] = (
            round(m.measured_white, 3) if m.measured_white is not None else None
        )
        pos["measured_n"] = m.n
        results.append((pos["id"], f"{rating}->{band}", hyp, m))
    doc["_measured_comment"] = (
        f"measured_practical_white_<band> = mean White result over dataset rows within "
        f"±{cp_window}cp of engine_cp in that rating band (source: {os.path.basename(data_path)}). "
        "null / measured_n=0 means the (committed sample) dataset had no row in the class; "
        "a full Lichess dump (task 0024) yields real numbers. Hypotheses are retained for comparison."
    )
    return doc, results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="path to a built 0002 dataset (csv/parquet)")
    ap.add_argument("--json", default=DEFAULT_JSON, help="failure_modes.json to update")
    ap.add_argument("--cp-window", type=float, default=75.0, help="cp band half-width")
    ap.add_argument("--write", action="store_true", help="write the updated JSON back")
    args = ap.parse_args()

    doc, results = measure(args.data, args.json, args.cp_window)
    for pid, band, hyp, m in results:
        meas = "—" if m.measured_white is None else f"{m.measured_white:.3f}"
        print(f"{pid:24s} @{band:>9s}  hypo {hyp:.2f}  measured {meas} (n={m.n})")
    if args.write:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        print(f"\nwrote {args.json}")
    else:
        print("\n(dry run — pass --write to update the JSON)")


if __name__ == "__main__":
    main()
