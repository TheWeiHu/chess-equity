#!/usr/bin/env python3
"""The 'before picture' for task 0003.

Runs the rating-BLIND Lichess Win% baseline over the curated failure-mode set and
prints, for each position, what the baseline claims versus the hypothesised
practical reality — making concrete the two failures the project aims to fix:

  1. "dead 0.00 but practically hard"  — the metric can't see who is playing.
  2. "unequal only via an absurd refutation" — the eval banks on a move no human
     of that rating finds.

The baseline is *exactly* Lichess's rating-blind logistic (the one 0001 ships as
``LichessBaselineModel`` / ``lichess_win_percent``). We import it from the
``chess_equity`` package when it is installed; otherwise we fall back to a local
copy of the same constant so this report runs standalone before 0001 merges.

Usage:  python3 report.py [--json failure_modes.json]
"""
from __future__ import annotations

import argparse
import json
import os
from math import exp

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SET = os.path.join(HERE, "failure_modes.json")

# Keep in lock-step with chess_equity.types.LICHESS_K (0001).
LICHESS_K = 0.00368208

try:  # prefer the package's implementation so there is one source of truth
    from chess_equity.types import lichess_win_percent  # type: ignore
except Exception:  # pragma: no cover - exercised only before 0001 is installed

    def lichess_win_percent(cp: float) -> float:
        """Lichess's rating-blind Win% for a centipawn eval, in [0, 100]."""
        return 50.0 + 50.0 * (2.0 / (1.0 + exp(-LICHESS_K * cp)) - 1.0)


def load_positions(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["positions"] if isinstance(data, dict) else data


def practical_field(pos):
    """The hypothesised practical White score and the rating band it is keyed to."""
    for key, val in pos.items():
        if key.startswith("hypothesized_practical_white_"):
            band = key.rsplit("_", 1)[-1]
            return float(val), band
    return None, None


def measured_field(pos):
    """The measured practical White score (task 0027) and its sample size, if present."""
    for key, val in pos.items():
        if key.startswith("measured_practical_white_"):
            return (float(val) if val is not None else None), int(pos.get("measured_n", 0))
    return None, None


def baseline_white_pct(pos) -> float:
    """Baseline White win% from the (White-POV) engine cp. All positions are W-to-move."""
    return lichess_win_percent(float(pos["engine_cp"]))


def render(positions) -> str:
    lines = []
    lines.append("Rating-blind baseline vs. practical reality (the 'before' picture)")
    lines.append("=" * 70)
    for pos in positions:
        base = baseline_white_pct(pos)
        practical, band = practical_field(pos)
        practical_pct = practical * 100.0 if practical is not None else float("nan")
        gap = abs(base - practical_pct) if practical is not None else float("nan")
        lines.append("")
        lines.append(f"[{pos['category']}] {pos['name']}")
        lines.append(f"  fen        : {pos['fen']}")
        lines.append(f"  engine     : cp={pos['engine_cp']}  ({pos['engine_note']})")
        lines.append(f"  baseline   : White {base:5.1f}%  (rating-blind)")
        lines.append(
            f"  practical* : White {practical_pct:5.1f}%  (hypothesis @~{band})"
            f"   -> gap {gap:4.1f} pts"
        )
        measured, m_n = measured_field(pos)
        if m_n and measured is not None:
            lines.append(
                f"  measured   : White {measured * 100:5.1f}%  (0002 data @~{band}, n={m_n})"
            )
        elif measured is not None or m_n == 0:
            lines.append("  measured   : (no rows in this class on the current dataset)")
        lines.append(f"  why wrong  : {pos['why_baseline_misleads']}")
    lines.append("")
    lines.append("* practical numbers are HYPOTHESES; `measured` is the rating-sliced mean")
    lines.append("  White result for the position CLASS (same engine_cp band) on the 0002")
    lines.append("  dataset (task 0027). Small-sample on the committed fixture; a full dump")
    lines.append("  (0024) makes it decisive — 0009 then settles the thesis.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=DEFAULT_SET)
    args = ap.parse_args()
    print(render(load_positions(args.json)))


if __name__ == "__main__":
    main()
