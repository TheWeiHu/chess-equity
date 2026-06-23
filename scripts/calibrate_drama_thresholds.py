#!/usr/bin/env python3
"""Calibrate the drama trigger thresholds against the REAL per-move Δequity distribution
(task 0170).

``chess_equity.drama`` fires clutch / missed_win / escape / scramble on hand-set
percentage-point constants (``CLUTCH_DELTA`` / ``SLIP_DELTA`` / ``SCRAMBLE_DELTA``).
This script derives those cutoffs from the *real* distribution of per-move practical-equity
swings on a cached Lichess dump, so the highlight reel fires at a principled tail rarity
instead of a guess.

How the swing is computed (the "baseline model" Δequity, per the task):
- each position's White-POV equity is the rating-blind Lichess logistic over the dump's
  real Stockfish ``cp_eval`` — i.e. ``chess_equity.types.lichess_win_percent`` (in [0, 100]),
  which is exactly :class:`~chess_equity.models.LichessBaselineModel`'s cp→equity map;
- the per-move Δequity is the *mover*'s POV change across one half-move, matching
  ``broadcast.GameTracker`` (after − before, mover POV).

KEY ASYMMETRY (documented in the report): under a rating-*blind* best-play eval the mover can
almost never raise their own eval — the eval already assumed their best reply — so positive
Δequity is bounded by eval depth-noise (p99 ≈ +1.5pt) while blunders form a long negative
tail. The clutch *positive-swing* bar therefore cannot be calibrated on the positive-only
quantiles (degenerate); it is set on the |Δequity| magnitude scale, the same scale that is
real on a rating-conditioned model (Maia-2), where a below-rating-typical baseline lets a
strong move genuinely lift practical equity.

Data policy (CLAUDE.md): real dump only. Point ``--data`` at a dataset built via
``chess-equity data build --month YYYY-MM`` (cached under ``~/.cache/chess-equity/dumps``);
the header records the dump label + n so provenance travels with the numbers.

    uv run --extra data python scripts/calibrate_drama_thresholds.py \
        --data /tmp/drama_calib_2016-05/dataset.parquet \
        --dump lichess_db_standard_rated_2016-05 \
        --out reports/drama_thresholds_real.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from chess_equity.data.build import load_rows
from chess_equity.types import lichess_win_percent

# Tail percentiles the thresholds target, on the |Δequity| magnitude distribution.
# Quiet-but-not-silent: a "real" let-it-slip / claw-back swing is the top 5% of moves by
# magnitude; a clutch find is a lower bar (top 10%); a clock-scramble swing is lower still
# (top 15% — the clock is the story, not the size).
SLIP_PCT = 95.0
CLUTCH_PCT = 90.0
SCRAMBLE_PCT = 85.0

# Round each derived cutoff to this granularity for a clean default constant.
ROUND_TO = 0.5

# Position-level "practically winning / losing" gates (mover POV) kept as principled round
# numbers — they are equity *levels*, not swing magnitudes, so the Δequity distribution does
# not inform them. Used here only to report the realised missed_win / escape firing rates.
WIN_LEVEL = 70.0
LOSS_LEVEL = 30.0
SCRAMBLE_SECS = 20.0


def percentile(sorted_vals, q):
    """Linear-interpolated percentile q in [0, 100] over a pre-sorted list (no numpy dep)."""
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 100:
        return sorted_vals[-1]
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= len(sorted_vals):
        return sorted_vals[lo]
    return sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac


def roundto(x, step=ROUND_TO):
    return round(x / step) * step


def per_move_deltas(rows):
    """Mover-POV per-move Δequity (pts) + the mover's pre-move equity, over every game.

    Each ``PositionRow`` is a position with ``side_to_move`` = the side about to move; the
    transition to the next ply is *that* side's move. White-POV equity is the baseline
    logistic over the real ``cp_eval``; the mover reads its own POV (White-POV or its
    complement).
    """
    games = defaultdict(list)
    for r in rows:
        gid = r.game_id if r.game_id is not None else "∅"
        games[gid].append(r)

    deltas = []
    before_levels = []
    for gid, sub in games.items():
        sub.sort(key=lambda r: r.ply)
        eq_w = [lichess_win_percent(float(r.cp_eval)) for r in sub]
        for i in range(len(sub) - 1):
            mover_white = sub[i].side_to_move == "white"
            before = eq_w[i] if mover_white else 100.0 - eq_w[i]
            after = eq_w[i + 1] if mover_white else 100.0 - eq_w[i + 1]
            deltas.append(after - before)
            before_levels.append(before)
    return deltas, before_levels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="real dataset parquet/csv (or partition dir)")
    ap.add_argument("--dump", default="", help="dump label for the header")
    ap.add_argument("--out", default="reports/drama_thresholds_real.md", help="report path")
    args = ap.parse_args(argv)

    rows = load_rows(args.data)
    n_games = len({r.game_id for r in rows})
    deltas, before = per_move_deltas(rows)
    n = len(deltas)

    mags = sorted(abs(d) for d in deltas)
    signed = sorted(deltas)

    slip = roundto(percentile(mags, SLIP_PCT))
    clutch = roundto(percentile(mags, CLUTCH_PCT))
    scramble = roundto(percentile(mags, SCRAMBLE_PCT))

    # Realised firing rates on this dump with the derived cutoffs (drama priority order:
    # missed_win / escape > clutch; scramble is clock-gated and not measurable here — the
    # dataset's clock columns are sparse, so we report only the swing-driven kinds).
    n_missed = n_escape = n_clutch = 0
    for d, b in zip(deltas, before):
        if b >= WIN_LEVEL and d <= -slip:
            n_missed += 1
        elif b <= LOSS_LEVEL and d >= slip:
            n_escape += 1
        elif d >= clutch:
            n_clutch += 1

    def pct(x):
        return f"{100.0 * x / n:.2f}%"

    qs_mag = [50, 75, 80, 85, 90, 95, 97.5, 99]
    qs_signed = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    mag_lines = "\n".join(
        f"| p{q} | {percentile(mags, q):.2f} |" for q in qs_mag
    )
    signed_lines = "\n".join(
        f"| p{q} | {percentile(signed, q):+.2f} |" for q in qs_signed
    )

    dump = args.dump or "(unspecified dump)"
    report = f"""# Drama trigger thresholds — calibrated on the real Δequity distribution (task 0170)

**Provenance.** Built from the cached real Lichess monthly dump **`{dump}`**
via `chess-equity data build --month <YYYY-MM>` (cached under
`~/.cache/chess-equity/dumps/`). Per-move practical-equity swings computed by
`scripts/calibrate_drama_thresholds.py`. **n = {n:,} per-move transitions** over
**{n_games:,} games**. Real positions, real Stockfish `cp_eval`, no synthetic data.

**Δequity definition.** Each position's White-POV equity is the rating-blind Lichess
logistic over the real `cp_eval` (`chess_equity.types.lichess_win_percent`, in [0,100] —
exactly `LichessBaselineModel`'s cp→equity map). The per-move Δequity is the **mover**'s
POV change across one half-move (after − before), matching `broadcast.GameTracker`.

## The key asymmetry (why clutch is calibrated on magnitude, not the positive tail)

Signed mover-POV Δequity (pts):

| quantile | Δ |
|---|---|
{signed_lines}

The distribution is heavily **negative-skewed**: under a rating-*blind* best-play eval the
mover can almost never raise their *own* eval — the eval already assumed their best reply —
so the positive side is bounded by eval depth-noise (p99 ≈ +1.5pt) while blunders form a
long negative tail. A clutch's *positive-swing* bar therefore **cannot** be set from the
positive-only quantiles (degenerate on the baseline). It is set on the **|Δequity| magnitude**
scale below — the same scale that becomes real on a rating-conditioned model (Maia-2), where
a below-rating-typical baseline lets a strong move genuinely lift practical equity. So on the
baseline these constants make the reel *quiet* (the documented "muted on baseline" behaviour);
on Maia-2 they fire at the calibrated rarity.

## |Δequity| magnitude distribution (the calibration basis)

| quantile | \\|Δ\\| (pts) |
|---|---|
{mag_lines}

## Derived thresholds (rounded to {ROUND_TO})

| constant | target tail | percentile of \\|Δ\\| | derived | realised rate on this dump |
|---|---|---|---|---|
| `SLIP_DELTA` | top {100 - SLIP_PCT:.0f}% | p{SLIP_PCT:.0f} = {percentile(mags, SLIP_PCT):.2f} | **{slip}** | missed_win {pct(n_missed)}, escape {pct(n_escape)} |
| `CLUTCH_DELTA` | top {100 - CLUTCH_PCT:.0f}% | p{CLUTCH_PCT:.0f} = {percentile(mags, CLUTCH_PCT):.2f} | **{clutch}** | clutch {pct(n_clutch)} (positive-swing, ≈eval-noise on baseline) |
| `SCRAMBLE_DELTA` | top {100 - SCRAMBLE_PCT:.0f}% | p{SCRAMBLE_PCT:.0f} = {percentile(mags, SCRAMBLE_PCT):.2f} | **{scramble}** | clock-gated; not measurable here (sparse clocks) |

`WIN_LEVEL` (70) / `LOSS_LEVEL` (30) are equity *levels*, not swing magnitudes, so the
Δequity distribution does not inform them — kept as principled round numbers ("practically
winning/losing"). `SCRAMBLE_SECS` (20) is a clock gate, unchanged.

Ordering holds: `SCRAMBLE_DELTA` ({scramble}) < `CLUTCH_DELTA` ({clutch}) < `SLIP_DELTA` ({slip}) —
the scramble bar is the lowest (clock is the story), slip the highest (a "real" let-it-slip).

_Regenerate: `uv run --extra data python scripts/calibrate_drama_thresholds.py --data <parquet> --dump {dump} --out {args.out}`._
"""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"n={n:,} games={n_games:,}")
    print(f"SLIP_DELTA={slip}  CLUTCH_DELTA={clutch}  SCRAMBLE_DELTA={scramble}")
    print(f"realised: missed_win {pct(n_missed)}  escape {pct(n_escape)}  clutch {pct(n_clutch)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
