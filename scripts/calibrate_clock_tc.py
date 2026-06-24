#!/usr/bin/env python3
"""Calibrate the per-time-control flag-risk multipliers against REAL time-forfeit
outcomes on a cached Lichess dump (task 0268).

``chess_equity.clock._TC_FLAG_MULTIPLIER`` scales the modelled per-move flag risk by
time control (a bullet scramble is far deadlier than a classical one). The values were
hand-set with sensible *shape* (bullet 1.0 > blitz > rapid > classical) but never fit.
This script derives the relative deadliness from the *real* rate at which games of each
time control actually end in a time forfeit, so the knob reflects measured Lichess
behaviour instead of a guess.

What it measures (game-level, no ``[%clk]`` needed):
- every game in the dump carries ``[TimeControl]`` (bucketed with the SAME
  :func:`chess_equity.data.schema.tc_bucket` production uses) and ``[Termination]``;
- ``Termination "Time forfeit"`` is a real flag loss. The per-bucket flag rate is
  ``time-forfeit games / total games`` — the empirical analogue of "how often does a
  side flag in this time control".
- the multiplier is that rate normalised so **bullet = 1.0** (the model's reference),
  i.e. ``rate[bucket] / rate[bullet]``.

WHY THE PER-CLOCK-BAND KNOBS ARE NOT TOUCHED HERE: ``SCRAMBLE_SCALE`` (the shape of
``time_pressure`` vs seconds remaining) and the band split of ``MAX_FLAG_RISK`` /
``FLAG_RISK_ALERT_THRESHOLD`` need per-move ``[%clk]`` clocks to fit. The cached
2016-05 dump predates ``[%clk]`` (see ``clock_coverage.py``), so the clock-band slice
is degenerate on it — that half of the calibration is BLOCKED on a >=2017-04 dump and
is left for a follow-up. This script deliberately calibrates only the knob that real
*cached* data can justify.

Data policy (CLAUDE.md): real dump only. Point ``--dump`` at a ``.pgn.zst`` cached under
``~/.cache/chess-equity/dumps``; the report records the dump label + n so provenance
travels with the numbers.

    uv run python scripts/calibrate_clock_tc.py \
        --dump ~/.cache/chess-equity/dumps/lichess_db_standard_rated_2016-05.pgn.zst \
        --out reports/clock_calibration_real.md
"""

from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

from chess_equity.clock import _TC_FLAG_MULTIPLIER
from chess_equity.data.schema import tc_bucket

# Display / calibration order. ``correspondence`` is reported for completeness but its
# multiplier stays pinned at 0.0 by design (days per move -> no flag pressure), so it is
# never derived from data.
BUCKET_ORDER = ("bullet", "blitz", "rapid", "classical", "correspondence")
REFERENCE_BUCKET = "bullet"
ROUND_TO = 2  # decimals for the derived multiplier constants


def stream_header_lines(dump_path: Path):
    """Yield the dump's text lines, decompressing the ``.zst`` with ``zstdcat``.

    Streamed (the multi-GB dump is never fully held in memory), header-line only work is
    done by the caller. ``zstdcat`` is the portable decompressor already required to read
    these dumps.
    """
    proc = subprocess.Popen(
        ["zstdcat", str(dump_path)],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1 << 20,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            yield line
    finally:
        proc.stdout.close()
        proc.wait()


def _tag_value(line: str) -> str:
    """Extract the quoted value from a PGN header line ``[Key "value"]``."""
    first = line.find('"')
    last = line.rfind('"')
    if first == -1 or last <= first:
        return ""
    return line[first + 1 : last]


def tally(dump_path: Path):
    """One streaming pass: per tc_bucket counts of total games and time-forfeit games."""
    total: dict[str, int] = defaultdict(int)
    forfeit: dict[str, int] = defaultdict(int)
    games = 0
    cur_tc = ""
    cur_term = ""

    def flush():
        nonlocal games
        if not cur_tc:
            return
        bucket = tc_bucket(cur_tc)
        total[bucket] += 1
        if cur_term == "Time forfeit":
            forfeit[bucket] += 1
        games += 1

    for line in stream_header_lines(dump_path):
        if line.startswith("[Event "):
            flush()
            cur_tc = ""
            cur_term = ""
        elif line.startswith("[TimeControl "):
            cur_tc = _tag_value(line)
        elif line.startswith("[Termination "):
            cur_term = _tag_value(line)
    flush()  # last game has no trailing [Event
    return total, forfeit, games


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, type=Path, help="path to a lichess .pgn.zst")
    ap.add_argument("--out", required=True, type=Path, help="report path (markdown)")
    args = ap.parse_args()

    total, forfeit, games = tally(args.dump)

    rate = {b: (forfeit[b] / total[b] if total[b] else 0.0) for b in total}
    ref_rate = rate.get(REFERENCE_BUCKET, 0.0)

    def derived_mult(bucket: str) -> float:
        if bucket == "correspondence":
            return 0.0  # by design, never from data
        if ref_rate <= 0.0:
            return float("nan")
        return round(rate.get(bucket, 0.0) / ref_rate, ROUND_TO)

    label = args.dump.name.replace(".pgn.zst", "")
    lines = [
        "# Clock flag-risk: per-time-control multiplier calibration (real data)",
        "",
        f"**Dump:** `{label}`  ·  **games (n):** {games:,}  ·  task 0268",
        "",
        "Real time-forfeit rates by time control, used to set",
        "`chess_equity.clock._TC_FLAG_MULTIPLIER`. The multiplier is the per-bucket",
        "time-forfeit rate normalised so **bullet = 1.0** (the model's reference TC).",
        "",
        "## Measured time-forfeit rate by time control",
        "",
        "| time control | games | time forfeits | forfeit rate | current mult | measured mult |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for b in BUCKET_ORDER:
        if total.get(b, 0) == 0 and b != "correspondence":
            continue
        cur = _TC_FLAG_MULTIPLIER.get(b, float("nan"))
        meas = derived_mult(b)
        lines.append(
            f"| {b} | {total.get(b, 0):,} | {forfeit.get(b, 0):,} | "
            f"{rate.get(b, 0.0) * 100:.1f}% | {cur:.2f} | {meas:.2f} |"
        )

    lines += [
        "",
        "**Reading it:** the forfeit rate falls steeply from bullet to classical — the",
        "same low clock is far more often fatal in bullet. Normalised to bullet=1.0 the",
        "measured multipliers replace the hand-set guesses. `correspondence` stays pinned",
        "at 0.0 by design (days per move -> no flag pressure), never derived from data.",
        "",
        "## What is NOT calibrated here (blocked on a [%clk] dump)",
        "",
        "`SCRAMBLE_SCALE` (the decay of `time_pressure` vs seconds remaining) and the",
        "band split of `MAX_FLAG_RISK` / `FLAG_RISK_ALERT_THRESHOLD` need per-move",
        "`[%clk]` clocks to fit by clock band. **The cached 2016-05 dump predates",
        "`[%clk]`** (confirmed: zero `%clk` tags; see `clock_coverage.py`), so the",
        "clock-band slice is degenerate on it. That half of the calibration needs a",
        ">=2017-04 dump (a download) and is left for a follow-up — it cannot be done on",
        "cached data and is not faked here (CLAUDE.md: real data only).",
        "",
        "## Reproduce",
        "",
        "```",
        "uv run python scripts/calibrate_clock_tc.py \\",
        f"    --dump ~/.cache/chess-equity/dumps/{label}.pgn.zst \\",
        "    --out reports/clock_calibration_real.md",
        "```",
        "",
    ]

    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
