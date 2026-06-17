"""Tests for the high-rating resolution instrument (task 0016).

The MVP slice of 0016 is the *measurement* step: slice the 0009 calibration at fine
bands above 2000 so the gap at titled / super-GM level is quantified before anyone
trains a finer model. These tests pin the band boundaries, the focused
high-rating-only report, and the instrument's ability to surface a per-band gap.
"""

from __future__ import annotations

from chess_equity.data.schema import PositionRow
from chess_equity.validate.harness import (
    SLICERS,
    baseline_cp,
    evaluate,
    high_rating_band,
    high_rating_calibration,
)


def _row(*, cp=0.0, we=1500, be=1500, result=0.5) -> PositionRow:
    return PositionRow(
        cp_eval=cp,
        white_elo=we,
        black_elo=be,
        ply=20,
        phase="middlegame",
        time_control="600+0",
        tc_bucket="rapid",
        clock_remaining=None,
        side_to_move="white",
        result=result,
    )


# --- band boundaries -----------------------------------------------------------

def test_high_rating_band_boundaries():
    assert high_rating_band(_row(we=1500, be=1500)) == "<2000"
    assert high_rating_band(_row(we=1990, be=1990)) == "<2000"
    assert high_rating_band(_row(we=2100, be=2100)) == "2000-2199"
    assert high_rating_band(_row(we=2300, be=2300)) == "2200-2399"
    assert high_rating_band(_row(we=2500, be=2500)) == "2400-2599"
    assert high_rating_band(_row(we=2750, be=2750)) == "2600+"


def test_high_rating_band_distinguishes_within_old_coarse_bucket():
    # rating_band lumps both of these into "2400+"; the high-rating instrument splits
    # the 2500 game from the 2700 game — the whole point of 0016.
    assert high_rating_band(_row(we=2500, be=2500)) != high_rating_band(_row(we=2700, be=2700))


# --- registered as a standard slicer ------------------------------------------

def test_high_rating_registered_as_slicer():
    assert SLICERS.get("high_rating") is high_rating_band
    rep = evaluate([_row(we=2300, be=2300, result=1.0)], {"baseline": baseline_cp})[0]
    assert set(rep.slices["high_rating"]) == {"2200-2399"}


# --- focused high-rating-only report ------------------------------------------

def test_high_rating_calibration_filters_and_slices():
    rows = [
        _row(we=1400, be=1400, result=1.0),   # dropped: below the bar
        _row(we=2100, be=2100, cp=300, result=1.0),
        _row(we=2700, be=2700, cp=-300, result=0.0),
    ]
    reports = high_rating_calibration(rows, {"baseline": baseline_cp})
    rep = reports[0]
    # Only the two high-rated rows survive; only the high_rating slicer is reported.
    assert rep.overall.n == 2
    assert set(rep.slices) == {"high_rating"}
    assert set(rep.slices["high_rating"]) == {"2000-2199", "2600+"}


def test_high_rating_calibration_empty_when_no_high_rated_rows():
    # The committed sample barely reaches 2000 — an honest empty result, not a crash.
    assert high_rating_calibration([_row(we=1500, be=1500)], {"baseline": baseline_cp}) == []


# --- the instrument actually surfaces a per-band gap ---------------------------

def test_instrument_surfaces_per_band_miscalibration():
    # Same objective eval (+200cp -> baseline predicts ~0.77 for White), but the
    # high band converts it far more reliably than the low-high band. A trustworthy
    # gap-measurement tool must show a worse Brier where reality diverges from the
    # rating-blind prediction.
    band_a = [_row(we=2100, be=2100, cp=200, result=float(i % 2)) for i in range(10)]  # ~50% real
    band_b = [_row(we=2700, be=2700, cp=200, result=1.0) for _ in range(10)]           # always wins
    rep = high_rating_calibration(band_a + band_b, {"baseline": baseline_cp})[0]
    brier_a = rep.slices["high_rating"]["2000-2199"].brier
    brier_b = rep.slices["high_rating"]["2600+"].brier
    # Rating-blind eval is badly off in band A (predicts 0.77, truth 0.5) and close
    # in band B (predicts 0.77, truth 1.0) — the instrument exposes the difference.
    assert brier_a > brier_b
