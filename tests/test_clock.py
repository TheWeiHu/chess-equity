"""Tests for the clock / time-pressure dimension (task 0015).

Two acceptance criteria, both checked here:

1. Equity *responds to clock state* — low time on the side to move lowers their
   equity, holding eval and ratings fixed (both colours), and faster time controls
   bite harder at the same clock.
2. A *measured calibration gain* on low-clock positions — the clock-aware predictor
   beats the clock-blind baseline (lower Brier) on a synthetic scramble slice where
   winning positions actually flag, while leaving comfortable positions untouched.
"""

from __future__ import annotations

from math import isclose

from chess_equity.clock import (
    FLAG_RISK_ALERT_THRESHOLD,
    clock_adjusted_white_equity,
    flag_risk,
    is_flag_risk_alert,
    time_pressure,
)
from chess_equity.data.schema import PositionRow
from chess_equity.validate.harness import (
    PREDICTORS,
    baseline_cp,
    baseline_cp_clock,
    clock_band,
    evaluate,
)
from chess_equity.validate.metrics import brier_score


def _row(*, cp=0.0, clock=None, tc_bucket="blitz", stm="white", result=0.5) -> PositionRow:
    return PositionRow(
        cp_eval=cp,
        white_elo=1500,
        black_elo=1500,
        ply=20,
        phase="middlegame",
        time_control="180+0",
        tc_bucket=tc_bucket,
        clock_remaining=clock,
        side_to_move=stm,
        result=result,
    )


# --- time_pressure / flag_risk shape -------------------------------------------

def test_time_pressure_decreases_with_clock():
    assert time_pressure(2.0) > time_pressure(20.0) > time_pressure(120.0)
    # Dead clock -> maximal; minutes left -> negligible.
    assert isclose(time_pressure(0.0), 1.0)
    assert time_pressure(180.0) < 0.01


def test_time_pressure_no_clock_is_noop():
    assert time_pressure(None) == 0.0
    assert flag_risk(None, "bullet") == 0.0


def test_flag_risk_faster_tc_bites_harder():
    # Same few seconds: bullet is deadlier than blitz, which beats rapid/classical.
    assert (
        flag_risk(5.0, "bullet")
        > flag_risk(5.0, "blitz")
        > flag_risk(5.0, "rapid")
        > flag_risk(5.0, "classical")
    )
    # Correspondence has no flag pressure at all.
    assert flag_risk(5.0, "correspondence") == 0.0


# --- equity responds to clock (criterion 1) ------------------------------------

def test_low_clock_lowers_side_to_move_equity_white():
    # White to move, winning eval, fixed: less time -> lower White equity.
    safe = clock_adjusted_white_equity(0.85, 300.0, "blitz", white_to_move=True)
    scramble = clock_adjusted_white_equity(0.85, 4.0, "blitz", white_to_move=True)
    assert scramble < safe <= 0.85


def test_low_clock_lowers_side_to_move_equity_black():
    # Black to move and winning (White equity 0.2): Black's scramble *raises* White.
    safe = clock_adjusted_white_equity(0.2, 300.0, "blitz", white_to_move=False)
    scramble = clock_adjusted_white_equity(0.2, 4.0, "blitz", white_to_move=False)
    # Black's equity = 1 - white_equity; it must drop, so White's must rise.
    assert scramble > safe >= 0.2


def test_comfortable_clock_is_essentially_unchanged():
    eq = clock_adjusted_white_equity(0.85, 600.0, "rapid", white_to_move=True)
    assert isclose(eq, 0.85, abs_tol=1e-3)


def test_clock_band_buckets():
    assert clock_band(_row(clock=None)) == "no-clock"
    assert clock_band(_row(clock=8.0)) == "scramble(<15s)"
    assert clock_band(_row(clock=40.0)) == "low(<60s)"
    assert clock_band(_row(clock=300.0)) == "comfortable(60s+)"


# --- measured calibration gain on low-clock positions (criterion 2) ------------

def test_clock_aware_predictor_improves_low_clock_calibration():
    # A bullet scramble: White is objectively winning (+400cp -> baseline ~0.86),
    # but with ~4s left half of these games are actually lost on time. Truth ~0.5.
    rows = [
        _row(cp=400.0, clock=4.0, tc_bucket="bullet", stm="white", result=float(i % 2))
        for i in range(20)
    ]
    blind = brier_score([baseline_cp(r) for r in rows], [r.result for r in rows])
    aware = brier_score([baseline_cp_clock(r) for r in rows], [r.result for r in rows])
    # The clock-aware bar is pulled toward the real 50/50 -> demonstrably better.
    assert aware < blind


# --- flag-risk alert threshold (task 0243) -------------------------------------

def test_flag_risk_alert_threshold_lights_on_real_scramble():
    # A bullet side with ~4s flags often: flag_risk well over the alert threshold.
    scramble = flag_risk(4.0, "bullet")
    assert scramble >= FLAG_RISK_ALERT_THRESHOLD
    assert is_flag_risk_alert(scramble) is True


def test_flag_risk_alert_off_for_comfortable_clock():
    # Minutes to spare -> risk ~0, well under the threshold -> no alert.
    comfortable = flag_risk(300.0, "bullet")
    assert comfortable < FLAG_RISK_ALERT_THRESHOLD
    assert is_flag_risk_alert(comfortable) is False


def test_flag_risk_alert_is_clock_blind_safe_and_threshold_tunable():
    # None (clock-blind side) never alerts; the threshold is an overridable knob.
    assert is_flag_risk_alert(None) is False
    assert is_flag_risk_alert(0.5, threshold=0.6) is False
    assert is_flag_risk_alert(0.5, threshold=0.4) is True
    # Boundary: exactly at the threshold trips (>=, not >).
    assert is_flag_risk_alert(FLAG_RISK_ALERT_THRESHOLD) is True


def test_clock_aware_predictor_registered_and_harness_slices_by_clock():
    assert "baseline+clock" in PREDICTORS
    rows = [
        _row(cp=400.0, clock=4.0, tc_bucket="bullet", result=0.0),
        _row(cp=400.0, clock=300.0, tc_bucket="bullet", result=1.0),
    ]
    rep = evaluate(rows, {"baseline+clock": baseline_cp_clock})[0]
    assert set(rep.slices["clock"]) == {"scramble(<15s)", "comfortable(60s+)"}
