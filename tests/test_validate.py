"""Tests for the validation gate (task 0009).

The metrics are checked against hand-computed values (including the draw / soft-label
cases that are the whole point), then the harness is exercised end-to-end on
synthetic rows and the committed 0002 sample, plus the rating-band slicer and the
baseline predictor's "rating-blind" behaviour.
"""

from __future__ import annotations

from math import isclose, log

import pytest

from chess_equity.data.schema import PositionRow
from chess_equity.validate.harness import (
    PREDICTORS,
    baseline_cp,
    evaluate,
    format_report,
    rating_band,
)
from chess_equity.validate.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_table,
)


def _row(*, cp=0.0, we=1500, be=1500, phase="middlegame", result=0.5) -> PositionRow:
    return PositionRow(
        cp_eval=cp,
        white_elo=we,
        black_elo=be,
        ply=10,
        phase=phase,
        time_control="600+0",
        tc_bucket="rapid",
        clock_remaining=None,
        side_to_move="white",
        result=result,
    )


# --- metrics -------------------------------------------------------------------

def test_brier_basic_and_draw():
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0
    # A 0.5 prediction on a draw is perfect under Brier.
    assert brier_score([0.5], [0.5]) == 0.0
    assert isclose(brier_score([0.8], [1.0]), 0.04)


def test_log_loss_perfect_and_soft_draw():
    # Predicting 0.5 on a draw: -(0.5*ln0.5 + 0.5*ln0.5) = ln2.
    assert isclose(log_loss([0.5], [0.5]), log(2.0))
    # Confident and correct -> near zero.
    assert log_loss([0.999999], [1.0]) < 1e-5


def test_log_loss_punishes_confident_wrong_finitely():
    val = log_loss([0.0], [1.0])  # clipped, not infinite
    assert val > 10 and val < 100


def test_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        brier_score([0.5], [0.5, 0.5])
    with pytest.raises(ValueError):
        log_loss([], [])


def test_reliability_and_ece():
    preds = [0.1, 0.1, 0.9, 0.9]
    labels = [0.0, 0.0, 1.0, 1.0]
    table = reliability_table(preds, labels, bins=10)
    assert len(table) == 2  # two non-empty bins
    # Each bin is off by 0.1 (pred 0.1 vs actual 0.0, pred 0.9 vs actual 1.0) -> ECE 0.1.
    assert isclose(expected_calibration_error(preds, labels), 0.1, abs_tol=1e-9)
    # A perfectly calibrated bin (pred 0.2, actual mean 0.2) -> ECE 0.
    cal = expected_calibration_error([0.2, 0.2, 0.2, 0.2, 0.2], [1.0, 0.0, 0.0, 0.0, 0.0])
    assert isclose(cal, 0.0, abs_tol=1e-9)
    # A systematically over-confident predictor has large ECE.
    assert expected_calibration_error([0.9, 0.9], [0.0, 0.0]) > 0.5


# --- predictors & slicing ------------------------------------------------------

def test_baseline_is_rating_blind():
    # Same cp, wildly different ratings -> identical prediction (the baseline's flaw).
    a = baseline_cp(_row(cp=100, we=800, be=800))
    b = baseline_cp(_row(cp=100, we=2600, be=2600))
    assert a == b
    # Even cp -> 0.5; White-favoured cp -> > 0.5.
    assert isclose(baseline_cp(_row(cp=0)), 0.5)
    assert baseline_cp(_row(cp=300)) > 0.5
    assert baseline_cp(_row(cp=-300)) < 0.5


def test_rating_band():
    assert rating_band(_row(we=1000, be=1000)) == "<1200"
    assert rating_band(_row(we=1500, be=1500)) == "1200-1599"
    assert rating_band(_row(we=2500, be=2500)) == "2400+"


# --- harness end to end --------------------------------------------------------

def test_evaluate_overall_and_slices():
    rows = [
        _row(cp=500, we=1000, be=1000, phase="opening", result=1.0),
        _row(cp=-500, we=2500, be=2500, phase="endgame", result=0.0),
    ]
    reports = evaluate(rows, {"baseline": baseline_cp})
    assert len(reports) == 1
    rep = reports[0]
    assert rep.overall.n == 2
    # rating slice has the two distinct bands; phase slice has the two phases.
    assert set(rep.slices["rating"]) == {"<1200", "2400+"}
    assert set(rep.slices["phase"]) == {"opening", "endgame"}
    assert rep.slices["rating"]["<1200"].n == 1


def test_format_report_is_markdown():
    rows = [_row(cp=100, result=1.0), _row(cp=-100, result=0.0)]
    md = format_report(evaluate(rows, {"baseline": baseline_cp}))
    assert md.startswith("# ")
    assert "log-loss" in md and "Brier" in md and "ECE" in md
    assert "## By rating" in md and "## By phase" in md


def test_runs_on_committed_sample():
    from pathlib import Path

    from chess_equity.data.build import load_rows

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset.csv"
    rows = load_rows(str(sample))
    reports = evaluate(rows, {"baseline": baseline_cp})
    assert reports[0].overall.n == len(rows) > 0


def test_baseline_registered():
    assert "baseline" in PREDICTORS
