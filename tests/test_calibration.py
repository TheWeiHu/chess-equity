"""Tests for the rating-band calibration report + failure-mode measurement (task 0027)."""

from __future__ import annotations

from math import isclose

from chess_equity.validate.calibration import (
    band_reliability,
    format_calibration_report,
    measure_position_classes,
)
from chess_equity.validate.harness import baseline_cp, band_for_avg, rating_band
from chess_equity.data.schema import PositionRow


def _row(*, cp=0.0, we=1500, be=1500, result=0.5) -> PositionRow:
    return PositionRow(
        cp_eval=cp,
        white_elo=we,
        black_elo=be,
        ply=10,
        phase="middlegame",
        time_control="600+0",
        tc_bucket="rapid",
        clock_remaining=None,
        side_to_move="white",
        result=result,
    )


def test_band_for_avg_matches_rating_band():
    # The scalar helper and the row-based slicer must agree on the same thresholds.
    assert band_for_avg(1000) == "<1200"
    assert band_for_avg(1500) == "1200-1599"
    assert band_for_avg(2500) == "2400+"
    assert band_for_avg(1500) == rating_band(_row(we=1500, be=1500))


def test_band_reliability_groups_by_band_and_scores():
    rows = [
        _row(we=1000, be=1000, cp=0, result=1.0),   # <1200
        _row(we=1000, be=1000, cp=0, result=1.0),   # <1200
        _row(we=2500, be=2500, cp=0, result=0.5),   # 2400+
    ]
    bands = band_reliability(rows, baseline_cp)
    by_band = {b.band: b for b in bands}
    assert set(by_band) == {"<1200", "2400+"}
    assert by_band["<1200"].scores.n == 2
    # cp=0 -> baseline predicts 0.5, but the <1200 rows are White wins -> miscalibrated.
    assert by_band["<1200"].scores.ece > 0.4
    # The 2400+ band's single drawn game at cp=0 is perfectly predicted by 0.5.
    assert isclose(by_band["2400+"].scores.ece, 0.0, abs_tol=1e-9)


def test_format_calibration_report_is_markdown():
    rows = [_row(we=1000, be=1000, cp=0, result=1.0), _row(we=2500, be=2500, cp=0, result=0.5)]
    md = format_calibration_report(band_reliability(rows, baseline_cp))
    assert md.startswith("# ")
    assert "ECE by rating band" in md
    assert "Reliability curves" in md
    assert "<1200" in md and "2400+" in md


def test_measure_position_class_hits_and_misses():
    rows = [
        _row(we=1500, be=1500, cp=10, result=1.0),    # in band 1200-1599, |cp-0|<=75
        _row(we=1500, be=1500, cp=-30, result=0.0),   # in band, in window
        _row(we=1500, be=1500, cp=500, result=1.0),   # in band but outside cp window
        _row(we=2500, be=2500, cp=0, result=0.5),     # right cp, wrong band
    ]
    m = measure_position_classes(rows, engine_cp=0.0, band="1200-1599", cp_window=75.0)
    assert m.n == 2                       # only the two in-band, in-window rows
    assert isclose(m.measured_white, 0.5)  # (1.0 + 0.0) / 2

    miss = measure_position_classes(rows, engine_cp=0.0, band="<1200", cp_window=75.0)
    assert miss.n == 0 and miss.measured_white is None


def test_calibration_runs_on_committed_sample():
    from pathlib import Path

    from chess_equity.data.build import load_rows

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset.csv"
    rows = load_rows(str(sample))
    bands = band_reliability(rows, baseline_cp)
    assert bands and sum(b.scores.n for b in bands) == len(rows)
    md = format_calibration_report(bands)
    assert "# " in md


# --- per-rating-band ECE confidence intervals (task 0076) ------------------------

def test_band_reliability_no_ci_by_default():
    # Backward-compat: without bootstrap each band carries no CI and the report keeps
    # the original point-ECE-only column.
    rows = [_row(we=1000, be=1000, cp=0, result=1.0), _row(we=2500, be=2500, cp=0, result=0.5)]
    bands = band_reliability(rows, baseline_cp)
    assert all(b.ece_ci is None for b in bands)
    md = format_calibration_report(bands)
    assert "ECE 95% CI" not in md


def test_band_reliability_pins_deterministic_ece_cis_on_committed_sample():
    # Acceptance (0076): seeded per-band ECE CIs are byte-reproducible on the committed
    # sample, and they tell the thesis story — the rating-blind baseline is well
    # calibrated in the ~2300 band it was fit on (ECE ~0, tight CI) but badly
    # miscalibrated for weaker players (high ECE, CI well clear of zero).
    from pathlib import Path

    from chess_equity.data.build import load_rows

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset.csv"
    rows = load_rows(str(sample))
    bands = band_reliability(rows, baseline_cp, bootstrap=2000, seed=0)
    by_band = {b.band: b for b in bands}
    assert set(by_band) == {"1200-1599", "2000-2399"}

    low = by_band["1200-1599"].ece_ci
    assert low is not None
    assert isclose(low.ece, 0.370504, abs_tol=1e-6)
    assert isclose(low.lo, 0.213027, abs_tol=1e-6)
    assert isclose(low.hi, 0.479564, abs_tol=1e-6)
    assert low.lo > 0.0  # miscalibration is significant, not sample noise

    high = by_band["2000-2399"].ece_ci
    assert high is not None
    assert isclose(high.ece, 0.009970, abs_tol=1e-6)
    assert isclose(high.lo, 0.005369, abs_tol=1e-6)
    assert isclose(high.hi, 0.014570, abs_tol=1e-6)

    # The report renders the CI column when bands carry one.
    md = format_calibration_report(bands)
    assert "ECE 95% CI" in md
    assert "[0.2130, 0.4796]" in md


def test_validate_cli_calibration_report_has_ece_cis(tmp_path):
    from pathlib import Path

    from chess_equity.cli import main

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset.csv"
    cal = tmp_path / "cal.md"
    rc = main(
        ["validate", "--data", str(sample), "--models", "baseline",
         "--bootstrap", "300", "--calibration", str(cal)]
    )
    assert rc == 0
    text = cal.read_text()
    assert "ECE 95% CI" in text


def test_validate_cli_calibration_report_no_ci_when_bootstrap_zero(tmp_path):
    from pathlib import Path

    from chess_equity.cli import main

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset.csv"
    cal = tmp_path / "cal.md"
    rc = main(
        ["validate", "--data", str(sample), "--models", "baseline",
         "--bootstrap", "0", "--calibration", str(cal)]
    )
    assert rc == 0
    assert "ECE 95% CI" not in cal.read_text()
