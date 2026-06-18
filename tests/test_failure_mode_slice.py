"""The validation gate slices by the two named 0003 failure modes (task 0111).

The most direct proof of the headline thesis is equity-vs-baseline *on* the failure
modes objective 0003 names. These tests pin that the curated set
(``baseline/failure_modes.json``) drives a ``failure_mode`` slicer, that it is wired
into the gate's SLICERS, and that the rendered report grows a per-mode
baseline-vs-model section.
"""

from __future__ import annotations

from chess_equity.data.schema import PositionRow
from chess_equity.validate.failure_modes import NONE, failure_mode
from chess_equity.validate.harness import (
    SLICERS,
    baseline_cp,
    evaluate,
    format_report,
)


def _row(*, cp: float, result: float = 0.5) -> PositionRow:
    return PositionRow(
        cp_eval=cp,
        white_elo=1500,
        black_elo=1500,
        ply=10,
        phase="endgame",
        time_control="600+0",
        tc_bucket="rapid",
        clock_remaining=None,
        side_to_move="white",
        result=result,
    )


def test_slicer_tags_curated_cp_regions():
    # engine_cp 0 (the drawn studies) -> hard-0.00; +1000 (Saavedra) -> absurd-refutation.
    assert failure_mode(_row(cp=0.0)) == "hard-0.00"
    assert failure_mode(_row(cp=40.0)) == "hard-0.00"  # within the ±75cp window
    assert failure_mode(_row(cp=1000.0)) == "absurd-refutation"
    assert failure_mode(_row(cp=970.0)) == "absurd-refutation"
    # A clearly-winning-but-ordinary eval sits near no anchor.
    assert failure_mode(_row(cp=400.0)) == NONE


def test_failure_mode_registered_in_slicers():
    assert "failure_mode" in SLICERS
    assert SLICERS["failure_mode"] is failure_mode


def test_evaluate_emits_per_mode_slice():
    rows = [
        _row(cp=0.0, result=1.0),
        _row(cp=20.0, result=0.5),
        _row(cp=1000.0, result=0.5),
        _row(cp=300.0, result=1.0),
    ]
    # baseline plus a second predictor, so the slice carries a model row to compare.
    predictors = {"baseline": baseline_cp, "shifted": lambda _r: 0.5}
    reports = evaluate(rows, predictors)
    base = next(r for r in reports if r.name == "baseline")
    modes = set(base.slices["failure_mode"])
    assert {"hard-0.00", "absurd-refutation", NONE} <= modes
    assert base.slices["failure_mode"]["hard-0.00"].n == 2
    # every predictor gets the same per-mode breakdown
    for rep in reports:
        assert set(rep.slices["failure_mode"]) == modes


def test_report_renders_failure_mode_section():
    rows = [_row(cp=0.0, result=1.0), _row(cp=1000.0, result=0.5), _row(cp=300.0)]
    reports = evaluate(rows, {"baseline": baseline_cp, "shifted": lambda _r: 0.5})
    md = format_report(reports)
    assert "## By failure_mode" in md
    assert "hard-0.00" in md
    assert "absurd-refutation" in md
