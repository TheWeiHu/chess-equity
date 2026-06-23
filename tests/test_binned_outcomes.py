"""Tests for the binned-outcomes failure-modes report (task 0151).

Pure-function tests over tiny constructed PositionRows — these are *fixtures for unit
tests*, NOT evidence (the evidence run feeds a real Lichess dump via
``scripts/failure_modes_real.py``; see CLAUDE.md's no-synthetic-data rule). They check the
aggregation (binning, measured rate, per-predictor means, n), the two failure-mode region
selectors, and the report's structure — not any thesis number.
"""

from __future__ import annotations

from chess_equity.data.schema import PositionRow
from chess_equity.validate.binned_outcomes import (
    DECISIVE_MIN,
    HARD_DRAW_MAX,
    bin_outcomes,
    format_report,
)


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


def test_measured_rate_and_n_are_the_real_mean():
    # Two wins and a loss in one cp×band cell -> measured 2/3, n=3.
    rows = [_row(cp=0.0, result=1.0), _row(cp=10.0, result=1.0), _row(cp=-10.0, result=0.0)]
    cells = bin_outcomes(rows)
    assert len(cells) == 1
    cell = cells[0]
    assert cell.n == 3
    assert abs(cell.measured - 2 / 3) < 1e-9
    # Both named predictors are present and in [0, 1].
    assert set(cell.predicted) == {"baseline", "wdl-a"}
    assert all(0.0 <= v <= 1.0 for v in cell.predicted.values())


def test_min_n_drops_underpowered_cells():
    rows = [_row(cp=0.0, result=1.0)]
    assert bin_outcomes(rows, min_n=1)  # kept
    assert bin_outcomes(rows, min_n=2) == []  # dropped


def test_cp_and_band_split_into_distinct_cells():
    rows = [
        _row(cp=0.0, we=1300, be=1300),       # dead-draw, 1200-1599
        _row(cp=2000.0, we=1300, be=1300),    # decisive, 1200-1599
        _row(cp=0.0, we=1700, be=1700),       # dead-draw, 1600-1999
    ]
    cells = bin_outcomes(rows)
    assert len(cells) == 3
    # Sorted by cp bin (low->high) then band: the negative/zero-cp cells precede decisive.
    assert cells[-1].cp_bin.startswith(">")


def test_dead_draw_band_uses_the_cp_window():
    # A row just inside the window stays in the hard-draw region; one just outside doesn't.
    inside = bin_outcomes([_row(cp=HARD_DRAW_MAX)])[0]
    outside = bin_outcomes([_row(cp=HARD_DRAW_MAX + 1)])[0]
    assert inside.cp_bin == f"({-int(HARD_DRAW_MAX)}, {int(HARD_DRAW_MAX)}]"
    assert outside.cp_bin != inside.cp_bin


def test_report_has_both_failure_mode_sections_and_provenance():
    rows = (
        [_row(cp=0.0, result=1.0) for _ in range(40)]
        + [_row(cp=2000.0, result=1.0) for _ in range(40)]
    )
    cells = bin_outcomes(rows)
    md = format_report(cells, dump="fixture-not-evidence", n=len(rows))
    assert "fixture-not-evidence" in md
    assert f"n={len(rows)}" in md
    assert f"hard 0.00 isn't 50/50\" (|cp| ≤ {int(HARD_DRAW_MAX)})" in md
    assert f"good moves read as good\" (|cp| ≥ {int(DECISIVE_MIN)})" in md
    assert "| cp bin | rating | n | measured |" in md
    # The verdict line names a predictor that tracks closer (data-driven, not asserted).
    assert "tracks the measured rate closer" in md
