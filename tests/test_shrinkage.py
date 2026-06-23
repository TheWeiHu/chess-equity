"""Tests for the n-aware wdl-a shrinkage knob (task 0163).

Pure-function tests over tiny constructed PositionRows — *fixtures for unit tests*, NOT
evidence (the effect on the thesis is measured on the real cached dump; see CLAUDE.md's
no-synthetic-data rule). They check the shrinkage weight algebra, that k=0 is an exact
no-op, that sparse cells fall toward the baseline, and that a well-populated cell is left
on the model.
"""

from __future__ import annotations

from chess_equity.data.schema import PositionRow
from chess_equity.validate.harness import baseline_cp, wdl_a
from chess_equity.validate.shrinkage import (
    cell_counts,
    cell_key,
    make_shrunk_predictor,
    shrinkage_weight,
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


def test_weight_algebra():
    # w = n / (n + k)
    assert shrinkage_weight(0, 0) == 1.0  # k=0 is a no-op even for an unseen cell
    assert shrinkage_weight(35, 0) == 1.0
    assert shrinkage_weight(0, 50) == 0.0  # no support, k>0 -> fully baseline
    assert abs(shrinkage_weight(50, 50) - 0.5) < 1e-12
    assert abs(shrinkage_weight(150, 50) - 0.75) < 1e-12


def test_weight_rejects_negative_k():
    import pytest

    with pytest.raises(ValueError):
        shrinkage_weight(10, -1)


def test_k_zero_is_an_exact_noop():
    rows = [_row(cp=-1500.0, we=2200, be=2200), _row(cp=0.0, we=1500, be=1500)]
    shrunk = make_shrunk_predictor(rows, 0.0)
    for r in rows:
        assert shrunk(r) == wdl_a(r)


def test_sparse_cell_falls_toward_baseline():
    # One lonely high-rating decisive row (n=1 in its cell) under a strong shrinkage:
    # w = 1/(1+50) ~= 0.0196, so the blend sits ~98% of the way from wdl-a to baseline.
    sparse = _row(cp=-1500.0, we=2200, be=2200, result=0.0)
    k = 50.0
    shrunk = make_shrunk_predictor([sparse], k)
    w = shrinkage_weight(1, k)
    expected = w * wdl_a(sparse) + (1.0 - w) * baseline_cp(sparse)
    assert abs(shrunk(sparse) - expected) < 1e-12
    # and it has moved most of the way to the baseline
    assert abs(shrunk(sparse) - baseline_cp(sparse)) < abs(wdl_a(sparse) - baseline_cp(sparse))


def test_dense_cell_stays_on_the_model():
    # A cell with many rows is barely shrunk: w = 400/(400+50) ~= 0.889.
    dense = [_row(cp=0.0, we=1700, be=1700) for _ in range(400)]
    probe = dense[0]
    shrunk = make_shrunk_predictor(dense, 50.0)
    # Much closer to wdl-a than to the baseline.
    assert abs(shrunk(probe) - wdl_a(probe)) < abs(shrunk(probe) - baseline_cp(probe))


def test_cell_counts_and_key_match_binning():
    rows = [
        _row(cp=0.0, we=1300, be=1300),    # dead-draw, 1200-1599
        _row(cp=10.0, we=1300, be=1300),   # same cell
        _row(cp=2000.0, we=1300, be=1300), # decisive, 1200-1599 (distinct cp bin)
    ]
    counts = cell_counts(rows)
    assert counts[cell_key(rows[0])] == 2
    assert counts[cell_key(rows[2])] == 1
    assert len(counts) == 2
