"""Tests for the per-time-control-bucket gate (task 0155).

Slices the thesis gate by ``tc_bucket`` (bullet/blitz/rapid/classical): does the best
rating-conditioned challenger beat the rating-blind centipawn baseline *within* each
time-control class, on both log-loss and Brier? Buckets below the head-to-head
underpowered floor must be flagged, never silently passed (task 0146 convention).

The grouping and verdict logic is exercised on a tiny fixture with custom predictors so
the win/underpowered/sorting behaviour is deterministic without loading the fitted model.
"""

from __future__ import annotations

from chess_equity.data.schema import PositionRow
from chess_equity.validate.harness import (
    baseline_cp,
    format_tc_bucket_gate,
    tc_bucket_gate,
)


def _row(*, bucket: str, result: float, cp: float = 0.0) -> PositionRow:
    return PositionRow(
        cp_eval=cp,
        white_elo=1500,
        black_elo=1500,
        ply=10,
        phase="middlegame",
        time_control="60+0" if bucket == "bullet" else "600+0",
        tc_bucket=bucket,
        clock_remaining=None,
        side_to_move="white",
        result=result,
    )


def _perfect(row: PositionRow) -> float:
    """A challenger that predicts the actual result exactly — clearly beats baseline."""
    return row.result


def test_groups_by_bucket_with_correct_counts():
    # 4 blitz rows + 2 bullet rows; cp_eval=0 so the rating-blind baseline always says 0.5.
    rows = (
        [_row(bucket="blitz", result=1.0) for _ in range(4)]
        + [_row(bucket="bullet", result=1.0) for _ in range(2)]
    )
    gate = tc_bucket_gate(
        rows, {"baseline": baseline_cp, "good": _perfect}, underpowered_n=3
    )
    assert gate is not None
    counts = {d.bucket: d.n for d in gate.buckets}
    assert counts == {"blitz": 4, "bullet": 2}


def test_powered_bucket_reads_beats_underpowered_flagged():
    rows = (
        [_row(bucket="blitz", result=1.0) for _ in range(4)]
        + [_row(bucket="bullet", result=1.0) for _ in range(2)]
    )
    gate = tc_bucket_gate(
        rows, {"baseline": baseline_cp, "good": _perfect}, underpowered_n=3
    )
    by_bucket = {d.bucket: d for d in gate.buckets}
    # The perfect challenger cuts both metrics on the adequately-powered blitz bucket.
    blitz = by_bucket["blitz"]
    assert blitz.verdict == "beats"
    assert blitz.log_loss_delta < 0 and blitz.brier_delta < 0
    assert not blitz.underpowered
    # The 2-row bullet bucket is below the floor: flagged, never a beats/loses claim.
    bullet = by_bucket["bullet"]
    assert bullet.underpowered
    assert bullet.verdict == "underpowered"


def test_sorted_biggest_equity_win_first():
    # Blitz: baseline very wrong (says 0.5, white wins) -> big equity win.
    # Rapid: baseline already right (cp huge so it says ~1.0) -> small/no win.
    rows = (
        [_row(bucket="blitz", result=1.0, cp=0.0) for _ in range(4)]
        + [_row(bucket="rapid", result=1.0, cp=2000.0) for _ in range(4)]
    )
    gate = tc_bucket_gate(
        rows, {"baseline": baseline_cp, "good": _perfect}, underpowered_n=0
    )
    # Sorted ascending by Δ log-loss = most negative (biggest equity win) first.
    deltas = [d.log_loss_delta for d in gate.buckets]
    assert deltas == sorted(deltas)
    assert gate.buckets[0].bucket == "blitz"


def test_returns_none_without_challenger():
    rows = [_row(bucket="blitz", result=1.0)]
    assert tc_bucket_gate(rows, {"baseline": baseline_cp}) is None


def test_format_renders_table_and_summary():
    rows = (
        [_row(bucket="blitz", result=1.0) for _ in range(4)]
        + [_row(bucket="bullet", result=1.0) for _ in range(2)]
    )
    gate = tc_bucket_gate(
        rows, {"baseline": baseline_cp, "good": _perfect}, underpowered_n=3
    )
    md = format_tc_bucket_gate(gate)
    assert "## By time-control bucket" in md
    assert "| time control | n | Δ log-loss | Δ Brier | verdict |" in md
    assert "underpowered (n=2)" in md
    # 1 powered bucket (blitz), 1 underpowered (bullet) excluded.
    assert "1/1 adequately-powered" in md
    assert "1 bucket(s) below n=3 excluded as underpowered" in md
