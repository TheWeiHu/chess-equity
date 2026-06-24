"""Tests for the equity-bar-vs-Stockfish-bar divergence measurement (task 0171).

Two layers: hand-built fixtures that pin the *machinery* (signed gap direction, the
absolute-gap magnitude, the deadband-gated rank-disagreement), then a real-shape smoke
check on the committed 0002 sample that the CLI-facing predictors plug in and produce a
populated report. The sample numbers are a machinery smoke test, NOT evidence — real-data
evidence lives in ``reports/divergence_real.md``.
"""

from __future__ import annotations

from chess_equity.data.build import load_rows
from chess_equity.data.schema import PositionRow
from chess_equity.validate.divergence import (
    FLIP_DEADBAND,
    format_divergence,
    format_max_divergence,
    measure_divergence,
    select_max_divergence,
)
from chess_equity.validate.harness import baseline_cp, wdl_a


def _row(*, cp, we=1500, be=1500, tc="rapid", ply=10) -> PositionRow:
    return PositionRow(
        cp_eval=cp,
        white_elo=we,
        black_elo=be,
        ply=ply,
        phase="middlegame",
        time_control="600+0",
        tc_bucket=tc,
        clock_remaining=None,
        side_to_move="white" if ply % 2 == 0 else "black",
        result=0.5,
        game_id="g",
    )


# --- machinery (hand-built predictors) ----------------------------------------

def test_signed_gap_direction_and_magnitude():
    # equity reads each position 0.10 (10pp) MORE White-favorable than the stockfish bar.
    rows = [_row(cp=0.0), _row(cp=200.0)]
    report = measure_divergence(
        rows,
        equity=lambda r: 0.60,
        stockfish=lambda r: 0.50,
    )
    # signed gap = (equity − stockfish) in pp = +10; |gap| = 10.
    assert abs(report.overall.mean_signed_gap - 10.0) < 1e-9
    assert abs(report.overall.mean_abs_gap - 10.0) < 1e-9
    # A bar that pulls toward Black is a negative signed gap, same magnitude.
    flipped = measure_divergence(rows, equity=lambda r: 0.40, stockfish=lambda r: 0.50)
    assert abs(flipped.overall.mean_signed_gap + 10.0) < 1e-9


def test_rank_disagree_counts_only_opposite_favorites_past_the_deadband():
    rows = [_row(cp=0.0), _row(cp=1.0), _row(cp=2.0), _row(cp=3.0)]
    # stockfish: clearly White (0.70). equity: clearly Black (0.30) -> they name
    # different favorites, and both clear the ±FLIP_DEADBAND band -> rankable + disagree.
    d = measure_divergence(rows, equity=lambda r: 0.30, stockfish=lambda r: 0.70)
    assert d.overall.n_rankable == 4
    assert d.overall.rank_disagree_rate == 1.0
    # Same side (both White) -> rankable but never a disagreement.
    agree = measure_divergence(rows, equity=lambda r: 0.65, stockfish=lambda r: 0.70)
    assert agree.overall.n_rankable == 4
    assert agree.overall.rank_disagree_rate == 0.0


def test_near_50_positions_are_not_rankable():
    rows = [_row(cp=0.0)]
    # Both bars within the deadband of 50% -> not rankable, rate undefined (None).
    eps = FLIP_DEADBAND / 2
    d = measure_divergence(
        rows, equity=lambda r: 0.5 + eps, stockfish=lambda r: 0.5 - eps
    )
    assert d.overall.n_rankable == 0
    assert d.overall.rank_disagree_rate is None


def test_slices_partition_rows_by_tc_and_rating():
    rows = [_row(cp=0.0, tc="bullet"), _row(cp=0.0, tc="blitz"), _row(cp=0.0, tc="blitz")]
    d = measure_divergence(rows, equity=lambda r: 0.5, stockfish=lambda r: 0.5)
    by_tc = {c.label: c.n for c in d.by_tc}
    assert by_tc == {"bullet": 1, "blitz": 2}
    assert sum(c.n for c in d.by_tc) == d.overall.n == 3


# --- max-divergence caster callout (task 0215) --------------------------------

def test_select_max_divergence_picks_the_largest_absolute_gap():
    # The middle row has the biggest |equity − stockfish| even though its sign differs.
    rows = [_row(cp=0.0, ply=4), _row(cp=200.0, ply=7), _row(cp=-50.0, ply=12)]
    # equity reads: small +, big − for the ply-7 row, small + again; stockfish flat at 0.5.
    equities = {4: 0.55, 7: 0.10, 12: 0.58}
    md = select_max_divergence(
        rows, equity=lambda r: equities[r.ply], stockfish=lambda r: 0.5
    )
    assert md is not None
    assert md.ply == 7  # |0.10 − 0.50| = 0.40 is the largest gap
    assert abs(md.gap - (0.10 - 0.5) * 100.0) < 1e-9  # signed gap, pp (negative)
    assert md.cp_eval == 200.0
    assert md.equity == 0.10 and md.stockfish == 0.5


def test_select_max_divergence_empty_and_tie_stability():
    assert select_max_divergence([], equity=lambda r: 0.6, stockfish=lambda r: 0.5) is None
    # Equal gaps -> the first row seen wins (stable).
    rows = [_row(cp=0.0, ply=3), _row(cp=0.0, ply=9)]
    md = select_max_divergence(rows, equity=lambda r: 0.7, stockfish=lambda r: 0.5)
    assert md is not None and md.ply == 3


def test_max_divergence_derived_fields_and_callout_line():
    # ply 7 -> White just moved, side_to_move black, move number (7+1)//2 = 4.
    rows = [_row(cp=12.0, ply=7)]
    md = select_max_divergence(rows, equity=lambda r: 0.38, stockfish=lambda r: 0.52)
    assert md is not None
    assert md.move_number == 4
    assert md.favored_side == "black"  # equity 0.38 < 0.5
    assert abs(md.favored_pct - 62.0) < 1e-9  # (1 − 0.38) * 100
    line = format_max_divergence(md)
    # The one line surfaces ply, cp, equity-as-win%, the favoured side, and the gap.
    assert "ply 7" in line
    assert "move 4" in line
    assert "cp +12" in line
    assert "62% for Black" in line
    assert "gap -14.0pp" in line


def test_report_includes_the_caster_callout_section():
    rows = [_row(cp=0.0, ply=5), _row(cp=300.0, ply=8)]
    report = measure_divergence(rows, equity=lambda r: 0.2, stockfish=lambda r: 0.5)
    assert report.max_divergence is not None
    text = format_divergence(report, header="# h")
    assert "## Caster callout" in text
    assert "Biggest human-edge moment" in text


# --- real-shape smoke check on the committed sample ---------------------------

def test_real_predictors_produce_a_populated_report_on_the_sample():
    rows = load_rows("data/sample/dataset.csv")
    report = measure_divergence(
        rows,
        equity=wdl_a,
        equity_name="wdl-a",
        stockfish=baseline_cp,
        stockfish_name="baseline",
    )
    assert report.overall.n == len(rows)
    # The two bars are not identical on real positions — there is some disagreement.
    assert report.overall.mean_abs_gap > 0.0
    text = format_divergence(report, header="# header line")
    assert text.startswith("# header line")
    assert "## By time control" in text
    assert "rank-disagree" in text
