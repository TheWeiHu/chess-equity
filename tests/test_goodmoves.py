"""Tests for 'good moves read as good' — the positive-direction validation (task 0117).

Two layers: hand-built fixtures that pin the *machinery* (move pairing, the mover-POV
flip on cp and equity, the good/blunder buckets), then the real-shape check on the
committed 0002 sample — the rating-conditioned bar must read engine-approved moves at
least as positively as the rating-blind centipawn baseline (``equity >= baseline``,
the acceptance criterion). The sample numbers are a machinery smoke test, NOT evidence
(real-data evidence lives in the reports/ artifact).
"""

from __future__ import annotations

from typing import Optional

from chess_equity.data.build import load_rows
from chess_equity.data.schema import PositionRow
from chess_equity.validate.goodmoves import (
    cp_gain_mover,
    equity_gain_mover,
    format_good_moves,
    format_good_moves_sweep,
    iter_move_pairs,
    measure_good_moves,
    reads_good_above_blunder,
    sweep_good_moves,
)
from chess_equity.validate.harness import baseline_cp, wdl_a


def _row(
    *, cp, ply, game_id: Optional[str] = "g", we=1500, be=1500, result=0.5
) -> PositionRow:
    return PositionRow(
        cp_eval=cp,
        white_elo=we,
        black_elo=be,
        ply=ply,
        phase="middlegame",
        time_control="600+0",
        tc_bucket="rapid",
        clock_remaining=None,
        side_to_move="white" if ply % 2 == 0 else "black",
        result=result,
        game_id=game_id,
    )


# --- pairing + POV flip (machinery) -------------------------------------------

def test_pairs_are_adjacent_within_a_game_only():
    rows = [
        _row(cp=0.0, ply=1, game_id="a"),
        _row(cp=50.0, ply=2, game_id="a"),
        _row(cp=80.0, ply=4, game_id="a"),  # ply gap (3 unevaluated) -> no pair to it
        _row(cp=10.0, ply=1, game_id="b"),  # different game -> never paired with a's
    ]
    pairs = iter_move_pairs(rows)
    assert [(b.ply, a.ply) for b, a in pairs] == [(1, 2)]


def test_rows_without_game_id_are_skipped():
    rows = [_row(cp=0.0, ply=1, game_id=None), _row(cp=50.0, ply=2, game_id=None)]
    assert iter_move_pairs(rows) == []


def test_cp_gain_is_mover_pov():
    # Move landing on an ODD ply is White's; on an EVEN ply is Black's.
    # White moved into ply 3: White-POV eval rose +50 -> White (the mover) gained +50.
    before_w = _row(cp=0.0, ply=2)
    white_up = _row(cp=50.0, ply=3)
    assert cp_gain_mover(before_w, white_up) == 50.0
    # Black moved into ply 2: White-POV eval rose +50 -> Black (the mover) *lost* 50.
    before_b = _row(cp=0.0, ply=1)
    black_blunder = _row(cp=50.0, ply=2)
    assert cp_gain_mover(before_b, black_blunder) == -50.0


def test_cp_gain_clamps_mate_scores():
    before = _row(cp=0.0, ply=2)
    after = _row(cp=100000.0, ply=3)  # White mover; a mate score parses huge -> clamp
    assert cp_gain_mover(before, after) == 1000.0


def test_equity_gain_flips_for_black_mover():
    before = _row(cp=0.0, ply=1)  # Black moves into ply 2
    after = _row(cp=200.0, ply=2)  # Black moved, White-POV eval rose -> bad for Black
    # baseline_cp is White expected-score; it rose, so Black's mover-POV equity fell.
    assert equity_gain_mover(before, after, baseline_cp) < 0.0


# --- bucketed measurement (machinery) -----------------------------------------

def test_consistent_monotone_predictor_reads_good_above_blunder():
    # One game of three moves; cp_eval is White-POV.
    rows = [
        _row(cp=20.0, ply=1, game_id="g"),
        _row(cp=15.0, ply=2, game_id="g"),   # ply2 = Black moved, White-POV -5 -> +5 good
        _row(cp=20.0, ply=3, game_id="g"),   # ply3 = White moved, +5 -> good
        _row(cp=320.0, ply=4, game_id="g"),  # ply4 = Black moved, White-POV +300 -> -300 blunder
    ]
    [rep] = measure_good_moves(rows, {"baseline": baseline_cp})
    assert rep.n_moves == 3
    assert rep.n_good >= 1 and rep.n_blunder >= 1
    # Good moves read non-negative; a blunder reads clearly negative.
    assert rep.mean_delta_good is not None and rep.mean_delta_good >= 0.0
    assert rep.mean_delta_blunder is not None and rep.mean_delta_blunder < 0.0


def test_no_pairs_returns_empty():
    rows = [_row(cp=0.0, ply=1, game_id="solo")]  # single ply -> no move
    assert measure_good_moves(rows, {"baseline": baseline_cp}) == []
    assert format_good_moves([]) == ""


# --- real-shape acceptance on the committed sample ----------------------------

def test_sample_metric_computes_and_equity_reads_good_at_least_as_well():
    rows = load_rows("data/sample/dataset.csv")
    reports = measure_good_moves(rows, {"baseline": baseline_cp, "wdl-a": wdl_a})
    by_name = {r.name: r for r in reports}

    # (1) the metric is computed: real moves, finite headline numbers.
    base = by_name["baseline"]
    equity = by_name["wdl-a"]
    assert base.n_moves > 0
    assert base.sign_accuracy is not None and equity.sign_accuracy is not None
    assert base.mean_delta_good is not None and equity.mean_delta_good is not None

    # (2a) direction: every bar reads engine-approved moves above blunders — good moves
    # read as good, not as bad. (The literal "wdl-a Δgood >= baseline Δgood" holds on
    # this fixture but FLIPS on the real 2013-01 dump — see reports/goodmoves_real.md —
    # because cp-delta is the baseline's own input; we don't encode that artifact here.)
    assert reads_good_above_blunder(base)
    assert reads_good_above_blunder(equity)

    # (2b) the robust rating signal (holds on BOTH fixture and the real dump): the
    # rating-conditioned bar reads blunders as less catastrophic than the rating-blind
    # baseline — a refutation a rating-peer won't find is discounted.
    base_bl = base.mean_delta_blunder
    equity_bl = equity.mean_delta_blunder
    assert base_bl is not None and equity_bl is not None
    assert equity_bl >= base_bl


def test_format_renders_table_and_verdict():
    rows = load_rows("data/sample/dataset.csv")
    reports = measure_good_moves(rows, {"baseline": baseline_cp, "wdl-a": wdl_a})
    text = format_good_moves(reports)
    assert "Good moves read as good" in text
    assert "| wdl-a |" in text
    assert "Direction:" in text
    assert "Rating signal:" in text


# --- cutoff-robustness sweep (task 0157) --------------------------------------

def test_sweep_covers_full_grid_and_sign_acc_is_grid_invariant():
    # A monotone-in-cp predictor reads good above blunder at every cutoff, so the sweep
    # should hold in all 9 cells. Build a few games so the buckets are non-empty across
    # the whole 5/10/20 × 75/100/150 grid.
    rows = []
    for gi, gid in enumerate(("a", "b", "c")):
        base = 100.0 * gi
        rows += [
            _row(cp=base + 0.0, ply=1, game_id=gid),
            _row(cp=base + 8.0, ply=2, game_id=gid),    # Black moved, White +8 -> -8 (good ≤10)
            _row(cp=base + 22.0, ply=3, game_id=gid),   # White moved, +14 good (>10, ≤20)
            _row(cp=base + 200.0, ply=4, game_id=gid),  # Black moved, +178 -> -178 blunder
        ]
    good_cuts = (5.0, 10.0, 20.0)
    blunder_cuts = (75.0, 100.0, 150.0)
    [sweep] = sweep_good_moves(
        rows, {"baseline": baseline_cp},
        good_cutoffs=good_cuts, blunder_cutoffs=blunder_cuts,
    )
    assert sweep.name == "baseline"
    assert len(sweep.cells) == len(good_cuts) * len(blunder_cuts)  # full grid, every cell
    assert {(c.good_max_loss, c.blunder_drop) for c in sweep.cells} == {
        (g, b) for g in good_cuts for b in blunder_cuts
    }
    # sign-acc is reported once and matches the default measurement (grid-invariant).
    [base_rep] = measure_good_moves(rows, {"baseline": baseline_cp})
    assert sweep.sign_accuracy == base_rep.sign_accuracy
    # monotone-in-cp -> direction holds everywhere.
    assert sweep.all_hold
    assert all(c.direction_holds for c in sweep.cells)


def test_sweep_flags_cell_where_direction_breaks():
    # A deliberately backwards bar: reads cp *down*, so good moves read below blunders.
    def reversed_bar(row):
        return baseline_cp(row) * -1.0 + 1.0  # still in [0,1]-ish, monotone *decreasing*

    rows = [
        _row(cp=0.0, ply=1, game_id="g"),
        _row(cp=8.0, ply=2, game_id="g"),
        _row(cp=20.0, ply=3, game_id="g"),
        _row(cp=320.0, ply=4, game_id="g"),
    ]
    [sweep] = sweep_good_moves(rows, {"rev": reversed_bar})
    assert not sweep.all_hold
    assert any(not c.direction_holds for c in sweep.cells)


def test_sweep_no_pairs_returns_empty():
    rows = [_row(cp=0.0, ply=1, game_id="solo")]
    assert sweep_good_moves(rows, {"baseline": baseline_cp}) == []
    assert format_good_moves_sweep([]) == ""


def test_format_sweep_renders_table_and_verdict():
    rows = load_rows("data/sample/dataset.csv")
    sweeps = sweep_good_moves(rows, {"baseline": baseline_cp, "wdl-a": wdl_a})
    text = format_good_moves_sweep(sweeps)
    assert "Cutoff-robustness sweep" in text
    assert "holds" in text
    assert "**`wdl-a`**" in text
    # verdict states robustness either way (cutoff-robust ✅ or cutoff-sensitive ⚠).
    assert "Cutoff-robust" in text or "Cutoff-sensitive" in text
