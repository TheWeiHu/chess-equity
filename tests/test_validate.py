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
    gate_verdicts,
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


def test_report_carries_reliability_curve_with_per_bin_counts():
    # Task 0118: the validate report shows the binned empirical win-rate behind the
    # scalar ECE, with per-bin counts — so a reader can see *why* a bar is (un)calibrated.
    rows = [_row(cp=300, result=1.0), _row(cp=-300, result=0.0), _row(cp=0, result=0.5)]
    reports = evaluate(rows, {"baseline": baseline_cp})
    # The overall reliability table is populated and is exactly the metrics-level table.
    rep = reports[0]
    assert rep.reliability == reliability_table(
        [baseline_cp(r) for r in rows], [r.result for r in rows], bins=10
    )
    md = format_report(reports)
    assert "## Reliability curve" in md
    assert "mean obs" in md and "gap (obs−pred)" in md
    # Per-bin counts sum to the overall n (every scored row lands in exactly one bin).
    assert sum(count for *_rest, count in rep.reliability) == rep.overall.n


def test_calibrated_predictor_scores_ece_near_zero_through_report():
    # Task 0118: a perfectly-calibrated (but deliberately unsharp) predictor — it always
    # says 0.3 for a group that empirically scores 0.3 — must read ECE ~ 0 in the report
    # path. Calibration is not sharpness: a flat 0.3 on 3-win/7-loss rows is honest.
    rows = [_row(result=1.0)] * 3 + [_row(result=0.0)] * 7
    calibrated = lambda r: 0.3  # noqa: E731 — flat, matches the empirical 0.3 win-rate
    rep = evaluate(rows, {"calibrated": calibrated})[0]
    assert isclose(rep.overall.ece, 0.0, abs_tol=1e-9)
    # The single populated bin's observed rate equals its predicted rate (gap 0).
    assert len(rep.reliability) == 1
    _bin_lo, mean_pred, mean_obs, count = rep.reliability[0]
    assert count == 10 and isclose(mean_pred, 0.3) and isclose(mean_obs, 0.3)


# --- gate verdict (task 0058) --------------------------------------------------

def test_gate_verdict_reflects_computed_deltas():
    # An "oracle" predictor that nails every outcome strictly beats the baseline on
    # both log-loss and Brier -> PASS, with deltas equal to oracle - baseline scores.
    rows = [_row(cp=300, result=1.0), _row(cp=-300, result=0.0), _row(cp=0, result=0.5)]
    oracle = lambda r: r.result  # noqa: E731 — perfect predictor for the fixture
    reports = evaluate(rows, {"baseline": baseline_cp, "oracle": oracle})
    by_name = {r.name: r for r in reports}
    verdicts = gate_verdicts(reports)
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.name == "oracle"
    # Deltas are exactly model - baseline on the overall scores.
    assert isclose(
        v.log_loss_delta, by_name["oracle"].overall.log_loss - by_name["baseline"].overall.log_loss
    )
    assert isclose(
        v.brier_delta, by_name["oracle"].overall.brier - by_name["baseline"].overall.brier
    )
    # Strictly better on both -> PASS.
    assert v.log_loss_delta < 0 and v.brier_delta < 0
    assert v.passed is True


def test_gate_verdict_fails_when_worse_on_either_metric():
    # A predictor that is worse than the baseline must FAIL.
    rows = [_row(cp=300, result=1.0), _row(cp=-300, result=0.0)]
    worse = lambda r: 1.0 - baseline_cp(r)  # noqa: E731 — inverts the baseline
    verdicts = gate_verdicts(evaluate(rows, {"baseline": baseline_cp, "worse": worse}))
    assert len(verdicts) == 1
    assert verdicts[0].passed is False


def test_gate_verdict_empty_without_baseline():
    rows = [_row(cp=100, result=1.0), _row(cp=-100, result=0.0)]
    oracle = lambda r: r.result  # noqa: E731
    assert gate_verdicts(evaluate(rows, {"oracle": oracle})) == []


def test_report_opens_with_gate_verdict():
    rows = [_row(cp=300, result=1.0), _row(cp=-300, result=0.0), _row(cp=0, result=0.5)]
    oracle = lambda r: r.result  # noqa: E731
    md = format_report(evaluate(rows, {"baseline": baseline_cp, "oracle": oracle}))
    # The verdict is the report's first section, ahead of the metrics tables.
    assert md.index("## Gate verdict") < md.index("## Overall")
    assert "oracle" in md and "PASS" in md
    assert "-> **PASS**" in md


# --- significance-aware gate (task 0069) ---------------------------------------
#
# The point-only gate (0058) PASSes on a bare negative delta, but "proves equity beats
# centipawns" means a *significant* win. With paired-bootstrap CIs supplied, PASS now also
# requires the headline-metric (log-loss) delta CI to clear zero.

def _decisive_rows():
    # All-decisive rows (no draws) where an oracle nails every result: every per-row
    # log-loss delta is strictly negative, so the bootstrap CI clears zero (significant).
    return [
        _row(cp=250, result=1.0),
        _row(cp=-250, result=0.0),
        _row(cp=180, result=1.0),
        _row(cp=-180, result=0.0),
        _row(cp=300, result=1.0),
        _row(cp=-300, result=0.0),
    ]


def test_significance_gate_passes_oracle():
    # An oracle's log-loss delta CI clears zero, so the significance-aware gate still PASSes.
    from chess_equity.validate.harness import HEADLINE_METRIC, compare_to_baseline

    rows = _decisive_rows()
    oracle = lambda r: r.result  # noqa: E731
    predictors = {"baseline": baseline_cp, "oracle": oracle}
    reports = evaluate(rows, predictors)
    comps = compare_to_baseline(rows, predictors, n_resamples=500, seed=0)
    verdicts = gate_verdicts(reports, comparisons=comps)
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.name == "oracle"
    assert v.significant is True
    assert v.headline_metric == HEADLINE_METRIC
    assert v.headline_ci is not None and v.headline_ci.beats_baseline
    assert v.passed is True


def test_significance_gate_flips_marginal_point_win_to_fail():
    # On the committed 15-row FEN sample wdl-a is point-better on BOTH metrics, so the
    # point-only gate PASSes — but its log-loss CI straddles zero (see the 0060 pin test),
    # so the significance-aware gate honestly FLIPS it to FAIL. No hardcoded numbers: we
    # assert the verdict flips between the two gates, not the CI bounds.
    from pathlib import Path

    from chess_equity.data.build import load_rows
    from chess_equity.validate.harness import build_predictors, compare_to_baseline

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"
    rows = load_rows(str(sample))
    predictors = build_predictors(["baseline", "wdl-a"])
    reports = evaluate(rows, predictors)

    # Point-only gate (no CIs): a bare double point win -> PASS, significance unrecorded.
    point_only = gate_verdicts(reports)[0]
    assert point_only.name == "wdl-a"
    assert point_only.log_loss_delta < 0 and point_only.brier_delta < 0
    assert point_only.passed is True
    assert point_only.significant is None

    # Significance-aware gate: same point win, but the log-loss CI straddles zero -> FAIL.
    comps = compare_to_baseline(rows, predictors, n_resamples=2000, seed=0)
    gated = gate_verdicts(reports, comparisons=comps)[0]
    assert gated.significant is False
    assert gated.headline_ci is not None and not gated.headline_ci.beats_baseline
    assert gated.passed is False


def test_format_verdict_renders_ci_and_significance_criterion():
    # The rendered gate block states the significance requirement and shows the headline
    # CI inline with a "straddles zero" read for the marginal sample win.
    from pathlib import Path

    from chess_equity.data.build import load_rows
    from chess_equity.validate.harness import (
        build_predictors,
        compare_to_baseline,
        format_verdict,
    )

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"
    rows = load_rows(str(sample))
    predictors = build_predictors(["baseline", "wdl-a"])
    reports = evaluate(rows, predictors)
    comps = compare_to_baseline(rows, predictors, n_resamples=2000, seed=0)
    block = "\n".join(format_verdict(gate_verdicts(reports, comparisons=comps)))
    assert "CI clears zero" in block  # the criterion line
    assert "log_loss 95% CI [" in block and "(CI straddles zero)" in block
    assert "-> **FAIL**" in block

    # Point-only rendering (no CIs supplied) keeps the pre-0069 wording and shows no CI.
    plain = "\n".join(format_verdict(gate_verdicts(reports)))
    assert "95% CI clears" not in plain and "log_loss 95% CI" not in plain
    assert "-> **PASS**" in plain


def test_format_verdict_shows_percent_reduction_for_passing_predictor():
    # The headline effect size (task 0133): a passing predictor's line states the percent
    # reduction in log-loss (and Brier) vs the baseline, computed from the absolute deltas.
    from chess_equity.validate.harness import format_verdict

    rows = [_row(cp=300, result=1.0), _row(cp=-300, result=0.0), _row(cp=0, result=0.5)]
    oracle = lambda r: r.result  # noqa: E731 — perfect predictor PASSES
    reports = evaluate(rows, {"baseline": baseline_cp, "oracle": oracle})
    by_name = {r.name: r for r in reports}

    v = gate_verdicts(reports)[0]
    assert v.passed is True
    # Expected reduction = (baseline - model) / baseline * 100, on the overall scores.
    bl = by_name["baseline"].overall
    ml = by_name["oracle"].overall
    exp_ll = (bl.log_loss - ml.log_loss) / bl.log_loss * 100.0
    exp_br = (bl.brier - ml.brier) / bl.brier * 100.0

    block = "\n".join(format_verdict(gate_verdicts(reports)))
    assert f"cuts log-loss {exp_ll:.1f}% (Brier {exp_br:.1f}%) vs baseline" in block


def test_format_verdict_omits_percent_reduction_for_failing_predictor():
    # A failing predictor gets no "cuts log-loss …%" headline — the number would be
    # negative/misleading, so it's reserved for predictors that actually win.
    from chess_equity.validate.harness import format_verdict

    rows = [_row(cp=300, result=1.0), _row(cp=-300, result=0.0)]
    worse = lambda r: 1.0 - baseline_cp(r)  # noqa: E731 — strictly worse than baseline
    block = "\n".join(format_verdict(gate_verdicts(evaluate(rows, {"baseline": baseline_cp, "worse": worse}))))
    assert "-> **FAIL**" in block
    assert "cuts log-loss" not in block


def test_gate_cli_significance_flips_sample_to_fail(tmp_path, capsys):
    # End-to-end: with --bootstrap > 0 the --gate exit code is significance-aware, so the
    # marginal 15-row sample win (point-better, CI straddles zero) exits 2, not 0.
    from pathlib import Path

    from chess_equity.cli import main

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"
    rc = main(
        [
            "validate",
            "--data",
            str(sample),
            "--models",
            "baseline,wdl-a",
            "--bootstrap",
            "2000",
            "--seed",
            "0",
            "--gate",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "GATE: FAIL" in err and "significant" in err


def test_runs_on_committed_sample():
    from pathlib import Path

    from chess_equity.data.build import load_rows

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset.csv"
    rows = load_rows(str(sample))
    reports = evaluate(rows, {"baseline": baseline_cp})
    assert reports[0].overall.n == len(rows) > 0


# --- paired-bootstrap CIs on the model-vs-baseline delta (task 0060) -------------

def test_bootstrap_delta_equals_score_difference():
    # The point delta is exactly mean(model_terms) - mean(baseline_terms), i.e. the
    # difference of the two metric scores — the bootstrap only adds the interval.
    from chess_equity.validate.bootstrap import paired_bootstrap_ci

    model = [0.10, 0.20, 0.30]
    base = [0.40, 0.40, 0.40]
    ci = paired_bootstrap_ci(model, base, "brier", n_resamples=200, seed=0)
    assert isclose(ci.delta, (sum(model) - sum(base)) / 3)
    assert ci.lo <= ci.delta <= ci.hi  # the point estimate sits inside its own CI


def test_bootstrap_verdict_directions():
    # All per-row deltas strictly negative -> the model is unambiguously better, so the
    # whole CI clears zero and the verdict is "beats" (and never "worse").
    from chess_equity.validate.bootstrap import paired_bootstrap_ci

    better = paired_bootstrap_ci([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], "brier", seed=1)
    assert better.beats_baseline and not better.worse_than_baseline
    worse = paired_bootstrap_ci([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], "brier", seed=1)
    assert worse.worse_than_baseline and not worse.beats_baseline


def test_bootstrap_is_seed_deterministic():
    from chess_equity.validate.bootstrap import paired_bootstrap_ci

    args = ([0.1, 0.5, 0.2, 0.9], [0.3, 0.3, 0.3, 0.3], "log_loss")
    a = paired_bootstrap_ci(*args, n_resamples=500, seed=42)
    b = paired_bootstrap_ci(*args, n_resamples=500, seed=42)
    assert (a.delta, a.lo, a.hi) == (b.delta, b.lo, b.hi)


def test_bootstrap_rejects_bad_input():
    from chess_equity.validate.bootstrap import paired_bootstrap_ci

    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.1, 0.2], [0.1], "brier")  # length mismatch
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [], "brier")  # no rows


def test_compare_to_baseline_pins_deterministic_cis_on_fen_sample():
    # Acceptance (0060): seeded CIs are reproducible byte-for-byte on the committed
    # FEN sample. wdl-a beats the rating-blind baseline on Brier (CI fully below 0);
    # on only 15 rows the log-loss win is real but not yet significant (CI straddles 0).
    from pathlib import Path

    from chess_equity.data.build import load_rows
    from chess_equity.validate.harness import build_predictors, compare_to_baseline

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"
    rows = load_rows(str(sample))
    comps = compare_to_baseline(
        rows, build_predictors(["baseline", "wdl-a"]), n_resamples=2000, seed=0
    )
    assert len(comps) == 1 and comps[0].name == "wdl-a" and comps[0].baseline == "baseline"
    by_metric = {ci.metric: ci for ci in comps[0].cis}

    brier = by_metric["brier"]
    assert isclose(brier.delta, -0.026506, abs_tol=1e-6)
    assert isclose(brier.lo, -0.046396, abs_tol=1e-6)
    assert isclose(brier.hi, -0.005976, abs_tol=1e-6)
    assert brier.beats_baseline  # whole CI below zero

    ll = by_metric["log_loss"]
    assert isclose(ll.delta, -0.038490, abs_tol=1e-6)
    assert isclose(ll.lo, -0.088755, abs_tol=1e-6)
    assert isclose(ll.hi, 0.020267, abs_tol=1e-6)
    assert not ll.beats_baseline  # better on average, but CI straddles zero


def test_compare_to_baseline_empty_without_a_second_predictor():
    from chess_equity.validate.harness import compare_to_baseline

    rows = [_row(cp=100, result=1.0), _row(cp=-100, result=0.0)]
    assert compare_to_baseline(rows, {"baseline": baseline_cp}) == []


def test_validate_cli_emits_significance_section(tmp_path, capsys):
    # End-to-end: the validate command appends the paired-bootstrap section and labels
    # a CI-clears-zero win as "beats".
    from pathlib import Path

    from chess_equity.cli import main

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"
    out = tmp_path / "report.md"
    rc = main(
        [
            "validate",
            "--data",
            str(sample),
            "--models",
            "baseline,wdl-a",
            "--bootstrap",
            "300",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    text = out.read_text()
    assert "## Significance vs baseline" in text
    assert "| wdl-a | brier |" in text and "beats" in text


def test_validate_cli_bootstrap_zero_disables_section(tmp_path, capsys):
    from pathlib import Path

    from chess_equity.cli import main

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"
    out = tmp_path / "report.md"
    rc = main(
        ["validate", "--data", str(sample), "--models", "baseline,wdl-a",
         "--bootstrap", "0", "--out", str(out)]
    )
    assert rc == 0
    assert "Significance vs baseline" not in out.read_text()


# --- bin-resampling CI on ECE / calibration (task 0072) --------------------------

def test_ece_bootstrap_ci_brackets_the_point_ece():
    # The point ECE must sit inside its own CI, and the CI must be byte-reproducible
    # under a fixed seed. A predictor off by 0.1 in every bin has point ECE 0.1.
    from chess_equity.validate.bootstrap import ece_bootstrap_ci

    args = ([0.1, 0.1, 0.9, 0.9], [0.0, 0.0, 1.0, 1.0])
    a = ece_bootstrap_ci(*args, n_resamples=500, seed=42)
    b = ece_bootstrap_ci(*args, n_resamples=500, seed=42)
    assert a == b  # seed-deterministic, frozen dataclass equality
    assert isclose(a.ece, 0.1, abs_tol=1e-9)
    assert a.lo <= a.ece <= a.hi
    assert a.delta is None and a.delta_lo is None  # no baseline given


def test_ece_bootstrap_paired_delta_directions():
    # Perfectly calibrated model (pred == label, ECE 0) vs a systematically over-confident
    # baseline (always predicts 0.9 on a 50/50 outcome, so its bin mean is off by >=0.1 in
    # every resample): the paired ECE delta (model - baseline) is negative and the whole
    # CI clears zero -> beats. Note 0.5-everywhere would NOT work — ECE measures
    # calibration, not sharpness, so a 0.5 bin on 50/50 labels is "calibrated" (ECE 0).
    from chess_equity.validate.bootstrap import ece_bootstrap_ci

    preds = [0.0, 0.0, 1.0, 1.0]
    labels = [0.0, 0.0, 1.0, 1.0]
    base = [0.9, 0.9, 0.9, 0.9]
    ci = ece_bootstrap_ci(preds, labels, baseline_preds=base, n_resamples=500, seed=1)
    assert ci.delta is not None and ci.delta < 0.0
    assert ci.beats_baseline and not ci.worse_than_baseline
    # swap roles: the miscalibrated predictor is significantly worse.
    worse = ece_bootstrap_ci(base, labels, baseline_preds=preds, n_resamples=500, seed=1)
    assert worse.worse_than_baseline and not worse.beats_baseline


def test_ece_bootstrap_rejects_bad_input():
    from chess_equity.validate.bootstrap import ece_bootstrap_ci

    with pytest.raises(ValueError):
        ece_bootstrap_ci([0.1, 0.2], [0.1])  # preds/labels length mismatch
    with pytest.raises(ValueError):
        ece_bootstrap_ci([], [])  # no rows
    with pytest.raises(ValueError):
        ece_bootstrap_ci([0.1, 0.2], [0.0, 1.0], baseline_preds=[0.1])  # baseline mismatch


def test_compare_ece_pins_deterministic_cis_on_fen_sample():
    # Acceptance (0072): seeded ECE CIs are reproducible byte-for-byte on the committed
    # FEN sample. This is a reproducibility pin, not a calibration claim: on this 15-row
    # smoke fixture wdl-a's point ECE sits just above the rating-blind baseline and the
    # delta CI straddles zero -> not significant either way. (On the real 50k dataset
    # wdl-a is the better-calibrated model; the tiny fixture is not evidence.)
    from pathlib import Path

    from chess_equity.data.build import load_rows
    from chess_equity.validate.harness import build_predictors, compare_ece_to_baseline

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"
    rows = load_rows(str(sample))
    cis = compare_ece_to_baseline(
        rows, build_predictors(["baseline", "wdl-a"]), n_resamples=2000, seed=0
    )
    by_name = {c.predictor: c for c in cis}
    assert set(by_name) == {"baseline", "wdl-a"}

    base = by_name["baseline"]
    assert isclose(base.ece, 0.218315, abs_tol=1e-6)
    assert isclose(base.lo, 0.090494, abs_tol=1e-6)
    assert isclose(base.hi, 0.346437, abs_tol=1e-6)
    assert base.delta is None  # baseline has no delta vs itself

    wdl = by_name["wdl-a"]
    assert isclose(wdl.ece, 0.236108, abs_tol=1e-6)
    assert isclose(wdl.lo, 0.157331, abs_tol=1e-6)
    assert isclose(wdl.hi, 0.317493, abs_tol=1e-6)
    assert wdl.delta is not None and wdl.delta_lo is not None and wdl.delta_hi is not None
    assert isclose(wdl.delta, 0.017793, abs_tol=1e-6)
    assert isclose(wdl.delta_lo, -0.028163, abs_tol=1e-6)
    assert isclose(wdl.delta_hi, 0.065198, abs_tol=1e-6)
    assert not wdl.beats_baseline  # delta CI straddles zero -> not significant


def test_validate_cli_emits_ece_ci_section(tmp_path):
    from pathlib import Path

    from chess_equity.cli import main

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"
    out = tmp_path / "report.md"
    rc = main(
        ["validate", "--data", str(sample), "--models", "baseline,wdl-a",
         "--bootstrap", "300", "--out", str(out)]
    )
    assert rc == 0
    text = out.read_text()
    assert "## Calibration (ECE) confidence intervals" in text
    assert "| wdl-a |" in text and "Δ vs baseline" in text


def test_baseline_registered():
    assert "baseline" in PREDICTORS


def test_report_on_sample_has_gate_and_head_to_head_sections():
    """Regenerating the gate report on the committed sample yields the two headline
    sections — the thesis evidence the ``reports/`` artifact captures (task 0063).

    Guards the wiring/format, not the numbers (the 15-row sample is meaningless): a
    baseline + rating-conditioned challenger must always render a Gate verdict and a
    Head-to-head "where equity wins" section so the committed artifact can't silently
    lose them when the harness evolves.
    """
    from pathlib import Path

    from chess_equity.data.build import load_rows

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset.csv"
    rows = load_rows(str(sample))
    md = format_report(evaluate(rows, {"baseline": PREDICTORS["baseline"], "wdl-a": PREDICTORS["wdl-a"]}))
    assert md.strip(), "report must be non-empty"
    assert "## Gate verdict" in md
    assert "## Head-to-head: where equity wins" in md

    # The checked-in evidence artifact must carry the same two sections (so a stale
    # commit predating these sections fails CI). Section-presence only — not a byte diff.
    committed = Path(__file__).resolve().parents[1] / "reports" / "validation_sample.md"
    text = committed.read_text(encoding="utf-8")
    assert "## Gate verdict" in text and "## Head-to-head: where equity wins" in text


# --- head-to-head: where equity wins (task 0059) -------------------------------

def test_head_to_head_sign_convention_and_ranking():
    from chess_equity.validate.harness import head_to_head_deltas

    # Baseline is rating-blind; the "model" predicts the actual result perfectly, so it
    # must win (Δ > 0) in every slice and overall.
    rows = [
        _row(cp=0, we=1000, be=1000, phase="opening", result=1.0),
        _row(cp=0, we=2500, be=2500, phase="endgame", result=0.0),
    ]
    reports = evaluate(
        rows,
        {"baseline": baseline_cp, "model": lambda r: r.result},
    )
    h2h = head_to_head_deltas(reports)
    assert h2h is not None
    assert h2h.baseline == "baseline" and h2h.model == "model"
    # The oracle beats the rating-blind baseline everywhere.
    assert h2h.overall_delta > 0
    assert all(d.delta > 0 for d in h2h.slices), "oracle must win every slice (Δ > 0)"
    # Both rating bands and both phases appear as slices.
    assert {(d.slicer, d.value) for d in h2h.slices} >= {
        ("rating", "<1200"),
        ("rating", "2400+"),
        ("phase", "opening"),
        ("phase", "endgame"),
    }
    # Δ is the baseline-minus-model log-loss for that slice.
    d0 = h2h.slices[0]
    assert isclose(d0.delta, d0.baseline_log_loss - d0.model_log_loss)
    # Sorted biggest-win-first.
    assert [d.delta for d in h2h.slices] == sorted((d.delta for d in h2h.slices), reverse=True)


def test_head_to_head_needs_a_challenger():
    from chess_equity.validate.harness import head_to_head_deltas

    rows = [_row(cp=100, result=1.0), _row(cp=-100, result=0.0)]
    # Baseline only -> nothing to compare against.
    assert head_to_head_deltas(evaluate(rows, {"baseline": baseline_cp})) is None
    # No predictor named "baseline" -> no reference.
    assert head_to_head_deltas(evaluate(rows, {"model": lambda r: r.result})) is None


def test_format_report_includes_head_to_head_section():
    rows = [_row(cp=100, result=1.0), _row(cp=-100, result=0.0)]
    md = format_report(evaluate(rows, {"baseline": baseline_cp, "model": lambda r: r.result}))
    assert "## Head-to-head: where equity wins" in md
    assert "Δ > 0 means equity wins" in md
    # The worst-case slice verdict is surfaced inline (task 0121).
    assert "**Worst slice:**" in md
    assert "Equity wins on" in md
    # Single-predictor reports omit the section.
    solo = format_report(evaluate(rows, {"baseline": baseline_cp}))
    assert "Head-to-head" not in solo


def test_worst_slice_verdict_names_worst_and_counts_wins():
    from chess_equity.validate.harness import head_to_head_deltas, worst_slice_verdict

    # An oracle model wins every slice, so the worst slice is still an equity win and the
    # count is M/M.
    rows = [
        _row(cp=0, we=1000, be=1000, phase="opening", result=1.0),
        _row(cp=0, we=2500, be=2500, phase="endgame", result=0.0),
    ]
    h2h = head_to_head_deltas(evaluate(rows, {"baseline": baseline_cp, "model": lambda r: r.result}))
    assert h2h is not None
    line = worst_slice_verdict(h2h)
    # Names the worst slice (the last, lowest-Δ entry) and reports wins/total.
    worst = h2h.slices[-1]
    assert f"`{worst.slicer}` `{worst.value}`" in line
    assert f"Δ={worst.delta:+.4f}" in line
    total = len(h2h.slices)
    assert f"Equity wins on {total}/{total} slices." in line
    # All Δ > 0 here, so the verdict says equity still wins everywhere.
    assert "equity still wins every slice" in line


def test_head_to_head_on_committed_sample():
    from pathlib import Path

    from chess_equity.data.build import load_rows
    from chess_equity.validate.harness import head_to_head_deltas

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset.csv"
    rows = load_rows(str(sample))
    # wdl-a is a rating-conditioned model; compare it head-to-head against the baseline.
    reports = evaluate(rows, {"baseline": PREDICTORS["baseline"], "wdl-a": PREDICTORS["wdl-a"]})
    h2h = head_to_head_deltas(reports)
    assert h2h is not None and h2h.slices, "sample must yield at least one slice"
    # Every slice's Δ is exactly baseline-minus-model log-loss on that slice (sign convention).
    for d in h2h.slices:
        assert isclose(d.delta, d.baseline_log_loss - d.model_log_loss)
        assert d.n > 0
    # The rating slicer is among the reported slices (the thesis's headline axis).
    assert any(d.slicer == "rating" for d in h2h.slices)


# --- board-model predictor (task 0029) -----------------------------------------

class _FakeBoardModel:
    """A board-needing EquityModel: equity rises with white_elo, so it must read the
    FEN+ratings (a (cp, ratings) predictor signature can't express it)."""

    def evaluate(self, fen, white_elo, black_elo):
        from chess_equity.types import WDL, Equity

        p = min(0.99, white_elo / 4000.0)
        return Equity(wdl=WDL(p, 0.0, 1 - p), equity_white=100.0 * p, source="fake")


def _row_fen(**kw):
    base = dict(cp=0.0, we=1500, be=1500, result=0.5)
    base.update(kw)
    row = _row(cp=base["cp"], we=base["we"], be=base["be"], result=base["result"])
    return PositionRow(**{**row.as_dict(), "fen": "8/8/8/8/8/8/8/K6k w - - 0 1"})


def test_model_predictor_reads_fen_and_scores():
    from chess_equity.validate.harness import model_predictor

    predict = model_predictor(_FakeBoardModel())
    assert isclose(predict(_row_fen(we=2000)), 0.5)
    # Different ratings -> different prediction (the thing the baseline can't do).
    assert predict(_row_fen(we=3000)) > predict(_row_fen(we=1000))


def test_model_predictor_raises_without_fen():
    from chess_equity.validate.harness import model_predictor

    predict = model_predictor(_FakeBoardModel())
    with pytest.raises(ValueError):
        predict(_row())  # no fen on the row


def test_model_predictor_runs_through_evaluate():
    from chess_equity.validate.harness import model_predictor

    rows = [_row_fen(we=2600, result=1.0), _row_fen(we=900, result=0.0)]
    reports = evaluate(rows, {"fake": model_predictor(_FakeBoardModel())})
    assert reports[0].overall.n == 2


def test_board_predictor_scores_on_committed_fen_sample():
    """The end-to-end proof of task 0023: a board-needing model is scored straight
    off the committed FEN fixture, with no PGN rebuild.

    ``data/sample/dataset.csv`` carries no FEN (kept small for the cp-only models), so
    ``data/sample/dataset_fen.csv`` is the committed companion that lets the 0009
    harness exercise the board path on real, checked-in rows. Maia-2 (0005/0031) plugs
    into exactly this loop; ``_FakeBoardModel`` stands in so the test needs no weights.
    """
    from pathlib import Path

    from chess_equity.data.build import load_rows
    from chess_equity.validate.harness import model_predictor

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"
    rows = load_rows(str(sample))
    assert rows and all(r.fen is not None for r in rows), "committed FEN sample must carry FENs"

    reports = evaluate(rows, {"fake": model_predictor(_FakeBoardModel())})
    assert reports[0].overall.n == len(rows) > 0


# --- registering maia2 as a 0009 predictor (task 0031) -------------------------

def _fake_maia2():
    """A Maia2Equity wired to a fake backend (no torch) whose win_prob rises with the
    side-to-move's rating, so it conditions on ratings like the real value head."""
    from chess_equity.maia2 import Maia2Equity

    def backend(fen, elo_self, elo_oppo):
        return {}, min(0.99, elo_self / 4000.0)

    return Maia2Equity(backend=backend)


def test_build_predictors_mixes_row_and_board_models(monkeypatch):
    import chess_equity.validate.harness as h

    monkeypatch.setitem(h.BOARD_MODELS, "maia2", _fake_maia2)
    preds = h.build_predictors(["baseline", "maia2"])
    assert set(preds) == {"baseline", "maia2"}
    # The maia2 predictor reads the board (fen) — a row without fen must raise.
    assert isclose(preds["baseline"](_row_fen(cp=0)), 0.5)
    with pytest.raises(ValueError):
        preds["maia2"](_row())


def test_build_predictors_rejects_unknown():
    from chess_equity.validate.harness import build_predictors

    with pytest.raises(KeyError):
        build_predictors(["baseline", "nope"])


def test_maia2_registered_as_board_model():
    from chess_equity.validate.harness import BOARD_MODELS

    assert "maia2" in BOARD_MODELS


def test_validate_cli_scores_maia2_against_baseline(tmp_path, monkeypatch, capsys):
    """End-to-end: `validate --models baseline,maia2` over a --with-fen dataset."""
    from pathlib import Path

    import chess_equity.validate.harness as h
    from chess_equity.cli import main
    from chess_equity.data.build import build_dataset

    monkeypatch.setitem(h.BOARD_MODELS, "maia2", _fake_maia2)
    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_games.pgn"
    data = build_dataset(str(sample), str(tmp_path), name="fen", include_fen=True)

    rc = main(["validate", "--data", str(data), "--models", "baseline,maia2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "baseline" in out and "maia2" in out


def test_validate_cli_scores_maia2_on_committed_fen_sample(monkeypatch, capsys):
    """Runbook step-3 smoke: `validate --models maia2` end-to-end on the committed
    ``data/sample/dataset_fen.csv`` via the injectable fake backend (no torch/weights).

    The attended runbook scores ``baseline,wdl-a,maia2``, but the maia2 column only has
    real numbers with torch+weights — so the *plumbing* (committed FEN fixture ->
    Maia2Equity -> metrics) can rot silently in the sandbox. This drives it on the
    checked-in fixture (not a freshly-built dataset) and asserts maia2 yields a scored
    column, so a fixture that loses its FENs or broken maia2 wiring fails CI loudly.
    """
    from pathlib import Path

    import chess_equity.validate.harness as h
    from chess_equity.cli import main

    monkeypatch.setitem(h.BOARD_MODELS, "maia2", _fake_maia2)
    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"

    rc = main(["validate", "--data", str(sample), "--models", "maia2"])
    assert rc == 0
    out = capsys.readouterr().out
    # The Overall table must carry a maia2 row with a real (parseable) numeric log-loss
    # and a positive n — i.e. the column was actually scored, not just named.
    maia2_rows = [ln for ln in out.splitlines() if ln.strip().startswith("| maia2 |")]
    assert maia2_rows, "maia2 must appear as a scored row in the report"
    cells = [c.strip() for c in maia2_rows[0].strip("|").split("|")]
    # | maia2 | n | log-loss | Brier | ECE |
    assert int(cells[1]) > 0  # n scored rows
    assert float(cells[2]) >= 0.0  # finite, parseable log-loss


def test_validate_cli_maia2_needs_fen(tmp_path, monkeypatch, capsys):
    """A FEN-less dataset makes the maia2 predictor fail with a clean error, not a trace."""
    from pathlib import Path

    import chess_equity.validate.harness as h
    from chess_equity.cli import main
    from chess_equity.data.build import build_dataset

    monkeypatch.setitem(h.BOARD_MODELS, "maia2", _fake_maia2)
    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_games.pgn"
    data = build_dataset(str(sample), str(tmp_path), name="nofen")  # no fen column

    rc = main(["validate", "--data", str(data), "--models", "maia2"])
    assert rc == 1
    assert "include_fen" in capsys.readouterr().err


# --- registering maia-search as a 0009 predictor (task 0037) -------------------

def _fake_maia_search():
    """A MaiaSearchModel with no torch: uniform move priors + a fake rating-conditioned
    leaf, so the expectimax conditions on ratings like the real Maia-backed search."""
    from chess_equity.grading import UniformPolicy
    from chess_equity.maia2 import Maia2Equity
    from chess_equity.search import MaiaSearchModel

    def backend(fen, elo_self, elo_oppo):
        return {}, min(0.99, elo_self / 4000.0)

    return MaiaSearchModel(UniformPolicy(), Maia2Equity(backend=backend), depth=1, k=2)


def test_maia_search_registered_as_board_model():
    from chess_equity.validate.harness import BOARD_MODELS

    assert "maia-search" in BOARD_MODELS


def test_build_predictors_includes_maia_search(monkeypatch):
    import chess_equity.validate.harness as h

    monkeypatch.setitem(h.BOARD_MODELS, "maia-search", _fake_maia_search)
    preds = h.build_predictors(["baseline", "maia-search"])
    assert set(preds) == {"baseline", "maia-search"}
    # It reads the board (fen) — a row without fen must raise, like any board model.
    with pytest.raises(ValueError):
        preds["maia-search"](_row())


def test_validate_cli_scores_maia_search_against_baseline(tmp_path, monkeypatch, capsys):
    """End-to-end: `validate --models baseline,maia-search` over a --with-fen dataset."""
    from pathlib import Path

    import chess_equity.validate.harness as h
    from chess_equity.cli import main
    from chess_equity.data.build import build_dataset

    monkeypatch.setitem(h.BOARD_MODELS, "maia-search", _fake_maia_search)
    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_games.pgn"
    data = build_dataset(str(sample), str(tmp_path), name="fen", include_fen=True)

    rc = main(["validate", "--data", str(data), "--models", "baseline,maia-search"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "baseline" in out and "maia-search" in out


# --- runbook CLI smoke (task 0061) ---------------------------------------------

def test_runbook_validate_cli_end_to_end(tmp_path, capsys):
    """Drive the documented runbook command path end to end on the committed fixture.

    Mirrors the `validation-proof-runbook` devbrain page's smoke command — the dependency-free
    `baseline,wdl-a` gate with a game-level holdout — plus the `--out`/`--calibration`
    artifact flags. Guards the real CLI invocation: a broken flag, a renamed report
    section, or a dropped metric column fails here before the attended multi-GB run.
    """
    from pathlib import Path

    from chess_equity.cli import main

    data = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"
    report = tmp_path / "validation-report.md"
    calibration = tmp_path / "validation-calibration.md"

    rc = main(
        [
            "validate",
            "--data", str(data),
            "--models", "baseline,wdl-a",
            "--holdout", "0.5",
            "--seed", "0",
            "--out", str(report),
            "--calibration", str(calibration),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert f"wrote {report}" in out and f"wrote {calibration}" in out

    # The report opens with the gate verdict and carries the metric table columns.
    report_text = report.read_text(encoding="utf-8")
    assert report_text.startswith("# Validation report")
    assert "held-out test:" in report_text  # the --holdout split annotation
    for section in ("## Gate verdict", "## Overall", "## By rating", "## By phase"):
        assert section in report_text
    assert "| predictor | n | log-loss | Brier | ECE |" in report_text
    assert "wdl-a" in report_text and "baseline" in report_text

    # The calibration artifact carries its per-band reliability table.
    cal_text = calibration.read_text(encoding="utf-8")
    assert cal_text.startswith("# Calibration by rating band")
    assert "ECE by rating band" in cal_text


# --- tunable ECE bin count (task 0079) -------------------------------------------

def test_evaluate_threads_ece_bins():
    # preds sit in two extreme bins (0.1, 0.9), each off the true rate by 0.1.
    rows = [
        _row(cp=0.1, result=0.0),
        _row(cp=0.1, result=0.0),
        _row(cp=0.9, result=1.0),
        _row(cp=0.9, result=1.0),
    ]
    pred = {"p": lambda r: r.cp_eval}
    # Default (10 bins): each non-empty bin is off by 0.1 -> ECE 0.1 (behaviour unchanged).
    assert isclose(evaluate(rows, pred)[0].overall.ece, 0.1, abs_tol=1e-9)
    # One bin collapses everything: mean_pred 0.5 == mean_label 0.5 -> ECE 0.
    assert isclose(evaluate(rows, pred, bins=1)[0].overall.ece, 0.0, abs_tol=1e-9)


def test_validate_cli_ece_bins_changes_binning(tmp_path):
    from pathlib import Path

    from chess_equity.cli import main

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset.csv"

    def _ece(bins):
        out = tmp_path / f"r{bins}.md"
        rc = main(
            ["validate", "--data", str(sample), "--models", "baseline",
             "--bootstrap", "0", "--ece-bins", str(bins), "--out", str(out)]
        )
        assert rc == 0
        for line in out.read_text().splitlines():
            if line.startswith("| baseline |"):
                return line
        raise AssertionError("no baseline metrics row in report")

    # The default (10) is exercised elsewhere; here two different bin counts must move
    # the ECE column, proving --ece-bins threads all the way through.
    assert _ece(2) != _ece(5)


# --- per-slice head-to-head significance CIs (task 0068) -----------------------

def test_head_to_head_slice_cis_sign_and_floor():
    """Per-slice paired-bootstrap CIs: convention, ranking, and the small-n floor."""
    from chess_equity.validate.harness import head_to_head_slice_cis

    # An oracle predicts the result perfectly, so it must win (Δ > 0) in every large
    # enough slice; the floor keeps singletons from reading as significant.
    rows = [_row(cp=0, we=1000, be=1000, phase="opening", result=1.0) for _ in range(40)]
    rows += [_row(cp=0, we=2500, be=2500, phase="endgame", result=0.0) for _ in range(40)]
    rows += [_row(cp=0, we=1500, be=1500, phase="middlegame", result=1.0)]  # singleton slice
    preds = {"baseline": baseline_cp, "oracle": lambda r: r.result}
    h2h = head_to_head_slice_cis(rows, preds, min_n=10, n_resamples=500, seed=0)
    assert h2h is not None
    assert h2h.baseline == "baseline" and h2h.model == "oracle"
    assert h2h.metric == "log_loss" and h2h.min_n == 10
    # Sorted biggest-equity-win first.
    assert [d.delta for d in h2h.slices] == sorted((d.delta for d in h2h.slices), reverse=True)
    by_key = {(d.slicer, d.value): d for d in h2h.slices}
    # A big slice the oracle dominates: whole CI clears zero -> `equity`.
    big = by_key[("phase", "opening")]
    assert big.n == 40 and big.lo is not None and big.lo > 0.0 and big.verdict == "equity"
    # The singleton middlegame slice is below the floor: no CI, never significant.
    tiny = by_key[("phase", "middlegame")]
    assert tiny.n == 1 and tiny.lo is None and tiny.hi is None
    assert tiny.verdict == "inconclusive"


def test_head_to_head_slice_cis_needs_baseline_and_challenger():
    from chess_equity.validate.harness import head_to_head_slice_cis

    rows = [_row(cp=100, result=1.0), _row(cp=-100, result=0.0)]
    # No challenger -> nothing to compare against.
    assert head_to_head_slice_cis(rows, {"baseline": baseline_cp}) is None
    # No predictor named "baseline" -> no reference.
    assert head_to_head_slice_cis(rows, {"model": lambda r: r.result}) is None


def test_head_to_head_slice_cis_deterministic_on_sample():
    """A seeded run pins exact per-slice CIs on the committed FEN sample (task 0068)."""
    from pathlib import Path

    from chess_equity.data.build import load_rows
    from chess_equity.validate.harness import head_to_head_slice_cis

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"
    rows = load_rows(str(sample))
    preds = {
        "baseline": PREDICTORS["baseline"],
        "baseline+clock": PREDICTORS["baseline+clock"],
        "wdl-a": PREDICTORS["wdl-a"],
    }
    # min_n=8: the larger slices (n>=9) get real CIs; n<=6 slices stay inconclusive.
    h2h = head_to_head_slice_cis(rows, preds, min_n=8, n_resamples=2000, seed=0)
    assert h2h is not None and h2h.model == "wdl-a"
    by_key = {(d.slicer, d.value): d for d in h2h.slices}

    # The clock 'comfortable' slice (n=13) is a pinned, significant equity win: the whole
    # 95% CI sits above zero. These bounds are byte-reproducible under the seed.
    comf = by_key[("clock", "comfortable(60s+)")]
    assert comf.n == 13 and comf.verdict == "equity"
    assert isclose(comf.delta, 0.058342, abs_tol=1e-5)
    assert isclose(comf.lo, 0.0023495, abs_tol=1e-6)
    assert isclose(comf.hi, 0.1061781, abs_tol=1e-6)

    # A 6-row rating slice is below the floor: no CI, reads inconclusive (not spuriously
    # significant), even though its point delta favours the baseline.
    small = by_key[("rating", "2000-2399")]
    assert small.n == 6 and small.lo is None and small.verdict == "inconclusive"
    assert small.delta < 0.0  # baseline is better here, but we don't call it significant

    # Determinism: a second seeded run yields identical CIs.
    again = head_to_head_slice_cis(rows, preds, min_n=8, n_resamples=2000, seed=0)
    assert [(d.lo, d.hi, d.verdict) for d in again.slices] == [
        (d.lo, d.hi, d.verdict) for d in h2h.slices
    ]


def test_head_to_head_slice_cis_default_floor_guards_tiny_sample():
    """The default floor (30) leaves the 15-row sample with no significant slice."""
    from pathlib import Path

    from chess_equity.data.build import load_rows
    from chess_equity.validate.harness import head_to_head_slice_cis

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"
    rows = load_rows(str(sample))
    preds = {"baseline": PREDICTORS["baseline"], "wdl-a": PREDICTORS["wdl-a"]}
    h2h = head_to_head_slice_cis(rows, preds, n_resamples=200, seed=0)
    assert h2h is not None and h2h.min_n == 30
    # Every slice is below the default floor on this tiny fixture -> all inconclusive,
    # none gets a CI. The thesis can't be "proven" on 15 rows by accident.
    assert all(d.lo is None and d.verdict == "inconclusive" for d in h2h.slices)


def test_format_head_to_head_cis_table():
    from chess_equity.validate.harness import format_head_to_head_cis, head_to_head_slice_cis

    rows = [_row(cp=0, we=1000, be=1000, phase="opening", result=1.0) for _ in range(40)]
    rows += [_row(cp=0, we=2500, be=2500, phase="endgame", result=0.0) for _ in range(40)]
    h2h = head_to_head_slice_cis(
        rows, {"baseline": baseline_cp, "oracle": lambda r: r.result},
        min_n=10, n_resamples=300, seed=0,
    )
    md = format_head_to_head_cis(h2h)
    assert "## Head-to-head significance: per-slice CIs (baseline vs oracle)" in md
    assert "Δ > 0 = equity wins" in md
    assert "| slice | value | n | Δ log-loss | 95% CI | verdict |" in md
    assert "equity" in md
