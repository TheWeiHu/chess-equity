"""Positive control: the thesis gate must PASS — with significance — on a committed,
network-free fixture whose outcomes obey a KNOWN rating-conditioned WDL law (task 0131).

This is the complement of the negative control (``test_gate_negative_control.py``). That
test proves the gate can say *no* (it FAILs deliberately-broken models). This one proves
the other half is real too: there is a checked-in, torch-free dataset on which the gate
says *yes* and the win is statistically significant — not just a point delta, but the
headline log-loss 95% CI clearing zero. Together they show the gate both
passes-when-it-should and fails-when-it-should, entirely offline.

The only other committed PASS is ``reports/validation_sample.md`` (15 rows, explicitly
illustrative-not-proof, and its CI actually *straddles* zero — see ``test_validate``'s
``test_significance_gate_flips_marginal_point_win_to_fail``); the real n=8000 Lichess run
is human-approval-gated (task 0128). So before this fixture, CI had no committed dataset
on which the *significance-aware* gate genuinely PASSed.

By construction (see ``validate/ground_truth.py``): outcomes are sampled from a logistic
whose slope steepens with rating, so the rating-blind baseline cannot fit every band, and
the rating-aware ``oracle`` (the law's own conditional mean) beats it. n=2400 is well above
the rows the paired bootstrap needs to clear zero, so the PASS is not boundary-flaky.
"""

from __future__ import annotations

from chess_equity.data.build import load_rows
from chess_equity.validate.ground_truth import (
    DEFAULT_N,
    DEFAULT_SEED,
    FIXTURE_PATH,
    generate_rows,
    oracle,
    write_fixture,
)
from chess_equity.validate.harness import (
    BOARD_MODELS,
    PREDICTORS,
    baseline_cp,
    build_predictors,
    compare_to_baseline,
    evaluate,
    gate_verdicts,
)


def test_fixture_is_committed_and_large_enough():
    # The acceptance bar: a checked-in fixture with enough rows (n >= 2000) for the paired
    # bootstrap to put a tight CI on the delta. Loaded through the real dataset loader, so
    # it travels the identical path a real Lichess dataset would.
    assert FIXTURE_PATH.exists(), f"committed fixture missing at {FIXTURE_PATH}"
    rows = load_rows(str(FIXTURE_PATH))
    assert len(rows) == DEFAULT_N >= 2000


def test_committed_fixture_matches_the_generator_byte_for_byte(tmp_path):
    # Drift guard: the committed CSV must be exactly what the seeded generator produces, so
    # the fixture can never silently diverge from the law the test claims it obeys. If this
    # fails, re-run `python -m chess_equity.validate.ground_truth` and commit the result.
    regenerated = tmp_path / "regen.csv"
    write_fixture(regenerated, n=DEFAULT_N, seed=DEFAULT_SEED)
    assert regenerated.read_text(encoding="utf-8") == FIXTURE_PATH.read_text(encoding="utf-8")


def test_gate_passes_with_significance_on_the_ground_truth_fixture():
    # The heart of 0131: same evaluate -> compare_to_baseline -> gate_verdicts path the
    # real proof run uses, on the committed fixture. The rating-aware oracle must beat the
    # rating-blind baseline on BOTH point metrics AND have its headline log-loss 95% CI
    # clear zero — a significant PASS, not a point delta that could be noise.
    rows = load_rows(str(FIXTURE_PATH))
    predictors = {"baseline": baseline_cp, "oracle": oracle}
    reports = evaluate(rows, predictors)
    comps = compare_to_baseline(rows, predictors, n_resamples=2000, seed=0)
    verdicts = gate_verdicts(reports, comparisons=comps)

    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.name == "oracle"
    # Point win on both gate metrics (negative delta = lower loss = better).
    assert v.log_loss_delta < 0 and v.brier_delta < 0
    # ...and the win is significant: the whole log-loss CI sits below zero.
    assert v.significant is True
    assert v.headline_ci is not None and v.headline_ci.metric == "log_loss"
    assert v.headline_ci.hi < 0.0  # the headline CI is BELOW zero, as the task requires
    assert v.headline_ci.beats_baseline
    assert v.passed is True


def test_gate_cli_exit_code_passes_on_the_ground_truth_fixture(capsys, monkeypatch):
    # The machine-checkable contract (task 0115): `validate --gate` must exit 0 on this
    # fixture with a significance-aware bootstrap. The oracle is injected into the CLI
    # registry only for this test (monkeypatch), so production --models choices are
    # unchanged. --bootstrap 2000 makes the exit code significance-aware, not point-only.
    from chess_equity.cli import main

    monkeypatch.setitem(PREDICTORS, "oracle", oracle)
    rc = main(
        [
            "validate",
            "--data",
            str(FIXTURE_PATH),
            "--models",
            "baseline,oracle",
            "--bootstrap",
            "2000",
            "--seed",
            "0",
            "--gate",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "GATE: PASS" in out and "oracle" in out


def test_oracle_is_not_a_production_predictor():
    # Guard the registry: the ground-truth oracle is a test instrument, not a shippable
    # model — it must never be selectable in a real run (it knows the synthetic answer).
    assert "oracle" not in PREDICTORS and "oracle" not in BOARD_MODELS
    try:
        build_predictors(["oracle"])
    except KeyError:
        pass  # expected — unknown predictor name
    else:  # pragma: no cover - only trips if the oracle leaks into the registry
        raise AssertionError("'oracle' should not be a buildable predictor")


def test_generator_is_deterministic():
    # The fixture's reproducibility rests on the generator being a pure function of
    # (n, seed); pin that directly so a future refactor can't introduce hidden randomness.
    a = generate_rows(64, seed=7)
    b = generate_rows(64, seed=7)
    assert [r.as_dict() for r in a] == [r.as_dict() for r in b]
    # A different seed gives different rows (the seed actually drives the sampling).
    c = generate_rows(64, seed=8)
    assert [r.as_dict() for r in a] != [r.as_dict() for r in c]
