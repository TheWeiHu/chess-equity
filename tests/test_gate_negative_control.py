"""Negative control: the thesis gate must FAIL on a deliberately-broken model (task 0130).

A PASS verdict only means something if the gate can also FAIL. Every checked-in run
PASSes, which leaves open the worry that the gate is vacuously green — that it would
wave through *any* predictor. These tests close that hole by registering deliberately-
miscalibrated predictors (a constant 0.5, and an equity-inverter) and asserting the gate
FAILs them, while `wdl-a` PASSes on the *same* committed FEN fixture in the *same* run.
Same data, same baseline, same code path: the only thing that changes is the model, so a
FAIL here is the gate's discriminating power, not a quirk of the fixture.

The gate is exercised both ways the codebase offers it:
- the `gate_verdicts` Python verdict (point-only, the pre-significance gate); and
- the `validate --gate` exit-code contract (0 PASS / 2 FAIL) via the CLI registry.
"""

from __future__ import annotations

from pathlib import Path

from chess_equity.cli import main
from chess_equity.data.build import load_rows
from chess_equity.validate.harness import (
    PREDICTORS,
    baseline_cp,
    build_predictors,
    evaluate,
    gate_verdicts,
    wdl_a,
)

# The committed FEN fixture wdl-a is point-better on (see test_validate's significance
# test): enough rows that wdl-a wins on both log-loss and Brier under the point-only gate.
FIXTURE = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"

# --- deliberately-broken predictors (the negative controls) --------------------------
#
# Each is registered into the predictors dict exactly like a real model, so it travels
# the identical evaluate() -> gate_verdicts() path. They are intentionally NOT added to
# the production PREDICTORS registry — they exist only to prove the gate can say no.


def constant_half(_) -> float:
    """Predicts a coin-flip for every position — maximally uninformative."""
    return 0.5


def inverted_equity(row) -> float:
    """Flips the baseline's call: says White is losing exactly when it is winning."""
    return 1.0 - baseline_cp(row)


def test_gate_fails_broken_models_but_passes_wdl_a_on_the_same_fixture():
    rows = load_rows(str(FIXTURE))
    # One run, one fixture, one baseline: the real model and the broken ones side by side.
    predictors = {
        "baseline": baseline_cp,
        "wdl-a": wdl_a,
        "constant-0.5": constant_half,
        "inverted": inverted_equity,
    }
    verdicts = {v.name: v for v in gate_verdicts(evaluate(rows, predictors))}

    # The gate has teeth: both deliberately-broken predictors FAIL.
    assert verdicts["constant-0.5"].passed is False
    assert verdicts["inverted"].passed is False
    # ...and the negative control is real because the same gate, same fixture, PASSes
    # the legitimate rating-conditioned model. (Point-only gate, as `--gate` uses.)
    assert verdicts["wdl-a"].passed is True


def test_inverted_model_is_worse_on_both_metrics():
    # Document *why* the inverter fails: it is strictly worse than the baseline it inverts
    # on both gate metrics, so there is no metric on which it could sneak a point win.
    rows = load_rows(str(FIXTURE))
    verdicts = {
        v.name: v
        for v in gate_verdicts(
            evaluate(rows, {"baseline": baseline_cp, "inverted": inverted_equity})
        )
    }
    v = verdicts["inverted"]
    assert v.log_loss_delta > 0 and v.brier_delta > 0
    assert v.passed is False


def test_validate_gate_exit_code_fails_on_broken_model(capsys, monkeypatch):
    # The machine-checkable contract (task 0115): `validate --gate` must exit 2 on a broken
    # model and 0 on wdl-a, over the SAME fixture. The broken model is injected into the CLI
    # registry only for this test (monkeypatch), so production `--models` choices are unchanged.
    monkeypatch.setitem(PREDICTORS, "constant-0.5", constant_half)

    def run(models):
        # --bootstrap 0: the exit-code gate reads the point verdict, not the CIs (fast).
        # --min-n 0: this asserts the PASS/FAIL teeth on the small fixture, so disable the
        # underpowered guard (task 0132) that would otherwise read the tiny n as INCONCLUSIVE.
        return main(
            ["validate", "--data", str(FIXTURE), "--bootstrap", "0", "--min-n", "0", "--models", models, "--gate"]
        )

    fail_rc = run("baseline,constant-0.5")
    fail_err = capsys.readouterr().err
    assert fail_rc == 2
    assert "GATE: FAIL" in fail_err and "constant-0.5" in fail_err

    pass_rc = run("baseline,wdl-a")
    pass_out = capsys.readouterr().out
    assert pass_rc == 0
    assert "GATE: PASS" in pass_out and "wdl-a" in pass_out


def test_build_predictors_does_not_expose_the_negative_controls():
    # Guard the registry: the broken models must never be selectable in a real run.
    for junk in ("constant-0.5", "inverted"):
        assert junk not in PREDICTORS
        try:
            build_predictors([junk])
        except KeyError:
            pass  # expected — unknown predictor name
        else:  # pragma: no cover - only trips if a junk model leaks into the registry
            raise AssertionError(f"{junk!r} should not be a buildable predictor")
