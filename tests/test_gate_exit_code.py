"""The machine-checkable thesis gate (task 0115): `validate --gate` exit-code contract.

The prose PASS/FAIL verdict in the report is not assertable by CI or the autonomous
loop. `--gate` makes the *exit code* carry it: 0 only if every rating-conditioned
predictor beats `baseline` on log-loss AND Brier, 2 if any FAILS, 3 if there is no
challenger to gate. These tests pin both directions on the committed 15-row sample
(where `wdl-a` beats baseline but `baseline+clock` does not) plus the misuse case.
"""

from pathlib import Path

from chess_equity.cli import main

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset.csv"


def _validate(*extra):
    # --bootstrap 0 keeps these fast; the gate reads the point verdict, not the CIs.
    # --min-n 0 disables the underpowered guard (task 0132) so these PASS/FAIL-mechanics
    # tests still exercise the verdict on the 15-row sample; the guard itself is covered
    # in test_gate_underpowered.py.
    return main(
        ["validate", "--data", str(SAMPLE), "--bootstrap", "0", "--min-n", "0", *extra]
    )


def test_gate_pass_exits_zero(capsys):
    rc = _validate("--models", "baseline,wdl-a", "--gate")
    out = capsys.readouterr().out
    assert rc == 0
    assert "GATE: PASS" in out and "wdl-a" in out


def test_gate_fail_exits_nonzero(capsys):
    # baseline+clock is a no-op on the clock-blind sample, so it cannot beat baseline.
    rc = _validate("--models", "baseline,baseline+clock", "--gate")
    err = capsys.readouterr().err
    assert rc == 2
    assert "GATE: FAIL" in err and "baseline+clock" in err


def test_gate_fails_if_any_challenger_fails(capsys):
    # A mixed run: wdl-a passes but baseline+clock fails → the gate as a whole FAILS,
    # and the exit code reflects the failing predictor (the scope is whatever --models ran).
    rc = _validate("--models", "baseline,baseline+clock,wdl-a", "--gate")
    err = capsys.readouterr().err
    assert rc == 2
    assert "baseline+clock" in err


def test_gate_without_challenger_is_misuse(capsys):
    rc = _validate("--models", "baseline", "--gate")
    err = capsys.readouterr().err
    assert rc == 3
    assert "--gate needs a rating-conditioned predictor" in err


def test_no_gate_flag_never_changes_exit_code(capsys):
    # Without --gate, a FAILing verdict still exits 0 — the report is the only output.
    rc = _validate("--models", "baseline,baseline+clock")
    out = capsys.readouterr().out
    assert rc == 0
    assert "GATE:" not in out  # no gate line emitted when the flag is off
