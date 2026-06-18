"""The ``gate-check`` consumer of verdict.json (task 0136).

Task 0135 writes a machine-readable ``verdict.json`` beside each report; ``gate-check``
turns its top-level ``pass`` into a process exit code so CI / a README badge can assert
the thesis gate *without re-parsing markdown*. These tests pin that contract: a passing
verdict exits 0, a failing one exits 2 (mirroring ``validate --gate``), and anything the
consumer can't trust — missing file, malformed JSON, an unrecognised schema — fails loudly
(exit 3) rather than silently passing.
"""

from __future__ import annotations

import json
from pathlib import Path

from chess_equity.cli import (
    GATE_CHECK_DEFAULT_VERDICT,
    GATE_CHECK_ERROR,
    GATE_CHECK_FAIL,
    GATE_CHECK_PASS,
    main,
)
from chess_equity.validate.harness import GATE_VERDICT_SCHEMA

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "sample" / "dataset.csv"


def _validate(out: Path, *extra) -> int:
    return main(
        ["validate", "--data", str(SAMPLE), "--out", str(out), "--bootstrap", "0", *extra]
    )


def _verdict_path(out: Path) -> Path:
    return out.with_name(out.stem + ".verdict.json")


def test_gate_check_passes_on_real_passing_verdict(tmp_path):
    # wdl-a beats baseline on the committed sample → verdict.json says pass → gate-check exits 0.
    # --min-n 0 turns off the underpowered floor (task 0132) for this tiny fixture.
    out = tmp_path / "report.md"
    _validate(out, "--models", "baseline,wdl-a", "--min-n", "0")
    rc = main(["gate-check", str(_verdict_path(out))])
    assert rc == GATE_CHECK_PASS


def test_gate_check_fails_on_real_failing_verdict(tmp_path):
    # baseline+clock is a no-op on the clock-blind sample → cannot beat baseline → pass:false.
    out = tmp_path / "report.md"
    _validate(out, "--models", "baseline,baseline+clock", "--min-n", "0")
    rc = main(["gate-check", str(_verdict_path(out))])
    assert rc == GATE_CHECK_FAIL


def test_gate_check_exit_code_agrees_with_gate(tmp_path):
    # The whole point: gate-check on the json must agree with `validate --gate`'s exit code,
    # so the two are interchangeable for CI.
    out = tmp_path / "report.md"
    gate_rc = _validate(out, "--models", "baseline,wdl-a", "--gate", "--min-n", "0")
    check_rc = main(["gate-check", str(_verdict_path(out))])
    assert gate_rc == 0 and check_rc == GATE_CHECK_PASS


def test_gate_check_defaults_to_headline_verdict(tmp_path, monkeypatch):
    # A bare `gate-check` (no path arg) reads reports/validation_headline.verdict.json,
    # relative to cwd, so CI can assert the headline gate with zero arguments (task 0143).
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "reports" / "validation_headline.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    _validate(out, "--models", "baseline,wdl-a", "--min-n", "0")
    assert _verdict_path(out) == tmp_path / GATE_CHECK_DEFAULT_VERDICT
    rc = main(["gate-check"])
    assert rc == GATE_CHECK_PASS


def test_gate_check_default_errors_when_headline_absent(tmp_path, monkeypatch):
    # No headline report yet → bare `gate-check` fails loudly (missing file), never silent pass.
    monkeypatch.chdir(tmp_path)
    rc = main(["gate-check"])
    assert rc == GATE_CHECK_ERROR


def test_gate_check_errors_on_missing_file(tmp_path):
    rc = main(["gate-check", str(tmp_path / "nope.verdict.json")])
    assert rc == GATE_CHECK_ERROR


def test_gate_check_errors_on_unknown_schema(tmp_path):
    bad = tmp_path / "bad.verdict.json"
    bad.write_text(json.dumps({"schema": "something-else/v9", "pass": True}), encoding="utf-8")
    rc = main(["gate-check", str(bad)])
    assert rc == GATE_CHECK_ERROR


def test_gate_check_errors_on_malformed_json(tmp_path):
    bad = tmp_path / "bad.verdict.json"
    bad.write_text("{not valid json", encoding="utf-8")
    rc = main(["gate-check", str(bad)])
    assert rc == GATE_CHECK_ERROR


def test_gate_check_errors_on_missing_pass_field(tmp_path):
    bad = tmp_path / "bad.verdict.json"
    bad.write_text(json.dumps({"schema": GATE_VERDICT_SCHEMA}), encoding="utf-8")
    rc = main(["gate-check", str(bad)])
    assert rc == GATE_CHECK_ERROR
