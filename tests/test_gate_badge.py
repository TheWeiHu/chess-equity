"""The ``gate-badge`` renderer and the committed README gate badge (task 0141).

The README's **gate** badge is a shields.io *endpoint* badge pointed at a committed
``reports/validation_real.badge.json``. ``gate_badge_payload`` / the ``gate-badge`` command
regenerate that file from the same ``verdict.json`` ``gate-check`` asserts, so the badge's
green/red state is *driven by* ``verdict.json['pass']`` rather than hand-edited. These tests
pin that contract — pass → green ``passing`` / fail → red ``failing`` — and, crucially, the
last test fails CI if the committed badge and the committed verdict ever disagree (the
acceptance criterion: badge state can never drift from the verdict).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_equity.cli import GATE_CHECK_ERROR, GATE_CHECK_PASS, main
from chess_equity.validate.harness import GATE_VERDICT_SCHEMA, gate_badge_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_VERDICT = REPO_ROOT / "reports" / "validation_real.verdict.json"
REAL_BADGE = REPO_ROOT / "reports" / "validation_real.badge.json"
README = REPO_ROOT / "README.md"


def _verdict(passed: bool) -> dict:
    return {"schema": GATE_VERDICT_SCHEMA, "pass": passed}


def test_payload_passing_is_green():
    badge = gate_badge_payload(_verdict(True))
    assert badge == {
        "schemaVersion": 1,
        "label": "gate",
        "message": "passing",
        "color": "brightgreen",
    }


def test_payload_failing_is_red():
    badge = gate_badge_payload(_verdict(False))
    assert badge["message"] == "failing"
    assert badge["color"] == "red"


def test_payload_rejects_unknown_schema():
    with pytest.raises(ValueError):
        gate_badge_payload({"schema": "something-else/v9", "pass": True})


def test_payload_rejects_non_boolean_pass():
    with pytest.raises(ValueError):
        gate_badge_payload({"schema": GATE_VERDICT_SCHEMA, "pass": "yes"})


def test_cli_writes_badge_beside_report(tmp_path):
    verdict = tmp_path / "validation_real.verdict.json"
    verdict.write_text(json.dumps(_verdict(True)), encoding="utf-8")
    rc = main(["gate-badge", str(verdict)])
    assert rc == GATE_CHECK_PASS
    # Default name is parallel to the verdict, not `*.verdict.badge.json`.
    badge_path = tmp_path / "validation_real.badge.json"
    assert badge_path.exists()
    assert json.loads(badge_path.read_text())["color"] == "brightgreen"


def test_cli_stdout(tmp_path, capsys):
    verdict = tmp_path / "v.verdict.json"
    verdict.write_text(json.dumps(_verdict(False)), encoding="utf-8")
    rc = main(["gate-badge", str(verdict), "--out", "-"])
    assert rc == GATE_CHECK_PASS
    assert json.loads(capsys.readouterr().out)["message"] == "failing"


def test_cli_errors_on_unknown_schema(tmp_path):
    bad = tmp_path / "bad.verdict.json"
    bad.write_text(json.dumps({"schema": "nope/v1", "pass": True}), encoding="utf-8")
    rc = main(["gate-badge", str(bad), "--out", "-"])
    assert rc == GATE_CHECK_ERROR


def test_committed_real_verdict_is_a_passing_mirror():
    # The committed real-evidence verdict (mirror of reports/validation_real.md) must say PASS.
    verdict = json.loads(REAL_VERDICT.read_text(encoding="utf-8"))
    assert verdict["schema"] == GATE_VERDICT_SCHEMA
    assert verdict["pass"] is True


def test_committed_badge_matches_committed_verdict():
    # The acceptance criterion: CI fails if the badge and the verdict.json disagree. The
    # committed badge must be exactly what `gate-badge` would regenerate from the verdict.
    verdict = json.loads(REAL_VERDICT.read_text(encoding="utf-8"))
    badge = json.loads(REAL_BADGE.read_text(encoding="utf-8"))
    assert badge == gate_badge_payload(verdict)


def test_readme_renders_the_committed_badge():
    # The badge in the README must point a shields endpoint at the committed badge.json.
    readme = README.read_text(encoding="utf-8")
    assert "img.shields.io/endpoint" in readme
    assert "reports/validation_real.badge.json" in readme
