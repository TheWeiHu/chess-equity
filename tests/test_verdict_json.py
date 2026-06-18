"""Machine-readable gate verdict.json beside the markdown report (task 0135).

The gate result used to be consumable only as markdown prose (a `## Gate verdict` block)
plus a `--gate` exit code. CI, a README status badge, or a dashboard had to re-parse the
markdown to assert the proof. Now any `validate --out` (and the `headline` recipe, which
sets `--out`) writes a sibling `verdict.json` with the structured result. These tests pin
its schema and — the load-bearing contract — that its `pass` field agrees with both the
markdown verdict and the `--gate` exit code, on the committed torch-free sample and on the
fake-Maia-2 headline smoke.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_equity.cli import main
from chess_equity.types import WDL, Equity
from chess_equity.validate.harness import GATE_VERDICT_SCHEMA
from chess_equity.validate.headline import SMOKE_DATA, run_headline

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "sample" / "dataset.csv"


def _validate(out: Path, *extra) -> int:
    # --bootstrap 0 keeps these fast and deterministic; the verdict mirrors the point gate.
    return main(
        ["validate", "--data", str(SAMPLE), "--out", str(out), "--bootstrap", "0", *extra]
    )


def _verdict_beside(out: Path) -> dict:
    sibling = out.with_name(out.stem + ".verdict.json")
    assert sibling.exists(), f"expected sibling verdict.json at {sibling}"
    return json.loads(sibling.read_text(encoding="utf-8"))


def test_verdict_json_schema_and_fields(tmp_path):
    out = tmp_path / "report.md"
    _validate(out, "--models", "baseline,wdl-a")
    payload = _verdict_beside(out)

    assert payload["schema"] == GATE_VERDICT_SCHEMA
    assert payload["baseline"] == "baseline"
    assert payload["headline_metric"] == "log_loss"
    assert payload["significance_gated"] is False  # --bootstrap 0
    assert isinstance(payload["n"], int) and payload["n"] > 0
    assert isinstance(payload["pass"], bool)

    assert len(payload["predictors"]) == 1
    wdl = payload["predictors"][0]
    assert wdl["name"] == "wdl-a"
    for key in ("pass", "significant", "log_loss_delta", "brier_delta", "pct_improvement", "n"):
        assert key in wdl, f"predictor entry missing {key}"
    assert wdl["n"] == payload["n"]
    assert wdl["significant"] is None  # no paired-bootstrap CIs on a --bootstrap 0 run


def test_pass_agrees_with_markdown_and_exit_code(tmp_path):
    # wdl-a beats baseline on the committed sample → PASS everywhere.
    out = tmp_path / "report.md"
    rc = _validate(out, "--models", "baseline,wdl-a", "--gate")
    payload = _verdict_beside(out)
    report = out.read_text(encoding="utf-8")

    assert rc == 0  # exit code says PASS
    assert payload["pass"] is True  # json says PASS
    assert payload["predictors"][0]["pass"] is True
    # the markdown gate block agrees
    assert "## Gate verdict" in report and "**PASS**" in report
    # a real, positive effect size landed in the structured payload
    assert payload["predictors"][0]["log_loss_delta"] < 0
    assert payload["predictors"][0]["pct_improvement"] > 0


def test_fail_agrees_with_markdown_and_exit_code(tmp_path):
    # baseline+clock is a no-op on the clock-blind sample → it cannot beat baseline → FAIL.
    out = tmp_path / "report.md"
    rc = _validate(out, "--models", "baseline,baseline+clock", "--gate")
    payload = _verdict_beside(out)
    report = out.read_text(encoding="utf-8")

    assert rc == 2  # exit code says FAIL
    assert payload["pass"] is False  # json says FAIL
    assert payload["predictors"][0]["name"] == "baseline+clock"
    assert payload["predictors"][0]["pass"] is False
    assert "**FAIL**" in report  # the markdown gate block agrees


def test_no_challenger_writes_no_verdict(tmp_path):
    # baseline-only: nothing to gate, so no sibling json (mirrors the --gate misuse exit 3).
    out = tmp_path / "report.md"
    _validate(out, "--models", "baseline")
    assert out.exists()
    assert not out.with_name(out.stem + ".verdict.json").exists()


class _FakeMaia2:
    """Torch-free Maia-2 stand-in (mirrors tests/test_headline.py): equity tracks rating."""

    def evaluate(self, fen, white_elo, black_elo):
        p = min(0.99, max(0.01, white_elo / 4000.0))
        return Equity(wdl=WDL(p, 0.0, 1 - p), equity_white=100.0 * p, source="fake-maia2")


@pytest.fixture
def fake_maia2(monkeypatch):
    from chess_equity.validate import harness

    monkeypatch.setitem(harness.BOARD_MODELS, "maia2", lambda: _FakeMaia2())


def test_headline_smoke_emits_verdict_json(tmp_path, fake_maia2):
    # The pinned headline recipe sets --out, so it gets a sibling verdict.json for free —
    # the task's headline path, exercised torch-free on the committed fen sample.
    out = tmp_path / "validation_headline.md"
    rc = run_headline(str(REPO_ROOT / SMOKE_DATA), out=str(out), bootstrap=0)
    assert rc == 0
    payload = _verdict_beside(out)
    assert payload["schema"] == GATE_VERDICT_SCHEMA
    names = {p["name"] for p in payload["predictors"]}
    assert {"wdl-a", "maia2"} <= names  # both rating-conditioned legs gated against baseline
    assert "baseline" not in names  # the baseline is the thing gated against, not a row
