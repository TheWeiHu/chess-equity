"""The canonical headline-run recipe + its dry-run smoke test (task 0114).

The real headline comparison (0087) is HELD on torch + Maia weights + a full dump, so
here the Maia-2 leg runs against a fake injected backend: the recipe (models, slicers,
out path) is exercised end-to-end on the committed fen sample with **no torch**, so the
moment a human approves the real run it is copy-paste.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_equity.types import WDL, Equity
from chess_equity.validate.headline import (
    HEADLINE_MODELS,
    HEADLINE_OUT,
    SMOKE_DATA,
    headline_namespace,
    run_headline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeMaia2:
    """A torch-free stand-in for Maia-2's value head: equity tracks White's rating, so it
    reads the FEN+ratings (a board model) rather than just cp."""

    def evaluate(self, fen, white_elo, black_elo):
        p = min(0.99, max(0.01, white_elo / 4000.0))
        return Equity(wdl=WDL(p, 0.0, 1 - p), equity_white=100.0 * p, source="fake-maia2")


@pytest.fixture
def fake_maia2(monkeypatch):
    """Swap the lazy maia2 factory for the fake, so build_predictors('maia2') needs no torch."""
    from chess_equity.validate import harness

    monkeypatch.setitem(harness.BOARD_MODELS, "maia2", lambda: _FakeMaia2())


def test_recipe_pins_the_three_headline_models():
    ns = headline_namespace("some/dump.csv")
    assert ns.models == HEADLINE_MODELS == "baseline,wdl-a,maia2"
    assert ns.out == HEADLINE_OUT == "reports/validation_headline.md"
    assert ns.data == "some/dump.csv"


def test_smoke_data_is_committed_and_has_fen():
    # The maia2 leg needs row.fen, so the pinned dry-run sample must carry it.
    sample = REPO_ROOT / SMOKE_DATA
    assert sample.exists(), f"{SMOKE_DATA} must be committed for the dry-run"


def test_headline_runs_end_to_end_with_fake_maia2(tmp_path, fake_maia2):
    out = tmp_path / "validation_headline.md"
    # bootstrap=0 keeps the smoke run fast and deterministic; the recipe is what's pinned.
    rc = run_headline(str(REPO_ROOT / SMOKE_DATA), out=str(out), bootstrap=0)
    assert rc == 0
    report = out.read_text(encoding="utf-8")
    # All three headline legs scored, and the slice axes the thesis hangs on are present.
    for leg in ("baseline", "wdl-a", "maia2"):
        assert leg in report
    assert "## By rating" in report
    assert "## Overall" in report


def test_headline_creates_missing_out_dir(tmp_path, fake_maia2):
    out = tmp_path / "nested" / "deep" / "report.md"
    rc = run_headline(str(REPO_ROOT / SMOKE_DATA), out=str(out), bootstrap=0)
    assert rc == 0 and out.exists()
