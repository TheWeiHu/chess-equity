"""Tests for the reliability-curve plots (task 0036)."""

from __future__ import annotations

import builtins

import pytest

from chess_equity.data.schema import PositionRow
from chess_equity.validate.calibration import band_reliability
from chess_equity.validate.harness import baseline_cp
from chess_equity.validate.plots import MatplotlibNotInstalled, save_reliability_plot


def _row(*, cp=0.0, we=1500, be=1500, result=0.5) -> PositionRow:
    return PositionRow(
        cp_eval=cp,
        white_elo=we,
        black_elo=be,
        ply=10,
        phase="middlegame",
        time_control="600+0",
        tc_bucket="rapid",
        clock_remaining=None,
        side_to_move="white",
        result=result,
    )


def _bands():
    # Two rating bands so the plot has more than one curve.
    rows = [
        _row(cp=200, we=1000, be=1000, result=1.0),
        _row(cp=-200, we=1000, be=1000, result=0.0),
        _row(cp=50, we=2500, be=2500, result=0.5),
        _row(cp=-50, we=2500, be=2500, result=0.5),
    ]
    return band_reliability(rows, baseline_cp)


def test_save_reliability_plot_writes_a_png(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")  # noqa: F841 - gate on the extra
    out = tmp_path / "reliability.png"
    path = save_reliability_plot(_bands(), str(out), title="test")
    assert path == str(out)
    assert out.exists()
    # PNG magic number — confirm we actually wrote an image, not an empty file.
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_empty_bands_raises():
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="no bands"):
        save_reliability_plot([], "ignored.png")


def test_missing_matplotlib_raises_clear_error(tmp_path, monkeypatch):
    # Simulate matplotlib being absent: make its import fail, regardless of install.
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(MatplotlibNotInstalled, match="matplotlib"):
        save_reliability_plot(_bands(), str(tmp_path / "x.png"))
