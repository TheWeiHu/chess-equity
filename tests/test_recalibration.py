"""Tests for the post-hoc Platt recalibrator (task 0166).

Pure-function tests over tiny constructed PositionRows — *fixtures for unit tests*, NOT
evidence (the real effect on maia2's ECE is measured on the cached Lichess dump via
`validate --recalibrate-maia2`; see CLAUDE.md's no-synthetic-data rule). They check the
Platt algebra (identity start, monotonicity, length/empty guards), that recalibration
lowers ECE on a deliberately *overconfident* base predictor, and that the CLI knob is wired
end-to-end (torch-free, via a fake Maia-2) and refuses without a held-out split.
"""

from __future__ import annotations

from math import exp, log

import pytest

from chess_equity.data.schema import PositionRow
from chess_equity.validate.metrics import expected_calibration_error
from chess_equity.validate.recalibration import (
    PlattRecalibrator,
    fit_platt,
    make_recalibrated_predictor,
)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + exp(-z))


def _logit(p: float) -> float:
    return log(p / (1.0 - p))


def _row(*, cp=0.0, we=1500, be=1500, result=0.5, gid="g") -> PositionRow:
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
        game_id=gid,
    )


# --- Platt algebra ---------------------------------------------------------------


def test_identity_recalibrator_is_a_noop():
    r = PlattRecalibrator(1.0, 0.0)
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert abs(r(p) - p) < 1e-9


def test_recalibrator_is_monotonic_for_positive_slope():
    r = PlattRecalibrator(0.5, 0.3)
    ps = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    out = [r(p) for p in ps]
    assert out == sorted(out)  # strictly increasing in p


def test_fit_recovers_a_known_logit_transform():
    # Generate labels from q = sigmoid(a*logit(p) + b) at high count so the fit is sharp.
    a_true, b_true = 0.5, 0.4
    base_preds, labels = [], []
    for i in range(1, 100):
        p = i / 100.0
        q = _sigmoid(a_true * _logit(p) + b_true)
        n = 200
        wins = round(n * q)
        base_preds += [p] * n
        labels += [1.0] * wins + [0.0] * (n - wins)
    scaler = fit_platt(base_preds, labels)
    assert abs(scaler.a - a_true) < 0.05
    assert abs(scaler.b - b_true) < 0.05


def test_fit_on_already_calibrated_data_stays_near_identity():
    base_preds, labels = [], []
    for i in range(1, 100):
        p = i / 100.0
        n = 200
        wins = round(n * p)  # base preds already match outcomes
        base_preds += [p] * n
        labels += [1.0] * wins + [0.0] * (n - wins)
    scaler = fit_platt(base_preds, labels)
    assert abs(scaler.a - 1.0) < 0.05 and abs(scaler.b) < 0.05


def test_fit_rejects_length_mismatch_and_empty():
    with pytest.raises(ValueError):
        fit_platt([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        fit_platt([], [])


def test_fit_survives_a_degenerate_constant_calibration_set():
    # Every base pred identical (zero logit variance) — the ridge keeps the Hessian solvable.
    scaler = fit_platt([0.5] * 50, [1.0] * 25 + [0.0] * 25)
    assert scaler(0.5) == pytest.approx(0.5, abs=0.05)


# --- ECE improvement on a deliberately overconfident base (fixtures, not evidence) ---


def _overconfident_dataset(gid_prefix):
    """Rows whose true White win-prob is sigmoid(cp/300) but a base predictor reads them as
    sigmoid(2*cp/300) — overconfident by a factor of 2 in logit space. Labels are placed at
    the TRUE frequency, so a perfect recalibrator (slope ~0.5) restores calibration.
    Returns (rows, base_predictor)."""
    rows = []
    for cp in range(-600, 601, 50):
        true_p = _sigmoid(cp / 300.0)
        n = 60
        wins = round(n * true_p)
        for j in range(n):
            res = 1.0 if j < wins else 0.0
            rows.append(_row(cp=float(cp), result=res, gid=f"{gid_prefix}-{cp}-{j}"))

    def base_predictor(row):
        return _sigmoid(2.0 * row.cp_eval / 300.0)

    return rows, base_predictor


def test_recalibration_lowers_ece_on_a_heldout_split():
    calib_rows, base = _overconfident_dataset("calib")
    eval_rows, _ = _overconfident_dataset("eval")  # disjoint game ids

    recal = make_recalibrated_predictor(calib_rows, base)

    labels = [r.result for r in eval_rows]
    base_ece = expected_calibration_error([base(r) for r in eval_rows], labels)
    recal_ece = expected_calibration_error([recal(r) for r in eval_rows], labels)

    assert recal_ece < base_ece
    # The overconfidence is large, so the repair should be substantial, not marginal.
    assert recal_ece < 0.5 * base_ece


def test_make_recalibrated_predictor_rejects_empty_calibration():
    _, base = _overconfident_dataset("x")
    with pytest.raises(ValueError):
        make_recalibrated_predictor([], base)


# --- CLI wiring (torch-free via a fake Maia-2) ------------------------------------


class _FakeMaia2:
    """Torch-free stand-in: overconfident in White's rating, so the recalibrator has work."""

    def evaluate(self, fen, white_elo, black_elo):
        from chess_equity.types import WDL, Equity

        p = min(0.99, max(0.01, white_elo / 4000.0))
        return Equity(wdl=WDL(p, 0.0, 1 - p), equity_white=100.0 * p, source="fake-maia2")


@pytest.fixture
def fake_maia2(monkeypatch):
    from chess_equity.validate import harness

    monkeypatch.setitem(harness.BOARD_MODELS, "maia2", lambda: _FakeMaia2())


def _fen_sample():
    from pathlib import Path

    return str(Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv")


def test_cli_recalibrate_maia2_runs_end_to_end(tmp_path, fake_maia2):
    from chess_equity.cli import main

    out = tmp_path / "report.md"
    rc = main(
        [
            "validate",
            "--data",
            _fen_sample(),
            "--models",
            "baseline,maia2",
            "--holdout",
            "0.5",
            "--recalibrate-maia2",
            "--bootstrap",
            "0",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    text = out.read_text()
    assert "Platt-recalibrated" in text
    assert "maia2" in text


def test_cli_recalibrate_maia2_requires_holdout(capsys, fake_maia2):
    from chess_equity.cli import main

    rc = main(
        ["validate", "--data", _fen_sample(), "--models", "baseline,maia2",
         "--recalibrate-maia2", "--bootstrap", "0"]
    )
    assert rc == 1
    assert "needs --holdout" in capsys.readouterr().err


def test_cli_recalibrate_maia2_requires_maia2_model(capsys):
    from chess_equity.cli import main

    rc = main(
        ["validate", "--data", _fen_sample(), "--models", "baseline,wdl-a",
         "--holdout", "0.5", "--recalibrate-maia2", "--bootstrap", "0"]
    )
    assert rc == 1
    assert "needs maia2" in capsys.readouterr().err
