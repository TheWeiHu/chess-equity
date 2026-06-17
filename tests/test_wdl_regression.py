"""Tests for Approach A — the rating-conditioned WDL regression (task 0004).

The committed artifact is a placeholder fit on the 15-row sample, so the *learning*
behaviour is proven here on synthetic data where the rating-conditioning signal is
constructible: a model fit on it must come out monotone in cp, genuinely
rating-conditioned (not rating-blind), and well-formed (valid WDL, round-trips, drops
into both the EquityModel and the validation registry).
"""

from __future__ import annotations

import chess
import pytest

from chess_equity.data.schema import PositionRow
from chess_equity.validate.harness import PREDICTORS, wdl_a
from chess_equity.wdl_regression import (
    FEATURE_NAMES,
    N_CLASSES,
    WdlRegression,
    WdlRegressionModel,
    build_wdl_a_equity,
    features,
    fit,
    load_wdl_a_model,
)


def _row(*, cp=0.0, we=1500, be=1500, ply=10, tc="blitz", result=0.5) -> PositionRow:
    return PositionRow(
        cp_eval=cp,
        white_elo=we,
        black_elo=be,
        ply=ply,
        phase="middlegame",
        time_control="180+2",
        tc_bucket=tc,
        clock_remaining=None,
        side_to_move="white",
        result=result,
    )


# --- features ------------------------------------------------------------------

def test_features_shape_and_bias():
    x = features(0.0, 1500, 1500, 10, "blitz")
    assert len(x) == len(FEATURE_NAMES)
    assert x[0] == 1.0  # bias
    # At the reference (1500/1500, blitz, cp 0) every non-bias feature except ply is 0.
    assert x[1] == pytest.approx(0.0)  # cp_sat
    assert x[2] == pytest.approx(0.0)  # avg_skill
    assert x[3] == pytest.approx(0.0)  # rating_delta


def test_features_saturate_on_mate_sentinel():
    # The 9999cp mate sentinel must not blow up — tanh saturates near 1.
    x = features(9999.0, 1500, 1500, 10, "blitz")
    assert 0.99 < x[1] <= 1.0
    # The reference time control (blitz) is the all-zero one-hot.
    assert x[6:] == [0.0, 0.0, 0.0, 0.0]
    assert features(0.0, 1500, 1500, 10, "rapid")[7] == 1.0  # tc_rapid


# --- learning behaviour --------------------------------------------------------

def _cp_separable_rows():
    """Outcome follows cp: White wins when ahead, loses when behind. Many copies → signal."""
    rows = []
    for _ in range(8):
        for cp in (-400, -200, 0, 200, 400):
            res = 1.0 if cp > 0 else (0.0 if cp < 0 else 0.5)
            rows.append(_row(cp=cp, result=res))
    return rows


def test_fit_is_monotone_in_cp():
    model = fit(_cp_separable_rows(), iters=2000)
    eqs = [model.predict_white_equity(cp, 1500, 1500, 10, "blitz") for cp in (-400, -200, 0, 200, 400)]
    assert eqs == sorted(eqs)  # non-decreasing
    assert eqs[0] < 0.5 < eqs[-1]


def test_fit_predicts_valid_wdl():
    model = fit(_cp_separable_rows(), iters=1000)
    wdl = model.predict_white_wdl(150.0, 1800, 1600, 12, "rapid")
    assert wdl.p_win >= 0 and wdl.p_draw >= 0 and wdl.p_loss >= 0
    assert wdl.p_win + wdl.p_draw + wdl.p_loss == pytest.approx(1.0)
    assert wdl.equity == pytest.approx(wdl.p_win + 0.5 * wdl.p_draw)


def test_fit_is_rating_conditioned():
    """Same cp, different ratings → different prediction (the whole point vs baseline).

    Construct a set where a +200cp edge converts for weak players but the strong game
    is drawn — the model must then read +200cp as less decisive at 2400 than at 1200.
    """
    rows = []
    for _ in range(10):
        rows.append(_row(cp=200, we=1200, be=1200, result=1.0))  # weak: converts
        rows.append(_row(cp=200, we=2400, be=2400, result=0.5))  # strong: held
    model = fit(rows, iters=2500)
    weak = model.predict_white_equity(200, 1200, 1200, 10, "blitz")
    strong = model.predict_white_equity(200, 2400, 2400, 10, "blitz")
    assert weak > strong  # rating-conditioned, not rating-blind
    assert abs(weak - strong) > 0.05


def test_fit_empty_raises():
    with pytest.raises(ValueError):
        fit([])


# --- serialization -------------------------------------------------------------

def test_serialization_round_trips(tmp_path):
    model = fit(_cp_separable_rows(), iters=500)
    path = tmp_path / "m.json"
    model.save(str(path))
    loaded = WdlRegression.load(str(path))
    for cp in (-300, 0, 300):
        a = model.predict_white_equity(cp, 1500, 1500, 10, "blitz")
        b = loaded.predict_white_equity(cp, 1500, 1500, 10, "blitz")
        assert a == pytest.approx(b)


def test_from_dict_version_mismatch_raises():
    model = fit(_cp_separable_rows(), iters=10)
    payload = model.to_dict()
    payload["feature_version"] = 999
    with pytest.raises(ValueError):
        WdlRegression.from_dict(payload)


def test_from_dict_bad_shape_raises():
    with pytest.raises(ValueError):
        WdlRegression.from_dict({"feature_version": 1, "weights": [[0.0, 0.0]]})


# --- EquityModel adapter -------------------------------------------------------

def test_adapter_white_pov_stable():
    """White up a rook reads > 50% regardless of whose turn it is (bar is White-POV)."""
    model = fit(_cp_separable_rows(), iters=1500)
    adapter = WdlRegressionModel(model)
    white_turn = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
    black_turn = "4k3/8/8/8/8/8/8/R3K3 b - - 0 1"
    assert adapter.evaluate(white_turn, 1500, 1500).equity_white > 50.0
    assert adapter.evaluate(black_turn, 1500, 1500).equity_white > 50.0


def test_adapter_reports_source_and_valid_wdl():
    adapter = WdlRegressionModel(fit(_cp_separable_rows(), iters=500))
    eq = adapter.evaluate(chess.STARTING_FEN, 1500, 1500)
    assert eq.source == "wdl-a"
    total = eq.wdl.p_win + eq.wdl.p_draw + eq.wdl.p_loss
    assert total == pytest.approx(1.0)


# --- registry + committed artifact ---------------------------------------------

def test_wdl_a_predictor_registered():
    assert "wdl-a" in PREDICTORS


def test_committed_artifact_loads_and_predicts():
    model = load_wdl_a_model()
    assert len(model.weights) == N_CLASSES
    eq = model.predict_white_equity(300, 1500, 1500, 10, "blitz")
    assert 0.0 <= eq <= 1.0


def test_committed_artifact_drives_eval_model():
    adapter = build_wdl_a_equity()
    eq = adapter.evaluate(chess.STARTING_FEN, 1500, 1500)
    assert eq.source == "wdl-a"
    assert 0.0 <= eq.equity_white <= 100.0


def test_predictor_matches_model_on_row():
    row = _row(cp=120, we=1700, be=1500, ply=18, tc="rapid")
    expected = load_wdl_a_model().predict_white_equity(
        row.cp_eval, row.white_elo, row.black_elo, row.ply, row.tc_bucket
    )
    assert wdl_a(row) == pytest.approx(expected)
