"""Tests for the validation gate (task 0009).

The metrics are checked against hand-computed values (including the draw / soft-label
cases that are the whole point), then the harness is exercised end-to-end on
synthetic rows and the committed 0002 sample, plus the rating-band slicer and the
baseline predictor's "rating-blind" behaviour.
"""

from __future__ import annotations

from math import isclose, log

import pytest

from chess_equity.data.schema import PositionRow
from chess_equity.validate.harness import (
    PREDICTORS,
    baseline_cp,
    evaluate,
    format_report,
    rating_band,
)
from chess_equity.validate.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_table,
)


def _row(*, cp=0.0, we=1500, be=1500, phase="middlegame", result=0.5) -> PositionRow:
    return PositionRow(
        cp_eval=cp,
        white_elo=we,
        black_elo=be,
        ply=10,
        phase=phase,
        time_control="600+0",
        tc_bucket="rapid",
        clock_remaining=None,
        side_to_move="white",
        result=result,
    )


# --- metrics -------------------------------------------------------------------

def test_brier_basic_and_draw():
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0
    # A 0.5 prediction on a draw is perfect under Brier.
    assert brier_score([0.5], [0.5]) == 0.0
    assert isclose(brier_score([0.8], [1.0]), 0.04)


def test_log_loss_perfect_and_soft_draw():
    # Predicting 0.5 on a draw: -(0.5*ln0.5 + 0.5*ln0.5) = ln2.
    assert isclose(log_loss([0.5], [0.5]), log(2.0))
    # Confident and correct -> near zero.
    assert log_loss([0.999999], [1.0]) < 1e-5


def test_log_loss_punishes_confident_wrong_finitely():
    val = log_loss([0.0], [1.0])  # clipped, not infinite
    assert val > 10 and val < 100


def test_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        brier_score([0.5], [0.5, 0.5])
    with pytest.raises(ValueError):
        log_loss([], [])


def test_reliability_and_ece():
    preds = [0.1, 0.1, 0.9, 0.9]
    labels = [0.0, 0.0, 1.0, 1.0]
    table = reliability_table(preds, labels, bins=10)
    assert len(table) == 2  # two non-empty bins
    # Each bin is off by 0.1 (pred 0.1 vs actual 0.0, pred 0.9 vs actual 1.0) -> ECE 0.1.
    assert isclose(expected_calibration_error(preds, labels), 0.1, abs_tol=1e-9)
    # A perfectly calibrated bin (pred 0.2, actual mean 0.2) -> ECE 0.
    cal = expected_calibration_error([0.2, 0.2, 0.2, 0.2, 0.2], [1.0, 0.0, 0.0, 0.0, 0.0])
    assert isclose(cal, 0.0, abs_tol=1e-9)
    # A systematically over-confident predictor has large ECE.
    assert expected_calibration_error([0.9, 0.9], [0.0, 0.0]) > 0.5


# --- predictors & slicing ------------------------------------------------------

def test_baseline_is_rating_blind():
    # Same cp, wildly different ratings -> identical prediction (the baseline's flaw).
    a = baseline_cp(_row(cp=100, we=800, be=800))
    b = baseline_cp(_row(cp=100, we=2600, be=2600))
    assert a == b
    # Even cp -> 0.5; White-favoured cp -> > 0.5.
    assert isclose(baseline_cp(_row(cp=0)), 0.5)
    assert baseline_cp(_row(cp=300)) > 0.5
    assert baseline_cp(_row(cp=-300)) < 0.5


def test_rating_band():
    assert rating_band(_row(we=1000, be=1000)) == "<1200"
    assert rating_band(_row(we=1500, be=1500)) == "1200-1599"
    assert rating_band(_row(we=2500, be=2500)) == "2400+"


# --- harness end to end --------------------------------------------------------

def test_evaluate_overall_and_slices():
    rows = [
        _row(cp=500, we=1000, be=1000, phase="opening", result=1.0),
        _row(cp=-500, we=2500, be=2500, phase="endgame", result=0.0),
    ]
    reports = evaluate(rows, {"baseline": baseline_cp})
    assert len(reports) == 1
    rep = reports[0]
    assert rep.overall.n == 2
    # rating slice has the two distinct bands; phase slice has the two phases.
    assert set(rep.slices["rating"]) == {"<1200", "2400+"}
    assert set(rep.slices["phase"]) == {"opening", "endgame"}
    assert rep.slices["rating"]["<1200"].n == 1


def test_format_report_is_markdown():
    rows = [_row(cp=100, result=1.0), _row(cp=-100, result=0.0)]
    md = format_report(evaluate(rows, {"baseline": baseline_cp}))
    assert md.startswith("# ")
    assert "log-loss" in md and "Brier" in md and "ECE" in md
    assert "## By rating" in md and "## By phase" in md


def test_runs_on_committed_sample():
    from pathlib import Path

    from chess_equity.data.build import load_rows

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset.csv"
    rows = load_rows(str(sample))
    reports = evaluate(rows, {"baseline": baseline_cp})
    assert reports[0].overall.n == len(rows) > 0


def test_baseline_registered():
    assert "baseline" in PREDICTORS


# --- board-model predictor (task 0029) -----------------------------------------

class _FakeBoardModel:
    """A board-needing EquityModel: equity rises with white_elo, so it must read the
    FEN+ratings (a (cp, ratings) predictor signature can't express it)."""

    def evaluate(self, fen, white_elo, black_elo):
        from chess_equity.types import WDL, Equity

        p = min(0.99, white_elo / 4000.0)
        return Equity(wdl=WDL(p, 0.0, 1 - p), equity_white=100.0 * p, source="fake")


def _row_fen(**kw):
    base = dict(cp=0.0, we=1500, be=1500, result=0.5)
    base.update(kw)
    row = _row(cp=base["cp"], we=base["we"], be=base["be"], result=base["result"])
    return PositionRow(**{**row.as_dict(), "fen": "8/8/8/8/8/8/8/K6k w - - 0 1"})


def test_model_predictor_reads_fen_and_scores():
    from chess_equity.validate.harness import model_predictor

    predict = model_predictor(_FakeBoardModel())
    assert isclose(predict(_row_fen(we=2000)), 0.5)
    # Different ratings -> different prediction (the thing the baseline can't do).
    assert predict(_row_fen(we=3000)) > predict(_row_fen(we=1000))


def test_model_predictor_raises_without_fen():
    from chess_equity.validate.harness import model_predictor

    predict = model_predictor(_FakeBoardModel())
    with pytest.raises(ValueError):
        predict(_row())  # no fen on the row


def test_model_predictor_runs_through_evaluate():
    from chess_equity.validate.harness import model_predictor

    rows = [_row_fen(we=2600, result=1.0), _row_fen(we=900, result=0.0)]
    reports = evaluate(rows, {"fake": model_predictor(_FakeBoardModel())})
    assert reports[0].overall.n == 2


def test_board_predictor_scores_on_committed_fen_sample():
    """The end-to-end proof of task 0023: a board-needing model is scored straight
    off the committed FEN fixture, with no PGN rebuild.

    ``data/sample/dataset.csv`` carries no FEN (kept small for the cp-only models), so
    ``data/sample/dataset_fen.csv`` is the committed companion that lets the 0009
    harness exercise the board path on real, checked-in rows. Maia-2 (0005/0031) plugs
    into exactly this loop; ``_FakeBoardModel`` stands in so the test needs no weights.
    """
    from pathlib import Path

    from chess_equity.data.build import load_rows
    from chess_equity.validate.harness import model_predictor

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"
    rows = load_rows(str(sample))
    assert rows and all(r.fen is not None for r in rows), "committed FEN sample must carry FENs"

    reports = evaluate(rows, {"fake": model_predictor(_FakeBoardModel())})
    assert reports[0].overall.n == len(rows) > 0


# --- registering maia2 as a 0009 predictor (task 0031) -------------------------

def _fake_maia2():
    """A Maia2Equity wired to a fake backend (no torch) whose win_prob rises with the
    side-to-move's rating, so it conditions on ratings like the real value head."""
    from chess_equity.maia2 import Maia2Equity

    def backend(fen, elo_self, elo_oppo):
        return {}, min(0.99, elo_self / 4000.0)

    return Maia2Equity(backend=backend)


def test_build_predictors_mixes_row_and_board_models(monkeypatch):
    import chess_equity.validate.harness as h

    monkeypatch.setitem(h.BOARD_MODELS, "maia2", _fake_maia2)
    preds = h.build_predictors(["baseline", "maia2"])
    assert set(preds) == {"baseline", "maia2"}
    # The maia2 predictor reads the board (fen) — a row without fen must raise.
    assert isclose(preds["baseline"](_row_fen(cp=0)), 0.5)
    with pytest.raises(ValueError):
        preds["maia2"](_row())


def test_build_predictors_rejects_unknown():
    from chess_equity.validate.harness import build_predictors

    with pytest.raises(KeyError):
        build_predictors(["baseline", "nope"])


def test_maia2_registered_as_board_model():
    from chess_equity.validate.harness import BOARD_MODELS

    assert "maia2" in BOARD_MODELS


def test_validate_cli_scores_maia2_against_baseline(tmp_path, monkeypatch, capsys):
    """End-to-end: `validate --models baseline,maia2` over a --with-fen dataset."""
    from pathlib import Path

    import chess_equity.validate.harness as h
    from chess_equity.cli import main
    from chess_equity.data.build import build_dataset

    monkeypatch.setitem(h.BOARD_MODELS, "maia2", _fake_maia2)
    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_games.pgn"
    data = build_dataset(str(sample), str(tmp_path), name="fen", include_fen=True)

    rc = main(["validate", "--data", str(data), "--models", "baseline,maia2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "baseline" in out and "maia2" in out


def test_validate_cli_maia2_needs_fen(tmp_path, monkeypatch, capsys):
    """A FEN-less dataset makes the maia2 predictor fail with a clean error, not a trace."""
    from pathlib import Path

    import chess_equity.validate.harness as h
    from chess_equity.cli import main
    from chess_equity.data.build import build_dataset

    monkeypatch.setitem(h.BOARD_MODELS, "maia2", _fake_maia2)
    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_games.pgn"
    data = build_dataset(str(sample), str(tmp_path), name="nofen")  # no fen column

    rc = main(["validate", "--data", str(data), "--models", "maia2"])
    assert rc == 1
    assert "include_fen" in capsys.readouterr().err


# --- registering maia-search as a 0009 predictor (task 0037) -------------------

def _fake_maia_search():
    """A MaiaSearchModel with no torch: uniform move priors + a fake rating-conditioned
    leaf, so the expectimax conditions on ratings like the real Maia-backed search."""
    from chess_equity.grading import UniformPolicy
    from chess_equity.maia2 import Maia2Equity
    from chess_equity.search import MaiaSearchModel

    def backend(fen, elo_self, elo_oppo):
        return {}, min(0.99, elo_self / 4000.0)

    return MaiaSearchModel(UniformPolicy(), Maia2Equity(backend=backend), depth=1, k=2)


def test_maia_search_registered_as_board_model():
    from chess_equity.validate.harness import BOARD_MODELS

    assert "maia-search" in BOARD_MODELS


def test_build_predictors_includes_maia_search(monkeypatch):
    import chess_equity.validate.harness as h

    monkeypatch.setitem(h.BOARD_MODELS, "maia-search", _fake_maia_search)
    preds = h.build_predictors(["baseline", "maia-search"])
    assert set(preds) == {"baseline", "maia-search"}
    # It reads the board (fen) — a row without fen must raise, like any board model.
    with pytest.raises(ValueError):
        preds["maia-search"](_row())


def test_validate_cli_scores_maia_search_against_baseline(tmp_path, monkeypatch, capsys):
    """End-to-end: `validate --models baseline,maia-search` over a --with-fen dataset."""
    from pathlib import Path

    import chess_equity.validate.harness as h
    from chess_equity.cli import main
    from chess_equity.data.build import build_dataset

    monkeypatch.setitem(h.BOARD_MODELS, "maia-search", _fake_maia_search)
    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_games.pgn"
    data = build_dataset(str(sample), str(tmp_path), name="fen", include_fen=True)

    rc = main(["validate", "--data", str(data), "--models", "baseline,maia-search"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "baseline" in out and "maia-search" in out
