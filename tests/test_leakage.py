"""Leakage guard (task 0112): the eval set must not be a model's own training month.

`wdl-a` is fit on Lichess `2016-05`; held-out evidence must use a different month.
These tests pin the pure detector, the path inference, the committed artifact's recorded
provenance, and the end-to-end CLI behaviour (warn by default, refuse with --strict).
"""

from pathlib import Path

import pytest

from chess_equity.cli import main
from chess_equity.validate.leakage import (
    Leak,
    detect_leakage,
    format_leakage_warning,
    infer_month_from_path,
    model_fit_months,
)
from chess_equity.wdl_regression import fit, load_wdl_a_model

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset.csv"


# --- pure detector --------------------------------------------------------------


def test_detect_leakage_same_month_warns():
    leaks = detect_leakage("2016-05", {"wdl-a": "2016-05"})
    assert leaks == [Leak(model="wdl-a", month="2016-05")]


def test_detect_leakage_different_month_clean():
    assert detect_leakage("2013-01", {"wdl-a": "2016-05"}) == []


def test_detect_leakage_unknown_eval_month_clean():
    # No declared/inferred month → nothing to compare, so the guard stays silent.
    assert detect_leakage(None, {"wdl-a": "2016-05"}) == []


def test_detect_leakage_ignores_models_without_provenance():
    # Rating-blind predictors carry no fit month and cannot leak.
    assert detect_leakage("2016-05", {}) == []


# --- path inference -------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        ("data/dataset_2016-05.csv", "2016-05"),
        ("/tmp/lichess_db_standard_rated_2013-01.pgn.zst", "2013-01"),
        ("data/sample/dataset.csv", None),  # no month token
        ("data/dataset_12016-050.csv", None),  # digits glued on → not a clean YYYY-MM
        ("data/dataset_2016-13.csv", None),  # month 13 is not 01–12
    ],
)
def test_infer_month_from_path(path, expected):
    assert infer_month_from_path(path) == expected


# --- recorded provenance --------------------------------------------------------


def test_committed_wdl_a_records_its_fit_month():
    # The committed artifact must carry the month the leakage guard reads.
    assert (load_wdl_a_model().meta or {}).get("fit_month") == "2016-05"


def test_model_fit_months_reads_committed_artifact():
    assert model_fit_months(["baseline", "wdl-a"]) == {"wdl-a": "2016-05"}
    assert model_fit_months(["baseline"]) == {}


def test_model_fit_months_honors_custom_artifact(tmp_path):
    # Task 0164: a refit wdl-a artifact (different month) must drive the guard, so a
    # cross-dump held-out run reads as held-out, not in-distribution.
    refit = fit(load_wdl_a_rows_stub(), iters=1, source_month="2013-01")
    path = tmp_path / "wdl_a_2013-01.json"
    refit.save(str(path))
    assert model_fit_months(["wdl-a"], wdl_a_path=str(path)) == {"wdl-a": "2013-01"}
    # Eval on 2016-05 is now clean (the refit month differs), where the committed
    # artifact would have leaked.
    assert detect_leakage("2016-05", model_fit_months(["wdl-a"], wdl_a_path=str(path))) == []


def test_fit_stamps_source_month():
    rows = load_wdl_a_rows_stub()
    model = fit(rows, iters=1, source_month="2020-02")
    assert (model.meta or {}).get("fit_month") == "2020-02"
    # Omitting it leaves no stale month behind.
    assert "fit_month" not in (fit(rows, iters=1).meta or {})


def load_wdl_a_rows_stub():
    from chess_equity.data.build import load_rows

    return load_rows(str(SAMPLE))


# --- rendering ------------------------------------------------------------------


def test_format_leakage_warning_is_loud_or_empty():
    assert format_leakage_warning([], "2013-01") == ""
    block = format_leakage_warning([Leak("wdl-a", "2016-05")], "2016-05")
    assert "LEAKAGE" in block and "`wdl-a`" in block and "2016-05" in block


# --- CLI end-to-end -------------------------------------------------------------


def test_cli_warns_when_eval_month_is_training_month(capsys):
    rc = main(
        ["validate", "--data", str(SAMPLE), "--models", "baseline,wdl-a",
         "--eval-month", "2016-05", "--bootstrap", "0"]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "LEAKAGE" in captured.err  # loud on stderr
    assert "LEAKAGE" in captured.out  # and led into the report body


def test_cli_strict_refuses_on_leakage(capsys):
    rc = main(
        ["validate", "--data", str(SAMPLE), "--models", "baseline,wdl-a",
         "--eval-month", "2016-05", "--strict", "--bootstrap", "0"]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "refusing" in captured.err


def test_cli_clean_on_held_out_month(capsys):
    rc = main(
        ["validate", "--data", str(SAMPLE), "--models", "baseline,wdl-a",
         "--eval-month", "2013-01", "--bootstrap", "0"]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "LEAKAGE" not in captured.out and "LEAKAGE" not in captured.err


def test_cli_infers_month_from_dataset_path(tmp_path, capsys):
    # A dataset named after its source month gets the guard for free.
    leaky = tmp_path / "dataset_2016-05.csv"
    leaky.write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    rc = main(
        ["validate", "--data", str(leaky), "--models", "baseline,wdl-a",
         "--bootstrap", "0"]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "LEAKAGE" in captured.err
