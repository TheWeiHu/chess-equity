"""Dataset source-month provenance + the 0112 leakage guard reading it (task 0116).

`build_dataset(month=...)` stamps the Lichess source month into a sidecar so `validate`
detects eval-vs-training-month overlap WITHOUT the caller passing `--eval-month`, and
even if the dataset file is renamed (the month no longer has to live in the path).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_equity.data.build import build_dataset, dataset_source_month
from chess_equity.validate.leakage import detect_leakage, model_fit_months

SAMPLE_PGN = str(Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_games.pgn")


def test_build_stamps_and_reloads_source_month(tmp_path):
    out = build_dataset(SAMPLE_PGN, str(tmp_path), sample=5, name="ds", month="2016-05")
    assert out.is_file()
    # The month round-trips through the sidecar, not the dataset filename.
    assert dataset_source_month(str(out)) == "2016-05"
    assert "2016-05" not in out.name


def test_no_month_means_no_sidecar(tmp_path):
    out = build_dataset(SAMPLE_PGN, str(tmp_path), sample=5, name="ds")
    assert dataset_source_month(str(out)) is None


def test_malformed_month_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        build_dataset(SAMPLE_PGN, str(tmp_path), sample=5, name="ds", month="2016/5")


def test_guard_auto_detects_overlap_without_eval_month(tmp_path):
    # A dataset stamped with wdl-a's own training month (2016-05) is memorization.
    out = build_dataset(SAMPLE_PGN, str(tmp_path), sample=5, name="ds", month="2016-05")
    eval_month = dataset_source_month(str(out))  # what validate uses when --eval-month is absent
    leaks = detect_leakage(eval_month, model_fit_months(["baseline", "wdl-a"]))
    assert [lk.model for lk in leaks] == ["wdl-a"]
    # A genuinely held-out month does not trip the guard.
    held = build_dataset(SAMPLE_PGN, str(tmp_path), sample=5, name="held", month="2013-01")
    assert detect_leakage(dataset_source_month(str(held)), model_fit_months(["wdl-a"])) == []


def test_stamp_survives_a_rename(tmp_path):
    out = build_dataset(SAMPLE_PGN, str(tmp_path), sample=5, name="ds", month="2016-05")
    sidecar = out.with_name(out.name + ".meta.json")
    assert sidecar.is_file()
    # Rename the dataset AND its sidecar (the path no longer encodes the month).
    renamed = out.with_name("anonymous_dump.csv")
    out.rename(renamed)
    sidecar.rename(renamed.with_name(renamed.name + ".meta.json"))
    assert dataset_source_month(str(renamed)) == "2016-05"


def test_partitioned_dataset_stamps_meta(tmp_path):
    out = build_dataset(
        SAMPLE_PGN, str(tmp_path), sample=5, name="parts", month="2016-05", partition=True
    )
    assert out.is_dir()
    assert dataset_source_month(str(out)) == "2016-05"
