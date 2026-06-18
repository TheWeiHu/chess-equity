"""The dataset source-month sidecar (task 0127): the "data stamp" that tells the
validation leakage guard which Lichess month a built dataset came from.

Pure code on the committed sample PGN + tiny synthetic files — no Lichess download,
no torch. Covers the two round trips the task asks for: build->read and stamp->read,
plus the validation surface that reads the stamp.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_equity.data.build import build_dataset
from chess_equity.data.source_month import (
    normalize_month,
    read_source_month,
    sidecar_path,
    write_source_month,
)

SAMPLE_PGN = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_games.pgn"


# --- month validation ----------------------------------------------------------

@pytest.mark.parametrize("good", ["2016-05", "2013-01", "2026-12", " 2020-09 "])
def test_normalize_month_accepts_valid(good):
    assert normalize_month(good) == good.strip()


@pytest.mark.parametrize("bad", ["", "2016", "2016-13", "2016-00", "16-05", "2016/05", "nope"])
def test_normalize_month_rejects_malformed(bad):
    with pytest.raises(ValueError):
        normalize_month(bad)


# --- build -> read --------------------------------------------------------------

def test_build_stamps_and_reads_back(tmp_path):
    out = build_dataset(
        str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="ds", source_month="2016-05"
    )
    # The sidecar sits next to the dataset and round-trips the month.
    assert sidecar_path(out).is_file()
    assert read_source_month(out) == "2016-05"


def test_build_without_month_leaves_no_sidecar(tmp_path):
    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="ds")
    assert not sidecar_path(out).exists()
    assert read_source_month(out) is None


def test_build_partitioned_dir_stamps_sibling(tmp_path):
    out = build_dataset(
        str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="ds",
        partition=True, source_month="2013-01",
    )
    assert out.is_dir()
    # The sidecar is a sibling file, not inside the partitioned tree.
    assert sidecar_path(out) == Path(str(out) + ".source.json")
    assert read_source_month(out) == "2013-01"


# --- stamp -> read (backfill) ---------------------------------------------------

def test_stamp_backfills_existing_dataset(tmp_path):
    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="legacy")
    assert read_source_month(out) is None  # built before stamping
    side = write_source_month(out, "2016-05")
    assert side.is_file()
    assert read_source_month(out) == "2016-05"


def test_stamp_rejects_bad_month(tmp_path):
    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="legacy")
    with pytest.raises(ValueError):
        write_source_month(out, "2016-13")
    assert read_source_month(out) is None


def test_read_ignores_corrupt_sidecar(tmp_path):
    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="ds")
    sidecar_path(out).write_text("{ not json", encoding="utf-8")
    # A corrupt advisory stamp reads as "unknown", never crashes a validation run.
    assert read_source_month(out) is None


# --- CLI surfaces (data stamp, validate reads it) -------------------------------

def test_cli_data_stamp_then_validate_shows_month(tmp_path, capsys):
    from chess_equity.cli import main

    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="ds")
    assert main(["data", "stamp", str(out), "2016-05"]) == 0
    assert read_source_month(out) == "2016-05"

    capsys.readouterr()  # drop the stamp output
    # validate reads the sidecar and names the data month in its report header.
    rc = main(["validate", "--data", str(out), "--models", "baseline", "--bootstrap", "0"])
    assert rc == 0
    assert "data month: 2016-05" in capsys.readouterr().out


def test_cli_data_stamp_missing_path_errors(tmp_path):
    from chess_equity.cli import main

    assert main(["data", "stamp", str(tmp_path / "nope.csv"), "2016-05"]) == 1
