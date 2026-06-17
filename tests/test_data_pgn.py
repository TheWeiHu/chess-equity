"""Tests for the Lichess data pipeline (task 0002).

The pure schema helpers are checked directly; the parser is checked against
hand-written PGNs covering the cases that bite in real dumps — mate evals, a move
with no eval, clock tags, drawn vs decisive results, and games that must be dropped
(no ratings, unfinished ``*``). Finally a build -> load round trip over the committed
sample fixture.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pytest

from chess_equity.data.build import build_dataset, load_rows
from chess_equity.data.pgn import iter_rows, rows_from_game
from chess_equity.data.schema import (
    MATE_CP,
    game_phase,
    mate_to_cp,
    parse_clock,
    parse_eval,
    tc_bucket,
    tc_seconds,
)

import chess.pgn

SAMPLE_PGN = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_games.pgn"


# --- pure helpers ----------------------------------------------------------------

def test_parse_eval_pawns_to_centipawns():
    assert parse_eval("2.35") == 235.0
    assert parse_eval("-0.5") == -50.0
    assert parse_eval("0.0") == 0.0


def test_parse_eval_mate_is_clamped_and_signed():
    assert parse_eval("#3") == MATE_CP - 3
    assert parse_eval("#-1") == -(MATE_CP - 1)
    # Shorter mate outranks longer mate, sign preserved.
    assert parse_eval("#1") > parse_eval("#8") > 0
    assert parse_eval("#-1") < parse_eval("#-8") < 0


def test_parse_eval_garbage_is_none():
    assert parse_eval("") is None
    assert parse_eval("   ") is None
    assert parse_eval("notanumber") is None
    assert parse_eval("#x") is None


def test_mate_to_cp_sign():
    assert mate_to_cp(5) > 0
    assert mate_to_cp(-5) < 0


def test_parse_clock():
    assert parse_clock("0:03:00") == 180.0
    assert parse_clock("1:00:00") == 3600.0
    assert parse_clock("0:00:30") == 30.0
    assert parse_clock("bad") is None


def test_tc_seconds_and_buckets():
    assert tc_seconds("180+2") == 180 + 80
    assert tc_seconds("-") is None
    assert tc_bucket("60+0") == "bullet"
    assert tc_bucket("180+2") == "blitz"
    assert tc_bucket("600+0") == "rapid"
    assert tc_bucket("1800+0") == "classical"
    assert tc_bucket("-") == "correspondence"


def test_game_phase():
    assert game_phase(4, 30) == "opening"
    assert game_phase(40, 28) == "middlegame"
    assert game_phase(60, 5) == "endgame"


# --- parser --------------------------------------------------------------------

def _rows(pgn_text: str):
    return list(iter_rows(io.StringIO(pgn_text)))


_MOVETEXT = "1. e4 { [%eval 0.3] [%clk 0:05:00] } e5 { [%eval 0.2] } 2. Nf3 { [%eval 0.25] [%clk 0:04:55] } "


def _pgn(result: str = "1-0", *, white_elo: Optional[str] = "1500") -> str:
    """A 3-eval game whose header (and movetext terminator) carry ``result``."""
    headers = [
        '[White "a"]',
        '[Black "b"]',
        f'[Result "{result}"]',
        f'[WhiteElo "{white_elo}"]' if white_elo is not None else "",
        '[BlackElo "1500"]',
        '[TimeControl "300+3"]',
    ]
    head = "\n".join(h for h in headers if h)
    return f"{head}\n\n{_MOVETEXT}{result}\n"


def test_eval_is_white_pov_and_label_is_white_result():
    rows = _rows(_pgn("1-0"))
    # 3 evaluated half-moves -> 3 rows.
    assert len(rows) == 3
    assert all(r.result == 1.0 for r in rows)
    assert rows[0].cp_eval == 30.0
    # side to move alternates: after 1.e4 it's Black, after 1...e5 it's White, etc.
    assert [r.side_to_move for r in rows] == ["black", "white", "black"]


def test_clock_remaining_is_side_to_move_and_carries_forward():
    # _MOVETEXT: 1.e4 {clk 5:00} e5 {no clk} 2.Nf3 {clk 4:55}.
    rows = _rows(_pgn("1-0"))
    # Row 0 is after 1.e4 -> Black to move; only White's clock is known so far, so the
    # side-to-move's (Black's) clock is still None. White's clock IS captured though.
    assert rows[0].white_clock == 300.0
    assert rows[0].black_clock is None
    assert rows[0].clock_remaining is None  # = stm (black) clock
    # Row 1 is after 1...e5 (no [%clk]) -> White to move; White's 5:00 carries forward
    # across the clock-less ply, so the side-to-move's clock is 300.
    assert rows[1].white_clock == 300.0
    assert rows[1].clock_remaining == 300.0  # = stm (white) clock
    # Row 2 is after 2.Nf3 {clk 4:55} -> Black to move; White's clock updated to 295,
    # Black still has none, so the stm clock is None again.
    assert rows[2].white_clock == 295.0
    assert rows[2].clock_remaining is None
    # The derived stm_clock property agrees with clock_remaining.
    assert all(r.stm_clock == r.clock_remaining for r in rows)


def test_both_clocks_tracked_independently():
    # A game where both sides carry [%clk]: each player's clock is recorded separately,
    # and clock_remaining always reflects whoever is to move.
    pgn = (
        '[Result "1-0"]\n[WhiteElo "1500"]\n[BlackElo "1500"]\n[TimeControl "300+0"]\n\n'
        "1. e4 { [%eval 0.3] [%clk 0:05:00] } "
        "e5 { [%eval 0.2] [%clk 0:04:50] } "
        "2. Nf3 { [%eval 0.25] [%clk 0:04:55] } 1-0\n"
    )
    rows = _rows(pgn)
    # After 1...e5 -> White to move: white 5:00, black 4:50, stm = white.
    assert rows[1].white_clock == 300.0 and rows[1].black_clock == 290.0
    assert rows[1].clock_remaining == 300.0
    # After 2.Nf3 -> Black to move: white updated to 4:55, black still 4:50, stm = black.
    assert rows[2].white_clock == 295.0 and rows[2].black_clock == 290.0
    assert rows[2].clock_remaining == 290.0


def test_unfinished_game_is_dropped():
    assert _rows(_pgn("*")) == []  # Result "*"


def test_missing_ratings_drops_game():
    assert _rows(_pgn("1-0", white_elo=None)) == []


def test_draw_result_label():
    rows = _rows(_pgn("1/2-1/2"))
    assert rows and all(r.result == 0.5 for r in rows)


def test_rows_from_game_directly():
    game = chess.pgn.read_game(io.StringIO(_pgn("0-1")))
    rows = list(rows_from_game(game))
    assert rows and all(r.result == 0.0 for r in rows)


# --- sample fixture + build round trip -----------------------------------------

def test_sample_fixture_parses():
    rows = _rows(SAMPLE_PGN.read_text(encoding="utf-8"))
    assert len(rows) > 0
    # Game 1 has a mate-in-1 for White (#1) on Black's blundering 3...Nf6.
    assert any(r.cp_eval > MATE_CP - 10 for r in rows)
    # Game 3 has a mate-in-1 for Black (#-1).
    assert any(r.cp_eval < -(MATE_CP - 10) for r in rows)
    buckets = {r.tc_bucket for r in rows}
    assert {"blitz", "rapid", "bullet"} <= buckets


def test_build_and_load_round_trip(tmp_path):
    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="sample")
    assert out.exists()
    loaded = load_rows(str(out))
    direct = _rows(SAMPLE_PGN.read_text(encoding="utf-8"))
    assert len(loaded) == len(direct)
    assert loaded[0].cp_eval == direct[0].cp_eval
    assert loaded[0].white_elo == direct[0].white_elo


def test_sample_cap_limits_rows(tmp_path):
    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), sample=2, fmt="csv", name="capped")
    assert len(load_rows(str(out))) == 2


def test_unknown_format_rejected(tmp_path):
    with pytest.raises(ValueError):
        build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="xml")


# --- optional FEN column (task 0029) -------------------------------------------

def test_fen_off_by_default():
    rows = _rows(_pgn("1-0"))
    assert all(r.fen is None for r in rows)


def test_include_fen_records_valid_positions():
    import chess

    rows = list(iter_rows(io.StringIO(_pgn("1-0")), include_fen=True))
    assert rows and all(r.fen is not None for r in rows)
    # After 1. e4 it is Black to move; the FEN must round-trip into a real board.
    first = chess.Board(rows[0].fen)
    assert first.turn == chess.BLACK
    assert rows[0].side_to_move == "black"


def test_build_with_fen_round_trips_the_column(tmp_path):
    out = build_dataset(
        str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="withfen", include_fen=True
    )
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header.endswith(",fen")
    loaded = load_rows(str(out))
    assert loaded and all(r.fen for r in loaded)


def test_load_dataset_without_fen_column_is_backward_compatible(tmp_path):
    # A dataset built the old way (no fen column) must still load, with fen=None.
    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="nofen")
    assert "fen" not in out.read_text(encoding="utf-8").splitlines()[0]
    loaded = load_rows(str(out))
    assert loaded and all(r.fen is None for r in loaded)


# --- partitioned dataset (task 0025) -------------------------------------------

def test_rating_bucket_floors_mean_to_band():
    from chess_equity.data.schema import rating_bucket

    assert rating_bucket(1690, 1690) == "1600"  # mean 1690 -> band [1600,1800)
    assert rating_bucket(1500, 1480) == "1400"  # mean 1490 -> [1400,1600)
    assert rating_bucket(2000, 2000) == "2000"
    assert rating_bucket(1500, 1500, width=100) == "1500"


def test_partitioned_build_creates_hive_tree(tmp_path):
    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="part", partition=True)
    assert out.is_dir() and out.name == "part"
    parts = sorted(out.rglob("part.csv"))
    assert parts, "expected at least one partition file"
    # Every part lives under a tc_bucket=…/rating_bucket=… hive path.
    for p in parts:
        rel = p.relative_to(out).parts
        assert rel[0].startswith("tc_bucket=")
        assert rel[1].startswith("rating_bucket=")


def test_partitioned_round_trips_same_rows(tmp_path):
    flat = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="flat")
    part = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="part", partition=True)
    flat_rows = load_rows(str(flat))
    part_rows = load_rows(str(part))
    assert len(part_rows) == len(flat_rows)
    # Same multiset of rows, partition order aside.
    key = lambda r: (r.game_id, r.ply, r.cp_eval, r.white_elo, r.black_elo)
    assert sorted(map(key, part_rows)) == sorted(map(key, flat_rows))


def test_partitioned_directory_groups_by_bucket(tmp_path):
    from chess_equity.data.schema import rating_bucket

    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="part", partition=True)
    for p in out.rglob("part.csv"):
        tcb = p.parent.parent.name.split("=", 1)[1]
        rb = p.parent.name.split("=", 1)[1]
        for row in load_rows(str(p)):  # every row in a part matches its dir's keys
            assert row.tc_bucket == tcb
            assert rating_bucket(row.white_elo, row.black_elo) == rb


def test_partitioned_with_fen_round_trips(tmp_path):
    out = build_dataset(
        str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="pf", partition=True, include_fen=True
    )
    loaded = load_rows(str(out))
    assert loaded and all(r.fen for r in loaded)


# --- partition selection / predicate pushdown (task 0040) ---------------------

def test_load_rows_selects_by_tc_bucket(tmp_path):
    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="part", partition=True)
    blitz = load_rows(str(out), tc_bucket="blitz")
    assert blitz and all(r.tc_bucket == "blitz" for r in blitz)
    # It's a strict subset of the full read (the sample spans bullet/blitz/rapid).
    assert len(blitz) < len(load_rows(str(out)))


def test_load_rows_selects_by_rating_bucket(tmp_path):
    from chess_equity.data.schema import rating_bucket

    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="part", partition=True)
    rows = load_rows(str(out), rating_bucket="1400")
    assert rows and all(rating_bucket(r.white_elo, r.black_elo) == "1400" for r in rows)


def test_load_rows_selector_accepts_iterable(tmp_path):
    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="part", partition=True)
    multi = load_rows(str(out), tc_bucket=["blitz", "rapid"])
    assert {r.tc_bucket for r in multi} == {"blitz", "rapid"}


def test_load_rows_nonmatching_selector_is_empty(tmp_path):
    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="part", partition=True)
    assert load_rows(str(out), tc_bucket="correspondence") == []


def test_pushdown_skips_nonmatching_part_files(tmp_path, monkeypatch):
    # The whole point of partitioning: a selector must not OPEN non-matching parts.
    from chess_equity.data import build as build_mod

    out = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="part", partition=True)
    opened = []
    real_read = build_mod._read_file
    monkeypatch.setattr(build_mod, "_read_file", lambda p: (opened.append(str(p)), real_read(p))[1])

    build_mod.load_rows(str(out), tc_bucket="blitz")
    assert opened, "expected at least one part to be read"
    assert all("tc_bucket=blitz" in p for p in opened)  # only the blitz partition opened


def test_selector_on_flat_file_filters_rows(tmp_path):
    # No partitions to prune, but the selector still applies row-wise (consistent result).
    flat = build_dataset(str(SAMPLE_PGN), str(tmp_path), fmt="csv", name="flat")
    blitz = load_rows(str(flat), tc_bucket="blitz")
    assert blitz and all(r.tc_bucket == "blitz" for r in blitz)
    assert len(blitz) < len(load_rows(str(flat)))
