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


def test_missing_eval_position_is_skipped_but_clock_optional():
    rows = _rows(_pgn("1-0"))
    # 1...e5 has an eval but no clock -> kept, clock None.
    assert rows[1].clock_remaining is None
    assert rows[0].clock_remaining == 300.0


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
