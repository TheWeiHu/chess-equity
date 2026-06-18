import json
import re

import chess

from chess_equity.cli import main

_SCHEMA_KEYS = {
    "fen", "model", "white_elo", "black_elo",
    "pov", "equity", "p_win", "p_draw", "p_loss", "cp",
}


def _assert_record_schema(rec):
    assert _SCHEMA_KEYS <= set(rec), f"missing keys: {_SCHEMA_KEYS - set(rec)}"
    assert rec["pov"] == "white"
    assert 0.0 <= rec["equity"] <= 100.0
    for p in ("p_win", "p_draw", "p_loss"):
        assert 0.0 <= rec[p] <= 1.0
    assert abs(rec["p_win"] + rec["p_draw"] + rec["p_loss"] - 1.0) < 1e-6
    # White-POV invariant: the bar value is the white-POV expected score.
    assert abs(rec["equity"] - 100.0 * (rec["p_win"] + 0.5 * rec["p_draw"])) < 1e-4
    assert rec["cp"] is None or isinstance(rec["cp"], (int, float))
    assert isinstance(rec["model"], str) and rec["model"]


def test_eval_json_emits_stable_schema(capsys):
    rc = main(["eval", chess.STARTING_FEN, "--white-elo", "1500", "--black-elo", "1500", "--json"])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    _assert_record_schema(rec)
    assert rec["fen"] == chess.STARTING_FEN
    assert rec["white_elo"] == 1500 and rec["black_elo"] == 1500


def test_eval_json_equity_agrees_with_text_render(capsys):
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    main(["eval", fen, "--json"])
    rec = json.loads(capsys.readouterr().out)
    main(["eval", fen])
    text = capsys.readouterr().out
    # The text bar renders the same white-POV percent as JSON `equity` (one decimal).
    pct = float(re.search(r"(\d+\.\d)%", text).group(1))
    assert abs(pct - rec["equity"]) < 0.06


def test_eval_pgn_json_one_record_per_ply(tmp_path, capsys):
    pgn = tmp_path / "game.pgn"
    pgn.write_text("1. e4 e5 2. Nf3 Nc6 *\n")
    rc = main(["eval", "--pgn", str(pgn), "--json"])
    assert rc == 0
    records = json.loads(capsys.readouterr().out)
    # start position + 4 half-moves = 5 records.
    assert len(records) == 5
    assert [r["ply"] for r in records] == [0, 1, 2, 3, 4]
    assert records[0]["san"] is None
    assert records[1]["san"] == "e4"
    for rec in records:
        _assert_record_schema(rec)


def test_eval_startpos_runs_and_prints_bar(capsys):
    rc = main(["eval", chess.STARTING_FEN, "--white-elo", "1500", "--black-elo", "1500"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "%" in out and "W/D/L" in out


def test_eval_defaults_to_startpos(capsys):
    rc = main(["eval"])
    assert rc == 0
    assert "%" in capsys.readouterr().out


def test_eval_bad_fen_errors_cleanly(capsys):
    rc = main(["eval", "not-a-fen"])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


def test_eval_pgn_annotates_every_move(tmp_path, capsys):
    pgn = tmp_path / "game.pgn"
    pgn.write_text("1. e4 e5 2. Nf3 Nc6 *\n")
    rc = main(["eval", "--pgn", str(pgn)])
    out = capsys.readouterr().out
    assert rc == 0
    # start line + 4 half-moves = 5 annotated lines.
    assert len([ln for ln in out.splitlines() if "%" in ln]) == 5
    assert "e4" in out and "Nf3" in out


def test_precompute_warns_default_model_is_placeholder(tmp_path, capsys):
    """precompute on the default baseline tells the user the bar is NOT Maia (task 0081)."""
    pgn = tmp_path / "game.pgn"
    pgn.write_text("1. e4 e5 *\n")
    out_json = tmp_path / "out.json"
    rc = main(["precompute", "--pgn", str(pgn), "--out", str(out_json)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "maia2" in err and "rating-blind" in err


# --- --depth threads through grade/broadcast (task 0044) ------------------------
# The Stockfish baseline depth comes from build_model(depth=...); grade/broadcast used
# to call build_model(args.model) with no depth, leaving the engine stuck at depth=2.

def _spy_build_model(monkeypatch):
    """Replace cli.build_model with a recorder returning a cheap material baseline."""
    import chess_equity.cli as cli
    from chess_equity.models import LichessBaselineModel, MaterialEngine

    calls = []

    def spy(name="baseline", **kwargs):
        calls.append((name, kwargs))
        return LichessBaselineModel(MaterialEngine())

    monkeypatch.setattr(cli, "build_model", spy)
    return calls


def test_grade_threads_depth_through(tmp_path, monkeypatch, capsys):
    calls = _spy_build_model(monkeypatch)
    pgn = tmp_path / "g.pgn"
    pgn.write_text("1. e4 e5 2. Nf3 Nc6 *\n")
    rc = main(["grade", "--pgn", str(pgn), "--depth", "9"])
    assert rc == 0
    assert calls and calls[0][1].get("depth") == 9


def test_broadcast_threads_depth_through(tmp_path, monkeypatch, capsys):
    calls = _spy_build_model(monkeypatch)
    pgn = tmp_path / "g.pgn"
    pgn.write_text("1. e4 e5 2. Nf3 Nc6 *\n")
    rc = main(["broadcast", "--pgn", str(pgn), "--depth", "11", "--max-polls", "1"])
    assert rc == 0
    assert calls and calls[0][1].get("depth") == 11


def test_grade_depth_defaults_to_two(tmp_path, monkeypatch):
    calls = _spy_build_model(monkeypatch)
    pgn = tmp_path / "g.pgn"
    pgn.write_text("1. e4 e5 *\n")
    rc = main(["grade", "--pgn", str(pgn)])
    assert rc == 0
    assert calls[0][1].get("depth") == 2


# --- --depth threads through highlights/precompute (task 0070) ------------------
# highlights/precompute used to call build_model(args.model) with no depth, leaving
# their Stockfish baseline stuck at depth=2 once a real engine is in use (0064 fixed
# grade/broadcast but missed these two).

def test_highlights_threads_depth_through(tmp_path, monkeypatch):
    calls = _spy_build_model(monkeypatch)
    pgn = tmp_path / "g.pgn"
    pgn.write_text("1. e4 e5 2. Nf3 Nc6 *\n")
    rc = main(["highlights", "--pgn", str(pgn), "--depth", "9"])
    assert rc == 0
    assert calls and calls[0][1].get("depth") == 9


def test_highlights_depth_defaults_to_two(tmp_path, monkeypatch):
    calls = _spy_build_model(monkeypatch)
    pgn = tmp_path / "g.pgn"
    pgn.write_text("1. e4 e5 2. Nf3 Nc6 *\n")
    rc = main(["highlights", "--pgn", str(pgn)])
    assert rc == 0
    assert calls[0][1].get("depth") == 2


def test_precompute_threads_depth_through(tmp_path, monkeypatch):
    calls = _spy_build_model(monkeypatch)
    pgn = tmp_path / "g.pgn"
    pgn.write_text("1. e4 e5 2. Nf3 Nc6 *\n")
    rc = main(["precompute", "--pgn", str(pgn), "--depth", "7"])
    assert rc == 0
    assert calls and calls[0][1].get("depth") == 7


def test_precompute_depth_defaults_to_two(tmp_path, monkeypatch):
    calls = _spy_build_model(monkeypatch)
    pgn = tmp_path / "g.pgn"
    pgn.write_text("1. e4 e5 2. Nf3 Nc6 *\n")
    rc = main(["precompute", "--pgn", str(pgn)])
    assert rc == 0
    assert calls[0][1].get("depth") == 2
