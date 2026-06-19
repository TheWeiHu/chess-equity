import chess

from chess_equity.cli import main


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


def test_score_reports_real_result_and_prediction(capsys):
    rc = main(["score", "--pgn", "data/sample/sample_games.pgn"])
    out = capsys.readouterr().out
    assert rc == 0
    # Answers the four questions: a game, the score (cp), the real result, a prediction.
    assert "alice (1500) vs bob (1480)" in out
    assert "real score: 1-0" in out
    assert "equity" in out and "cp" in out
    assert "validate" in out  # stays honest about what one game proves


def test_score_rating_overrides(capsys):
    rc = main([
        "score", "--pgn", "data/sample/sample_games.pgn",
        "--white-elo", "800", "--black-elo", "2400",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "alice (800) vs bob (2400)" in out


def test_score_missing_pgn_errors_cleanly(capsys):
    rc = main(["score", "--pgn", "does-not-exist.pgn"])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


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
