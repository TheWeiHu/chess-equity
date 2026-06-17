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
