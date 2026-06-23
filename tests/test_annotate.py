"""Equity-annotated PGN export round-trips through python-chess (task 0197)."""

import io
from pathlib import Path

import chess
import chess.pgn

from chess_equity.annotate import annotate_game, annotate_pgn_file, white_pov_equity
from chess_equity.grading import EquityGrader
from chess_equity.models import LichessBaselineModel

SAMPLE_PGN = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_games.pgn"


def _read_first_game(text: str) -> chess.pgn.Game:
    game = chess.pgn.read_game(io.StringIO(text))
    assert game is not None
    return game


def test_annotate_game_adds_parseable_equity_comments():
    """Every mainline move carries a re-parseable [%equity 0..1] White-POV tag."""
    pgn = "1. e4 e5 2. Nf3 Nc6 *"
    game = annotate_game(_read_first_game(pgn), LichessBaselineModel(), 1500, 1500)

    # Re-serialize and re-parse: the comments must survive a full round-trip.
    reparsed = _read_first_game(str(game))
    n = 0
    for node in reparsed.mainline():
        assert "[%equity " in node.comment, f"missing equity tag on {node.comment!r}"
        # The tag value is a White-POV probability in [0, 1].
        tag = node.comment.split("[%equity ")[1].split("]")[0]
        val = float(tag)
        assert 0.0 <= val <= 1.0
        n += 1
    assert n == 4  # e4 e5 Nf3 Nc6


def test_annotated_equity_matches_eval_white_pov():
    """The [%equity ...] value equals the model's White-POV equity for that FEN."""
    model = LichessBaselineModel()
    pgn = "1. e4 e5 2. Nf3 Nc6 *"
    game = _read_first_game(pgn)

    grades = EquityGrader(model).grade_game(game, 1500, 1500)
    # equity after 1. e4: White just moved, so mover-POV == White-POV.
    e4 = grades[0]
    assert e4.san == "e4"
    expected = model.evaluate(chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").fen(), 1500, 1500)
    assert abs(white_pov_equity(e4) - expected.equity_white / 100.0) < 1e-9

    # equity after 1... e5: Black just moved, so White-POV == 100 - mover-POV.
    e5 = grades[1]
    assert e5.san == "e5"
    assert white_pov_equity(e5) == (100.0 - e5.equity_after) / 100.0


def test_annotate_preserves_existing_eval_and_clk_tags():
    """Appending [%equity] must not clobber the PGN's existing [%eval]/[%clk]."""
    game = annotate_game(
        _read_first_game(SAMPLE_PGN.read_text(encoding="utf-8")),
        LichessBaselineModel(), 1500, 1480,
    )
    first = game.mainline().__iter__().__next__()  # node after 1. e4
    assert "[%eval 0.2]" in first.comment   # original survives
    assert "[%clk 0:03:00]" in first.comment
    assert "[%equity " in first.comment     # ours is appended


def test_annotate_pgn_file_round_trips_sample(tmp_path):
    """End-to-end: annotate the sample fixture to a file, re-parse, a known move is tagged."""
    out = tmp_path / "annotated.pgn"
    n = annotate_pgn_file(str(SAMPLE_PGN), str(out), LichessBaselineModel(), 1500, 1480)
    assert n == 7  # game 1 (fool's mate) has 7 half-moves

    reparsed = _read_first_game(out.read_text(encoding="utf-8"))
    first = next(iter(reparsed.mainline()))  # 1. e4
    assert "[%equity " in first.comment
    assert "[%grade " in first.comment
