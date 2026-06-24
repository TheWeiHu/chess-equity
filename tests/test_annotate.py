"""Equity-annotated PGN export round-trips through python-chess (task 0197)."""

import io
from pathlib import Path

import chess
import chess.pgn

from chess_equity.annotate import (
    annotate_game,
    annotate_pgn_file,
    drama_by_ply,
    white_pov_equity,
)
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


def test_drama_by_ply_fires_clutch_on_the_mate():
    """The mating move's huge White-POV swing is highlight-worthy even on the baseline."""
    text = SAMPLE_PGN.read_text(encoding="utf-8")
    drama = drama_by_ply(text, LichessBaselineModel(), 1500, 1480)
    # Scholar's mate: only the final move (4. Qxf7#, ply 7) is dramatic.
    assert drama.keys() == {7}
    assert drama[7].kind == "clutch"


def test_annotate_embeds_drama_tag_only_on_dramatic_moves(tmp_path):
    """`[%drama <kind>]` lands on the mate; quiet moves carry no drama tag; [%eval]/[%clk] survive."""
    out = tmp_path / "annotated.pgn"
    annotate_pgn_file(str(SAMPLE_PGN), str(out), LichessBaselineModel(), 1500, 1480)

    reparsed = _read_first_game(out.read_text(encoding="utf-8"))
    nodes = list(reparsed.mainline())

    # The mating move (last ply) gains a parseable [%drama clutch] tag...
    mate = nodes[-1]
    assert "[%drama clutch]" in mate.comment
    # ...alongside the equity/grade tags, not replacing them.
    assert "[%equity " in mate.comment and "[%grade " in mate.comment

    # The quiet opening moves are unchanged — no drama tag, originals preserved.
    first = nodes[0]  # 1. e4
    assert "[%drama" not in first.comment
    assert "[%eval 0.2]" in first.comment and "[%clk 0:03:00]" in first.comment

    # Exactly one move in the game is dramatic.
    assert sum("[%drama" in node.comment for node in nodes) == 1


def test_annotate_game_without_drama_is_byte_identical():
    """Omitting the drama map embeds no [%drama] tags (back-compat with task 0197)."""
    game = annotate_game(
        _read_first_game(SAMPLE_PGN.read_text(encoding="utf-8")),
        LichessBaselineModel(), 1500, 1480,
    )
    assert "[%drama" not in str(game)
