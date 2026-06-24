"""Live caster captions on the broadcast console (task 0190).

The offline reel (task 0184) ranks a *finished* game's drama into OBS lower-thirds;
this is the *live* counterpart: ``broadcast --captions`` prints one human caster
sentence per graded move as the feed streams, TTS/chat-ready, with no new model calls.
These tests pin two acceptance facts:

* a caption line is emitted for *every graded move* (one per half-move past the
  opening), composed from the move, its grade, the practical swing and the mover's
  rating; and
* a *dramatic* move surfaces the drama classifier's caster ``headline`` inline.

Fixture: ``data/sample/sample_games.pgn`` — the sanctioned offline-smoke fixture
(illustrative, not evidence; see project CLAUDE.md). Its first game is a 7-ply
scholar's mate ending in ``Qxf7#``.

The drama assertion drives the ingestor directly with an *engine-free*
:class:`LichessBaselineModel`, so the result is deterministic and independent of
whether Stockfish is installed (the CLI's default ``baseline`` wraps a real engine
when one is present, which would shift the swing onto a different ply). The CLI test
exercises the ``--captions`` flag wiring without pinning which move is dramatic.
"""
import io
import os

import chess
import chess.pgn

from chess_equity.broadcast import (
    BroadcastIngestor,
    LocalPgnFeed,
    MoveEvent,
    live_caption,
)
from chess_equity.cli import main
from chess_equity.models import LichessBaselineModel

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PGN = os.path.join(HERE, "..", "data", "sample", "sample_games.pgn")


def _first_game_sans():
    """SAN moves of the sample's first game (LocalPgnFeed replays only that game)."""
    with open(SAMPLE_PGN, "r", encoding="utf-8") as fh:
        game = chess.pgn.read_game(io.StringIO(fh.read()))
    assert game is not None
    board = game.board()
    sans = []
    for move in game.mainline_moves():
        sans.append(board.san(move))
        board.push(move)
    return sans


def _replay_events():
    """Replay the sample's first game through the engine-free baseline model."""
    with open(SAMPLE_PGN, "r", encoding="utf-8") as fh:
        text = fh.read()
    # LocalPgnFeed replays only the first game; ingest_snapshot would process both, so
    # render the first game alone via the feed and ingest each successive snapshot.
    feed = LocalPgnFeed(text)
    ingestor = BroadcastIngestor(
        feed, LichessBaselineModel(), white_elo=1800, black_elo=1800
    )
    events = []
    while True:
        snap = feed.poll()
        if snap is None:
            break
        events.extend(ingestor.ingest_snapshot(snap))
    return events


def test_caption_per_graded_move_and_drama_headline():
    events = _replay_events()
    sans = _first_game_sans()
    captions = [live_caption(e) for e in events]

    # One caption per move (every move past the opening is graded on the baseline), each
    # naming its move with a grade and a signed practical swing for the mover's rating.
    assert len(captions) == len(sans)
    for san, cap in zip(sans, captions):
        assert cap is not None and cap.startswith(f"{san} — "), cap
        assert "%" in cap and "1800" in cap, cap

    # The scholar's mate Qxf7# is a >50% swing → the drama classifier fires and its
    # caster headline is appended inline after the bare grade.
    mate_cap = captions[-1]
    assert mate_cap is not None
    assert "·" in mate_cap and "clutch move" in mate_cap, mate_cap
    assert sans[-1] in mate_cap


def test_cli_captions_flag_streams_one_line_per_move(capsys):
    rc = main(
        [
            "broadcast",
            "--pgn",
            SAMPLE_PGN,
            "--captions",
            "--white-elo",
            "1800",
            "--black-elo",
            "1800",
            "--interval",
            "0",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    sans = _first_game_sans()

    # A "🎙 ... vs ..." caster intro, then one caption line per move.
    lines = [ln for ln in out.splitlines() if ln]
    assert lines[0].startswith("🎙") and "vs" in lines[0], lines[0]
    caption_lines = lines[1:]
    assert len(caption_lines) == len(sans), (caption_lines, sans)
    for san, line in zip(sans, caption_lines):
        assert line.startswith(f"{san} — "), line


def test_live_caption_is_none_for_opening_position():
    """No prior move to grade → no caption (caller cleanly skips the tick)."""
    opening = MoveEvent(
        game_id="g",
        ply=0,
        san="",
        uci="",
        fen=chess.STARTING_FEN,
        white_to_move=True,
        white_clock=None,
        black_clock=None,
        white_elo=1800,
        black_elo=1800,
        equity=50.0,
        delta_equity=None,
        last_move_grade=None,
        source="test",
        compute_ms=0.0,
    )
    assert live_caption(opening) is None
