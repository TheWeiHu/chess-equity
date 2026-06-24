"""Anti-repetition in the live caster caption stream (task 0274).

``broadcast --captions`` prints one :func:`live_caption` line per graded move as the
feed streams. ``live_caption`` is *stateless* — it sees only the current move — so when
two consecutive plies compose the byte-identical line (e.g. White ``O-O`` then Black
``O-O`` with the same grade and swing) the overlay reads as stuck/broken on stream.
:class:`CaptionDeduper` is the thin stateful guard: it varies a caption that would repeat
the immediately-preceding line with a short rotating lead-in, so every move still gets a
caption but no two consecutive emitted lines are equal.

These tests pin three facts:

* the deduper varies a back-to-back *identical* caption and leaves distinct ones alone;
* a real castling-into-castling event pair (the canonical duplicate trigger) is de-duped
  while still naming both moves; and
* replaying the in-repo sample fixtures through the stream emits no two consecutive
  identical captions. A real cached Lichess dump, when present, gets the same check.

Engine-free throughout (``LichessBaselineModel``) so the result is deterministic and
independent of whether Stockfish is installed.
"""
import os

import chess
import pytest

from chess_equity.broadcast import (
    BroadcastIngestor,
    CaptionDeduper,
    LocalPgnFeed,
    MoveEvent,
    live_caption,
)
from chess_equity.models import LichessBaselineModel

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(HERE, "..", "data", "sample")
CACHED_REAL_PGN = os.path.expanduser(
    "~/.cache/chess-equity/lichess/kAdOQKeh.pgn"
)


def _event(san, *, white_to_move, grade, delta):
    """A minimal graded MoveEvent — only the fields live_caption reads matter here."""
    return MoveEvent(
        game_id="g",
        ply=1,
        san=san,
        uci="",
        fen=chess.STARTING_FEN,
        white_to_move=white_to_move,
        white_clock=None,
        black_clock=None,
        white_elo=1800,
        black_elo=1800,
        equity=50.0,
        delta_equity=delta,
        last_move_grade=grade,
        source="test",
        compute_ms=0.0,
    )


def _no_consecutive_identical(lines):
    return all(a != b for a, b in zip(lines, lines[1:]))


def test_deduper_varies_identical_and_passes_distinct():
    """Identical back-to-back lines get a varying lead-in; distinct lines pass through."""
    d = CaptionDeduper()
    assert d.feed("Qxf7 — brilliant here") == "Qxf7 — brilliant here"
    # Same line again → must differ from what we just emitted.
    second = d.feed("Qxf7 — brilliant here")
    assert second is not None and second != "Qxf7 — brilliant here"
    assert second.endswith("Qxf7 — brilliant here")  # the move is still named
    # A genuinely different caption is never altered.
    assert d.feed("Rd1 — ok here") == "Rd1 — ok here"
    # None (ungraded tick) passes straight through.
    assert d.feed(None) is None


def test_deduper_breaks_a_long_identical_run():
    """Even a run of identical captions never emits two equal consecutive lines."""
    d = CaptionDeduper()
    out = [d.feed("O-O — book here") for _ in range(5)]
    assert _no_consecutive_identical(out)
    assert all(line is not None and line.endswith("O-O — book here") for line in out)


def test_castling_into_castling_is_deduped():
    """The canonical trigger: White O-O then Black O-O compose the identical line."""
    white_castle = _event("O-O", white_to_move=False, grade="book", delta=0.0)
    black_castle = _event("O-O", white_to_move=True, grade="book", delta=0.0)
    # Stateless live_caption produces the same string for both → the bug.
    assert live_caption(white_castle) == live_caption(black_castle)

    d = CaptionDeduper()
    first = d.feed(live_caption(white_castle))
    second = d.feed(live_caption(black_castle))
    assert first is not None and second is not None
    assert first != second
    assert "O-O" in first and "O-O" in second  # both moves still captioned


def _stream_captions(pgn_text):
    """Replay one game's captions through the live emit path (deduper applied)."""
    feed = LocalPgnFeed(pgn_text)
    ingestor = BroadcastIngestor(
        feed, LichessBaselineModel(), white_elo=1800, black_elo=1800
    )
    deduper = CaptionDeduper()
    lines = []
    while True:
        snap = feed.poll()
        if snap is None:
            break
        for event in ingestor.ingest_snapshot(snap):
            line = deduper.feed(live_caption(event))
            if line is not None:
                lines.append(line)
    return lines


@pytest.mark.parametrize(
    "fixture",
    ["sample_games.pgn", "otb_classical_no_clock.pgn", "round_games.pgn"],
)
def test_sample_fixtures_have_no_consecutive_identical_captions(fixture):
    path = os.path.join(SAMPLE_DIR, fixture)
    with open(path, "r", encoding="utf-8") as fh:
        lines = _stream_captions(fh.read())
    assert lines  # the fixture produced captions at all
    assert _no_consecutive_identical(lines)


@pytest.mark.skipif(
    not os.path.exists(CACHED_REAL_PGN),
    reason="real cached Lichess dump not present (offline CI)",
)
def test_real_cached_pgn_has_no_consecutive_identical_captions():
    """Acceptance: replay a real cached --pgn; no identical caption on consecutive plies."""
    with open(CACHED_REAL_PGN, "r", encoding="utf-8") as fh:
        lines = _stream_captions(fh.read())
    assert lines
    assert _no_consecutive_identical(lines)
