"""Live running per-side accuracy in the broadcast event stream (task 0245).

``grade --round`` computes per-player accuracy *post-hoc*; this threads the same
ok-or-better figure onto every published :class:`MoveEvent` so the overlay can show
"White 94% / Black 88%" live and update each ply. The contract:

- every event carries ``cumulative_accuracy_white``/``black`` (0..100, or ``None`` for a
  side yet to move), each equal to :func:`cumulative_accuracy` over the stream *through
  that move* — so the live figure on the last move of a game matches the post-hoc one;
- the accurate count per side is monotonic non-decreasing as the game progresses;
- a resync (walk-back / correction) rebuilds the tally from scratch, never double-counts;
- ``to_overlay_event`` surfaces it under ``accuracy: {white, black}`` for the widget.

The replay drives the engine-free :class:`LichessBaselineModel`, so it is deterministic
regardless of whether Stockfish is installed. Fixture: ``data/sample/sample_games.pgn``
(illustrative offline smoke, not evidence; see project CLAUDE.md).
"""
import io
import os

import chess
import chess.pgn

from chess_equity.broadcast import (
    ACCURATE_LABELS,
    GameTracker,
    LocalPgnFeed,
    cumulative_accuracy,
)
from chess_equity.models import LichessBaselineModel

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PGN = os.path.join(HERE, "..", "data", "sample", "sample_games.pgn")


def _first_game_pgn():
    with open(SAMPLE_PGN, "r", encoding="utf-8") as fh:
        return chess.pgn.read_game(io.StringIO(fh.read()))


def _replay_first_game():
    """Replay the sample's first game move-by-move through one GameTracker."""
    game = _first_game_pgn()
    feed = LocalPgnFeed(str(game))
    tracker = GameTracker(
        "g1", LichessBaselineModel(), white_elo=1800, black_elo=1800
    )
    events = []
    while True:
        snap = feed.poll()
        if snap is None:
            break
        events.extend(tracker.ingest(snap))
    return events


def test_every_event_accuracy_matches_post_hoc_prefix():
    """Running accuracy through move i == post-hoc accuracy over the first i+1 events."""
    events = _replay_first_game()
    assert events, "fixture must yield move events"
    for i, event in enumerate(events):
        ref = cumulative_accuracy(events[: i + 1])
        assert event.cumulative_accuracy_white == ref["white"]
        assert event.cumulative_accuracy_black == ref["black"]


def test_game_end_matches_post_hoc_full_stream():
    """The acceptance headline: the last move's live figure == the post-hoc round figure."""
    events = _replay_first_game()
    final = cumulative_accuracy(events)
    assert events[-1].cumulative_accuracy_white == final["white"]
    assert events[-1].cumulative_accuracy_black == final["black"]
    # Both sides moved in a real game, so neither figure is the "yet to move" None.
    assert final["white"] is not None and final["black"] is not None
    assert 0.0 <= final["white"] <= 100.0 and 0.0 <= final["black"] <= 100.0


def test_accurate_count_is_monotonic_per_side():
    """The ok-or-better numerator per side never decreases as the game progresses."""
    events = _replay_first_game()
    accurate = {True: 0, False: 0}  # mover_white -> running ok-or-better count
    prev = {True: 0, False: 0}
    for event in events:
        mover_white = not event.white_to_move
        if event.last_move_grade in ACCURATE_LABELS:
            accurate[mover_white] += 1
        # The numerator only ever grows.
        assert accurate[mover_white] >= prev[mover_white]
        prev[mover_white] = accurate[mover_white]


def test_accuracy_threads_to_overlay_event():
    """The overlay reads evt.accuracy.white / evt.accuracy.black on a position event."""
    events = _replay_first_game()
    # By the final move both sides have moved, so the nested accuracy block is present.
    overlay = events[-1].to_overlay_event()
    assert "accuracy" in overlay
    assert overlay["accuracy"]["white"] == events[-1].cumulative_accuracy_white
    assert overlay["accuracy"]["black"] == events[-1].cumulative_accuracy_black


def test_resync_rebuilds_tally_without_double_counting():
    """A walk-back resync re-emits from the start; the final figure is unchanged."""
    game = _first_game_pgn()
    nodes = list(game.mainline())
    full = chess.pgn.Game.from_board(nodes[-1].board())  # full game as one PGN
    full_pgn = str(full)

    # Baseline: ingest the whole game at once.
    t1 = GameTracker("g", LichessBaselineModel(), white_elo=1800, black_elo=1800)
    once = t1.ingest(full_pgn)

    # Now feed a longer prefix, then a SHORTER one (a walk-back), then the full game.
    t2 = GameTracker("g", LichessBaselineModel(), white_elo=1800, black_elo=1800)
    half = chess.pgn.Game.from_board(nodes[len(nodes) // 2].board())
    t2.ingest(str(half))
    short = chess.pgn.Game.from_board(nodes[2].board())  # shorter => resync
    resynced = t2.ingest(str(short))
    assert resynced and resynced[0].resync, "shorter snapshot must trigger a resync"
    final_events = t2.ingest(full_pgn)

    # The figure on the last move is identical whether or not a resync intervened —
    # the tally was rebuilt from scratch, not double-counted.
    assert (
        final_events[-1].cumulative_accuracy_white
        == once[-1].cumulative_accuracy_white
    )
    assert (
        final_events[-1].cumulative_accuracy_black
        == once[-1].cumulative_accuracy_black
    )
