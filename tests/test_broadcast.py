"""Tests for live broadcast ingestion (task 0018).

All offline: a finished PGN (with [%clk] tags + ratings) is replayed as if live, and
flaky/multi-game feeds are simulated in-process. No network.
"""

import chess
import pytest

from chess_equity.broadcast import (
    BroadcastFeed,
    BroadcastIngestor,
    FeedError,
    GameTracker,
    LocalPgnFeed,
    MoveEvent,
    grade_delta,
    split_games,
)
from chess_equity.models import LichessBaselineModel

# A short game with clock tags and ratings — the shape a Lichess broadcast emits.
GAME_PGN = """[Event "Test Broadcast"]
[Site "https://lichess.org/abcd1234"]
[White "Carlsen"]
[Black "Nakamura"]
[Round "1"]
[WhiteElo "2850"]
[BlackElo "2780"]
[Result "*"]

1. e4 { [%clk 0:03:00] } e5 { [%clk 0:02:58] } 2. Nf3 { [%clk 0:02:55] } Nc6 { [%clk 0:02:50] } *
"""


def _model():
    return LichessBaselineModel()


# --------------------------------------------------------------------------- #
# grade_delta
# --------------------------------------------------------------------------- #


def test_grade_delta_bands():
    assert grade_delta(None) is None
    assert grade_delta(10.0) == "brilliant"
    assert grade_delta(3.0) == "good"
    assert grade_delta(0.0) == "ok"
    assert grade_delta(-3.0) == "inaccuracy"
    assert grade_delta(-8.0) == "mistake"
    assert grade_delta(-20.0) == "blunder"


# --------------------------------------------------------------------------- #
# LocalPgnFeed replay
# --------------------------------------------------------------------------- #


def test_local_feed_reveals_one_move_per_poll():
    feed = LocalPgnFeed(GAME_PGN)
    first = feed.poll()
    assert first is not None
    # Exactly one half-move revealed so far.
    game = chess.pgn.read_game(__import__("io").StringIO(first))
    assert len(list(game.mainline_moves())) == 1


def test_local_feed_exhausts_and_returns_none():
    feed = LocalPgnFeed(GAME_PGN)
    polls = 0
    while feed.poll() is not None:
        polls += 1
        assert polls < 100
    # 4 half-moves in the fixture.
    assert polls == 4


def test_local_feed_rejects_empty_pgn():
    with pytest.raises(ValueError):
        LocalPgnFeed("")


def test_local_feed_replay_preserves_clocks_end_to_end():
    # Clocks set in the source PGN must survive replay -> ingestion (a live feed
    # carries [%clk]; the replay must too, or the streaming-wedge clock is lost).
    feed = LocalPgnFeed(GAME_PGN)
    ingestor = BroadcastIngestor(feed, _model())
    events = []
    ingestor.run(events.append, interval=0.0, sleep=lambda _: None)
    assert events[0].white_clock == 180.0
    assert events[1].black_clock == 178.0


# --------------------------------------------------------------------------- #
# GameTracker — incremental diffing, clocks, equity, dedup, resync
# --------------------------------------------------------------------------- #


def test_tracker_emits_only_new_moves():
    tracker = GameTracker("g", _model(), white_elo=None, black_elo=None)
    feed = LocalPgnFeed(GAME_PGN)

    all_events = []
    while True:
        snap = feed.poll()
        if snap is None:
            break
        all_events.extend(tracker.ingest(snap))

    # One event per half-move, plies strictly increasing, no duplicates.
    assert [e.ply for e in all_events] == [1, 2, 3, 4]
    assert [e.san for e in all_events] == ["e4", "e5", "Nf3", "Nc6"]


def test_tracker_dedup_on_repeat_snapshot():
    tracker = GameTracker("g", _model(), white_elo=None, black_elo=None)
    snap_full = GAME_PGN
    first = tracker.ingest(snap_full)
    second = tracker.ingest(snap_full)  # identical poll — nothing new
    assert len(first) == 4
    assert second == []


def test_tracker_parses_clocks_and_ratings():
    tracker = GameTracker("g", _model(), white_elo=None, black_elo=None)
    events = tracker.ingest(GAME_PGN)
    e1, e2 = events[0], events[1]
    assert e1.white_clock == 180.0 and e1.black_clock is None
    assert e2.white_clock == 180.0 and e2.black_clock == 178.0
    assert e1.white_elo == 2850 and e1.black_elo == 2780


def test_tracker_equity_is_white_pov_and_delta_is_mover_pov():
    tracker = GameTracker("g", _model(), white_elo=None, black_elo=None)
    events = tracker.ingest(GAME_PGN)
    # Symmetric opening stays near 50% White-POV throughout.
    for e in events:
        assert 40.0 < e.equity < 60.0
    # Every move has a delta (a grade) except there is always a prior position.
    assert all(e.delta_equity is not None for e in events)
    assert all(e.last_move_grade is not None for e in events)


def test_tracker_resync_on_truncated_pgn():
    tracker = GameTracker("g", _model(), white_elo=None, black_elo=None)
    full = tracker.ingest(GAME_PGN)
    assert len(full) == 4

    # Feed a shorter PGN (a broadcast correction): only 2 half-moves.
    short = """[Event "Test Broadcast"]
[White "Carlsen"]
[Black "Nakamura"]
[WhiteElo "2850"]
[BlackElo "2780"]
[Result "*"]

1. e4 e5 *
"""
    resynced = tracker.ingest(short)
    assert [e.ply for e in resynced] == [1, 2]
    assert all(e.resync for e in resynced)


def test_tracker_missing_ratings_default_but_report_none():
    no_elo = """[Event "OTB"]
[White "A"]
[Black "B"]
[Result "*"]

1. e4 e5 *
"""
    tracker = GameTracker("g", _model(), white_elo=None, black_elo=None)
    events = tracker.ingest(no_elo)
    # Equity still computes (model falls back to 1500), but the event is honest.
    assert events[0].white_elo is None and events[0].black_elo is None
    assert 0.0 <= events[0].equity <= 100.0


def test_tracker_elo_override_wins():
    tracker = GameTracker("g", _model(), white_elo=1200, black_elo=2400)
    events = tracker.ingest(GAME_PGN)  # header says 2850/2780, override should win
    assert events[0].white_elo == 1200 and events[0].black_elo == 2400


# --------------------------------------------------------------------------- #
# split_games
# --------------------------------------------------------------------------- #


def test_split_games_separates_two_games():
    two = GAME_PGN + "\n" + GAME_PGN.replace("Carlsen", "Ding").replace("Round \"1\"", "Round \"2\"")
    games = split_games(two)
    assert len(games) == 2
    assert "Carlsen" in games[0] and "Ding" in games[1]


def test_split_games_single():
    assert len(split_games(GAME_PGN)) == 1


# --------------------------------------------------------------------------- #
# BroadcastIngestor — the loop, reconnects, multi-game
# --------------------------------------------------------------------------- #


def test_ingestor_streams_replay_end_to_end():
    feed = LocalPgnFeed(GAME_PGN)
    ingestor = BroadcastIngestor(feed, _model())
    events = []
    stats = ingestor.run(events.append, interval=0.0, sleep=lambda _: None)
    assert [e.san for e in events] == ["e4", "e5", "Nf3", "Nc6"]
    assert stats.events == 4
    assert stats.max_compute_ms >= 0.0
    # Each event carries everything the overlay needs.
    for e in events:
        d = e.to_dict()
        for key in ("fen", "white_clock", "black_clock", "white_elo", "black_elo",
                    "equity", "delta_equity", "last_move_grade"):
            assert key in d


class _FlakyFeed(BroadcastFeed):
    """Raises FeedError on the 2nd poll, otherwise replays GAME_PGN move by move."""

    def __init__(self):
        self._inner = LocalPgnFeed(GAME_PGN)
        self._n = 0

    def poll(self):
        self._n += 1
        if self._n == 2:
            raise FeedError("transient")
        return self._inner.poll()


def test_ingestor_survives_feed_error_and_reconnects():
    ingestor = BroadcastIngestor(_FlakyFeed(), _model())
    events = []
    stats = ingestor.run(
        events.append, interval=0.0, max_polls=10, sleep=lambda _: None
    )
    # A poll was dropped, but reconnect loses no moves: the underlying game state is
    # intact, so the next successful poll catches up and all 4 moves still stream.
    assert stats.errors >= 1
    assert [e.san for e in events] == ["e4", "e5", "Nf3", "Nc6"]


def test_ingestor_routes_multiple_games():
    game2 = GAME_PGN.replace("Carlsen", "Ding").replace(
        'Site "https://lichess.org/abcd1234"', 'Site "https://lichess.org/wxyz9999"'
    )
    snapshot = GAME_PGN + "\n" + game2
    feed = _OneShotFeed(snapshot)
    ingestor = BroadcastIngestor(feed, _model())
    events = []
    ingestor.run(events.append, interval=0.0, sleep=lambda _: None)
    game_ids = {e.game_id for e in events}
    assert len(game_ids) == 2
    # Each game contributed its 4 moves.
    assert len(events) == 8


class _OneShotFeed(BroadcastFeed):
    """Returns a snapshot once, then None (a completed round)."""

    def __init__(self, snapshot):
        self._snapshot = snapshot
        self._done = False

    def poll(self):
        if self._done:
            return None
        self._done = True
        return self._snapshot


def test_move_event_is_json_serializable():
    import json

    feed = LocalPgnFeed(GAME_PGN)
    ingestor = BroadcastIngestor(feed, _model())
    events = []
    ingestor.run(events.append, interval=0.0, sleep=lambda _: None)
    blob = json.dumps(events[0].to_dict())
    assert "fen" in blob
    assert isinstance(events[0], MoveEvent)
