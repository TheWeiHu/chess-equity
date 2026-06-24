"""Tests for live broadcast ingestion (task 0018).

All offline: a finished PGN (with [%clk] tags + ratings) is replayed as if live, and
flaky/multi-game feeds are simulated in-process. No network.
"""

import chess
import pytest

from chess_equity.broadcast import (
    FOCUS_BIAS,
    FOCUS_BIAS_BONUS,
    FOCUS_DECAY,
    FOCUS_MARGIN,
    BroadcastFeed,
    BroadcastIngestor,
    FeedError,
    FocusDirector,
    GameEvent,
    GameTracker,
    LocalPgnFeed,
    MoveEvent,
    PinChannel,
    focus_recap_md,
    game_event,
    grade_delta,
    overlay_events,
    parse_focus_bias,
    split_games,
)
from chess_equity.adapters import EquityModel
from chess_equity.models import LichessBaselineModel
from chess_equity.types import Equity, WDL

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


class _ConnectThenErrorFeed(BroadcastFeed):
    """Connects once (so trackers exist), then raises FeedError on every later poll.

    Models a live feed that drops mid-stream and stays down — the worst case for the
    reconnect backoff, since the ingestor must keep retrying (it connected) instead of
    giving up, ramping the wait each time.
    """

    def __init__(self):
        self._n = 0

    def poll(self):
        self._n += 1
        if self._n == 1:
            return GAME_PGN
        raise FeedError("feed down")


def test_reconnect_backoff_is_exponential_and_bounded():
    # A feed that drops after connecting should be retried with a geometric, capped
    # backoff, and announce each reconnect attempt — no network, sleep is recorded.
    delays: list[float] = []
    reconnects: list[tuple[int, float]] = []
    ingestor = BroadcastIngestor(_ConnectThenErrorFeed(), _model())
    stats = ingestor.run(
        lambda _: None,
        interval=2.0,
        max_polls=7,
        sleep=delays.append,
        reconnect_backoff=1.0,
        backoff_factor=2.0,
        backoff_max=8.0,
        on_reconnect=lambda attempt, delay: reconnects.append((attempt, delay)),
    )
    # poll 1 connects (no sleep before it); the first inter-poll sleep is the normal
    # interval, then each subsequent sleep is the prior error's backoff.
    assert delays[0] == 2.0  # healthy cadence before the drop
    assert delays[1:] == [1.0, 2.0, 4.0, 8.0, 8.0]  # geometric, capped at backoff_max
    # on_reconnect fires once per FeedError with the growing (capped) delay.
    assert reconnects == [(1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0), (5, 8.0), (6, 8.0)]
    assert stats.errors == 6
    assert stats.max_backoff_s == 8.0
    # It never crashes and never gives up once connected (kept polling to max_polls).
    assert stats.polls == 7


class _ErrorsThenRecoversFeed(BroadcastFeed):
    """Errors twice, recovers and serves a move, then errors again — a flicker.

    Lets us assert the backoff *resets* on a successful poll instead of ramping forever.
    """

    def __init__(self):
        self._inner = LocalPgnFeed(GAME_PGN)
        self._n = 0

    def poll(self):
        self._n += 1
        if self._n in (2, 3, 5):
            raise FeedError("flicker")
        return self._inner.poll()


def test_reconnect_backoff_resets_after_a_successful_poll():
    reconnects: list[tuple[int, float]] = []
    ingestor = BroadcastIngestor(_ErrorsThenRecoversFeed(), _model())
    stats = ingestor.run(
        lambda _: None,
        interval=0.0,
        max_polls=10,
        sleep=lambda _: None,
        reconnect_backoff=1.0,
        backoff_factor=2.0,
        on_reconnect=lambda attempt, delay: reconnects.append((attempt, delay)),
    )
    # Two consecutive errors ramp to 1,2; the recovery (poll 4) resets the attempt
    # counter, so the later lone error starts again at attempt 1 (delay 1.0).
    assert reconnects == [(1, 1.0), (2, 2.0), (1, 1.0)]
    # Two separate recovery episodes (after the 2-error burst, after the lone error).
    assert stats.reconnects == 2


def test_reconnect_resumes_from_last_seen_move_losing_nothing():
    # A drop mid-stream must not lose moves: the tracker keeps its emitted prefix, so
    # after reconnecting the feed catches up and all four moves still arrive in order.
    ingestor = BroadcastIngestor(_ErrorsThenRecoversFeed(), _model())
    events: list[MoveEvent] = []
    ingestor.run(
        events.append, interval=0.0, max_polls=10, sleep=lambda _: None
    )
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


# --------------------------------------------------------------------------- #
# BoardSelector — follow one board of a multi-game round (task 0182)
# --------------------------------------------------------------------------- #


# Illustrative fixture only (NOT evidence): a tiny two-board round snapshot used to
# unit-test board selection. Real broadcasts carry the same multi-game shape.
def _two_board_round():
    game2 = GAME_PGN.replace("Carlsen", "Ding").replace("Nakamura", "Firouzja").replace(
        'Site "https://lichess.org/abcd1234"', 'Site "https://lichess.org/wxyz9999"'
    )
    return GAME_PGN + "\n" + game2


def test_parse_board_selector_modes():
    from chess_equity.broadcast import BoardSelector, parse_board_selector

    assert parse_board_selector(None) is None
    assert parse_board_selector("  ") is None
    assert parse_board_selector("1") == BoardSelector(index=1)
    assert parse_board_selector("Carlsen") == BoardSelector(player="Carlsen")


def test_parse_auto_spec_modes():
    """`--board auto[:player]` is recognised as the auto-follow form; everything else
    (incl. a fixed one-board follow) is not (tasks 0256 / 0262)."""
    from chess_equity.broadcast import parse_auto_spec

    assert parse_auto_spec("auto") == (True, None)
    assert parse_auto_spec("AUTO") == (True, None)  # keyword is case-insensitive
    assert parse_auto_spec("  auto  ") == (True, None)
    assert parse_auto_spec("auto:Carlsen") == (True, "Carlsen")  # name keeps its case
    assert parse_auto_spec("auto: ding ") == (True, "ding")
    assert parse_auto_spec("auto:") == (True, None)  # blank name degrades to plain auto
    # Not an auto spec — the caller falls back to parse_board_selector.
    assert parse_auto_spec(None) == (False, None)
    assert parse_auto_spec("Carlsen") == (False, None)
    assert parse_auto_spec("1") == (False, None)


def test_ingestor_follows_one_board_by_player():
    from chess_equity.broadcast import parse_board_selector

    feed = _OneShotFeed(_two_board_round())
    ingestor = BroadcastIngestor(feed, _model(), select=parse_board_selector("ding"))
    events = []
    ingestor.run(events.append, interval=0.0, sleep=lambda _: None)
    # Only Ding's board streams (game_id is its Site URL); Carlsen's is filtered out.
    game_ids = {e.game_id for e in events}
    assert game_ids == {"https://lichess.org/wxyz9999"}
    assert len(events) == 4  # the followed game's 4 moves only


def test_ingestor_follows_one_board_by_index():
    from chess_equity.broadcast import parse_board_selector

    feed = _OneShotFeed(_two_board_round())
    ingestor = BroadcastIngestor(feed, _model(), select=parse_board_selector("0"))
    events = []
    ingestor.run(events.append, interval=0.0, sleep=lambda _: None)
    # Board 0 is the first game (Carlsen's, Site abcd1234).
    assert {e.game_id for e in events} == {"https://lichess.org/abcd1234"}


def test_ingestor_no_selector_follows_all_boards():
    feed = _OneShotFeed(_two_board_round())
    ingestor = BroadcastIngestor(feed, _model())  # default: follow every board
    events = []
    ingestor.run(events.append, interval=0.0, sleep=lambda _: None)
    assert len({e.game_id for e in events}) == 2


def test_selector_matches_either_color():
    from chess_equity.broadcast import parse_board_selector

    # The needle matches the *Black* player of board 1 (Firouzja).
    feed = _OneShotFeed(_two_board_round())
    ingestor = BroadcastIngestor(feed, _model(), select=parse_board_selector("firouzja"))
    events = []
    ingestor.run(events.append, interval=0.0, sleep=lambda _: None)
    # Matched on Black; only board 1 (Ding vs Firouzja, Site wxyz9999) streams.
    assert {e.game_id for e in events} == {"https://lichess.org/wxyz9999"}


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


# --------------------------------------------------------------------------- #
# GameEvent — player names threaded to the overlay "game" metadata (task 0047)
# --------------------------------------------------------------------------- #


def _headers(pgn):
    import io

    return chess.pgn.read_headers(io.StringIO(pgn))


def test_game_event_parses_names_and_ratings():
    ge = game_event(_headers(GAME_PGN), "g")
    assert ge.white_name == "Carlsen" and ge.black_name == "Nakamura"
    assert ge.white_elo == 2850 and ge.black_elo == 2780
    players = ge.to_overlay()["players"]
    assert ge.to_overlay()["type"] == "game"
    assert players["white"] == {"name": "Carlsen", "rating": 2850}
    assert players["black"] == {"name": "Nakamura", "rating": 2780}


def test_game_event_missing_names_are_none():
    anon = """[Event "OTB"]
[White "?"]
[Black ""]
[Result "*"]

1. e4 e5 *
"""
    ge = game_event(_headers(anon), "g")
    assert ge.white_name is None and ge.black_name is None
    # overlay.js falls back to "White"/"Black" on a null name.
    assert ge.to_overlay()["players"]["white"]["name"] is None


def test_game_event_rating_override_wins():
    ge = game_event(_headers(GAME_PGN), "g", white_elo=1200, black_elo=2400)
    assert ge.white_elo == 1200 and ge.black_elo == 2400


def test_ingestor_emits_game_event_once_with_names_before_moves():
    feed = LocalPgnFeed(GAME_PGN)
    ingestor = BroadcastIngestor(feed, _model())
    games, moves = [], []
    ingestor.on_game = games.append
    ingestor.run(moves.append, interval=0.0, sleep=lambda _: None)
    # Exactly one game event, carrying the PGN's player names.
    assert len(games) == 1
    assert isinstance(games[0], GameEvent)
    assert games[0].white_name == "Carlsen" and games[0].black_name == "Nakamura"
    # ...and the moves still all stream (game event is additive, not a replacement).
    assert [e.san for e in moves] == ["e4", "e5", "Nf3", "Nc6"]


def test_ingestor_emits_one_game_event_per_game():
    game2 = GAME_PGN.replace("Carlsen", "Ding").replace(
        'Site "https://lichess.org/abcd1234"', 'Site "https://lichess.org/wxyz9999"'
    )
    feed = _OneShotFeed(GAME_PGN + "\n" + game2)
    ingestor = BroadcastIngestor(feed, _model())
    games = []
    ingestor.on_game = games.append
    ingestor.run(lambda _e: None, interval=0.0, sleep=lambda _: None)
    names = {g.white_name for g in games}
    assert names == {"Carlsen", "Ding"}


# --------------------------------------------------------------------------- #
# Live board switcher — boards roster + per-board routing (task 0185)
# --------------------------------------------------------------------------- #


def _overlay_events_for(snapshot):
    """Drive a snapshot through the overlay bridge and collect the emitted events."""
    feed = _OneShotFeed(snapshot)
    ingestor = BroadcastIngestor(feed, _model())
    return [
        e
        for e in overlay_events(ingestor, interval=0.0, sleep=lambda _: None)
        if isinstance(e, dict)
    ]


def test_overlay_bridge_announces_boards_for_a_multi_game_round():
    """A multi-game round emits a `boards` roster event listing every board
    (index + players), so the overlay can render a live board selector."""
    events = _overlay_events_for(_two_board_round())
    rosters = [e for e in events if e.get("type") == "boards"]
    assert rosters, "a multi-game round must announce a boards roster"
    # The final roster lists both boards with their index + players.
    boards = rosters[-1]["boards"]
    assert {b["index"] for b in boards} == {0, 1}
    by_index = {b["index"]: b for b in boards}
    assert by_index[0]["players"]["white"]["name"] == "Carlsen"
    assert by_index[1]["players"]["white"]["name"] == "Ding"


def test_overlay_bridge_stamps_board_index_on_each_event():
    """Every game/position event of a multi-game round carries its 0-based `board`
    index, so the overlay can route it to the chosen board."""
    events = _overlay_events_for(_two_board_round())
    games = {e["game_id"]: e["board"] for e in events if e.get("type") == "game"}
    # Two games, board 0 and board 1.
    assert set(games.values()) == {0, 1}
    positions = [e for e in events if e.get("type") == "position"]
    assert positions, "moves must stream"
    for e in positions:
        assert e["board"] in (0, 1), "each position routes to a known board"


def test_overlay_bridge_single_game_has_no_boards_or_index():
    """Default single-board behavior: a single-game feed announces no roster and tags
    no `board`, so the overlay shows no selector and renders every event."""
    events = _overlay_events_for(GAME_PGN)
    assert not any(e.get("type") == "boards" for e in events), "no roster for one board"
    for e in events:
        assert "board" not in e, "single-game events carry no board index"


# --------------------------------------------------------------------------- #
# Auto-advance off a finished board — game-end result signal (task 0189)
# --------------------------------------------------------------------------- #


# Board 0 has ended (Result "1-0"); board 1 is still in progress (Result "*"). The
# overlay bridge should announce board 0's result so the router can advance focus.
def _round_with_board0_finished():
    board1 = (
        GAME_PGN.replace("Carlsen", "Ding")
        .replace("Nakamura", "Firouzja")
        .replace('Site "https://lichess.org/abcd1234"', 'Site "https://lichess.org/wxyz9999"')
    )
    board0 = GAME_PGN.replace('[Result "*"]', '[Result "1-0"]')
    return board0 + "\n" + board1


def test_overlay_bridge_emits_result_when_a_board_finishes():
    """A board whose PGN reaches a terminal Result emits a `result` event carrying its
    board index, so the overlay can auto-advance focus off the finished game."""
    events = _overlay_events_for(_round_with_board0_finished())
    results = [e for e in events if e.get("type") == "result"]
    assert len(results) == 1, "exactly one board finished"
    assert results[0]["board"] == 0
    assert results[0]["result"] == "1-0"


def test_overlay_bridge_no_result_while_games_in_progress():
    """While every board's Result is still `*`, the bridge emits no result events —
    nothing has ended to advance off of."""
    events = _overlay_events_for(_two_board_round())
    assert not any(e.get("type") == "result" for e in events)


def test_overlay_bridge_single_game_finish_emits_no_result():
    """A finished single-game feed emits no result event — there's no other board to
    advance to, and its event stream stays unchanged (no `board`, no `result`)."""
    finished = GAME_PGN.replace('[Result "*"]', '[Result "1-0"]')
    events = _overlay_events_for(finished)
    assert not any(e.get("type") == "result" for e in events)


def test_ingestor_fires_on_result_once_per_finished_board():
    """`on_result` fires exactly once per game, the first time it reaches a terminal
    result — even across repeated snapshots of the same finished round."""
    from chess_equity.broadcast import ResultEvent

    snapshot = _round_with_board0_finished()

    class _RepeatFeed(BroadcastFeed):
        def __init__(self, snap, times):
            self._snap, self._times = snap, times

        def poll(self):
            if self._times <= 0:
                return None
            self._times -= 1
            return self._snap

    ingestor = BroadcastIngestor(_RepeatFeed(snapshot, 3), _model())
    results = []
    ingestor.on_result = results.append
    ingestor.run(lambda _e: None, interval=0.0, sleep=lambda _: None)
    assert len(results) == 1, "result announced once, not re-fired on every poll"
    assert isinstance(results[0], ResultEvent)
    assert results[0].board == 0 and results[0].result == "1-0"


# --------------------------------------------------------------------------- #
# Drama auto-follow — `broadcast --board auto` server-side focus (task 0256)
# --------------------------------------------------------------------------- #


def test_focus_director_adopts_first_board_silently():
    """The first board seen is adopted without a focus event — the overlay router
    already defaults to board 0, so an opening cut would be redundant."""
    d = FocusDirector()
    assert d.note(0, 0.0) is None
    assert d.focus == 0


def test_focus_director_follows_the_more_dramatic_board():
    """A rival board that out-dramas the focus by the margin steals the cut."""
    d = FocusDirector(margin=FOCUS_MARGIN)
    d.note(0, 0.0)  # adopt board 0 silently
    assert d.note(0, 0.0) is None  # quiet move on the focused board: no cut
    assert d.note(1, FOCUS_MARGIN + 0.1) == 1  # board 1 erupts -> cut to it
    assert d.focus == 1


def test_focus_director_hysteresis_ignores_tiny_swings():
    """A blip below the margin can't thrash the focus off the followed board."""
    d = FocusDirector(margin=FOCUS_MARGIN)
    d.note(0, 0.0)
    # Board 1 is barely more dramatic than board 0 (under the margin) — no cut.
    assert d.note(1, FOCUS_MARGIN - 0.01) is None
    assert d.focus == 0


def test_focus_director_no_cut_on_the_focused_board():
    """Repeated drama on the already-followed board never re-emits a focus event."""
    d = FocusDirector()
    d.note(0, 0.0)
    assert d.note(0, 0.9) is None  # huge swing, but we're already on board 0
    assert d.focus == 0


def test_focus_director_decays_a_stale_peak_so_an_active_rival_steals_focus():
    """A board whose drama PEAK was its final move must not hold focus forever: as it
    goes quiet its standing score decays each tick, so a board playing steady moderate
    drama eventually out-dramas the stale peak and steals the cut (task 0257)."""
    d = FocusDirector(margin=FOCUS_MARGIN, decay=0.5)
    d.note(0, 0.0)  # adopt board 0
    assert d.note(1, 0.9) == 1  # board 1 erupts on its final move -> cut to it
    assert d.focus == 1
    # Board 1 is now silent (no follow-up). Board 0 plays steady moderate drama; each
    # tick decays board 1's standing 0.9 (0.45, 0.225, 0.1125, ...) until board 0's
    # 0.3 clears it by the margin and steals focus back.
    assert d.note(0, 0.3) is None  # 0.3 - 0.45  -> still below margin
    assert d.note(0, 0.3) is None  # 0.3 - 0.225 = 0.075 < 0.15
    assert d.note(0, 0.3) == 0  # 0.3 - 0.1125 = 0.1875 >= 0.15 -> steal
    assert d.focus == 0


def test_focus_director_without_decay_a_stale_peak_holds_focus_forever():
    """Contrast: with ``decay=1.0`` (the old behaviour) the same stale peak is never
    eroded, so the steady-moderate rival can never out-drama it — proving the decay is
    what frees the cut."""
    d = FocusDirector(margin=FOCUS_MARGIN, decay=1.0)
    d.note(0, 0.0)
    assert d.note(1, 0.9) == 1
    # No matter how many moderate moves board 0 plays, board 1's 0.9 never fades.
    for _ in range(20):
        assert d.note(0, 0.3) is None
    assert d.focus == 1


def test_focus_decay_default_is_a_gentle_per_ply_factor():
    """The shipped default decays but stays well under 1 (a real recency window)."""
    assert 0.5 < FOCUS_DECAY < 1.0
    assert FocusDirector().decay == FOCUS_DECAY


# Player bias (tasks 0258/0262): `--board auto:<player>` softly biases the cut toward a
# named player's boards via an additive standing bonus, so a biased board wins ties/small
# margins while a rival must out-drama it by margin+bias to steal the cut.


def test_focus_bias_default_equals_the_margin():
    """The shipped bias is the margin, so a favoured board steals on a mere tie."""
    assert FOCUS_BIAS == FOCUS_MARGIN
    assert FOCUS_BIAS_BONUS == FOCUS_MARGIN


def test_parse_focus_bias_splits_auto_and_player():
    """`auto` -> follow-all no bias; `auto:<player>` -> follow-all biased to that name;
    anything else -> not auto (hard-filtered by the board selector instead)."""
    assert parse_focus_bias("auto") == (True, None)
    assert parse_focus_bias("AUTO") == (True, None)
    assert parse_focus_bias("auto:Carlsen") == (True, "Carlsen")
    assert parse_focus_bias("auto: Nakamura ") == (True, "Nakamura")
    assert parse_focus_bias("auto:") == (True, None)  # empty bias degrades to plain auto
    assert parse_focus_bias("Carlsen") == (False, None)
    assert parse_focus_bias("3") == (False, None)
    assert parse_focus_bias(None) == (False, None)


def test_focus_director_bias_wins_a_tie():
    """A biased rival steals the cut the moment it merely MATCHES the focus's drama —
    unbiased an exact tie can never out-drama the focus by the margin."""
    d = FocusDirector(margin=FOCUS_MARGIN, decay=1.0, bias={1: FOCUS_BIAS})
    d.note(0, 0.5)  # adopt board 0 (unbiased)
    # Board 1 ties board 0's 0.5 drama: standing 0.5 + bias(0.15) = 0.65 vs 0.5 -> cut.
    assert d.note(1, 0.5) == 1
    assert d.focus == 1


def test_focus_director_bias_wins_a_small_margin_unbiased_would_not_cut():
    """A biased board with only a hair of lead (well under the margin) steals — the same
    swing on an unbiased board leaves the focus put."""
    d = FocusDirector(margin=FOCUS_MARGIN, decay=1.0, bias={1: FOCUS_BIAS})
    d.note(0, 0.5)
    assert d.note(1, 0.55) == 1  # 0.55 + 0.15 - 0.5 = 0.20 >= margin -> cut
    # Contrast: no bias, the same 0.05 lead is under the margin, so no cut.
    d2 = FocusDirector(margin=FOCUS_MARGIN, decay=1.0)
    d2.note(0, 0.5)
    assert d2.note(1, 0.55) is None
    assert d2.focus == 0


def test_focus_director_bias_focus_held_until_rival_exceeds_margin_plus_bias():
    """A biased FOCUS board holds the cut until a rival out-dramas it by margin+bias —
    a bigger margin than the plain hysteresis, but still beatable (it's a bias, not a
    pin). The boundary swing steals; one hair under it does not."""
    d = FocusDirector(margin=FOCUS_MARGIN, decay=1.0, bias={0: FOCUS_BIAS})
    d.note(0, 0.5)  # board 0 is biased AND focus; effective standing 0.5 + 0.15 = 0.65
    # A rival needs to clear 0.65 by the margin -> reach 0.80 (= 0.5 + margin + bias).
    assert d.note(1, 0.79) is None  # just under: 0.79 - 0.65 = 0.14 < margin
    assert d.focus == 0
    assert d.note(1, 0.80) == 1  # at the threshold: 0.80 - 0.65 = 0.15 -> steal
    assert d.focus == 1


def test_focus_director_set_bias_registers_and_zero_is_a_noop():
    """`set_bias` records a per-board bonus; a zero bonus registers no preference."""
    d = FocusDirector()
    d.set_bias(2)  # defaults to FOCUS_BIAS
    assert d.bias == {2: FOCUS_BIAS}
    d.set_bias(3, 0.0)  # no-op
    assert 3 not in d.bias


# Caster pin (task 0259): a pin holds focus for N note() ticks regardless of rival
# drama, then auto-resumes drama-following; it also clears the moment the pinned
# board's game ends.


def test_focus_director_pin_holds_through_bigger_rival_drama():
    """While pinned, no rival can steal the cut no matter how dramatic it gets."""
    d = FocusDirector(margin=FOCUS_MARGIN)
    d.note(0, 0.0)  # adopt board 0
    assert d.pin(0, 2) is None  # pin the already-focused board: no cut emitted
    # Two huge swings on board 1 land inside the 2-ply pin window: both suppressed.
    assert d.note(1, 0.99) is None
    assert d.note(1, 0.99) is None
    assert d.focus == 0


def test_focus_director_pin_to_another_board_emits_a_cut():
    """Pinning a board other than the current focus moves the cut and returns it."""
    d = FocusDirector(margin=FOCUS_MARGIN)
    d.note(0, 0.0)
    assert d.pin(1, 3) == 1  # caster cuts to board 1 and pins it
    assert d.focus == 1
    assert d.pinned == 1


def test_focus_director_pin_expires_then_drama_following_resumes():
    """After the pin window elapses, normal margin logic resumes immediately."""
    d = FocusDirector(margin=FOCUS_MARGIN)
    d.note(0, 0.0)
    d.pin(0, 1)  # hold board 0 for exactly one tick
    assert d.note(1, 0.99) is None  # tick 1: suppressed by the pin
    assert d.pin_remaining == 0 and d.pinned is None  # pin has lifted
    # Next dramatic move on board 1 now steals focus under the usual margin rule.
    assert d.note(1, FOCUS_MARGIN + 0.1) == 1
    assert d.focus == 1


def test_focus_director_pin_cleared_on_result_of_pinned_board():
    """A result for the pinned board clears the pin so focus can auto-resume."""
    d = FocusDirector(margin=FOCUS_MARGIN)
    d.note(0, 0.0)
    d.pin(0, 5)  # long pin on board 0
    d.result(0)  # board 0's game ends -> pin lifts even though plies remained
    assert d.pinned is None and d.pin_remaining == 0
    assert d.note(1, FOCUS_MARGIN + 0.1) == 1  # drama-following resumes at once


def test_focus_director_result_for_other_board_keeps_the_pin():
    """A result for a *different* board must not disturb an active pin."""
    d = FocusDirector(margin=FOCUS_MARGIN)
    d.note(0, 0.0)
    d.pin(0, 5)
    d.result(1)  # some other board ended; our pin is untouched
    assert d.pinned == 0 and d.pin_remaining == 5
    assert d.note(1, 0.99) is None  # still suppressed by the pin


# Director cue (task 0260): every cut sets a human-readable `last_reason` explaining
# WHY focus moved, reusing the magnitudes already in `recent` so casters/captions can
# voice the cut. Board labels are 1-based ("Bd3" == 0-based index 2).
# --------------------------------------------------------------------------- #


def test_focus_director_cut_emits_a_reason_naming_board_and_magnitudes():
    """A drama cut records a cue that names the new (1-based) board and BOTH compared
    magnitudes — the erupting board's swing vs the board being left."""
    # decay=1.0 keeps the standing scores exact so the cue's magnitudes are unambiguous;
    # the cue always reports the *decayed* prev score (the same value the margin test uses).
    d = FocusDirector(margin=FOCUS_MARGIN, decay=1.0)
    d.note(0, 0.4)  # adopt board 0 with a moderate standing score
    assert d.last_reason is None  # silent adoption: no cut, no cue yet
    assert d.note(2, 0.9) == 2  # board 2 (index 2 -> "Bd3") erupts and steals focus
    reason = d.last_reason
    assert reason is not None
    assert "Bd3" in reason  # names the NEW board, 1-based
    assert "+0.9" in reason  # the erupting board's magnitude
    assert "+0.4" in reason  # the board being cut away from


def test_focus_director_reason_unset_when_no_cut():
    """Hysteresis blip below the margin leaves `last_reason` untouched (no cut fired)."""
    d = FocusDirector(margin=FOCUS_MARGIN)
    d.note(0, 0.0)
    assert d.note(1, FOCUS_MARGIN - 0.01) is None
    assert d.last_reason is None


def test_focus_director_pin_records_a_pin_reason():
    """Pinning a different board sets a caster-pin cue naming the pinned (1-based) board."""
    d = FocusDirector(margin=FOCUS_MARGIN)
    d.note(0, 0.0)
    assert d.pin(2, 3) == 2
    assert d.last_reason is not None and "Bd3" in d.last_reason
    assert "pin" in d.last_reason.lower()


def test_auto_follow_focus_event_carries_the_director_cue():
    """The overlay `focus` event threads the cue so the cut reason can be voiced/subtitled
    via the --captions-srt/vtt path."""
    events = _auto_overlay_events_for(_two_board_round_with_drama())
    focuses = [e for e in events if e.get("type") == "focus"]
    assert focuses, "a dramatic board must trigger a focus cut"
    reason = focuses[-1].get("reason")
    assert isinstance(reason, str) and reason, "focus event must carry a reason string"
    assert "Bd2" in reason  # cut to the dramatic board (0-based index 1 -> "Bd2")


# More player-bias coverage (tasks 0258/0262), adapted to the dict-bias FocusDirector API.


def test_focus_director_bias_yields_to_a_much_bigger_swing():
    """The bias is a thumb on the scale, not a hard filter: a rival board with a swing
    big enough to clear margin + bonus still steals focus off the favored board.
    ``decay=1.0`` keeps the held score steady so the threshold is exactly margin+bonus."""
    d = FocusDirector(bias={0: FOCUS_BIAS_BONUS}, decay=1.0)
    d.note(0, 0.5)  # adopt + bias the focused board 0
    # A sub-(margin+bonus) rival can't steal from the favored board...
    assert d.note(1, 0.5 + FOCUS_MARGIN + FOCUS_BIAS_BONUS - 0.01) is None
    assert d.focus == 0
    # ...but a swing clearing margin + bonus does.
    assert d.note(1, 0.5 + FOCUS_MARGIN + FOCUS_BIAS_BONUS + 0.01) == 1
    assert d.focus == 1


def test_focus_director_bias_still_fades_when_the_favorite_goes_quiet():
    """Bias is added on top of the DECAYED recency score, so a favored board that stops
    playing still loses the cut to a steadily-active rival (recency + bias compose)."""
    # Strong decay so the stale favorite erodes within a few ticks.
    d = FocusDirector(bias={0: FOCUS_BIAS_BONUS}, decay=0.5)
    d.note(0, 0.9)  # board 0 (the favorite) peaks, then goes silent
    stole = None
    for _ in range(6):
        # Board 1 plays steady moderate drama; its score holds while board 0's decays.
        stole = d.note(1, 0.5)
        if stole is not None:
            break
    assert stole == 1, "even a biased board must yield once its drama decays away"
    assert d.focus == 1


def test_focus_director_bias_routes_by_player_in_overlay():
    """End-to-end: `--board auto:Hero` biases the director toward Hero's board, so the
    favorite still cuts in via the overlay bridge (board 1's White is 'Hero')."""
    feed = _OneShotFeed(_two_board_round_with_drama())
    ingestor = BroadcastIngestor(feed, _model())
    events = [
        e
        for e in overlay_events(
            ingestor,
            auto_follow=True,
            bias_player="Hero",
            interval=0.0,
            sleep=lambda _: None,
        )
        if isinstance(e, dict)
    ]
    focuses = [e for e in events if e.get("type") == "focus"]
    assert focuses, "the biased dramatic board must still trigger a focus cut"
    assert focuses[-1]["board"] == 1


# A two-board round where board 0 is a quiet opening (no drama) and board 1 is a
# scholar's mate (a huge final-move swing the material baseline scores as drama). Tiny
# illustrative fixture for the routing unit test only — NOT thesis evidence.
def _two_board_round_with_drama():
    drama = """[Event "Test Broadcast"]
[Site "https://lichess.org/dramagame"]
[White "Hero"]
[Black "Victim"]
[Round "1"]
[WhiteElo "1500"]
[BlackElo "1500"]
[Result "1-0"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""
    return GAME_PGN + "\n" + drama


def _auto_overlay_events_for(snapshot, bias_player=None):
    """Drive a snapshot through the overlay bridge with `--board auto` on (optionally
    softly biased toward ``bias_player``'s boards, task 0262)."""
    feed = _OneShotFeed(snapshot)
    ingestor = BroadcastIngestor(feed, _model())
    return [
        e
        for e in overlay_events(
            ingestor,
            auto_follow=True,
            bias_player=bias_player,
            interval=0.0,
            sleep=lambda _: None,
        )
        if isinstance(e, dict)
    ]


def test_auto_follow_emits_focus_cut_to_the_dramatic_board():
    """`--board auto` emits a `focus` event cutting to the board with the biggest recent
    swing (the scholar's-mate board), routed to its game_id."""
    events = _auto_overlay_events_for(_two_board_round_with_drama())
    focuses = [e for e in events if e.get("type") == "focus"]
    assert focuses, "a dramatic board must trigger a focus cut"
    last = focuses[-1]
    assert last["board"] == 1, "focus should land on the dramatic board (index 1)"
    assert last["game_id"] == "https://lichess.org/dramagame"


def test_auto_follow_off_emits_no_focus_events():
    """Without `--board auto` the bridge emits no focus events (the default follow-all)."""
    events = _overlay_events_for(_two_board_round_with_drama())
    assert not any(e.get("type") == "focus" for e in events)


def test_auto_bias_registers_bias_for_the_named_players_board(monkeypatch):
    """`--board auto:<player>` wires a standing bias onto exactly the board(s) featuring
    that player — matched on either side's name-plate as the round is announced (task
    0262); a non-matching name registers nothing."""
    import chess_equity.broadcast as bx

    calls = []
    orig = bx.FocusDirector.set_bias

    def spy(self, board, bonus=bx.FOCUS_BIAS):
        calls.append(board)
        return orig(self, board, bonus)

    monkeypatch.setattr(bx.FocusDirector, "set_bias", spy)
    # _two_board_round: board 0 = Carlsen/Nakamura, board 1 = Ding/Firouzja.
    _auto_overlay_events_for(_two_board_round(), bias_player="ding")
    assert calls == [1], "only board 1 features Ding"

    calls.clear()
    _auto_overlay_events_for(_two_board_round(), bias_player="nobody")
    assert calls == [], "no board features 'nobody' -> no bias registered"


def test_auto_bias_still_yields_to_a_big_rival_swing():
    """The bias is soft, not a hard pin: biasing toward the QUIET board's player does not
    keep the cut there — board 1's scholar's-mate swing is big enough to steal anyway."""
    events = _auto_overlay_events_for(
        _two_board_round_with_drama(), bias_player="Carlsen"  # the quiet board 0's player
    )
    focuses = [e for e in events if e.get("type") == "focus"]
    assert focuses, "a big-enough rival swing must still cut away despite the bias"
    assert focuses[-1]["board"] == 1


# --------------------------------------------------------------------------- #
# Caster pin INPUT channel (task 0261)
# --------------------------------------------------------------------------- #


def test_pin_channel_delivers_directives_fifo_then_empties():
    """The channel is a FIFO mailbox: drain returns every queued directive in order
    (a None board is an unpin) and leaves the channel empty."""
    ch = PinChannel()
    ch.submit(1, 3)
    ch.submit(0, 5)
    ch.submit(None)  # explicit unpin
    assert ch.drain() == [(1, 3), (0, 5), (None, 0)]
    assert ch.drain() == []  # draining empties it


def test_overlay_events_applies_a_channel_pin_suppressing_the_auto_cut():
    """A pin delivered out-of-band on the channel reaches the live director: it cuts to
    the pinned (quiet) board and holds it, so the dramatic board never steals focus —
    the auto-cut that `test_auto_follow_emits_focus_cut_to_the_dramatic_board` proves."""
    ch = PinChannel()
    ch.submit(0, 50)  # caster pins the quiet board before the scholar's-mate fires
    feed = _OneShotFeed(_two_board_round_with_drama())
    ingestor = BroadcastIngestor(feed, _model())
    events = [
        e
        for e in overlay_events(
            ingestor, auto_follow=True, pin_channel=ch, interval=0.0, sleep=lambda _: None
        )
        if isinstance(e, dict)
    ]
    focuses = [e for e in events if e.get("type") == "focus"]
    assert focuses, "the channel pin must emit a focus cut to the pinned board"
    assert all(f["board"] == 0 for f in focuses), "the pin holds board 0; drama can't steal it"
    assert focuses[0]["game_id"] == "https://lichess.org/abcd1234"


def test_overlay_events_ignores_a_channel_when_auto_follow_is_off():
    """No director without `--board auto`, so channel pins are inert (no focus events)."""
    ch = PinChannel()
    ch.submit(1, 5)
    feed = _OneShotFeed(_two_board_round_with_drama())
    ingestor = BroadcastIngestor(feed, _model())
    events = [
        e
        for e in overlay_events(
            ingestor, auto_follow=False, pin_channel=ch, interval=0.0, sleep=lambda _: None
        )
        if isinstance(e, dict)
    ]
    assert not any(e.get("type") == "focus" for e in events)


def test_sse_server_accepts_a_pin_post_onto_the_channel():
    """`POST /pin {board, plies}` queues the directive on the shared channel (HTTP input
    side of the caster pin), returning 204."""
    import threading
    import urllib.request

    from chess_equity.broadcast import make_sse_server, overlay_events

    ch = PinChannel()
    server = make_sse_server(
        lambda: overlay_events(BroadcastIngestor(LocalPgnFeed(GAME_PGN), _model()), interval=0),
        port=0,
        pin_channel=ch,
    )
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/pin",
            data=b'{"board": 1, "plies": 4}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=10)
        assert resp.status == 204
    finally:
        server.shutdown()
        server.server_close()
    assert ch.drain() == [(1, 4)]


def test_sse_server_pin_post_rejects_bad_json():
    """A malformed pin body is a 400, not a crash or a silent drop."""
    import threading
    import urllib.error
    import urllib.request

    from chess_equity.broadcast import make_sse_server, overlay_events

    ch = PinChannel()
    server = make_sse_server(
        lambda: overlay_events(BroadcastIngestor(LocalPgnFeed(GAME_PGN), _model()), interval=0),
        port=0,
        pin_channel=ch,
    )
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/pin", data=b"not json", method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
    assert ch.drain() == []  # nothing queued from a rejected body


def test_sse_server_404s_pin_post_when_no_channel():
    """Without a pin channel the server has no input side: `POST /pin` is a 404."""
    import threading
    import urllib.error
    import urllib.request

    from chess_equity.broadcast import make_sse_server, overlay_events

    server = make_sse_server(
        lambda: overlay_events(BroadcastIngestor(LocalPgnFeed(GAME_PGN), _model()), interval=0),
        port=0,
    )
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/pin", data=b"{}", method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- #
# GET /focus status readback — caster OUTPUT side of the pin conduit (task 0265)
# --------------------------------------------------------------------------- #


def test_focus_status_holds_the_live_cut_or_none_until_set():
    """The thread-safe holder is empty until the first board is live, then snapshots
    the live cut as ``{board, label, reason}``."""
    from chess_equity.broadcast import FocusStatus

    fs = FocusStatus()
    assert fs.get() is None
    fs.set(2, "Bd3", "cut to Bd3: +0.9 swing vs +0.4")
    assert fs.get() == {
        "board": 2,
        "label": "Bd3",
        "reason": "cut to Bd3: +0.9 swing vs +0.4",
    }
    # A later cut overwrites; reason may be None (silent first-board adoption).
    fs.set(0, "Bd1", None)
    assert fs.get() == {"board": 0, "label": "Bd1", "reason": None}


def test_sse_server_serves_get_focus_status_json():
    """`GET /focus` returns the live director state as JSON when a status channel is
    wired (the caster control-surface readback, task 0265)."""
    import json as _json
    import threading
    import urllib.request

    from chess_equity.broadcast import FocusStatus, make_sse_server, overlay_events

    fs = FocusStatus()
    fs.set(1, "Bd2", "cut to Bd2: +0.7 swing vs +0.2")
    server = make_sse_server(
        lambda: overlay_events(BroadcastIngestor(LocalPgnFeed(GAME_PGN), _model()), interval=0),
        port=0,
        focus_status=fs,
    )
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/focus", timeout=10)
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "application/json"
        body = _json.loads(resp.read())
    finally:
        server.shutdown()
        server.server_close()
    assert body == {"board": 1, "label": "Bd2", "reason": "cut to Bd2: +0.7 swing vs +0.2"}


def test_sse_server_get_focus_reports_no_live_board_before_first_cut():
    """Before any board is live, `GET /focus` is still valid JSON with null fields
    (so a caster UI renders "no board yet" rather than erroring)."""
    import json as _json
    import threading
    import urllib.request

    from chess_equity.broadcast import FocusStatus, make_sse_server, overlay_events

    server = make_sse_server(
        lambda: overlay_events(BroadcastIngestor(LocalPgnFeed(GAME_PGN), _model()), interval=0),
        port=0,
        focus_status=FocusStatus(),
    )
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/focus", timeout=10)
        body = _json.loads(resp.read())
    finally:
        server.shutdown()
        server.server_close()
    assert body == {"board": None, "label": None, "reason": None}


def test_sse_server_404s_get_focus_when_no_status_channel():
    """Without a focus-status channel the server has no readback side: `GET /focus`
    is a 404 (mirrors the pinless `POST /pin` 404)."""
    import threading
    import urllib.error
    import urllib.request

    from chess_equity.broadcast import make_sse_server, overlay_events

    server = make_sse_server(
        lambda: overlay_events(BroadcastIngestor(LocalPgnFeed(GAME_PGN), _model()), interval=0),
        port=0,
    )
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/focus", timeout=10)
        assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_overlay_events_publishes_the_live_cut_to_focus_status():
    """End-to-end: a multi-board stream's `focus` cuts are mirrored onto the shared
    FocusStatus, so `GET /focus` reflects the director without reaching into it. The
    drama board steals the cut, so the status ends on it with the cut reason."""
    from chess_equity.broadcast import FocusStatus

    fs = FocusStatus()
    feed = _OneShotFeed(_two_board_round_with_drama())
    ingestor = BroadcastIngestor(feed, _model())
    events = [
        e
        for e in overlay_events(
            ingestor,
            auto_follow=True,
            focus_status=fs,
            interval=0.0,
            sleep=lambda _: None,
        )
        if isinstance(e, dict)
    ]
    focuses = [e for e in events if e.get("type") == "focus"]
    assert focuses, "the drama board must steal the cut"
    snap = fs.get()
    assert snap is not None
    # Status mirrors the last emitted focus cut: same board, caster label, and reason.
    last = focuses[-1]
    assert snap["board"] == last["board"]
    assert snap["label"] == f"Bd{last['board'] + 1}"
    assert snap["reason"] == last.get("reason")


# --------------------------------------------------------------------------- #
# Game-over signal -> overlay auto-advance off a finished board (task 0189), more cases
# --------------------------------------------------------------------------- #


def _round_board0_finished():
    """A two-board round where board 0 (Carlsen) has a terminal Result and board 1
    (Ding) is still ongoing — so the bridge should emit a result event for board 0."""
    finished = GAME_PGN.replace('[Result "*"]', '[Result "1-0"]').rstrip().rstrip("*").rstrip() + " 1-0\n"
    ongoing = GAME_PGN.replace("Carlsen", "Ding").replace("Nakamura", "Firouzja").replace(
        'Site "https://lichess.org/abcd1234"', 'Site "https://lichess.org/wxyz9999"'
    )
    return finished + "\n" + ongoing


def test_overlay_bridge_emits_result_event_for_a_finished_board():
    """When a multi-board game reaches a terminal PGN Result, the bridge emits a
    `{type:"result", board, result}` event so the overlay can advance focus off it."""
    events = _overlay_events_for(_round_board0_finished())
    results = [e for e in events if e.get("type") == "result"]
    assert results, "a finished board must emit a result event"
    assert results[0]["board"] == 0, "the result names the finished board's index"
    assert results[0]["result"] == "1-0"
    # The still-ongoing board emits no result event.
    assert {r["board"] for r in results} == {0}


def test_overlay_bridge_emits_result_event_once_per_finished_game():
    """A finished game's result fires once, not on every subsequent poll."""
    # Poll the same finished snapshot twice; the result event must not repeat.
    feed = _ScriptedFeed([_round_board0_finished(), _round_board0_finished(), None])
    ingestor = BroadcastIngestor(feed, _model())
    events = [
        e
        for e in overlay_events(ingestor, interval=0.0, sleep=lambda _: None)
        if isinstance(e, dict)
    ]
    results = [e for e in events if e.get("type") == "result"]
    assert len(results) == 1, "the result event fires exactly once for the finished game"


def test_single_game_feed_emits_no_result_event():
    """Single-game feeds are unchanged: no board index, so no result event even when
    the game has a terminal Result."""
    finished = GAME_PGN.replace('[Result "*"]', '[Result "1-0"]').rstrip().rstrip("*").rstrip() + " 1-0\n"
    events = _overlay_events_for(finished)
    assert not any(e.get("type") == "result" for e in events), "single game emits no result event"


# --------------------------------------------------------------------------- #
# Drama classifier attached to the overlay event (task 0053)
# --------------------------------------------------------------------------- #


def _move_event(delta_equity, *, equity=60.0, white_clock=120.0):
    """A MoveEvent with a tunable mover-POV swing (White to have just moved)."""
    return MoveEvent(
        game_id="g",
        ply=10,
        san="Qxf7",
        uci="d1f7",
        fen="8/8/8/8/8/8/8/8 b - - 0 1",
        white_to_move=False,  # Black to move => White just moved (the mover)
        white_clock=white_clock,
        black_clock=120.0,
        white_elo=1500,
        black_elo=1500,
        equity=equity,
        delta_equity=delta_equity,
        last_move_grade=grade_delta(delta_equity),
        source="test",
        compute_ms=0.0,
    )


def test_overlay_event_attaches_real_drama_on_a_sharp_swing():
    # A +15pt mover swing is a "clutch" find for chess_equity.drama.
    event = _move_event(15.0).to_overlay_event()
    assert "drama" in event
    drama = event["drama"]
    assert drama["kind"] == "clutch"
    assert drama["headline"]  # caster-facing one-liner
    assert 0.0 <= drama["magnitude"] <= 1.0  # matches overlay.js / test_overlay schema


def test_overlay_event_has_no_drama_when_quiet():
    # A small swing isn't highlight-worthy: no drama payload (overlay won't flare).
    event = _move_event(1.0).to_overlay_event()
    assert "drama" not in event


def test_move_event_is_json_serializable():
    import json

    feed = LocalPgnFeed(GAME_PGN)
    ingestor = BroadcastIngestor(feed, _model())
    events = []
    ingestor.run(events.append, interval=0.0, sleep=lambda _: None)
    blob = json.dumps(events[0].to_dict())
    assert "fen" in blob
    assert isinstance(events[0], MoveEvent)


# --------------------------------------------------------------------------- #
# SSE bridge: a round straight into the overlay (task 0094)
# --------------------------------------------------------------------------- #


def test_sse_frame_is_a_single_data_line_with_blank_terminator():
    from chess_equity.broadcast import sse_frame

    frame = sse_frame({"type": "position", "ply": 3})
    assert frame == 'data: {"type": "position", "ply": 3}\n\n'


def test_overlay_events_emits_game_metadata_before_its_positions():
    from chess_equity.broadcast import overlay_events

    ingestor = BroadcastIngestor(LocalPgnFeed(GAME_PGN), _model())
    events = list(overlay_events(ingestor, interval=0, max_polls=None, max_idle_polls=1))
    # First the one-time game event, then a position per ply (4 in GAME_PGN).
    assert events[0]["type"] == "game"
    assert events[0]["players"]["white"]["name"] == "Carlsen"
    positions = [e for e in events if e["type"] == "position"]
    assert [e["ply"] for e in positions] == [1, 2, 3, 4]
    # Overlay schema: White-POV equity in [0, 1], not the flat [0, 100] internal form.
    assert all(0.0 <= e["equity"] <= 1.0 for e in positions)


def test_serve_sse_streams_a_replay_to_an_eventsource_client():
    import json
    import threading
    import urllib.request

    from chess_equity.broadcast import make_sse_server, overlay_events

    def make_events():
        return overlay_events(
            BroadcastIngestor(LocalPgnFeed(GAME_PGN), _model()),
            interval=0,
            max_polls=None,
            max_idle_polls=1,
        )

    server = make_sse_server(make_events, port=0)  # OS-assigned port
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/sse", timeout=10).read().decode()
    finally:
        server.shutdown()
        server.server_close()

    frames = [f for f in body.split("\n\n") if f.strip()]
    parsed = [json.loads(f[len("data: "):]) for f in frames]
    assert parsed[0]["type"] == "game"
    assert [e["type"] for e in parsed[1:]] == ["position"] * 4


def test_sse_server_404s_non_sse_paths_when_no_static_dir():
    import threading
    import urllib.error
    import urllib.request

    from chess_equity.broadcast import make_sse_server, overlay_events

    server = make_sse_server(
        lambda: overlay_events(BroadcastIngestor(LocalPgnFeed(GAME_PGN), _model()), interval=0),
        port=0,
        directory=None,
    )
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html", timeout=10)
        assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- #
# SSE live-wait + heartbeat (task 0099)
# --------------------------------------------------------------------------- #


class _ScriptedFeed(BroadcastFeed):
    """A feed that returns a fixed sequence of snapshots ('' = idle), then nothing."""

    def __init__(self, snapshots):
        self._snaps = list(snapshots)
        self._i = 0

    def poll(self):
        if self._i < len(self._snaps):
            snap = self._snaps[self._i]
            self._i += 1
            return snap
        return None


def test_stream_emits_heartbeat_tick_on_idle_poll_when_enabled():
    feed = _ScriptedFeed(["", GAME_PGN])  # poll 1 idle, poll 2 has the game
    ingestor = BroadcastIngestor(feed, _model())
    items = list(
        ingestor.stream(
            interval=0, max_polls=2, max_idle_polls=None, heartbeat=True, sleep=lambda s: None
        )
    )
    assert items[0] is None  # idle poll → heartbeat tick (doesn't end the stream)
    assert any(isinstance(i, MoveEvent) for i in items)  # then the real moves arrive


def test_stream_has_no_heartbeat_tick_by_default():
    # Default heartbeat=False keeps the historical MoveEvent-only contract.
    feed = _ScriptedFeed(["", GAME_PGN])
    ingestor = BroadcastIngestor(feed, _model())
    items = list(
        ingestor.stream(interval=0, max_polls=2, max_idle_polls=None, sleep=lambda s: None)
    )
    assert all(isinstance(i, MoveEvent) for i in items)


def test_overlay_events_maps_idle_poll_to_heartbeat_sentinel():
    from chess_equity.broadcast import HEARTBEAT, overlay_events

    feed = _ScriptedFeed(["", GAME_PGN])
    ingestor = BroadcastIngestor(feed, _model())
    items = list(
        overlay_events(
            ingestor, interval=0, max_polls=2, max_idle_polls=None, heartbeat=True, sleep=lambda s: None
        )
    )
    assert items[0] is HEARTBEAT
    assert any(isinstance(i, dict) and i.get("type") == "position" for i in items)


def test_serve_sse_sends_keepalive_comment_during_idle():
    import threading
    import urllib.request

    from chess_equity.broadcast import make_sse_server, overlay_events

    def make_events():
        feed = _ScriptedFeed(["", GAME_PGN])
        return overlay_events(
            BroadcastIngestor(feed, _model()),
            interval=0,
            max_polls=2,
            max_idle_polls=None,
            heartbeat=True,
            sleep=lambda s: None,
        )

    server = make_sse_server(make_events, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/sse", timeout=10).read().decode()
    finally:
        server.shutdown()
        server.server_close()
    assert ": keepalive" in body  # the idle-poll heartbeat comment
    assert '"type": "position"' in body  # and the real move frames


# --------------------------------------------------------------------------- #
# Clock-aware equity warp (task 0097)
# --------------------------------------------------------------------------- #

# A bullet game (TimeControl -> bullet bucket, the deadliest multiplier) whose clocks
# fall to a few seconds — the regime where time pressure dominates practical results.
LOW_CLOCK_PGN = """[Event "Time scramble"]
[Site "https://lichess.org/lc000001"]
[White "A"]
[Black "B"]
[Round "1"]
[WhiteElo "2000"]
[BlackElo "2000"]
[TimeControl "60+0"]
[Result "*"]

1. e4 { [%clk 0:00:05] } e5 { [%clk 0:00:04] } 2. Nf3 { [%clk 0:00:03] } *
"""


def _last_equity(pgn, *, clock_aware):
    tracker = GameTracker("g", _model(), white_elo=2000, black_elo=2000, clock_aware=clock_aware)
    return tracker.ingest(pgn)[-1].equity


def test_clock_aware_shifts_equity_on_low_clock():
    # Acceptance: a low-clock position's published bar differs from the clock-blind value.
    blind = _last_equity(LOW_CLOCK_PGN, clock_aware=False)
    aware = _last_equity(LOW_CLOCK_PGN, clock_aware=True)
    assert abs(aware - blind) > 1.0, "seconds-left bullet position should move the bar"
    assert 0.0 <= aware <= 100.0
    # After 2.Nf3 Black is to move with ~4s: Black's practical chances fall, so the
    # White-POV bar rises above the clock-blind reading.
    assert aware > blind


def test_clock_aware_is_default_on():
    # The flag defaults on, so the plain tracker already warps a low-clock bar.
    default = _last_equity(LOW_CLOCK_PGN, clock_aware=True)
    explicit_off = _last_equity(LOW_CLOCK_PGN, clock_aware=False)
    assert default != explicit_off


def test_clock_aware_negligible_with_minutes_left():
    # The comfortable-clock GAME_PGN (3 minutes) should warp essentially not at all —
    # it's the *low* clock that matters, not merely the presence of clocks.
    blind = _last_equity(GAME_PGN, clock_aware=False)
    aware = _last_equity(GAME_PGN, clock_aware=True)
    assert abs(aware - blind) < 0.5


def test_clock_blind_when_no_clk_tags():
    # No [%clk] tags -> the side-to-move clock is None -> a no-op even with clock_aware on.
    no_clk = LOW_CLOCK_PGN.replace(" { [%clk 0:00:05] }", "").replace(
        " { [%clk 0:00:04] }", ""
    ).replace(" { [%clk 0:00:03] }", "")
    assert _last_equity(no_clk, clock_aware=True) == _last_equity(no_clk, clock_aware=False)


# --------------------------------------------------------------------------- #
# Objective cp fallback for cp-less models (task 0103)
# --------------------------------------------------------------------------- #

from chess_equity.adapters import EquityModel, ObjectiveEngine, ObjectiveEval
from chess_equity.types import WDL, Equity


class _CpLessModel(EquityModel):
    """A rating-blind model whose Equity carries no cp (mimics maia2's win-prob)."""

    def evaluate(self, fen, white_elo, black_elo):
        wdl = WDL.from_unnormalized(p_win=1 / 3, p_draw=1 / 3, p_loss=1 / 3)
        return Equity(wdl=wdl, equity_white=50.0, source="cpless", cp=None)


class _StubEngine(ObjectiveEngine):
    """Counts calls and returns a fixed side-to-move cp (or a mate -> cp None)."""

    def __init__(self, cp_value):
        self.cp_value = cp_value
        self.calls = 0

    def eval(self, fen):
        self.calls += 1
        return ObjectiveEval(cp=self.cp_value)


def test_cp_fallback_populates_cp_for_cpless_model():
    # Acceptance: a cp-less model + an objective engine -> every event carries a cp,
    # so the overlay's ghost tick + divergence badge work on a maia2-style feed.
    engine = _StubEngine(123.0)
    tracker = GameTracker("g", _CpLessModel(), white_elo=2000, black_elo=2000, engine=engine)
    events = tracker.ingest(GAME_PGN)
    assert events
    assert all(e.cp is not None for e in events)
    # cp is rendered White-POV (the engine's side-to-move cp is flipped per ply), so the
    # magnitude is the stub value while the sign alternates.
    assert all(abs(e.cp) == 123.0 for e in events)
    assert engine.calls == len(events)


def test_cpless_model_without_engine_leaves_cp_none():
    # No engine -> nothing to fall back to; cp stays None (unchanged behaviour).
    tracker = GameTracker("g", _CpLessModel(), white_elo=2000, black_elo=2000)
    events = tracker.ingest(GAME_PGN)
    assert events and all(e.cp is None for e in events)


def test_model_with_cp_does_not_consult_engine():
    # A model that already exposes cp (the baseline) must not call the fallback engine.
    engine = _StubEngine(123.0)
    tracker = GameTracker("g", _model(), white_elo=2000, black_elo=2000, engine=engine)
    events = tracker.ingest(GAME_PGN)
    assert events
    assert engine.calls == 0
    assert any(e.cp is not None for e in events)


# --------------------------------------------------------------------------- #
# Clock-aware Δequity move grade (task 0106)
# --------------------------------------------------------------------------- #


def _last_delta(pgn, *, clock_aware):
    tracker = GameTracker("g", _model(), white_elo=2000, black_elo=2000, clock_aware=clock_aware)
    return tracker.ingest(pgn)[-1].delta_equity


def test_clock_aware_delta_differs_from_blind_on_low_clock():
    # The grade is computed from delta_equity, so a clock-warped delta is the grade going
    # clock-aware. In the bullet scramble the warp is live, so the mover's swing -- the
    # practical win-chance change the grade reflects -- differs from the clock-blind delta.
    blind = _last_delta(LOW_CLOCK_PGN, clock_aware=False)
    aware = _last_delta(LOW_CLOCK_PGN, clock_aware=True)
    assert blind is not None and aware is not None
    assert abs(aware - blind) > 0.5, "seconds-left bullet move should move the graded swing"


# --------------------------------------------------------------------------- #
# Scramble drama reachable on a low-clock broadcast replay (task 0108)
# --------------------------------------------------------------------------- #

# A scramble is a *modest* positional swing gated by the clock (SCRAMBLE_DELTA=6.5 <
# CLUTCH_DELTA=10, calibrated in task 0170), so to land in that band reliably we drive a
# stub model with a
# controlled ~7pt swing rather than a real engine (which on these toy positions only
# returns 0 or a large swing that escalates to clutch/escape). The point under test is
# the wiring: clocks now ride on every MoveEvent (0097), so the clock-gated scramble
# branch in chess_equity.drama is finally reachable on a replay.
class _ScrambleStubModel(EquityModel):
    """Returns 50% everywhere except the post-Nf3 position (knight on f3 -> '5N2' in the
    FEN), which reads 57% — a +7pt White swing for that one ply."""

    def evaluate(self, fen, white_elo, black_elo):
        equity_white = 57.0 if "5N2" in fen.split(" ")[0] else 50.0
        wdl = WDL.from_unnormalized(equity_white / 100.0, 0.0, 1.0 - equity_white / 100.0)
        return Equity(wdl=wdl, equity_white=equity_white, source="stub")


# White is down to 8s when it plays the +7pt Nf3; Black has comfortable time on its move.
SCRAMBLE_PGN = """[Event "Bullet scramble"]
[White "A"]
[Black "B"]
[WhiteElo "2000"]
[BlackElo "2000"]
[TimeControl "60+0"]
[Result "*"]

1. e4 { [%clk 0:00:10] } e5 { [%clk 0:00:30] } 2. Nf3 { [%clk 0:00:08] } *
"""


def test_low_clock_replay_surfaces_scramble_drama():
    # Clock-blind delta keeps the swing positional (clock_aware warps the bar itself,
    # which on a seconds-left game escalates every move to escape/clutch and swamps the
    # modest scramble band — see drama.py). The clock still rides on the MoveEvent, so
    # the clock-gated scramble branch fires.
    from chess_equity.drama import detect

    tracker = GameTracker(
        "g", _ScrambleStubModel(), white_elo=2000, black_elo=2000, clock_aware=False
    )
    events = tracker.ingest(SCRAMBLE_PGN)
    dramas = detect(events)
    kinds = {(d.ply, d.kind) for d in dramas}
    assert (3, "scramble") in kinds, f"expected a scramble on the low-clock Nf3, got {kinds}"
    scramble = next(d for d in dramas if d.kind == "scramble")
    assert scramble.mover_clock is not None and scramble.mover_clock < 20
    assert scramble.headline  # caster-facing one-liner


def test_clock_aware_grade_degrades_to_raw_when_blind():
    # The clock-blind path must still grade the *raw* equity delta unchanged: with the
    # warp off, before/after both pass through _clock_warp untouched, so the delta (and
    # thus the grade) is exactly the positional swing -- no clock contribution leaks in.
    blind = GameTracker("g", _model(), white_elo=2000, black_elo=2000, clock_aware=False)
    events = blind.ingest(LOW_CLOCK_PGN)
    last = events[-1]
    assert last.last_move_grade == grade_delta(last.delta_equity)
    # And a clock-blind run equals a no-[%clk] run's delta: clocks only ever enter via warp.
    no_clk = LOW_CLOCK_PGN.replace(" { [%clk 0:00:05] }", "").replace(
        " { [%clk 0:00:04] }", ""
    ).replace(" { [%clk 0:00:03] }", "")
    assert _last_delta(no_clk, clock_aware=True) == last.delta_equity


# Post-round director-cut recap markdown (task 0265)
# ---------------------------------------------------------------------------
def _pos(ply):
    """A minimal overlay `position` event carrying just the ply the recap needs."""
    return {"type": "position", "ply": ply, "move": {"san": "e4"}}


def test_focus_recap_md_tabulates_cuts_with_next_move_ply():
    """A recap row per `focus` event: # / ply (the move it precedes) / board / reason."""
    events = [
        _pos(1),  # board 0 adopted silently — no focus event before it
        _pos(2),
        {"type": "focus", "board": 2, "game_id": "g3", "reason": "cut to Bd3: +0.9 swing vs +0.4"},
        _pos(3),  # the dramatic move the cut precedes -> ply 3
        _pos(4),
        {"type": "focus", "board": 0, "game_id": "g1", "reason": "cut to Bd1: +1.2 swing vs +0.3"},
        _pos(5),
    ]
    md = focus_recap_md(events)
    lines = md.splitlines()
    assert lines[0] == "| # | Ply | Board | Reason |"
    assert lines[1] == "| --- | --- | --- | --- |"
    assert lines[2] == "| 1 | 3 | Bd3 | cut to Bd3: +0.9 swing vs +0.4 |"
    assert lines[3] == "| 2 | 5 | Bd1 | cut to Bd1: +1.2 swing vs +0.3 |"
    assert md.endswith("\n")
    assert len(lines) == 4  # header + separator + two cuts


def test_focus_recap_md_empty_stream_is_header_only():
    """No events (or no cuts) -> a paste-able header-only table, never a crash."""
    assert focus_recap_md([]) == (
        "| # | Ply | Board | Reason |\n| --- | --- | --- | --- |\n"
    )
    # A stream of positions with no focus events is still header-only.
    assert focus_recap_md([_pos(1), _pos(2)]).splitlines() == [
        "| # | Ply | Board | Reason |",
        "| --- | --- | --- | --- |",
    ]


def test_focus_recap_md_pin_without_following_move_falls_back_to_last_ply():
    """A caster pin landing on an idle tick (no later position) reuses the last ply seen,
    and its `caster pin` reason rides through verbatim."""
    events = [
        _pos(7),
        {"type": "focus", "board": 1, "game_id": "g2", "reason": "caster pin: hold Bd2"},
    ]
    rows = focus_recap_md(events).splitlines()[2:]
    assert rows == ["| 1 | 7 | Bd2 | caster pin: hold Bd2 |"]


def test_focus_recap_md_escapes_pipes_and_tolerates_missing_fields():
    """A `|` in a reason is escaped so it can't break the table; heartbeats/None reason
    are tolerated (sentinel non-dicts skipped, missing reason -> empty cell)."""
    events = [
        "HEARTBEAT",  # sentinel non-dict event from an idle poll
        {"type": "focus", "board": 0, "reason": "cut to Bd1: a|b"},
        _pos(4),
        {"type": "focus", "board": 1},  # no reason field
        _pos(5),
    ]
    rows = focus_recap_md(events).splitlines()[2:]
    assert rows[0] == "| 1 | 4 | Bd1 | cut to Bd1: a\\|b |"
    assert rows[1] == "| 2 | 5 | Bd2 |  |"
