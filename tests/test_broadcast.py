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
    GameEvent,
    GameTracker,
    LocalPgnFeed,
    MoveEvent,
    game_event,
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


def _last_delta(pgn, *, clock_aware):
    tracker = GameTracker("g", _model(), white_elo=2000, black_elo=2000, clock_aware=clock_aware)
    return tracker.ingest(pgn)[-1].delta_equity


def test_clock_aware_shifts_move_grade_delta_on_low_clock():
    # Task 0106: the Δequity move grade is clock-adjusted at both endpoints, so a low-clock
    # move's delta differs from the clock-blind (raw positional) delta. The last move in
    # LOW_CLOCK_PGN (2.Nf3, White at ~3s) warps White's pre- and post-move equity by its
    # own seconds-left clock, so the surviving move's grade shifts.
    blind = _last_delta(LOW_CLOCK_PGN, clock_aware=False)
    aware = _last_delta(LOW_CLOCK_PGN, clock_aware=True)
    assert abs(aware - blind) > 0.5, "low-clock survival should shift the grade vs clock-blind"
