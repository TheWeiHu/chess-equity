"""Tests for the broadcast → overlay SSE bridge (task 0021).

Covers the three things the bridge must get right:
  1. pure translation (percentage-point equity -> overlay [0,1] fraction);
  2. an end-to-end replay through a real ingestor yields the overlay's
     game/position events in order;
  3. the SSE server actually pushes those frames over a socket — no mock-game.json.

All offline: a finished PGN is replayed as if live.
"""

import dataclasses
import json
import threading
import urllib.request

import pytest

from chess_equity.broadcast import (
    BroadcastIngestor,
    LocalPgnFeed,
    MoveEvent,
)
from chess_equity.models import LichessBaselineModel
from chess_equity.overlay import (
    game_event,
    position_event,
    serve_overlay,
    stream_overlay_events,
)

# A short game with clock tags + ratings — the shape a Lichess broadcast emits.
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


def _ingestor():
    feed = LocalPgnFeed(GAME_PGN, moves_per_poll=1)
    return BroadcastIngestor(feed, LichessBaselineModel())


# --------------------------------------------------------------------------- #
# Pure translation
# --------------------------------------------------------------------------- #


_BASE_EVENT = MoveEvent(
    game_id="g1",
    ply=44,
    san="Rxd5",
    uci="d1d5",
    fen="8/8/8/3R4/8/8/8/8 b - - 0 1",
    white_to_move=False,
    white_clock=13.2,
    black_clock=1.6,
    white_elo=2850,
    black_elo=2780,
    equity=88.0,
    delta_equity=-22.0,
    last_move_grade="blunder",
    source="LichessBaselineModel",
    compute_ms=0.1,
)


def _move_event(**over):
    return dataclasses.replace(_BASE_EVENT, **over)


def test_position_event_rescales_to_overlay_fraction():
    out = position_event(_move_event())
    assert out["type"] == "position"
    assert out["ply"] == 44
    assert out["move"] == {"san": "Rxd5"}
    # 88.0 pct points -> 0.88 fraction; -22.0 -> -0.22.
    assert out["equity"] == pytest.approx(0.88)
    assert out["clock"] == {"white": 13.2, "black": 1.6}
    assert out["grade"] == {"label": "blunder", "delta": pytest.approx(-0.22)}
    # No centipawn from the equity models -> no ghost tick.
    assert "cp" not in out


def test_position_event_omits_missing_optional_fields():
    out = position_event(
        _move_event(white_clock=None, black_clock=None, last_move_grade=None, delta_equity=None)
    )
    assert "clock" not in out
    assert "grade" not in out
    assert out["equity"] == pytest.approx(0.88)


def test_game_event_carries_ratings():
    out = game_event(_move_event())
    assert out["type"] == "game"
    assert out["players"] == {"white": {"rating": 2850}, "black": {"rating": 2780}}


# --------------------------------------------------------------------------- #
# End-to-end: replay a PGN -> overlay events
# --------------------------------------------------------------------------- #


def test_stream_emits_game_then_positions():
    events = list(
        stream_overlay_events(_ingestor(), interval=0.0, max_polls=10, sleep=lambda _s: None)
    )
    games = [e for e in events if e["type"] == "game"]
    positions = [e for e in events if e["type"] == "position"]

    # Exactly one game event, emitted before any position.
    assert len(games) == 1
    assert events[0]["type"] == "game"

    # One position per half-move of the 4-ply game, in order.
    assert [p["ply"] for p in positions] == [1, 2, 3, 4]
    assert [p["move"]["san"] for p in positions] == ["e4", "e5", "Nf3", "Nc6"]

    # Every position carries a normalised equity and at least one parsed clock
    # (Black's appears once Black has moved); both are present by the second ply.
    for p in positions:
        assert 0.0 <= p["equity"] <= 1.0
        assert set(p["clock"]).issubset({"white", "black"}) and p["clock"]
    assert set(positions[-1]["clock"]) == {"white", "black"}


# --------------------------------------------------------------------------- #
# The SSE server actually pushes frames over a socket
# --------------------------------------------------------------------------- #


def _parse_sse(body: str):
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def test_sse_server_streams_overlay_events():
    def stream_factory():
        return stream_overlay_events(
            _ingestor(), interval=0.0, max_polls=10, sleep=lambda _s: None
        )

    server = serve_overlay(stream_factory, port=0)  # ephemeral port
    host, port = server.server_address[0], server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/sse", timeout=10) as resp:
            assert resp.headers["Content-Type"] == "text/event-stream"
            body = resp.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    events = _parse_sse(body)
    assert events[0]["type"] == "game"
    positions = [e for e in events if e["type"] == "position"]
    assert [p["ply"] for p in positions] == [1, 2, 3, 4]


def test_sse_server_serves_static_overlay_index():
    server = serve_overlay(lambda: iter(()), port=0)
    host, port = server.server_address[0], server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/index.html", timeout=10) as resp:
            html = resp.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert "overlay" in html.lower()
