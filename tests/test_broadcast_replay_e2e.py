"""End-to-end broadcast -> overlay replay smoke test (task 0167).

``test_broadcast_overlay_contract.py`` (task 0050) pins the JSON *field contract*
between the producer and the overlay, but only on a couple of hand-fed moves via
``GameTracker.ingest`` — it never runs a whole game through the real SSE socket. So
nothing today catches a regression where the *stream as a whole* breaks: a ply gets
dropped, the series stops short of the final move, ordering scrambles, or the replay
never terminates. This test is that catch.

It drives the existing PGN replayer (:class:`LocalPgnFeed`) through the very SSE
handler :func:`serve_sse` runs (:func:`make_sse_server` builds the same
``_sse_handler``; ``serve_sse`` only adds the blocking ``serve_forever`` loop), and
reads the frames back over a real HTTP socket exactly like ``overlay/feed.js``'s
``EventSource`` would. The bar is *full-game coverage*: every half-move of a complete
committed game must surface as a ``position`` event with a usable equity, in order,
bracketed by the one-time ``game`` metadata and a clean end-of-stream.

Fixture: ``data/sample/sample_games.pgn`` — its first game is a 7-ply scholar's mate
ending in checkmate (``Qxf7#``). ``data/sample`` is the sanctioned offline-smoke
fixture (see project CLAUDE.md / ``reports/validation_sample.md``): illustrative, not
evidence. No browser, no torch, no network — unattended-safe.
"""
import io
import json
import os
import threading
import urllib.request

import chess.pgn

from chess_equity.broadcast import (
    BroadcastIngestor,
    LocalPgnFeed,
    make_sse_server,
    overlay_events,
)
from chess_equity.models import LichessBaselineModel

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PGN = os.path.join(HERE, "..", "data", "sample", "sample_games.pgn")


def _first_game_pgn():
    """The PGN text and the metadata (ply count, final SAN) of the first sample game.

    ``LocalPgnFeed`` replays only the first game of a multi-game snapshot, so the
    expectations the test asserts against are derived from that same first game rather
    than hard-coded — if the sample file changes, the test follows it.
    """
    with open(SAMPLE_PGN, "r", encoding="utf-8") as fh:
        text = fh.read()
    game = chess.pgn.read_game(io.StringIO(text))
    assert game is not None, "sample PGN must contain at least one game"
    board = game.board()
    sans = []
    for move in game.mainline_moves():
        sans.append(board.san(move))
        board.push(move)
    assert sans, "first sample game must have moves"
    return text, len(sans), sans[-1]


def _replay_over_sse(pgn_text):
    """Replay ``pgn_text`` through the SSE bridge and return the parsed overlay events.

    This is the real path a streamer hits: ``LocalPgnFeed`` -> ``BroadcastIngestor``
    -> ``overlay_events`` -> the SSE ``_sse_handler`` over a socket -> an EventSource
    client parsing ``data:`` frames. ``interval=0`` so the finite replay runs at full
    speed; ``max_idle_polls=1`` so the stream ends as soon as the game is exhausted.
    """

    def make_events():
        return overlay_events(
            BroadcastIngestor(
                LocalPgnFeed(pgn_text),
                LichessBaselineModel(),
                white_elo=1500,
                black_elo=1480,
            ),
            interval=0,
            max_polls=None,
            max_idle_polls=1,
        )

    server = make_sse_server(make_events, port=0)  # OS-assigned port
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            urllib.request.urlopen(f"http://127.0.0.1:{port}/sse", timeout=10)
            .read()
            .decode("utf-8")
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    # Parse like overlay/feed.js: each event is a "data: <json>" frame ended by a
    # blank line. Keepalive comments (": ...") and empty splits are ignored.
    events = []
    for frame in body.split("\n\n"):
        frame = frame.strip()
        if frame.startswith("data: "):
            events.append(json.loads(frame[len("data: ") :]))
    return events


# --------------------------------------------------------------------------- #
# Full-game end-to-end coverage
# --------------------------------------------------------------------------- #


def test_full_game_replays_to_the_overlay_over_sse():
    """A complete committed game streams through serve_sse's handler intact.

    The single assertion-rich smoke test: a leading ``game`` event, one ``position``
    per ply with a usable equity, strictly increasing ply with no gaps, and a clean
    terminating stream ending on the final (checkmate) move.
    """
    pgn_text, n_plies, final_san = _first_game_pgn()
    events = _replay_over_sse(pgn_text)

    assert events, "stream produced no events at all"

    # (1) game metadata leads the stream, before any position (overlay name-plates).
    assert events[0]["type"] == "game", "first event must be the one-time game metadata"
    players = events[0]["players"]
    assert players["white"]["name"] and players["black"]["name"], "game event must name both players"
    positions = [e for e in events if e["type"] == "position"]
    first_pos = events.index(positions[0])
    assert first_pos > 0, "game event must precede the first position event"

    # (2) full coverage: exactly one position per half-move, no ply dropped, in order.
    plies = [e["ply"] for e in positions]
    assert plies == list(range(1, n_plies + 1)), (
        f"expected a position for every ply 1..{n_plies}, got {plies}"
    )

    # (3) every position carries a usable White-POV equity in [0, 1] (the whole point
    #     of the overlay — a per-move equity series, not just well-named empty events).
    for evt in positions:
        eq = evt["equity"]
        assert isinstance(eq, float) and 0.0 <= eq <= 1.0, f"bad equity on ply {evt['ply']}: {eq!r}"

    # (4) the series reaches the game's actual end: the last position is the final
    #     (terminal) move, and the stream terminated cleanly (finite frames = "end").
    assert positions[-1]["ply"] == n_plies
    assert positions[-1]["move"]["san"] == final_san, "last position must be the game's final move"
    # No event after the last position (the replay ended rather than hanging open).
    assert events[-1] is positions[-1], "stream must end on the final move, not keep emitting"


def test_no_ply_is_dropped_or_duplicated_under_sse_framing():
    """Defense in depth: the per-ply equity series is gap-free AND duplicate-free.

    SSE framing (chunked writes, socket flushes) is exactly where a move can be lost
    or doubled; assert the multiset of plies is the clean 1..N with no repeats.
    """
    pgn_text, n_plies, _ = _first_game_pgn()
    events = _replay_over_sse(pgn_text)
    plies = [e["ply"] for e in events if e["type"] == "position"]
    assert len(plies) == n_plies, f"expected {n_plies} positions, got {len(plies)}"
    assert len(set(plies)) == len(plies), f"duplicate plies in the stream: {plies}"
    assert plies == sorted(plies), f"plies arrived out of order: {plies}"
