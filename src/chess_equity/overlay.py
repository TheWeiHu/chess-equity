"""Bridge: broadcast :class:`~chess_equity.broadcast.MoveEvent`\\ s → the OBS
overlay's wire schema, served over SSE so the overlay (task 0019) renders a live
(or replayed) game — *no* ``mock-game.json``. This is task 0021.

The two halves shipped in parallel with incompatible contracts (PR #8): broadcast
emits flat ``MoveEvent``s with equity in **percentage points** (`[0, 100]`,
White-POV); the overlay consumes ``"game"`` / ``"position"`` events with equity as a
**fraction** (`[0, 1]`). This module is the one place that translation lives. The
overlay contract is documented prose in ``overlay/README.md``; the functions here are
its single executable definition, imported by both the SSE server below and the
tests, so the two sides cannot drift.

Pieces:

- :func:`game_event` / :func:`position_event` — pure ``MoveEvent`` → overlay-dict
  translators (unit-tested, no I/O).
- :func:`stream_overlay_events` — drive a :class:`BroadcastIngestor` and yield overlay
  dicts, emitting a one-time ``"game"`` event the first time each game_id appears.
- :func:`serve_overlay` — a stdlib ``ThreadingHTTPServer`` that serves the static
  overlay files *and* pushes the live stream at ``/sse``.

Real equity numbers still wait on Maia-2 (task 0005); today the bar moves on the
placeholder baseline, but the wiring is exactly what 0005 will flow through.
"""

from __future__ import annotations

import http.server
import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Optional, Set

from chess_equity.broadcast import BroadcastIngestor, MoveEvent

# Repo-root ``overlay/`` dir (src/chess_equity/overlay.py -> parents[2] == repo root).
# Overridable so the server works from an installed wheel or a relocated checkout.
DEFAULT_OVERLAY_DIR = Path(__file__).resolve().parents[2] / "overlay"


# --------------------------------------------------------------------------- #
# Translation: MoveEvent -> overlay wire schema
# --------------------------------------------------------------------------- #


def game_event(event: MoveEvent) -> Dict[str, object]:
    """One-time ``"game"`` metadata event for a game_id.

    ``MoveEvent`` carries ratings but not player *names* (the broadcast layer doesn't
    surface them yet), so only ratings are filled — the overlay falls back to
    "White"/"Black" for the names. Adding names is a documented follow-up.
    """
    players: Dict[str, object] = {}
    if event.white_elo is not None:
        players["white"] = {"rating": event.white_elo}
    if event.black_elo is not None:
        players["black"] = {"rating": event.black_elo}
    out: Dict[str, object] = {"type": "game", "game_id": event.game_id}
    if players:
        out["players"] = players
    return out


def position_event(event: MoveEvent) -> Dict[str, object]:
    """Per-move ``"position"`` event: equity/clock/grade rescaled to the overlay.

    Equity and Δequity convert from percentage points ``[0, 100]`` to the overlay's
    fraction ``[0, 1]``. ``cp`` (classic centipawn eval) is omitted — the equity
    models don't expose it — so the overlay hides its ghost tick, which is correct.
    """
    out: Dict[str, object] = {
        "type": "position",
        "game_id": event.game_id,
        "ply": event.ply,
        "move": {"san": event.san},
        "equity": round(event.equity / 100.0, 4),
    }
    clock: Dict[str, float] = {}
    if event.white_clock is not None:
        clock["white"] = event.white_clock
    if event.black_clock is not None:
        clock["black"] = event.black_clock
    if clock:
        out["clock"] = clock
    if event.last_move_grade is not None:
        grade: Dict[str, object] = {"label": event.last_move_grade}
        if event.delta_equity is not None:
            grade["delta"] = round(event.delta_equity / 100.0, 4)
        out["grade"] = grade
    return out


def translate(event: MoveEvent, seen_games: Set[str]) -> Iterator[Dict[str, object]]:
    """Yield the overlay events for one ``MoveEvent``.

    Emits a ``"game"`` event the first time a game_id is seen (so the overlay gets its
    metadata), then always the ``"position"`` event. ``seen_games`` is the caller's
    running set of game_ids already announced.
    """
    if event.game_id not in seen_games:
        seen_games.add(event.game_id)
        yield game_event(event)
    yield position_event(event)


def stream_overlay_events(
    ingestor: BroadcastIngestor, **stream_kwargs: object
) -> Iterator[Dict[str, object]]:
    """Drive ``ingestor.stream(...)`` and yield overlay-schema dicts.

    ``stream_kwargs`` pass straight through to
    :meth:`BroadcastIngestor.stream` (``interval``, ``max_polls``, ``sleep`` …).
    """
    seen: Set[str] = set()
    for event in ingestor.stream(**stream_kwargs):  # type: ignore[arg-type]
        yield from translate(event, seen)


# --------------------------------------------------------------------------- #
# SSE server
# --------------------------------------------------------------------------- #


def _sse_frame(payload: Dict[str, object]) -> bytes:
    return ("data: " + json.dumps(payload) + "\n\n").encode("utf-8")


def make_handler(
    stream_factory: Callable[[], Iterable[Dict[str, object]]],
    overlay_dir: Path,
) -> type:
    """Build a request handler serving static overlay files + a live ``/sse`` push.

    ``stream_factory`` is called *per connection* and must return a fresh iterable of
    overlay events — so a replay restarts from move one for each OBS source that
    connects, and a live feed streams from "now".
    """

    directory = str(overlay_dir)

    class _OverlayHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=directory, **kwargs)  # type: ignore[arg-type]

        def do_GET(self) -> None:  # noqa: N802 (stdlib API)
            if self.path.split("?")[0] == "/sse":
                self._stream_sse()
                return
            super().do_GET()

        def _stream_sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            # Close at end-of-stream so a finite replay signals EOF to the client; a
            # live (unbounded) feed simply never reaches the end of the generator.
            self.close_connection = True
            self.end_headers()
            try:
                for payload in stream_factory():
                    self.wfile.write(_sse_frame(payload))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # OBS / browser closed the source — normal.

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 (stdlib API)
            pass  # quieter console

    return _OverlayHandler


def serve_overlay(
    stream_factory: Callable[[], Iterable[Dict[str, object]]],
    *,
    port: int = 8777,
    host: str = "127.0.0.1",
    overlay_dir: Optional[Path] = None,
) -> http.server.ThreadingHTTPServer:
    """Create (but do not start) a ThreadingHTTPServer for the live overlay.

    Static overlay at ``http://host:port/``; live SSE push at ``/sse``. Point an OBS
    Browser source at ``http://host:port/?src=/sse``. Call ``serve_forever()`` on the
    returned server (the CLI does); tests drive it from a thread and shut it down.
    """
    directory = overlay_dir or DEFAULT_OVERLAY_DIR
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"overlay dir not found: {directory}")
    handler = make_handler(stream_factory, Path(directory))
    return http.server.ThreadingHTTPServer((host, port), handler)
