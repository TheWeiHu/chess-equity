"""``chess-equity doctor`` — verify the optional external engines actually run (task 0073).

The core path needs none of this (see ``DEPENDENCIES.md``): the baseline CLI, tests, and
CI run on ``python-chess`` alone. But two bars depend on heavyweight, externally-provisioned
engines:

* the **classic centipawn bar** → a real **Stockfish** binary (``StockfishEngine``), and
* the **rating-conditioned equity bar** → **Maia-2** (``pip install maia2``, pulls torch,
  downloads a checkpoint on first use).

"Make Stockfish work, and install Maia" (task 0073) is really *provision + verify*. This
turns the verify half into one command: ``chess-equity doctor`` resolves Stockfish and runs
a real eval, imports Maia-2 and runs a real inference, and reports PASS/FAIL per engine with
the same install hint the adapters raise. Exit code is non-zero if any checked engine is
missing or broken, so it can gate a provisioning step.

The *reporting* logic (:func:`run_doctor`) is pure and the engine probes are injectable, so
the unit tests exercise it with fakes — no binary, no torch, no network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, List, Optional, TextIO

import chess

START_FEN = chess.STARTING_FEN


@dataclass
class Check:
    """The outcome of probing one optional engine."""

    name: str
    ok: bool
    detail: str


# A probe runs the real engine and returns a human-readable "it works" detail string,
# or raises (missing install, or installed-but-broken). Injectable so tests use fakes.
Probe = Callable[[], str]


def _probe_stockfish() -> str:
    """Resolve a real Stockfish and evaluate the start position."""
    from chess_equity.stockfish import StockfishEngine, StockfishNotFound, stockfish_path

    path = stockfish_path()
    if path is None:  # be explicit rather than relying on the engine to raise
        raise StockfishNotFound(
            "no Stockfish binary on PATH or $STOCKFISH_PATH — "
            "`brew install stockfish` / `apt-get install stockfish` (see DEPENDENCIES.md)"
        )
    ev = StockfishEngine(depth=8).eval(START_FEN)
    return f"{path}: startpos eval cp={ev.cp}"


def _probe_maia2() -> str:
    """Build the real Maia-2 model and run one rating-conditioned inference."""
    from chess_equity.cli import build_model

    model = build_model("maia2")
    eq = round(model.evaluate(START_FEN, 1500, 1500).equity_white, 1)
    return f"startpos equity(1500/1500) = {eq}% White"


def probe_broadcast(feed) -> str:
    """Poll a broadcast feed once and confirm it emits at least one parseable move.

    The go-live preflight (task 0183): before going on air a streamer needs the LIVE
    path to work, not just the engines. We reuse :mod:`broadcast` *parsing only* — poll
    the feed, split the snapshot into games, and parse the first with ``python-chess`` —
    so the check never touches a model (no torch / Maia-2 needed). Raises on every
    not-ready state so :func:`check` turns it into a FAIL with a streamer-readable hint:

    * unreachable feed → the underlying :class:`~chess_equity.broadcast.FeedError`,
    * reachable but no PGN / no moves yet → "round not started?" hints.
    """
    import io as _io

    import chess.pgn

    from chess_equity.broadcast import split_games

    pgn = feed.poll()
    if not pgn or not pgn.strip():
        raise RuntimeError("feed reachable but emitted no PGN yet (round not started?)")
    games = split_games(pgn)
    if not games:
        raise RuntimeError("feed returned data but no parseable PGN game (round not started?)")
    game = chess.pgn.read_game(_io.StringIO(games[0]))
    if game is None:
        raise RuntimeError("could not parse a PGN game from the feed")
    board = game.board()
    moves = 0
    last_san: Optional[str] = None
    for move in game.mainline_moves():
        last_san = board.san(move)
        board.push(move)
        moves += 1
    if moves == 0:
        raise RuntimeError("feed has a game header but no moves yet (round not started?)")
    return f"{len(games)} game(s), first game {moves} move(s) (last: {last_san})"


# --------------------------------------------------------------------------- #
# overlay bundle go-live preflight (task 0192)
# --------------------------------------------------------------------------- #


def _is_number(v: object) -> bool:
    # bool is a subclass of int; an equity/clock of True is a schema bug, not a number.
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_position(event: dict, loc: str) -> None:
    if "equity" not in event:
        raise ValueError(f"position event missing required 'equity'{loc}")
    equity = event["equity"]
    if not _is_number(equity):
        raise ValueError(f"position 'equity' must be a number{loc}, got {equity!r}")
    if not 0.0 <= float(equity) <= 1.0:
        raise ValueError(
            f"position 'equity' must be a White-POV win chance in [0,1]{loc}, got {equity}"
        )
    if "ply" in event and not (isinstance(event["ply"], int) and event["ply"] >= 0):
        raise ValueError(f"position 'ply' must be a non-negative int{loc}, got {event['ply']!r}")
    cp = event.get("cp")
    if cp is not None and not _is_number(cp):
        raise ValueError(f"position 'cp' must be a number or null{loc}, got {cp!r}")
    clock = event.get("clock")
    if clock is not None:
        if not isinstance(clock, dict):
            raise ValueError(f"position 'clock' must be an object{loc}, got {clock!r}")
        for side in ("white", "black"):
            v = clock.get(side)
            if v is not None and not _is_number(v):
                raise ValueError(
                    f"position clock.{side} must be seconds (number) or null{loc}, got {v!r}"
                )
    drama = event.get("drama")
    if drama is not None:
        if not isinstance(drama, dict):
            raise ValueError(f"position 'drama' must be an object{loc}, got {drama!r}")
        kind = drama.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"drama.kind must be a non-empty string{loc}, got {kind!r}")
        mag = drama.get("magnitude")
        if not _is_number(mag) or not 0.0 <= float(mag) <= 1.0:  # type: ignore[arg-type]
            raise ValueError(f"drama.magnitude must be a number in [0,1]{loc}, got {mag!r}")
        if not isinstance(drama.get("headline"), str):
            raise ValueError(f"drama.headline must be a string{loc}, got {drama.get('headline')!r}")


def _validate_game(event: dict, loc: str) -> None:
    players = event.get("players")
    if not isinstance(players, dict):
        raise ValueError(f"game event 'players' must be an object{loc}, got {players!r}")
    for side in ("white", "black"):
        if not isinstance(players.get(side), dict):
            raise ValueError(f"game event players.{side} must be an object{loc}")


def _validate_boards(event: dict, loc: str) -> None:
    boards = event.get("boards")
    if not isinstance(boards, list):
        raise ValueError(f"boards event 'boards' must be an array{loc}, got {boards!r}")


def validate_overlay_event(event: Any, where: str = "") -> None:
    """Assert one overlay event conforms to ``overlay/README.md``'s schema, or raise.

    The contract the streamer's overlay actually reads: ``type`` is one of
    ``game|position|boards``; a ``position`` carries a White-POV ``equity`` in
    ``[0,1]`` (the one required per-move field) plus optionally typed ``ply``/``cp``/
    ``clock``/``drama`` (see schema in the README). Unknown extra fields (e.g. a
    replay's ``delayMs``) degrade gracefully and are ignored. Raises ``ValueError``
    with a streamer-readable message on the first violation — used both for the
    bundled ``mock-game.json`` and for live ``to_overlay_event`` output.
    """
    loc = f" ({where})" if where else ""
    if not isinstance(event, dict):
        raise ValueError(f"overlay event must be a JSON object{loc}, got {type(event).__name__}")
    etype = event.get("type")
    if etype == "position":
        _validate_position(event, loc)
    elif etype == "game":
        _validate_game(event, loc)
    elif etype == "boards":
        _validate_boards(event, loc)
    else:
        raise ValueError(
            f"overlay event has unknown/missing 'type' {etype!r}{loc} (expected game|position|boards)"
        )


class _HTMLValidator(HTMLParser):
    """HTMLParser is lenient, but ``strict``-style breakage (e.g. a truncated tag)
    still surfaces as an exception from ``feed`` — enough to catch a corrupted asset."""


def _check_parses_html(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{path.name} is empty")
    parser = _HTMLValidator()
    parser.feed(text)  # raises on malformed markup
    parser.close()


def _check_nonempty(path: Path) -> None:
    if not path.read_text(encoding="utf-8").strip():
        raise ValueError(f"{path.name} is empty")


def overlay_dir() -> Optional[Path]:
    """The repo's ``overlay/`` dir, or ``None`` from an installed wheel without assets."""
    candidate = Path(__file__).resolve().parents[2] / "overlay"
    return candidate if candidate.is_dir() else None


def _read_overlay_events(mock: Path) -> list:
    """Parse a replay file into its event list (an array, or ``{"events": [...]}``)."""
    data = json.loads(mock.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("events", [])
    if not isinstance(data, list) or not data:
        raise ValueError(f"{mock.name} has no events array")
    return data


def probe_overlay(directory: Optional[Path] = None) -> str:
    """Assert the overlay bundle a streamer loads is shippable (task 0192).

    The broadcast preflight (:func:`probe_broadcast`) checks the *feed* side; this
    checks the *front-end* the OBS browser source actually loads:

    * ``index.html`` / ``config.html`` exist and parse, and ``overlay.js`` is present
      and non-empty (a corrupted/truncated bundle fails here, before air);
    * the bundled ``mock-game.json`` parses and every event conforms to the documented
      schema (:func:`validate_overlay_event`); and
    * a live ``MoveEvent.to_overlay_event()`` (driven through the pure baseline model —
      no torch/network) conforms to the *same* validator, so producer and bundle can't
      drift past the schema.

    Static + schema only — safe to run unattended. Raises on the first problem so
    :func:`check` reports a FAIL with the offending detail.
    """
    directory = directory or overlay_dir()
    if directory is None or not directory.is_dir():
        raise ValueError("overlay/ bundle not found (running from a wheel without assets?)")

    for name in ("index.html", "config.html"):
        path = directory / name
        if not path.is_file():
            raise ValueError(f"missing overlay asset {name}")
        _check_parses_html(path)
    js = directory / "overlay.js"
    if not js.is_file():
        raise ValueError("missing overlay asset overlay.js")
    _check_nonempty(js)

    mock = directory / "mock-game.json"
    if not mock.is_file():
        raise ValueError("missing bundled mock-game.json")
    events = _read_overlay_events(mock)
    for i, event in enumerate(events):
        validate_overlay_event(event, where=f"mock-game.json event {i}")

    # Pin the live producer to the same schema (pure baseline — no torch/network).
    produced = _probe_to_overlay_event()

    return (
        f"{directory.name}/ bundle OK — "
        f"index.html, config.html, overlay.js parse; "
        f"mock-game.json {len(events)} event(s) valid; to_overlay_event valid ({produced})"
    )


def _probe_to_overlay_event() -> str:
    """Drive a 2-move PGN through the bridge and validate the produced overlay event."""
    from chess_equity.broadcast import GameTracker
    from chess_equity.models import LichessBaselineModel

    pgn = (
        '[White "A"]\n[Black "B"]\n[Result "*"]\n\n'
        "1. e4 { [%clk 0:03:00] } e5 { [%clk 0:02:58] } *\n"
    )
    tracker = GameTracker("doctor-overlay", LichessBaselineModel(), white_elo=1500, black_elo=1500)
    move_events = tracker.ingest(pgn)
    if not move_events:
        raise ValueError("bridge produced no overlay events from a 2-move PGN")
    event = move_events[-1].to_overlay_event()
    validate_overlay_event(event, where="to_overlay_event")
    return f"ply {event.get('ply')}, equity {event.get('equity')}"


def check(name: str, probe: Probe) -> Check:
    """Run one probe, mapping success/exception to a :class:`Check`.

    A clean exception (e.g. ``StockfishNotFound`` / ``Maia2NotInstalled``) becomes a
    failed check carrying its install hint; any other exception is reported as
    installed-but-broken so the message distinguishes the two.
    """
    try:
        return Check(name, True, probe())
    except Exception as exc:  # noqa: BLE001 - the whole point is to report, not crash
        return Check(name, False, str(exc) or exc.__class__.__name__)


def run_doctor(checks: List[Check], out: Optional[TextIO] = None) -> int:
    """Print each check and return 0 iff every checked engine works."""
    import sys

    out = out if out is not None else sys.stdout
    failures = 0
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        print(f"[{mark}] {c.name}: {c.detail}", file=out)
        if not c.ok:
            failures += 1
    summary = "all engines OK" if failures == 0 else f"{failures} engine(s) need attention"
    print(f"\n{len(checks) - failures}/{len(checks)} engines OK — {summary}", file=out)
    return 1 if failures else 0


def doctor(
    out: Optional[TextIO] = None,
    probes: Optional[dict] = None,
    engines: Optional[List[str]] = None,
    broadcast_probe: Optional[Probe] = None,
    overlay_probe: Optional[Probe] = None,
) -> int:
    """Probe the optional engines with the real backends (override ``probes`` in tests).

    ``engines`` restricts the probes to a subset (e.g. ``["stockfish"]`` for a
    binary-only CI runner that never installs torch/Maia-2); ``None`` checks all.

    ``broadcast_probe`` (set by ``doctor --broadcast <spec>``) appends a go-live
    preflight that verifies the LIVE feed is reachable and emitting a parseable move
    (task 0183) — it runs alongside whatever engine checks ``engines`` selects.

    ``overlay_probe`` (set by ``doctor --overlay``) appends a check that the overlay
    bundle the streamer loads is shippable: its HTML/JS assets parse and the bundled
    replay + live ``to_overlay_event`` output conform to the documented event schema
    (task 0192). Static + schema only — no torch/network.
    """
    probes = probes or {"stockfish": _probe_stockfish, "maia2": _probe_maia2}
    if engines:
        probes = {name: probes[name] for name in engines if name in probes}
    checks = [check(name, probe) for name, probe in probes.items()]
    if broadcast_probe is not None:
        checks.append(check("broadcast", broadcast_probe))
    if overlay_probe is not None:
        checks.append(check("overlay", overlay_probe))
    return run_doctor(checks, out=out)
