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

from dataclasses import dataclass
from typing import Callable, List, Optional, TextIO

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
) -> int:
    """Probe the optional engines with the real backends (override ``probes`` in tests).

    ``engines`` restricts the probes to a subset (e.g. ``["stockfish"]`` for a
    binary-only CI runner that never installs torch/Maia-2); ``None`` checks all.

    ``broadcast_probe`` (set by ``doctor --broadcast <spec>``) appends a go-live
    preflight that verifies the LIVE feed is reachable and emitting a parseable move
    (task 0183) — it runs alongside whatever engine checks ``engines`` selects.
    """
    probes = probes or {"stockfish": _probe_stockfish, "maia2": _probe_maia2}
    if engines:
        probes = {name: probes[name] for name in engines if name in probes}
    checks = [check(name, probe) for name, probe in probes.items()]
    if broadcast_probe is not None:
        checks.append(check("broadcast", broadcast_probe))
    return run_doctor(checks, out=out)
