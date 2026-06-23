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
from math import isfinite
from pathlib import Path
from typing import Any, Callable, List, Optional, TextIO

import chess

START_FEN = chess.STARTING_FEN


@dataclass
class Check:
    """The outcome of probing one optional engine.

    ``warn`` flags a *soft* problem: the check still passes (``ok`` stays True, exit code
    unaffected) but something is off enough to surface as ``WARN`` rather than ``PASS``
    (e.g. a model artifact that works but lacks leakage-guard provenance — task 0199).
    """

    name: str
    ok: bool
    detail: str
    warn: bool = False


class DoctorWarning(Exception):
    """A soft preflight failure: the probe's subject works, but a non-fatal caveat should
    surface as ``WARN`` (passing) instead of ``PASS``. :func:`check` maps it to a passing
    :class:`Check` with ``warn=True``; any *other* exception is a hard ``FAIL``."""


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


# --- live SSE wiring go-live preflight (task 0209) -------------------------------------
#
# `probe_broadcast` checks the *feed* parses and `probe_overlay` checks the *bundle*
# loads, but neither exercises the actual `broadcast --serve-sse` HTTP path the overlay
# (OBS browser source) EventSources onto. A streamer who has both green can still go on
# air to a dead bar if the SSE server never binds or never emits. `probe_serve_sse` closes
# that gap: it binds the *real* SSE server on an ephemeral port over a finite local PGN
# replay, connects to `/sse` exactly as the overlay would, and confirms a real overlay
# `position` frame arrives within a timeout — torch-free (pure baseline model) and
# network-free (loopback + committed sample PGN), so it runs unattended.


def sample_pgn_path() -> Optional[Path]:
    """The repo's committed offline sample game, or ``None`` from a wheel without it."""
    candidate = Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_games.pgn"
    return candidate if candidate.is_file() else None


def probe_serve_sse(pgn_path: Optional[Path] = None, *, timeout: float = 5.0) -> str:
    """Assert the live ``broadcast --serve-sse`` wiring works before going on air (0209).

    Drives the smallest possible end-to-end of the live path a streamer points OBS at:

    * builds the same overlay event source ``broadcast --serve-sse`` uses (a finite
      :class:`~chess_equity.broadcast.LocalPgnFeed` replay through the pure baseline
      model — no torch, no network);
    * binds the *real* SSE server (:func:`~chess_equity.broadcast.make_sse_server`) on
      an ephemeral loopback port (``port=0``) in a background thread;
    * connects to ``/sse`` as the overlay's ``EventSource`` would, asserts the
      ``text/event-stream`` content type, and reads frames until the first overlay
      ``position`` event arrives — validated against the documented schema
      (:func:`validate_overlay_event`) — or ``timeout`` seconds elapse.

    A server that fails to bind, returns the wrong content type, or never emits a move
    turns doctor red so a dead feed is caught before air. Raises ``ValueError`` /
    ``RuntimeError`` on any failure so :func:`check` reports a FAIL with the detail.
    """
    import threading
    import urllib.request

    from chess_equity.broadcast import (
        BroadcastIngestor,
        LocalPgnFeed,
        make_sse_server,
        overlay_events,
    )
    from chess_equity.models import LichessBaselineModel

    path = Path(pgn_path) if pgn_path else sample_pgn_path()
    if path is None or not path.is_file():
        raise ValueError("sample PGN not found (running from a wheel without data/sample assets?)")
    pgn_text = path.read_text(encoding="utf-8")

    def make_events():
        # Fresh ingestor per connection; a high moves_per_poll reveals the whole replay
        # on the first poll, and a no-op sleep keeps the finite replay instant.
        ingestor = BroadcastIngestor(
            LocalPgnFeed(pgn_text, moves_per_poll=20),
            LichessBaselineModel(),
            white_elo=1500,
            black_elo=1500,
        )
        return overlay_events(
            ingestor, max_idle_polls=1, heartbeat=False, sleep=lambda _: None
        )

    httpd = make_sse_server(make_events, port=0, host="127.0.0.1")
    bound = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{bound}/sse"
        frames = 0
        first_position: Optional[dict] = None
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if "text/event-stream" not in ctype:
                raise RuntimeError(f"/sse returned Content-Type {ctype!r}, not text/event-stream")
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue  # SSE comments (": keepalive") and blank separators
                frames += 1
                payload = line[len("data:") :].strip()
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"/sse emitted a non-JSON data frame: {payload!r}") from exc
                validate_overlay_event(event, where="serve-sse /sse frame")
                if event.get("type") == "position":
                    first_position = event
                    break  # one real move proves the live wiring; stop reading
        if first_position is None:
            raise RuntimeError(
                f"/sse reachable on port {bound} but emitted no position event "
                f"({frames} frame(s)) within {timeout:g}s"
            )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2.0)

    return (
        f"/sse bound on 127.0.0.1:{bound}, streamed {frames} overlay frame(s); "
        f"first position ply {first_position.get('ply')}, equity {first_position.get('equity')}"
    )


# --- evidence gate preflight (task 0195) -----------------------------------------------
#
# `doctor` verifies the optional engines and the go-live bundle, but not the project's
# actual headline claim: that the committed real-data gate reports are present and still
# passing. Without this a repo could ship with a missing or regressed proof and doctor
# would stay green. `probe_evidence` reads `reports/SUMMARY.md` — the canonical gate index,
# whose verdicts are quoted/parsed from each report's own header — and confirms every listed
# report exists on disk and corroborates its stated verdict.
#
# Scope boundary with task 0194 (guards SUMMARY's verdicts MATCH the report headers): this
# trusts SUMMARY's verdict column and guards (a) no listed proof is missing and (b) no gate
# report has silently regressed to FAIL except the one deliberate negative result.

# The only report SUMMARY may legitimately mark FAIL: the end-to-end board→WDL net, kept on
# purpose as a negative result (Approach D loses to the centipawn baseline). Any *other* FAIL
# is a regression doctor must catch.
EVIDENCE_FAIL_ALLOWLIST = frozenset({"wdl_net_real.md"})

# A row whose verdict is PASS must have its report corroborate it. Markers differ across
# reports: most say "PASS"; goodmoves_real states its pass in prose with a "✅". Accept either.
_PASS_MARKERS = ("PASS", "✅")


def reports_dir() -> Optional[Path]:
    """The repo's ``reports/`` dir, or ``None`` from an installed wheel without assets."""
    candidate = Path(__file__).resolve().parents[2] / "reports"
    return candidate if candidate.is_dir() else None


def _parse_summary_rows(summary_text: str) -> list:
    """Extract ``(filename, verdict)`` for each report row in ``SUMMARY.md``'s table.

    ``verdict`` is normalised to ``PASS`` / ``FAIL`` / ``info`` (``PASS (caveat)`` → ``PASS``).
    Only table rows that link a ``*.md`` report are returned; prose and the header row are
    skipped. Raises if the table has no parseable rows (a gutted/renamed SUMMARY).
    """
    import re

    rows = []
    for line in summary_text.splitlines():
        line = line.strip()
        if not line.startswith("| ["):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 4:
            continue
        link = re.search(r"\(([^)]+\.md)\)", cols[0])
        if not link:
            continue
        filename = link.group(1)
        verdict_col = cols[-1].upper()
        if "**FAIL**" in verdict_col or verdict_col.startswith("FAIL"):
            verdict = "FAIL"
        elif "**PASS**" in verdict_col or "PASS" in verdict_col:
            verdict = "PASS"
        else:
            verdict = "info"
        rows.append((filename, verdict))
    if not rows:
        raise ValueError("SUMMARY.md has no parseable report rows (gate index empty or renamed?)")
    return rows


def probe_evidence(directory: Optional[Path] = None) -> str:
    """Assert the committed real-data gate reports are present and still passing (task 0195).

    Reads ``reports/SUMMARY.md`` (the gate index) and, for every report it lists:

    * confirms the linked ``*_real.md`` file exists on disk — a missing proof fails here;
    * for a **PASS** verdict, confirms the report itself states a pass (a ``PASS`` token or
      the prose ``✅`` goodmoves uses) — a report that regressed to no-longer-passing while
      SUMMARY still claims PASS is caught;
    * for a **FAIL** verdict, confirms the file is the one allowlisted deliberate negative
      result (:data:`EVIDENCE_FAIL_ALLOWLIST`) and that the report states ``FAIL`` — any
      *other* FAIL is an unintended regression;
    * **info** rows (calibration/disagreement/threshold reports that state no gate) are
      existence-checked only.

    Reads report text but no datasets — safe to run unattended. Raises on the first problem
    so :func:`check` reports a FAIL with the offending detail.
    """
    directory = directory or reports_dir()
    if directory is None or not directory.is_dir():
        raise ValueError("reports/ dir not found (running from a wheel without assets?)")
    summary = directory / "SUMMARY.md"
    if not summary.is_file():
        raise ValueError("missing reports/SUMMARY.md (the gate index)")

    rows = _parse_summary_rows(summary.read_text(encoding="utf-8"))
    pass_n = fail_n = info_n = 0
    for filename, verdict in rows:
        report = directory / filename
        if not report.is_file():
            raise ValueError(f"gate report listed in SUMMARY.md is missing on disk: {filename}")
        text = report.read_text(encoding="utf-8")
        if verdict == "PASS":
            if not any(marker in text for marker in _PASS_MARKERS):
                raise ValueError(
                    f"{filename}: SUMMARY.md marks it PASS but the report states no pass "
                    "(regressed proof?)"
                )
            pass_n += 1
        elif verdict == "FAIL":
            if filename not in EVIDENCE_FAIL_ALLOWLIST:
                raise ValueError(
                    f"{filename}: gate report regressed to FAIL (only "
                    f"{sorted(EVIDENCE_FAIL_ALLOWLIST)} is an allowed deliberate FAIL)"
                )
            if "FAIL" not in text.upper():
                raise ValueError(f"{filename}: SUMMARY.md marks it FAIL but the report says no FAIL")
            fail_n += 1
        else:
            info_n += 1

    return (
        f"SUMMARY.md gate index OK — {len(rows)} report(s) present: "
        f"{pass_n} PASS, {fail_n} deliberate FAIL, {info_n} info"
    )


# --- active equity-model preflight (task 0199) -----------------------------------------
#
# `doctor`'s engine checks prove Stockfish/Maia-2 *can* run and the bundle/feed are
# shippable, but not that the model the overlay is configured to use will actually
# produce a bar. `probe_model` loads the selected `--model` and evaluates one fixture FEN
# so a missing/garbled artifact (or a NaN/out-of-range bar) turns doctor red *before* air,
# while a model that works but lacks leakage-guard provenance surfaces as WARN.

# A wdl-a artifact fit on this few rows or fewer is the committed tiny smoke-test seed,
# not a real fit — the bar still renders, but the numbers aren't trustworthy on air (WARN).
# The real shipped artifact is n_train=50000, well clear of this floor.
_WDL_A_SEED_MAX_TRAIN = 1000

# Fixture inputs for the "does it produce a sane bar?" probe — startpos at a mid rating.
_MODEL_FIXTURE_FEN = START_FEN
_MODEL_FIXTURE_ELO = 1500


def _wdl_a_provenance_warnings(meta: dict) -> List[str]:
    """Soft caveats on a wdl-a artifact's fit metadata (absent → empty list = clean PASS).

    A missing ``fit_month`` means the 0112 leakage guard can't refuse an eval set that *is*
    the training month; an ``n_train`` at/under the seed floor means it's the committed
    overfit smoke seed. Either is a WARN, not a FAIL — the model still evaluates.
    """
    warnings: List[str] = []
    if not meta.get("fit_month"):
        warnings.append("artifact has no fit_month (the 0112 leakage guard can't run)")
    n_train = meta.get("n_train")
    if isinstance(n_train, (int, float)) and not isinstance(n_train, bool):
        if n_train <= _WDL_A_SEED_MAX_TRAIN:
            warnings.append(
                f"n_train={n_train} looks like the tiny overfit seed, not a real fit"
            )
    return warnings


def _default_build_model(model_name: str):
    """Construct the named model via the CLI registry (lazy import avoids a cycle)."""
    from chess_equity.cli import build_model

    return build_model(model_name)


def probe_model(
    model_name: str = "baseline",
    build: Optional[Callable[[str], Any]] = None,
    artifact_path: Optional[Path] = None,
) -> str:
    """Assert the ACTIVE equity model loads and produces a sane bar before going live (0199).

    The reliability gap doctor's engine checks leave: the overlay reads *one* configured
    ``--model``, and nothing verifies it actually works until the first live position. This
    closes it:

    * the model **constructs** — ``--model wdl-a`` loads + parses its committed artifact;
      ``--model baseline`` builds its objective engine. A missing/garbled artifact FAILs;
    * it evaluates one fixture FEN to a **finite White-POV bar in [0,100]** — a NaN or
      out-of-range equity FAILs (the overlay would render a broken bar);
    * (wdl-a only) the artifact carries **fit provenance**: a missing ``fit_month`` (the
      leakage guard can't run) or a seed-sized ``n_train`` is a WARN, not a FAIL.

    Loads the model but no datasets/network — safe to run unattended for ``baseline`` and
    ``wdl-a``. ``build``/``artifact_path`` are injectable for tests. Raises ``ValueError``
    on a hard problem (FAIL); raises :class:`DoctorWarning` for a soft one (WARN).
    """
    build = build or _default_build_model
    provenance: Optional[str] = None
    warnings: List[str] = []

    if model_name == "wdl-a":
        from chess_equity.wdl_regression import default_artifact_path, load_wdl_a_model

        path = Path(artifact_path) if artifact_path else default_artifact_path()
        if not path.is_file():
            raise ValueError(f"--model wdl-a artifact missing on disk: {path}")
        try:
            fitted = load_wdl_a_model(str(path))
        except Exception as exc:  # noqa: BLE001 - report the parse/shape failure as a FAIL
            raise ValueError(f"--model wdl-a artifact unreadable ({path.name}): {exc}") from exc
        meta = fitted.meta or {}
        warnings = _wdl_a_provenance_warnings(meta)
        provenance = f"n_train={meta.get('n_train')}, fit_month={meta.get('fit_month') or 'absent'}"

    model = build(model_name)  # unknown model / failed load → FAIL via check()
    equity = model.evaluate(_MODEL_FIXTURE_FEN, _MODEL_FIXTURE_ELO, _MODEL_FIXTURE_ELO)
    bar = equity.equity_white
    if not isinstance(bar, (int, float)) or isinstance(bar, bool) or not isfinite(bar):
        raise ValueError(f"--model {model_name} produced a non-finite bar: {bar!r}")
    if not 0.0 <= float(bar) <= 100.0:
        raise ValueError(f"--model {model_name} bar {bar} is outside [0,100]% White-POV")

    win = float(bar) / 100.0
    detail = f"--model {model_name} loads; startpos win-equity {win:.2f} (0..1)"
    if provenance is not None:
        detail += f"; {provenance}"
    if warnings:
        raise DoctorWarning(detail + " — WARN: " + "; ".join(warnings))
    return detail


def check(name: str, probe: Probe) -> Check:
    """Run one probe, mapping success/exception to a :class:`Check`.

    A :class:`DoctorWarning` becomes a *passing* check flagged ``warn`` (a soft caveat,
    exit code unaffected). A clean failure exception (e.g. ``StockfishNotFound`` /
    ``Maia2NotInstalled``) becomes a failed check carrying its install hint; any other
    exception is reported as installed-but-broken so the message distinguishes the two.
    """
    try:
        return Check(name, True, probe())
    except DoctorWarning as warn:
        return Check(name, True, str(warn) or warn.__class__.__name__, warn=True)
    except Exception as exc:  # noqa: BLE001 - the whole point is to report, not crash
        return Check(name, False, str(exc) or exc.__class__.__name__)


def run_doctor(checks: List[Check], out: Optional[TextIO] = None) -> int:
    """Print each check and return 0 iff every checked engine works."""
    import sys

    out = out if out is not None else sys.stdout
    failures = 0
    for c in checks:
        mark = "FAIL" if not c.ok else ("WARN" if c.warn else "PASS")
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
    serve_sse_probe: Optional[Probe] = None,
    evidence_probe: Optional[Probe] = None,
    model_probe: Optional[Probe] = None,
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

    ``evidence_probe`` (set by ``doctor --evidence``) appends a check that the committed
    real-data gate reports listed in ``reports/SUMMARY.md`` are present and still state
    their expected verdict (task 0195) — so a missing/regressed proof turns doctor red.
    Reads report text but no datasets — safe to run unattended.

    ``serve_sse_probe`` (set by ``doctor --serve-sse``) appends a go-live preflight that
    binds the real ``broadcast --serve-sse`` server on an ephemeral port over a local PGN
    replay and confirms ``/sse`` emits at least one overlay event (task 0209) — so a dead
    live feed is caught before air. Loopback + committed sample PGN, torch/network-free.

    ``model_probe`` (set by ``doctor --model NAME``) appends a preflight that the active
    equity model loads and produces a finite in-range bar (task 0199): a missing/garbled
    artifact FAILs, missing wdl-a fit provenance WARNs. Torch-free for baseline/wdl-a.
    """
    probes = probes or {"stockfish": _probe_stockfish, "maia2": _probe_maia2}
    if engines:
        probes = {name: probes[name] for name in engines if name in probes}
    checks = [check(name, probe) for name, probe in probes.items()]
    if broadcast_probe is not None:
        checks.append(check("broadcast", broadcast_probe))
    if overlay_probe is not None:
        checks.append(check("overlay", overlay_probe))
    if serve_sse_probe is not None:
        checks.append(check("serve-sse", serve_sse_probe))
    if evidence_probe is not None:
        checks.append(check("evidence", evidence_probe))
    if model_probe is not None:
        checks.append(check("model", model_probe))
    return run_doctor(checks, out=out)
