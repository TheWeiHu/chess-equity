"""Live broadcast ingestion: a game feed -> a stream of per-move equity events.

This is the core plumbing for the streaming wedge (task 0018). It turns a live (or
replayed) chess broadcast into a stream of events the overlay (task 0019) can
consume:

    {ply, san, uci, fen, white_clock, black_clock, white_elo, black_elo,
     equity, delta_equity, last_move_grade, ...}

Three pieces compose:

- :class:`BroadcastFeed` — *where the moves come from*. ``poll()`` returns the
  current full PGN of one or more games. :class:`LocalPgnFeed` replays a finished
  PGN one move at a time (for demos/tests, no network); :class:`LichessRoundFeed`
  polls a Lichess broadcast round's public PGN endpoint.
- :class:`GameTracker` — *incremental diffing*. Given the latest PGN for a game, it
  emits only the genuinely new moves, parsing ``[%clk]`` clock tags and computing
  equity + Δequity per move. Handles duplicate polls and out-of-order / truncated
  PGN (a correction) by resyncing.
- :class:`BroadcastIngestor` — *the loop*. Polls the feed, routes each game's PGN to
  its tracker, and emits events. Survives transient feed errors (reconnects).

Equity comes from any :class:`~chess_equity.adapters.EquityModel`; today that is the
placeholder baseline, but Maia-2 (task 0005) drops in unchanged. The clock is parsed,
carried on every event, **and** (task 0097) fed into the emitted bar: when a game has
``[%clk]`` tags, :class:`GameTracker` warps the published ``equity`` by the side-to-move's
time pressure via :func:`chess_equity.clock.clock_adjusted_white_equity`, so a won
position with seconds left reads as less safe on the live overlay. Gate with
``clock_aware`` (CLI ``--clock-aware`` / ``--no-clock-aware``); it is a no-op when no
clocks are present or for correspondence time controls.
"""

from __future__ import annotations

import http.server
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterator, List, Optional

import chess
import chess.pgn

from chess_equity.adapters import EquityModel, ObjectiveEngine
from chess_equity.clock import clock_adjusted_white_equity
from chess_equity.data.schema import tc_bucket

# --------------------------------------------------------------------------- #
# Event + move grading
# --------------------------------------------------------------------------- #

# Δequity grade bands, in equity *percentage points* from the mover's POV. A minimal
# stand-in for full move grading (task 0008): positive = the move helped the mover.
_GRADE_BANDS = [
    (8.0, "brilliant"),
    (2.0, "good"),
    (-2.0, "ok"),
    (-5.0, "inaccuracy"),
    (-12.0, "mistake"),
]


def grade_delta(delta_equity: Optional[float]) -> Optional[str]:
    """Coarse label for a Δequity (mover POV, in percentage points).

    ``None`` for the opening position (no prior move to grade). Full, model-aware
    grading lives in task 0008; this is enough for the overlay to colour a move.
    """
    if delta_equity is None:
        return None
    for threshold, label in _GRADE_BANDS:
        if delta_equity >= threshold:
            return label
    return "blunder"


def live_caption(event: "MoveEvent") -> Optional[str]:
    """One caster-facing sentence for a just-played move, or ``None`` if ungraded.

    The *live* counterpart to the offline reel's lower-thirds (``chess_equity.reel``,
    task 0184): where the reel ranks a finished game's drama into OBS captions, this
    composes a single spoken-style line per move as it streams — TTS/chat-ready, with
    no new model calls. It reuses only fields the event already carries:

    * the move (``san``), its grade (``last_move_grade``) and signed practical swing
      (``delta_equity``, in equity percentage points), and the *mover's* rating — e.g.
      ``"Qxf7 — brilliant, +12% for a 1800 here"``;
    * when the move is dramatic enough that :func:`chess_equity.drama.score_event`
      fires (a clutch / missed win / escape / scramble), the classifier's caster
      ``headline`` is appended after a separator, so a real swing reads as the story it
      is instead of a bare grade.

    Returns ``None`` for the opening position (no prior move to grade), so a caller can
    cleanly skip ungraded ticks.
    """
    if event.last_move_grade is None:
        return None
    # The mover is the side that just moved: in the post-move FEN the side *to* move is
    # the opponent, so the mover is White exactly when it's now Black to move.
    mover_white = not event.white_to_move
    elo = event.white_elo if mover_white else event.black_elo
    who = f"a {elo}" if elo else "an unrated player"
    delta = event.delta_equity
    swing = "" if delta is None else f", {delta:+.0f}% for {who}"
    base = f"{event.san} — {event.last_move_grade}{swing} here"

    # Lazy import: drama imports MoveEvent from this module, so a top-level import cycles.
    from chess_equity.drama import score_event

    drama = score_event(event)
    if drama is not None:
        return f"{base}  ·  {drama.headline}"
    return base


@dataclass(frozen=True)
class MoveEvent:
    """One published move: position, clocks, ratings, and equity.

    ``equity`` is the White-POV bar in [0, 100]% (stable as turns alternate, like the
    eval bar). ``delta_equity`` is the change from the *mover's* POV in percentage
    points — positive means the move improved the mover's practical chances, the
    whole point of the reframe. Clocks are remaining seconds, or ``None`` if the PGN
    carried no ``[%clk]`` tag. ``cp`` is the objective engine's classic centipawn
    eval **from White's POV** (so it lines up with ``equity``), or ``None`` when the
    model exposes no objective cp (e.g. a pure win-prob model, or a mate).
    """

    game_id: str
    ply: int
    san: str
    uci: str
    fen: str
    white_to_move: bool
    white_clock: Optional[float]
    black_clock: Optional[float]
    white_elo: Optional[int]
    black_elo: Optional[int]
    equity: float
    delta_equity: Optional[float]
    last_move_grade: Optional[str]
    source: str
    compute_ms: float
    cp: Optional[float] = None
    resync: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_overlay_event(self) -> Dict[str, object]:
        """Serialize to the overlay's documented ``position`` event.

        The overlay (``overlay/overlay.js``, schema in ``overlay/README.md``)
        consumes a *nested*, White-POV event with ``equity`` in ``[0, 1]`` — not the
        flat internal :class:`MoveEvent` (``equity`` in ``[0, 100]%``, flat
        ``white_clock``/``black_clock``/``last_move_grade`` fields). This is the one
        bridge between the two, so producer and consumer can't silently drift; the
        contract is pinned by ``tests/test_broadcast_overlay_contract.py``.

        ``cp`` is the White-POV objective centipawn eval (the overlay's classic
        ghost tick and the human-edge divergence badge), or ``None`` when no engine
        cp is available — the overlay then simply hides the tick.
        """
        event: Dict[str, object] = {
            "type": "position",
            "ply": self.ply,
            "move": {"san": self.san},
            "equity": self.equity / 100.0,
            "cp": self.cp,
            "clock": {"white": self.white_clock, "black": self.black_clock},
        }
        if self.last_move_grade is not None:
            event["grade"] = {
                "label": self.last_move_grade,
                "delta": None
                if self.delta_equity is None
                else self.delta_equity / 100.0,
            }
        # Real drama classification (tasks 0020/0053): attach the chess_equity.drama
        # verdict so the overlay flares on the actual classifier (clutch / missed_win /
        # escape / scramble) instead of its client-side equity-swing heuristic. Lazy
        # import: drama imports MoveEvent from here, so a top-level import would cycle.
        from chess_equity.drama import score_event

        drama = score_event(self)
        if drama is not None:
            event["drama"] = {
                "kind": drama.kind,
                "magnitude": drama.magnitude,
                "headline": drama.headline,
            }
        return event


@dataclass(frozen=True)
class GameEvent:
    """One-time game metadata in the overlay's ``"game"`` schema (task 0047).

    The bridge emits this once per game, before that game's first :class:`MoveEvent`,
    so the overlay's name-plates show *who* is playing — previously names were parsed
    only to build :func:`_game_id`, never surfaced, so the overlay always fell back to
    literal "White"/"Black". Ratings mirror the per-move ``white_elo``/``black_elo``.
    """

    game_id: str
    white_name: Optional[str]
    black_name: Optional[str]
    white_elo: Optional[int]
    black_elo: Optional[int]
    # 0-based board index within a multi-game round (task 0185). ``None`` for a
    # single-game feed; set to the game's position in the round PGN otherwise, so the
    # overlay can build a board selector and route each event to the chosen board.
    board: Optional[int] = None

    def to_overlay(self) -> Dict[str, object]:
        """Render as the overlay's ``{type: "game", players: {...}}`` event (see
        overlay/README.md). ``name``/``rating`` may be ``null``; overlay.js falls back
        to "White"/"Black" and a blank rating. ``board`` is the 0-based index in a
        multi-game round (omitted when single-game)."""
        event: Dict[str, object] = {
            "type": "game",
            "game_id": self.game_id,
            "players": {
                "white": {"name": self.white_name, "rating": self.white_elo},
                "black": {"name": self.black_name, "rating": self.black_elo},
            },
        }
        if self.board is not None:
            event["board"] = self.board
        return event


# --------------------------------------------------------------------------- #
# Clock / rating parsing from PGN
# --------------------------------------------------------------------------- #


def _parse_elo(headers: chess.pgn.Headers, key: str) -> Optional[int]:
    """Read an Elo header, tolerating ``?`` / blank / non-numeric (common OTB)."""
    raw = headers.get(key, "").strip()
    if not raw or raw == "?":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _player_name(headers: chess.pgn.Headers, key: str) -> Optional[str]:
    """Read a player-name header, tolerating ``?`` / blank (anonymous / OTB)."""
    raw = headers.get(key, "").strip()
    return raw or None if raw != "?" else None


def game_event(
    headers: chess.pgn.Headers,
    game_id: str,
    *,
    white_elo: Optional[int] = None,
    black_elo: Optional[int] = None,
    board: Optional[int] = None,
) -> GameEvent:
    """Build the one-time :class:`GameEvent` for a game from its PGN headers.

    An explicit ``white_elo``/``black_elo`` (the ingestor's override) wins over the
    header so the announced ratings match the ones the trackers actually evaluate at.
    ``board`` is the 0-based index of this game in a multi-game round (task 0185), or
    ``None`` for a single-game feed.
    """
    return GameEvent(
        game_id=game_id,
        white_name=_player_name(headers, "White"),
        black_name=_player_name(headers, "Black"),
        white_elo=white_elo if white_elo is not None else _parse_elo(headers, "WhiteElo"),
        black_elo=black_elo if black_elo is not None else _parse_elo(headers, "BlackElo"),
        board=board,
    )


@dataclass(frozen=True)
class BoardSelector:
    """Pick which board of a multi-game round to follow on the live feed.

    A round PGN (Titled Tuesday, a simul, any multi-board event) carries several
    simultaneous games; by default the ingestor follows *all* of them. A selector
    narrows the stream to one board, chosen either by **player name** (a
    case-insensitive substring matched against either side's name) or by **board
    index** (the 0-based position of the game in the round PGN). With both unset the
    selector matches everything (the default, follow-all behaviour).
    """

    player: Optional[str] = None
    index: Optional[int] = None

    def matches(self, headers: chess.pgn.Headers, index: int) -> bool:
        """True if the game at ``index`` with these ``headers`` should be followed."""
        if self.index is not None and index != self.index:
            return False
        if self.player is not None:
            needle = self.player.casefold()
            white = (headers.get("White", "") or "").casefold()
            black = (headers.get("Black", "") or "").casefold()
            if needle not in white and needle not in black:
                return False
        return True


def parse_board_selector(spec: Optional[str]) -> Optional[BoardSelector]:
    """Interpret a ``--board`` spec into a :class:`BoardSelector` (``None`` = follow all).

    An all-digits spec is a 0-based board index; anything else is a case-insensitive
    player-name substring. A blank/``None`` spec returns ``None`` (default behaviour).
    """
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    if spec.isdigit():
        return BoardSelector(index=int(spec))
    return BoardSelector(player=spec)


def _game_id(headers: chess.pgn.Headers, fallback: int) -> str:
    """Stable-ish identity for a game within a round.

    Prefer an explicit GameId / Site URL; else compose from the pairing so two games
    in the same round don't collide. ``fallback`` (the game's index in the PGN) keeps
    it unique if headers are sparse.
    """
    for key in ("GameId", "Site"):
        val = headers.get(key, "").strip()
        if val and val not in ("?", "https://lichess.org"):
            return val
    white = headers.get("White", "?")
    black = headers.get("Black", "?")
    rnd = headers.get("Round", "?")
    return f"{white}-{black}-R{rnd}#{fallback}"


# --------------------------------------------------------------------------- #
# Per-game incremental tracker
# --------------------------------------------------------------------------- #


class GameTracker:
    """Turns successive PGN snapshots of one game into new-move events.

    Keeps the moves already emitted; each :meth:`ingest` returns only moves beyond
    that point. A snapshot that *diverges* from the moves seen so far — whether it is
    shorter (a walk-back / mid-stream truncation) or replaces an earlier move at the
    same or greater length (an operator correction) — resets state and re-emits from
    the start, flagged ``resync=True`` so a consumer can reconcile by ``ply``. The
    common append-only case (the snapshot just grew) is byte-identical to before:
    nothing in the emitted prefix changed, so no resync fires.
    """

    def __init__(
        self,
        game_id: str,
        model: EquityModel,
        *,
        white_elo: Optional[int],
        black_elo: Optional[int],
        clock_aware: bool = True,
        engine: Optional[ObjectiveEngine] = None,
    ) -> None:
        self.game_id = game_id
        self.model = model
        self.white_elo = white_elo
        self.black_elo = black_elo
        self.clock_aware = clock_aware
        # Time-control bucket for the clock warp, read once from the PGN's TimeControl
        # header on first ingest. None until then; missing/unknown -> "correspondence",
        # whose flag multiplier is 0, so the warp is a safe no-op.
        self.tc_bucket: Optional[str] = None
        # Optional objective engine to fill the centipawn eval when the equity model
        # exposes none (e.g. Maia-2's win-prob has no cp), so the overlay's classic
        # ghost tick + human-edge divergence badge work on a maia2 feed (task 0103).
        # Only consulted when ``equity.cp is None``; models that carry cp are untouched.
        self.engine = engine
        self.emitted_ply = 0
        # The UCIs of every mainline move emitted so far, so a *correction* that
        # replaces an earlier move (not just a walk-back that shortens the PGN) is
        # caught: if the new mainline diverges from this prefix we resync. Rebuilt to
        # the full mainline on every ingest, so the common append-only poll never
        # touches it beyond a cheap matching-prefix comparison.
        self.emitted_ucis: List[str] = []

    def _elos(self) -> tuple[int, int]:
        # The model contract takes ints; default unknown ratings to a neutral 1500
        # (the event still reports the true None so the overlay can show "unrated").
        return (self.white_elo or 1500, self.black_elo or 1500)

    def ingest(self, pgn_text: str) -> List[MoveEvent]:
        """Parse the latest PGN and emit events for moves not yet seen."""
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return []

        # A rating may only appear once the broadcast has metadata; pick it up late.
        if self.white_elo is None:
            self.white_elo = _parse_elo(game.headers, "WhiteElo")
        if self.black_elo is None:
            self.black_elo = _parse_elo(game.headers, "BlackElo")
        if self.tc_bucket is None:
            self.tc_bucket = tc_bucket(game.headers.get("TimeControl", "-"))

        nodes = list(game.mainline())
        new_ucis = [node.move.uci() for node in nodes]
        # Resync when the snapshot diverges from what we've already emitted — either it
        # is shorter (a walk-back / truncated poll) or it replaces a move within the
        # overlapping prefix (an operator correction at the same-or-greater length).
        # Append-only growth leaves the prefix identical, so this is a no-op there.
        overlap = min(len(new_ucis), len(self.emitted_ucis))
        diverged = new_ucis[:overlap] != self.emitted_ucis[:overlap]
        resync = False
        if len(nodes) < self.emitted_ply or diverged:
            self.emitted_ply = 0
            resync = True

        white_elo, black_elo = self._elos()
        events: List[MoveEvent] = []

        # Running clocks + prior equity, rebuilt up to the last emitted ply so deltas
        # and clock carry-over are correct even on the first ingest after a resync.
        white_clock: Optional[float] = None
        black_clock: Optional[float] = None
        board = game.board()
        prev_equity_white = self._equity_white(board.fen(), white_elo, black_elo)

        for ply, node in enumerate(nodes, start=1):
            mover_white = node.parent.board().turn == chess.WHITE
            # The mover's clock *before* this move — the time pressure they were under
            # while facing the prior position (the "before" bar for the clock-aware delta).
            prev_mover_clock = white_clock if mover_white else black_clock
            clock = node.clock()
            if clock is not None:
                if mover_white:
                    white_clock = clock
                else:
                    black_clock = clock

            if ply <= self.emitted_ply:
                # Already emitted — just keep clocks/equity in sync for later deltas.
                prev_equity_white = self._equity_white(
                    node.board().fen(), white_elo, black_elo
                )
                continue

            san = node.parent.board().san(node.move)
            fen = node.board().fen()
            t0 = time.perf_counter()
            equity = self.model.evaluate(fen, white_elo, black_elo)
            compute_ms = (time.perf_counter() - t0) * 1000.0
            equity_white = equity.equity_white
            cp = self._white_pov_cp(equity, fen, mover_white)

            # The published bar reflects the side-to-move's time pressure (task 0097): in
            # the post-move FEN the side to move is the mover's opponent, so warp by their
            # remaining clock. A no-op without clocks / for correspondence / clock-blind.
            bar_equity = self._clock_warp(
                equity_white, white_clock, black_clock, stm_white=not mover_white
            )

            # Δequity from the mover's POV (task 0106): grade the swing in *practical*
            # win chance, so a low-clock survival reads as the save it is rather than the
            # raw positional dip. Both bars are clock-warped at their own ply/clock state —
            # the "before" position faced the mover (warp by their pre-move clock), the
            # "after" position faces the opponent (the already-warped published bar). When
            # clock-blind (no tc_bucket / no clocks / clock_aware off) ``_clock_warp`` is a
            # no-op, so this degrades to the plain raw-equity delta. White reads the
            # White-POV bar directly; Black reads its complement.
            before_bar_white = self._clock_warp(
                prev_equity_white,
                prev_mover_clock if mover_white else white_clock,
                black_clock if mover_white else prev_mover_clock,
                stm_white=mover_white,
            )
            after = bar_equity if mover_white else 100.0 - bar_equity
            before = before_bar_white if mover_white else 100.0 - before_bar_white
            delta = after - before

            events.append(
                MoveEvent(
                    game_id=self.game_id,
                    ply=ply,
                    san=san,
                    uci=node.move.uci(),
                    fen=fen,
                    white_to_move=(not mover_white),
                    white_clock=white_clock,
                    black_clock=black_clock,
                    white_elo=self.white_elo,
                    black_elo=self.black_elo,
                    equity=bar_equity,
                    delta_equity=delta,
                    last_move_grade=grade_delta(delta),
                    source=self.model.__class__.__name__,
                    compute_ms=compute_ms,
                    cp=cp,
                    resync=resync,
                )
            )
            prev_equity_white = equity_white

        self.emitted_ply = len(nodes)
        self.emitted_ucis = new_ucis
        return events

    def _equity_white(self, fen: str, white_elo: int, black_elo: int) -> float:
        return self.model.evaluate(fen, white_elo, black_elo).equity_white

    def _clock_warp(
        self,
        equity_white: float,
        white_clock: Optional[float],
        black_clock: Optional[float],
        *,
        stm_white: bool,
    ) -> float:
        """Warp a White-POV bar (in [0, 100]%) by the side-to-move's time pressure.

        Returns ``equity_white`` unchanged when clock-awareness is off, no tc_bucket has
        been read yet, or the side to move has no recorded clock — so clock-blind feeds
        and correspondence games pass through untouched. Otherwise scales through
        :func:`chess_equity.clock.clock_adjusted_white_equity` (which works in [0, 1]).
        """
        if not self.clock_aware or self.tc_bucket is None:
            return equity_white
        stm_clock = white_clock if stm_white else black_clock
        if stm_clock is None:
            return equity_white
        adjusted = clock_adjusted_white_equity(
            equity_white / 100.0, stm_clock, self.tc_bucket, white_to_move=stm_white
        )
        return adjusted * 100.0

    def _white_pov_cp(self, equity, fen: str, mover_white: bool) -> Optional[float]:
        """The classic centipawn eval for ``fen``, from White's POV (matches equity).

        Prefer the equity model's own ``cp``; when it has none (e.g. Maia-2's win-prob
        model) fall back to the optional objective ``engine`` so the overlay's cp ghost
        tick + divergence badge still work (task 0103). Both the model cp and the engine
        eval are *side-to-move* POV of the post-move ``fen`` (whose side to move is the
        mover's opponent), so the flip to White POV is the same for either source. A mate
        (engine returns ``cp=None``) stays ``None`` — the overlay then hides the tick.
        """
        cp_stm = equity.cp
        if cp_stm is None and self.engine is not None:
            cp_stm = self.engine.eval(fen).cp
        if cp_stm is None:
            return None
        return cp_stm if not mover_white else -cp_stm


# --------------------------------------------------------------------------- #
# Feeds
# --------------------------------------------------------------------------- #


class BroadcastFeed:
    """A source of PGN snapshots. ``poll()`` returns the current PGN, or None.

    A snapshot may contain several concatenated games (a whole broadcast round). The
    ingestor splits them and routes each to its own tracker.
    """

    def poll(self) -> Optional[str]:  # pragma: no cover - interface
        raise NotImplementedError


class LocalPgnFeed(BroadcastFeed):
    """Replay a finished PGN as if it were live, one move per poll.

    Useful for demos and tests with zero network. Each :meth:`poll` reveals one more
    half-move of the (single) game, so a tracker downstream sees the game grow move
    by move exactly as a live feed would. Returns ``None`` once the game is complete.
    """

    def __init__(self, pgn_text: str, *, moves_per_poll: int = 1) -> None:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None or not list(game.mainline_moves()):
            raise ValueError("no game (with moves) found in PGN")
        self._headers = game.headers
        # Keep each move's clock so the replayed snapshots carry [%clk] tags, exactly
        # like a live feed would (the whole point of streaming the clock downstream).
        self._moves = [(node.move, node.clock()) for node in game.mainline()]
        self._moves_per_poll = max(1, moves_per_poll)
        self._revealed = 0

    def poll(self) -> Optional[str]:
        if self._revealed >= len(self._moves):
            return None
        self._revealed = min(len(self._moves), self._revealed + self._moves_per_poll)
        return self._render(self._revealed)

    def _render(self, upto: int) -> str:
        game = chess.pgn.Game()
        game.headers.update(self._headers)
        node: chess.pgn.GameNode = game
        for move, clock in self._moves[:upto]:
            node = node.add_variation(move)
            if clock is not None:
                node.set_clock(clock)
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
        return game.accept(exporter)


class LichessRoundFeed(BroadcastFeed):
    """Poll a Lichess broadcast round's public PGN endpoint.

    ``round_id`` is the 8-char id from a broadcast round URL. The endpoint returns the
    concatenated PGN of every game in the round, updated as moves come in:
    ``https://lichess.org/api/broadcast/round/<id>.pgn`` (see the Lichess API docs).
    Network errors raise :class:`FeedError`; the ingestor catches them and retries
    (reconnect) rather than crashing.
    """

    BASE = "https://lichess.org/api/broadcast/round"

    def __init__(self, round_id: str, *, timeout: float = 10.0, token: Optional[str] = None) -> None:
        self.round_id = round_id
        self.timeout = timeout
        self.token = token

    def poll(self) -> Optional[str]:
        url = f"{self.BASE}/{self.round_id}.pgn"
        req = urllib.request.Request(url, headers={"Accept": "application/x-chess-pgn"})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise FeedError(f"lichess round {self.round_id}: {exc}") from exc


class UrlPgnFeed(BroadcastFeed):
    """Poll an arbitrary public PGN URL (chess.com export, a static file server, …).

    A generic fallback feed so the ingestor works for non-Lichess streams.
    """

    def __init__(self, url: str, *, timeout: float = 10.0) -> None:
        self.url = url
        self.timeout = timeout

    def poll(self) -> Optional[str]:
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise FeedError(f"{self.url}: {exc}") from exc


class FeedError(RuntimeError):
    """A transient feed failure the ingestor should retry rather than crash on."""


def feed_from_spec(
    spec: str, *, token: Optional[str] = None, moves_per_poll: int = 1
) -> BroadcastFeed:
    """Build the right :class:`BroadcastFeed` from a single source string.

    One front door for "point me at a feed" callers (e.g. the ``doctor`` go-live
    preflight) so they don't re-implement the --pgn/--round/--url dispatch:

    * an existing **file path** → :class:`LocalPgnFeed` (offline replay),
    * an **http(s):// URL** → :class:`UrlPgnFeed`,
    * anything else → a Lichess broadcast **round id** → :class:`LichessRoundFeed`.
    """
    if os.path.exists(spec):
        with open(spec, encoding="utf-8") as fh:
            return LocalPgnFeed(fh.read(), moves_per_poll=moves_per_poll)
    if spec.startswith("http://") or spec.startswith("https://"):
        return UrlPgnFeed(spec)
    return LichessRoundFeed(spec, token=token)


# --------------------------------------------------------------------------- #
# Splitting a multi-game PGN snapshot
# --------------------------------------------------------------------------- #


def split_games(pgn_text: str) -> List[str]:
    """Split a concatenated PGN snapshot into one PGN string per game.

    Each PGN game begins with an ``[Event ...]`` tag at the start of a line, so we cut
    on those boundaries. Robust to the blank-line / movetext variations a live feed
    produces.
    """
    starts = [m.start() for m in re.finditer(r"(?m)^\[Event ", pgn_text)]
    if not starts:
        return [pgn_text] if pgn_text.strip() else []
    games: List[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(pgn_text)
        chunk = pgn_text[start:end].strip()
        if chunk:
            games.append(chunk)
    return games


# --------------------------------------------------------------------------- #
# The ingestor loop
# --------------------------------------------------------------------------- #


@dataclass
class IngestStats:
    """What happened during a run — for logging the documented latency target."""

    polls: int = 0
    events: int = 0
    errors: int = 0
    max_compute_ms: float = 0.0
    # How many times the feed recovered after one or more consecutive errors (a
    # transient drop that self-healed), and the longest backoff delay we waited. Lets
    # the run summary show that a live stream rode out feed hiccups rather than crashing.
    reconnects: int = 0
    max_backoff_s: float = 0.0


class BroadcastIngestor:
    """Poll a feed and emit a real-time stream of :class:`MoveEvent`.

    Routes each game in the snapshot to a per-game :class:`GameTracker` (so a whole
    round streams at once), recovers from transient :class:`FeedError`\\ s by retrying
    on the next tick (reconnect), and stops after ``max_polls`` empty/None polls so a
    replay terminates cleanly.
    """

    def __init__(
        self,
        feed: BroadcastFeed,
        model: EquityModel,
        *,
        white_elo: Optional[int] = None,
        black_elo: Optional[int] = None,
        clock_aware: bool = True,
        engine: Optional[ObjectiveEngine] = None,
        select: Optional[BoardSelector] = None,
    ) -> None:
        self.feed = feed
        self.model = model
        self.white_elo = white_elo
        self.black_elo = black_elo
        self.clock_aware = clock_aware
        # Which board(s) of a multi-game round to follow; None = all (the default).
        self.select = select
        # Objective engine for the cp fallback on cp-less models (task 0103); threaded
        # to every per-game tracker.
        self.engine = engine
        self._trackers: Dict[str, GameTracker] = {}
        self.stats = IngestStats()
        # Fired once per game, the first time it is seen, with its :class:`GameEvent`
        # (overlay "game" metadata). Optional so the MoveEvent stream is unchanged when
        # a caller doesn't care (e.g. the existing tests). The CLI wires it to emit the
        # game line before that game's moves.
        self.on_game: Optional[Callable[["GameEvent"], None]] = None
        self._announced: set[str] = set()

    def _tracker_for(self, game_id: str) -> GameTracker:
        tracker = self._trackers.get(game_id)
        if tracker is None:
            tracker = GameTracker(
                game_id,
                self.model,
                white_elo=self.white_elo,
                black_elo=self.black_elo,
                clock_aware=self.clock_aware,
                engine=self.engine,
            )
            self._trackers[game_id] = tracker
        return tracker

    def ingest_snapshot(self, pgn_text: str) -> List[MoveEvent]:
        """Process one PGN snapshot (possibly many games) into new events."""
        events: List[MoveEvent] = []
        games = list(split_games(pgn_text))
        # A multi-board round (>1 game in the snapshot) tags each game with its 0-based
        # board index so the overlay can offer a live board selector (task 0185); a
        # single-game feed leaves board=None and the overlay shows no selector.
        multi_board = len(games) > 1
        for index, game_pgn in enumerate(games):
            headers = chess.pgn.read_headers(io.StringIO(game_pgn))
            if headers is None:
                continue
            # Multi-board round: skip games the streamer isn't following (task 0182).
            if self.select is not None and not self.select.matches(headers, index):
                continue
            gid = _game_id(headers, index)
            if gid not in self._announced:
                self._announced.add(gid)
                if self.on_game is not None:
                    self.on_game(
                        game_event(
                            headers,
                            gid,
                            white_elo=self.white_elo,
                            black_elo=self.black_elo,
                            board=index if multi_board else None,
                        )
                    )
            new = self._tracker_for(gid).ingest(game_pgn)
            events.extend(new)
        for ev in events:
            self.stats.max_compute_ms = max(self.stats.max_compute_ms, ev.compute_ms)
        self.stats.events += len(events)
        return events

    def stream(
        self,
        *,
        interval: float = 2.0,
        max_polls: Optional[int] = None,
        max_idle_polls: Optional[int] = 1,
        sleep: Callable[[float], None] = time.sleep,
        heartbeat: bool = False,
        reconnect_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        backoff_max: float = 30.0,
        on_reconnect: Optional[Callable[[int, float], None]] = None,
    ) -> Iterator[Optional[MoveEvent]]:
        """Yield events as they arrive. Generator so callers control the sink.

        ``interval`` seconds between polls; ``max_polls`` caps total polls (None =
        unbounded, for a true live stream); ``max_idle_polls`` stops after that many
        consecutive polls produced no PGN (so a finished replay or a dead round ends).
        ``sleep`` is injectable for tests.

        On a transient :class:`FeedError` (a dropped/erroring live feed) the loop does
        not crash: it waits a **bounded exponential backoff** and retries, resuming from
        the last seen move (each :class:`GameTracker` keeps its emitted prefix, so the
        next good poll re-emits only genuinely new moves). The delay starts at
        ``reconnect_backoff`` seconds, multiplies by ``backoff_factor`` per *consecutive*
        error, is capped at ``backoff_max``, and **resets the moment a poll succeeds** —
        so a feed that flickers doesn't ramp the wait forever. ``on_reconnect(attempt,
        delay)`` is called each time we schedule a retry, so a caller (the CLI) can log a
        visible 'reconnecting' state for the streamer. Healthy idle polls (a round that
        hasn't started) still wait the normal ``interval``, not a backoff.

        With ``heartbeat=True`` an idle poll that does *not* end the stream yields
        ``None`` — a tick the SSE bridge turns into a keep-alive comment so an
        early-tuned-in connection (a round that hasn't started) isn't dropped. Default
        ``False`` keeps the historical ``Iterator[MoveEvent]`` contract for the JSONL
        path and existing callers.
        """
        polls = 0
        idle = 0
        first = True
        # Pending delay before the *next* poll: 0 means "use the normal interval". It
        # grows geometrically per consecutive FeedError (the reconnect backoff) and is
        # reset to 0 by any successful poll, so a recovered feed returns to cadence.
        backoff = 0.0
        consecutive_errors = 0
        while max_polls is None or polls < max_polls:
            if not first:
                sleep(backoff if backoff > 0 else interval)
            first = False
            polls += 1
            self.stats.polls = polls
            try:
                snapshot = self.feed.poll()
            except FeedError:
                self.stats.errors += 1
                idle += 1
                consecutive_errors += 1
                # Bounded exponential backoff before the next reconnect attempt.
                backoff = min(
                    backoff_max,
                    reconnect_backoff * (backoff_factor ** (consecutive_errors - 1)),
                )
                self.stats.max_backoff_s = max(self.stats.max_backoff_s, backoff)
                if on_reconnect is not None:
                    on_reconnect(consecutive_errors, backoff)
                if max_idle_polls is not None and idle >= max_idle_polls and polls > 1:
                    # Keep retrying live feeds; only give up if we never connected.
                    if not self._trackers:
                        break
                if heartbeat:
                    yield None
                continue
            # A poll came back (even an empty one): the connection is healthy again, so
            # clear any reconnect backoff and count the recovery if we'd been erroring.
            if consecutive_errors:
                self.stats.reconnects += 1
            consecutive_errors = 0
            backoff = 0.0
            if not snapshot:
                idle += 1
                if max_idle_polls is not None and idle >= max_idle_polls:
                    break
                if heartbeat:
                    yield None
                continue
            idle = 0
            for event in self.ingest_snapshot(snapshot):
                yield event

    def run(
        self,
        emit: Callable[[MoveEvent], None],
        *,
        interval: float = 2.0,
        max_polls: Optional[int] = None,
        max_idle_polls: Optional[int] = 1,
        sleep: Callable[[float], None] = time.sleep,
        reconnect_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        backoff_max: float = 30.0,
        on_reconnect: Optional[Callable[[int, float], None]] = None,
    ) -> IngestStats:
        """Drive :meth:`stream`, calling ``emit`` for each event. Returns stats.

        ``reconnect_backoff`` / ``backoff_factor`` / ``backoff_max`` / ``on_reconnect``
        configure the reconnect behaviour documented on :meth:`stream`.
        """
        for event in self.stream(
            interval=interval,
            max_polls=max_polls,
            max_idle_polls=max_idle_polls,
            sleep=sleep,
            reconnect_backoff=reconnect_backoff,
            backoff_factor=backoff_factor,
            backoff_max=backoff_max,
            on_reconnect=on_reconnect,
        ):
            if event is not None:  # heartbeat is off here, but stay type-safe
                emit(event)
        return self.stats


# --------------------------------------------------------------------------- #
# Live SSE bridge: a round straight into the overlay (task 0094)
# --------------------------------------------------------------------------- #


# Sentinel yielded by overlay_events on an idle poll — the SSE bridge turns it into a
# keep-alive comment (": ...\n\n"), which EventSource ignores, so an idle connection
# (a round that hasn't started) stays open instead of being dropped by a proxy/OBS.
HEARTBEAT = object()


def sse_frame(event: Dict[str, object]) -> str:
    """Format one overlay event as a Server-Sent-Events ``data:`` frame.

    Matches what ``overlay/feed.js`` (``EventSource.onmessage``) parses: a single
    ``data: <json>`` line terminated by a blank line.
    """
    return "data: " + json.dumps(event) + "\n\n"


def overlay_events(ingestor: "BroadcastIngestor", **stream_kwargs) -> Iterator[object]:
    """Bridge a :class:`BroadcastIngestor` into the overlay's event schema.

    Yields overlay-shaped dicts in the order ``overlay.js`` expects: a one-time
    ``game`` event (player name-plates) before each game's first ``position`` event,
    then a ``position`` event per move. This is :meth:`BroadcastIngestor.stream`
    re-serialized through :meth:`MoveEvent.to_overlay_event` /
    :meth:`GameEvent.to_overlay` — the same bridge the JSONL path uses, but as a
    generator the SSE server can write frame-by-frame as moves arrive.

    ``stream_kwargs`` pass straight through to ``stream`` (``interval`` / ``max_polls``
    / ``max_idle_polls`` / ``sleep`` / ``heartbeat``). With ``heartbeat=True`` an idle
    poll yields the :data:`HEARTBEAT` sentinel instead of a ``position`` dict.
    """
    queued: List[Dict[str, object]] = []
    # Board roster for the overlay's live board selector (task 0185). As each game of a
    # multi-game round is announced we add it to the roster and re-emit a single
    # ``boards`` event listing every known board (index + players), so the overlay can
    # render/refresh its selector; ``board_of`` lets us stamp the board index onto each
    # game's position events so the overlay can route them to the chosen board. A
    # single-game feed never carries a board index, so no roster/selector appears.
    roster: List[Dict[str, object]] = []
    board_of: Dict[str, int] = {}

    def on_game(game: "GameEvent") -> None:
        ev = game.to_overlay()
        if game.board is not None:
            board_of[game.game_id] = game.board
            roster.append(
                {
                    "index": game.board,
                    "game_id": game.game_id,
                    "players": ev["players"],
                }
            )
            # Announce the full roster (in board order — games appear in index order in
            # the round PGN) before this board's game event.
            queued.append({"type": "boards", "boards": list(roster)})
        queued.append(ev)

    ingestor.on_game = on_game
    for move_event in ingestor.stream(**stream_kwargs):
        if move_event is None:  # idle-poll heartbeat tick from stream()
            yield HEARTBEAT
            continue
        while queued:  # game/boards announcements fire during the poll, before their moves
            yield queued.pop(0)
        event = move_event.to_overlay_event()
        board = board_of.get(move_event.game_id)
        if board is not None:
            event["board"] = board
        yield event
    while queued:
        yield queued.pop(0)


def _sse_handler(event_source: Callable[[], Iterator[object]], directory: Optional[str]):
    """Build a request handler that serves ``/sse`` as a live event stream.

    ``event_source`` is a zero-arg factory returning a *fresh* iterator of overlay
    events per connection (so each browser source replays/streams from the start).
    When ``directory`` is set the handler also serves the overlay's static files, so
    ``http://host:port/?src=/sse`` is a one-command overlay; otherwise only ``/sse``
    is served.
    """

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            if directory is not None:
                super().__init__(*args, directory=directory, **kwargs)
            else:
                super().__init__(*args, **kwargs)

        def do_GET(self):  # noqa: N802 (stdlib API)
            if self.path.split("?")[0] == "/sse":
                return self._stream_sse()
            if directory is None:
                self.send_error(404, "only /sse is served")
                return None
            return super().do_GET()

        def _stream_sse(self) -> None:
            # One stream per connection; close the socket when it ends (a finite replay
            # terminates, a live feed runs until the round ends) so clients see EOF.
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                for event in event_source():
                    if isinstance(event, dict):
                        self.wfile.write(sse_frame(event).encode("utf-8"))
                    else:
                        # HEARTBEAT sentinel → an SSE comment: ignored by EventSource,
                        # just keeps the idle socket warm.
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the overlay / OBS closed the source — normal.

        def log_message(self, format, *args):  # quieter console
            pass

    return _Handler


def make_sse_server(
    event_source: Callable[[], Iterator[object]],
    *,
    port: int = 0,
    host: str = "127.0.0.1",
    directory: Optional[str] = None,
) -> "http.server.ThreadingHTTPServer":
    """Build (but don't start) a threaded SSE server. ``port=0`` lets the OS pick one
    (the bound port is then ``server.server_address[1]`` — handy for tests)."""
    return http.server.ThreadingHTTPServer((host, port), _sse_handler(event_source, directory))


def serve_sse(
    event_source: Callable[[], Iterator[object]],
    *,
    port: int,
    host: str = "127.0.0.1",
    directory: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> None:
    """Serve overlay events as SSE on ``host:port`` until interrupted (Ctrl-C)."""
    httpd = make_sse_server(event_source, port=port, host=host, directory=directory)
    bound = httpd.server_address[1]
    log(f"chess-equity SSE bridge: http://localhost:{bound}/sse")
    if directory is not None:
        log(f"  one-command overlay : http://localhost:{bound}/?src=/sse")
    log("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
