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
placeholder baseline, but Maia-2 (task 0005) drops in unchanged. Clock is parsed and
carried on every event so a clock-aware model (task 0015) can use it later — this
module does not yet feed the clock *into* the equity computation.
"""

from __future__ import annotations

import io
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterator, List, Optional

import chess
import chess.pgn

from chess_equity.adapters import EquityModel

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


@dataclass(frozen=True)
class MoveEvent:
    """One published move: position, clocks, ratings, and equity.

    ``equity`` is the White-POV bar in [0, 100]% (stable as turns alternate, like the
    eval bar). ``delta_equity`` is the change from the *mover's* POV in percentage
    points — positive means the move improved the mover's practical chances, the
    whole point of the reframe. Clocks are remaining seconds, or ``None`` if the PGN
    carried no ``[%clk]`` tag.
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

        ``cp`` (the classic centipawn ghost tick) is not yet plumbed through the
        broadcast path, so it is emitted as ``None`` — optional in the schema, and
        the overlay simply hides the tick. Threading the objective engine's cp
        (already on :class:`~chess_equity.types.Equity`) is a follow-up.
        """
        event: Dict[str, object] = {
            "type": "position",
            "ply": self.ply,
            "move": {"san": self.san},
            "equity": self.equity / 100.0,
            "cp": None,
            "clock": {"white": self.white_clock, "black": self.black_clock},
        }
        if self.last_move_grade is not None:
            event["grade"] = {
                "label": self.last_move_grade,
                "delta": None
                if self.delta_equity is None
                else self.delta_equity / 100.0,
            }
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
    that point. A snapshot with *fewer* moves than seen (a broadcast correction /
    re-sync) resets state and re-emits from the start, flagged ``resync=True`` so a
    consumer can reconcile by ``ply``.
    """

    def __init__(
        self,
        game_id: str,
        model: EquityModel,
        *,
        white_elo: Optional[int],
        black_elo: Optional[int],
    ) -> None:
        self.game_id = game_id
        self.model = model
        self.white_elo = white_elo
        self.black_elo = black_elo
        self.emitted_ply = 0

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

        nodes = list(game.mainline())
        resync = False
        if len(nodes) < self.emitted_ply:
            # The feed walked the game back (correction). Replay from scratch.
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
            equity_white = self._equity_white(fen, white_elo, black_elo)
            compute_ms = (time.perf_counter() - t0) * 1000.0

            # Δequity from the mover's POV: White reads equity_white directly, Black
            # reads its complement.
            after = equity_white if mover_white else 100.0 - equity_white
            before = prev_equity_white if mover_white else 100.0 - prev_equity_white
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
                    equity=equity_white,
                    delta_equity=delta,
                    last_move_grade=grade_delta(delta),
                    source=self.model.__class__.__name__,
                    compute_ms=compute_ms,
                    resync=resync,
                )
            )
            prev_equity_white = equity_white

        self.emitted_ply = len(nodes)
        return events

    def _equity_white(self, fen: str, white_elo: int, black_elo: int) -> float:
        return self.model.evaluate(fen, white_elo, black_elo).equity_white


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
    ) -> None:
        self.feed = feed
        self.model = model
        self.white_elo = white_elo
        self.black_elo = black_elo
        self._trackers: Dict[str, GameTracker] = {}
        self.stats = IngestStats()

    def _tracker_for(self, game_id: str) -> GameTracker:
        tracker = self._trackers.get(game_id)
        if tracker is None:
            tracker = GameTracker(
                game_id,
                self.model,
                white_elo=self.white_elo,
                black_elo=self.black_elo,
            )
            self._trackers[game_id] = tracker
        return tracker

    def ingest_snapshot(self, pgn_text: str) -> List[MoveEvent]:
        """Process one PGN snapshot (possibly many games) into new events."""
        events: List[MoveEvent] = []
        for index, game_pgn in enumerate(split_games(pgn_text)):
            headers = chess.pgn.read_headers(io.StringIO(game_pgn))
            if headers is None:
                continue
            gid = _game_id(headers, index)
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
    ) -> Iterator[MoveEvent]:
        """Yield events as they arrive. Generator so callers control the sink.

        ``interval`` seconds between polls; ``max_polls`` caps total polls (None =
        unbounded, for a true live stream); ``max_idle_polls`` stops after that many
        consecutive polls produced no PGN (so a finished replay or a dead round ends).
        ``sleep`` is injectable for tests.
        """
        polls = 0
        idle = 0
        first = True
        while max_polls is None or polls < max_polls:
            if not first:
                sleep(interval)
            first = False
            polls += 1
            self.stats.polls = polls
            try:
                snapshot = self.feed.poll()
            except FeedError:
                self.stats.errors += 1
                idle += 1
                if max_idle_polls is not None and idle >= max_idle_polls and polls > 1:
                    # Keep retrying live feeds; only give up if we never connected.
                    if not self._trackers:
                        break
                continue
            if not snapshot:
                idle += 1
                if max_idle_polls is not None and idle >= max_idle_polls:
                    break
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
    ) -> IngestStats:
        """Drive :meth:`stream`, calling ``emit`` for each event. Returns stats."""
        for event in self.stream(
            interval=interval,
            max_polls=max_polls,
            max_idle_polls=max_idle_polls,
            sleep=sleep,
        ):
            emit(event)
        return self.stats
