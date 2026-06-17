"""Drama / clutch metrics — the swings the objective bar misses (task 0020).

A streaming differentiator ([[product-wedge-streaming]]): auto-highlight the
*practical* storylines a caster cares about, which a flat centipawn bar can't show.

Every input is a :class:`~chess_equity.broadcast.MoveEvent` — the same per-move record
the broadcast pipeline (0018) emits and the overlay (0019) renders — so drama composes
directly with the live feed: no extra model calls, no state beyond the single event.
A ``MoveEvent`` already carries the mover's practical-equity swing (``delta_equity``,
mover-POV percentage points), the White-POV ``equity`` after the move, and both
clocks, which is all four drama signals need:

- **clutch** — a move with strongly positive Δequity: the player found better than
  their rating/clock is expected to, the highlight-reel moment.
- **missed_win** — a player who was practically winning let it slip (big negative
  Δequity from a winning position).
- **escape** — a player who was practically losing claws back (big positive Δequity
  from a losing position) — the lucky escape / great defence.
- **scramble** — a clock-driven turning point: a real swing while the mover is low on
  time.

The detector is deliberately *quiet*: a single best event per ply, only above
thresholds, so it fires on the moments a human would call out and stays dark on dull
stretches (precision over flash). Tune :data:`THRESHOLDS` against annotated games.

Real numbers track the underlying model: with the placeholder baseline the *machinery*
runs but swings are muted; on Maia-2 (task 0005) the practical swings are real. The
"engine-blind drama" signal (practical swing vs *objective* centipawn swing) needs a
centipawn eval the equity models don't expose yet — deferred to the Stockfish wiring
(task 0028).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from chess_equity.broadcast import MoveEvent

# --------------------------------------------------------------------------- #
# Tunable thresholds (equity in percentage points; clock in seconds)
# --------------------------------------------------------------------------- #

# A move this much better than the mover's rating-typical play is a clutch find.
CLUTCH_DELTA = 8.0
# A swing of this size (either direction) is a "real" practical swing.
SLIP_DELTA = 12.0
# "Practically winning" / "practically losing" from the mover's POV, before the move.
WIN_LEVEL = 70.0
LOSS_LEVEL = 30.0
# A notable swing during time pressure (lower bar than SLIP_DELTA — the clock is the
# story here, not the size of the swing).
SCRAMBLE_DELTA = 6.0
# The mover is "in time trouble" under this many seconds.
SCRAMBLE_SECS = 20.0
# A swing of this many points is maximally dramatic (magnitude saturates at 1.0).
MAX_SWING = 40.0

THRESHOLDS = {
    "CLUTCH_DELTA": CLUTCH_DELTA,
    "SLIP_DELTA": SLIP_DELTA,
    "WIN_LEVEL": WIN_LEVEL,
    "LOSS_LEVEL": LOSS_LEVEL,
    "SCRAMBLE_DELTA": SCRAMBLE_DELTA,
    "SCRAMBLE_SECS": SCRAMBLE_SECS,
}


@dataclass(frozen=True)
class DramaEvent:
    """One highlight-worthy moment, ready for the overlay or a post-game reel.

    ``magnitude`` is a 0..1 drama score (bigger = more dramatic) for ranking a
    highlight reel and sizing an on-screen flash. ``headline`` is a caster-facing
    one-liner. ``equity`` is White-POV after the move; ``delta_equity`` is the mover's
    swing in percentage points.
    """

    game_id: str
    ply: int
    san: str
    kind: str  # clutch | missed_win | escape | scramble
    magnitude: float
    mover_white: bool
    equity: float
    delta_equity: float
    mover_clock: Optional[float]
    headline: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _norm(points: float) -> float:
    """Map a swing in percentage points to a 0..1 magnitude (saturating)."""
    return max(0.0, min(1.0, abs(points) / MAX_SWING))


def _side(mover_white: bool) -> str:
    return "White" if mover_white else "Black"


def _signed(points: float) -> str:
    return f"{points:+.0f}"


def score_event(event: MoveEvent) -> Optional[DramaEvent]:
    """Classify one move's drama, or ``None`` if it isn't highlight-worthy.

    Checks the signals in priority order (let-a-win-slip and lucky-escape are the
    biggest stories, then a clutch find, then a time-scramble swing) and returns at
    most one event, so the stream never flashes more than once per move.
    """
    delta = event.delta_equity
    if delta is None:
        return None  # opening position — nothing to grade.

    mover_white = not event.white_to_move
    after_mover = event.equity if mover_white else 100.0 - event.equity
    before_mover = after_mover - delta
    mover_clock = event.white_clock if mover_white else event.black_clock
    side = _side(mover_white)

    kind: Optional[str] = None
    headline = ""

    if before_mover >= WIN_LEVEL and delta <= -SLIP_DELTA:
        kind = "missed_win"
        headline = (
            f"{side} let a winning position slip on {event.san} "
            f"({before_mover:.0f}%→{after_mover:.0f}%)"
        )
    elif before_mover <= LOSS_LEVEL and delta >= SLIP_DELTA:
        kind = "escape"
        headline = (
            f"{side} claws back from a lost position with {event.san} "
            f"({before_mover:.0f}%→{after_mover:.0f}%)"
        )
    elif delta >= CLUTCH_DELTA:
        kind = "clutch"
        headline = f"{side} finds {event.san} — a clutch move ({_signed(delta)} pts)"
    elif (
        mover_clock is not None
        and mover_clock < SCRAMBLE_SECS
        and abs(delta) >= SCRAMBLE_DELTA
    ):
        kind = "scramble"
        headline = (
            f"Time scramble — {side} ({mover_clock:.0f}s) swings the bar "
            f"{_signed(delta)} pts on {event.san}"
        )

    if kind is None:
        return None

    return DramaEvent(
        game_id=event.game_id,
        ply=event.ply,
        san=event.san,
        kind=kind,
        magnitude=round(_norm(delta), 3),
        mover_white=mover_white,
        equity=event.equity,
        delta_equity=delta,
        mover_clock=mover_clock,
        headline=headline,
    )


def detect(events: Iterable[MoveEvent]) -> List[DramaEvent]:
    """Drama events for a stream of moves, in play order (drops the quiet ones)."""
    out: List[DramaEvent] = []
    for event in events:
        drama = score_event(event)
        if drama is not None:
            out.append(drama)
    return out


def highlights(events: Iterable[MoveEvent], *, top: Optional[int] = None) -> List[DramaEvent]:
    """Post-game highlight reel: drama events ranked by magnitude (ties keep order)."""
    ranked = sorted(detect(events), key=lambda d: d.magnitude, reverse=True)
    return ranked[:top] if top is not None else ranked
