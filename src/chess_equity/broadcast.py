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
from itertools import groupby
from typing import Callable, Dict, Iterable, Iterator, List, Optional, TextIO, Tuple

import chess
import chess.pgn

from chess_equity.adapters import EquityModel, ObjectiveEngine
from chess_equity.clock import (
    clock_adjusted_white_equity,
    flag_risk,
    is_flag_risk_alert,
)
from chess_equity.data.schema import tc_bucket
from chess_equity.grading import ACCURATE_LABELS

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


# Out-of-distribution high-rating flag (task 0255). Maia-2's highest rating embedding is
# a single coarse ``">2000"`` bucket (see product-wedge-streaming gap #1): it cannot tell
# a 2200 from a 2800, so when BOTH players sit above this threshold the equity bar is a
# coarse-bucket read, not a true 2200-vs-2800 distinction. The overlay marks the bar
# lower-confidence in that regime instead of implying a resolution the model lacks. A
# single named constant so the boundary is testable and tunable; ``> THRESHOLD`` (strict)
# so exactly-2000 players are still in-distribution.
RATING_OOD_THRESHOLD = 2000


def is_rating_ood(white_elo: Optional[int], black_elo: Optional[int]) -> bool:
    """True iff BOTH ratings are above :data:`RATING_OOD_THRESHOLD` (model out of bucket).

    Pure function of the two ratings — no model call. An unknown rating (``None``, common
    on OTB/anonymous feeds) is *not* out-of-distribution: we only flag when we can confirm
    both sides clear the coarse bucket, so a missing rating degrades to in-distribution
    rather than a bogus uncertainty mark.
    """
    if white_elo is None or black_elo is None:
        return False
    return white_elo > RATING_OOD_THRESHOLD and black_elo > RATING_OOD_THRESHOLD


def _accuracy_pct(accurate: int, total: int) -> Optional[float]:
    """Running ok-or-better accuracy %, or ``None`` before the side has a graded move.

    ``None`` (not ``0.0``) until ``total > 0`` so the overlay can show "—" for a side
    that has not moved yet rather than a misleading flat 0%.
    """
    return None if total <= 0 else round(100.0 * accurate / total, 1)


def cumulative_accuracy(events: "Iterable[MoveEvent]") -> Dict[str, Optional[float]]:
    """Per-side ok-or-better accuracy over a (finished) move-event stream.

    The post-hoc counterpart to the live running figure the broadcast threads onto each
    event: pools every graded move by its mover and reports the share graded
    ok-or-better (:data:`~chess_equity.grading.ACCURATE_LABELS`), the same definition the
    ``grade --round`` leaderboard uses. Returns ``{"white": pct|None, "black": pct|None}``
    in 0..100. Pure reduction over fields the events already carry — no model calls — so
    the live ``cumulative_accuracy_white``/``black`` on the final move of a game equals
    this over the whole stream (pinned by the broadcast tests).
    """
    accurate = {True: 0, False: 0}  # mover_white -> ok-or-better count
    total = {True: 0, False: 0}
    for event in events:
        if event.last_move_grade is None:
            continue
        mover_white = not event.white_to_move
        total[mover_white] += 1
        if event.last_move_grade in ACCURATE_LABELS:
            accurate[mover_white] += 1
    return {
        "white": _accuracy_pct(accurate[True], total[True]),
        "black": _accuracy_pct(accurate[False], total[False]),
    }


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
    # Running per-side move accuracy *through this move*, in 0..100 (task 0245): the
    # share of each side's moves so far graded ok-or-better, so the overlay can show
    # "White 94% / Black 88%" live and update every ply. ``None`` for a side that has
    # not moved yet. At a game's last move these equal :func:`cumulative_accuracy` over
    # the whole stream. Both sides ride every event so the overlay needs no extra state.
    cumulative_accuracy_white: Optional[float] = None
    cumulative_accuracy_black: Optional[float] = None
    # Per-side flag risk in [0, MAX_FLAG_RISK=0.6] (task 0243): each side's modelled
    # P(loses on time) from its own remaining clock + the game's time control, via
    # :func:`chess_equity.clock.flag_risk`. Distinct from the raw-seconds low-clock cue
    # (task 0105) — this surfaces the MODEL's time-trouble read so the overlay can light
    # a flag/flame alert (see :func:`chess_equity.clock.is_flag_risk_alert`). ``None`` for
    # a clock-blind side (no ``[%clk]``), so clock-blind feeds carry no flag-risk block.
    flag_risk_white: Optional[float] = None
    flag_risk_black: Optional[float] = None

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

        ``white_to_move`` is the authoritative side-to-move in the position this
        event describes (the post-move FEN's turn). The overlay's ``?pov=stm``
        readout reads it directly instead of guessing from ply parity, which is only
        a fallback for replay feeds that omit the flag.
        """
        event: Dict[str, object] = {
            "type": "position",
            "ply": self.ply,
            "move": {"san": self.san},
            "white_to_move": self.white_to_move,
            "equity": self.equity / 100.0,
            "cp": self.cp,
            "clock": {"white": self.white_clock, "black": self.black_clock},
            # Out-of-distribution high-rating flag (task 0255): True when both ratings clear
            # the coarse ``">2000"`` Maia-2 bucket, so the overlay can mark the bar
            # lower-confidence. Always present (a plain bool) so the overlay can clear a
            # stale marker on the next in-distribution move.
            "rating_ood": is_rating_ood(self.white_elo, self.black_elo),
        }
        if self.last_move_grade is not None:
            event["grade"] = {
                "label": self.last_move_grade,
                "delta": None
                if self.delta_equity is None
                else self.delta_equity / 100.0,
            }
        # Running per-side accuracy (task 0245), in 0..100. Emitted as soon as either
        # side has a graded move so the overlay's "White 94% / Black 88%" readout
        # updates live; a side that has not moved yet stays ``None`` (overlay shows "—").
        if (
            self.cumulative_accuracy_white is not None
            or self.cumulative_accuracy_black is not None
        ):
            event["accuracy"] = {
                "white": self.cumulative_accuracy_white,
                "black": self.cumulative_accuracy_black,
            }
        # Per-side flag-risk alert (task 0243): the MODEL's time-trouble read per side, so
        # the overlay can light a flag/flame badge when a side is in real danger of losing
        # on time — distinct from the raw-seconds low-clock nameplate cue (task 0105) and
        # the drama toast (0241). Emitted only when a side has a flag_risk (clocked +
        # time-control known); absent entirely on clock-blind feeds, so the overlay
        # degrades gracefully (no badge) rather than showing a bogus zero.
        if self.flag_risk_white is not None or self.flag_risk_black is not None:
            event["flag_risk"] = {
                "white": {
                    "risk": self.flag_risk_white,
                    "alert": is_flag_risk_alert(self.flag_risk_white),
                },
                "black": {
                    "risk": self.flag_risk_black,
                    "alert": is_flag_risk_alert(self.flag_risk_black),
                },
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


# Flat, spreadsheet-friendly column order for the post-show ledger (task 0204). One row
# per published move; reuses only fields the event already carries (no extra model calls).
# ``drama_label``/``drama_score`` come from the same :func:`chess_equity.drama.score_event`
# classifier the overlay/captions use, and are blank on a non-dramatic move.
LEDGER_COLUMNS: List[str] = [
    "ply",
    "side",
    "san",
    "equity",
    "delta_equity",
    "grade",
    "drama_label",
    "drama_score",
    "white_clock",
    "black_clock",
    "model",
]


def ledger_row(event: "MoveEvent") -> Dict[str, object]:
    """One flat CSV row (keyed by :data:`LEDGER_COLUMNS`) for a published move.

    ``side`` is the mover (in the post-move FEN the side *to* move is the opponent, so
    the mover is White exactly when it's now Black to move — same convention as
    :func:`live_caption`). Equities/deltas are rounded to whole percentage points to
    match the overlay bar; clocks pass through as remaining seconds (blank without
    ``[%clk]``). Drama columns are blank unless the classifier fires. ``model`` is the
    equity model's identity (``event.source``, e.g. ``LichessBaselineModel``/``Maia2Model``)
    so an archived ledger can be attributed to the model that produced its equities.
    """
    # Lazy import: drama imports MoveEvent from this module, so a top-level import cycles.
    from chess_equity.drama import score_event

    drama = score_event(event)
    mover_white = not event.white_to_move
    return {
        "ply": event.ply,
        "side": "white" if mover_white else "black",
        "san": event.san,
        "equity": round(event.equity, 1),
        "delta_equity": None
        if event.delta_equity is None
        else round(event.delta_equity, 1),
        "grade": event.last_move_grade or "",
        "drama_label": drama.kind if drama is not None else "",
        "drama_score": round(drama.magnitude, 3) if drama is not None else "",
        "white_clock": event.white_clock,
        "black_clock": event.black_clock,
        "model": event.source,
    }


def write_ledger(events: "Iterable[MoveEvent]", fh: "TextIO") -> int:
    """Write a per-move equity ledger CSV (header + one row per move) to ``fh``.

    Returns the number of move rows written (the header is not counted). The flat
    tabular counterpart to the equity-annotated PGN (task 0197): same per-move data,
    shaped for spreadsheets and post-show graphics instead of a chess GUI.
    """
    import csv

    writer = csv.DictWriter(fh, fieldnames=LEDGER_COLUMNS)
    writer.writeheader()
    rows = 0
    for event in events:
        writer.writerow(ledger_row(event))
        rows += 1
    return rows


# --------------------------------------------------------------------------- #
# WebVTT caption track (task 0211)
# --------------------------------------------------------------------------- #
#
# ``broadcast --captions`` prints the per-move caster sentence to stdout (task 0190);
# this turns that same sentence stream into a *timestamped* WebVTT subtitle track so
# the caster line becomes a real caption/TTS track for the recorded stream. One cue per
# graded move, keyed by the game's own clock: each cue starts at the elapsed game time
# the move was played (derived from the [%clk] tags), so the subtitles line up with a
# screen recording paced by the players' clocks. When a move carries no clock the cue
# falls back to a fixed dwell, so a clock-less PGN still produces sensible move-index
# spacing. Reuses the reel's WebVTT timestamp/escape helpers (task 0205) so both
# exporters speak the same cue dialect.

# Fixed per-move dwell (seconds) used when a move carries no [%clk] tag (so a clock-less
# PGN degrades to plain move-index spacing) and for the trailing cue's hold time.
CAPTION_CUE_SECONDS = 3.0


def _caption_cues(
    events: "Iterable[MoveEvent]", *, cue_seconds: float = CAPTION_CUE_SECONDS
) -> List[Tuple[float, float, str]]:
    """Shared source of truth for the caption timeline: ``(start, end, text)`` cues.

    Each graded move (the ones :func:`live_caption` voices) becomes one cue whose text
    is that caster sentence and whose start is the elapsed game time the move was made.
    See :func:`_game_caption_cues` for how one game's timeline is laid out from its
    [%clk] deltas. Both :func:`build_captions_vtt` and :func:`build_captions_srt` render
    these exact cues so the two tracks stay cue-for-cue identical and only differ in
    container/timestamp dialect.

    A snapshot can carry **several games** (a multi-board round, task 0185), whose events
    arrive grouped by ``game_id`` in board order. Clocks reset per game, so each game's
    timeline must restart at its own ``t=0`` — otherwise a 2nd/3rd board's cues pile up
    behind board 1's *total* elapsed time and land at ever-growing, wrong timestamps
    (task 0230). We therefore lay out **each game independently** (its own cue list, so a
    cue's end never crosses a game boundary onto the next game's smaller start) and
    concatenate; a single-game snapshot is one group, so its timeline is byte-identical
    to before. Cue numbering stays globally sequential (the renderers number the merged
    list). Cues from different games may overlap in time by design — they subtitle each
    board's own per-game recording, not one merged reel.
    """
    cues: List[Tuple[float, float, str]] = []
    for _gid, group in groupby(events, key=lambda e: e.game_id):
        cues.extend(_game_caption_cues(group, cue_seconds=cue_seconds))
    return cues


def _drama_callout(kind: str) -> str:
    """Emoji + caster label for a drama ``kind`` (e.g. ``"🎯 Clutch"``).

    Reuses the reel's shared ``_KIND_LABEL`` map (``chess_equity.reel``) so the caption
    track and the reel's lower-thirds never label the same drama kind differently.
    """
    from chess_equity.reel import _KIND_LABEL

    emoji, label = _KIND_LABEL.get(kind, ("", kind))
    return f"{emoji} {label}".strip()


def _game_caption_cues(
    events: "Iterable[MoveEvent]", *, cue_seconds: float = CAPTION_CUE_SECONDS
) -> List[Tuple[float, float, str]]:
    """Caption cues for the events of a **single** game (one ``game_id``).

    Elapsed time is accumulated from the per-side [%clk] deltas — the mover's think time
    is ``previous_remaining - current_remaining`` (clamped at 0; clock increments are
    folded into this net change). A move with no clock (or the first move of each side,
    which has no prior reading) advances by ``cue_seconds`` instead, so a clock-less PGN
    degrades to even move-index spacing. Cue starts are forced strictly increasing and
    each cue ends where the next begins (the last holds for ``cue_seconds``), keeping the
    output valid even when two clock readings collide.

    A move that fired a drama event (clutch / missed win / escape / scramble) gets its cue
    text **prefixed** with that kind's emoji+label callout (the reel's ``_KIND_LABEL``), so
    a dramatic cue reads e.g. ``"🎯 Clutch — Qxf7# — brilliant, +48% …"`` and a caster/
    viewer can scan the drama kind at a glance. Drama is keyed by ply (via the same
    :func:`chess_equity.drama.detect` source :func:`chess_equity.annotate.drama_by_ply`
    uses) so it never drifts from the move it labels; undramatic cues are unchanged.
    """
    from chess_equity.drama import detect

    events = list(events)
    drama_by_ply = {d.ply: d for d in detect(events)}

    prev_white: Optional[float] = None
    prev_black: Optional[float] = None
    elapsed = 0.0
    starts: List[float] = []
    texts: List[str] = []
    last_start = -1.0
    min_step = 0.001
    for event in events:
        mover_white = not event.white_to_move
        if mover_white:
            cur = event.white_clock
            dt = max(0.0, prev_white - cur) if (cur is not None and prev_white is not None) else cue_seconds
            if cur is not None:
                prev_white = cur
        else:
            cur = event.black_clock
            dt = max(0.0, prev_black - cur) if (cur is not None and prev_black is not None) else cue_seconds
            if cur is not None:
                prev_black = cur
        elapsed += dt
        text = live_caption(event)
        if text is None:
            continue
        drama = drama_by_ply.get(event.ply)
        if drama is not None:
            text = f"{_drama_callout(drama.kind)} — {text}"
        # Force strictly increasing starts so identical/empty clock deltas never yield
        # a zero-length or out-of-order cue.
        start = max(elapsed, last_start + min_step)
        last_start = start
        starts.append(start)
        texts.append(text)

    cues: List[Tuple[float, float, str]] = []
    for i, (start, text) in enumerate(zip(starts, texts)):
        end = starts[i + 1] if i + 1 < len(starts) else start + cue_seconds
        cues.append((start, end, text))
    return cues


def build_captions_vtt(
    events: "Iterable[MoveEvent]", *, cue_seconds: float = CAPTION_CUE_SECONDS
) -> str:
    """Render a graded game's caster captions as a timestamped WebVTT track.

    One cue per graded move (see :func:`_caption_cues` for the timeline). WebVTT cues
    carry a ``WEBVTT`` header, ``HH:MM:SS.mmm`` (dot-decimal) timestamps, and escape the
    three reserved characters ``& < >``.
    """
    from chess_equity.reel import _vtt_escape, _vtt_timestamp

    lines = ["WEBVTT", ""]
    for i, (start, end, text) in enumerate(_caption_cues(events, cue_seconds=cue_seconds), start=1):
        lines.append(str(i))
        lines.append(f"{_vtt_timestamp(start)} --> {_vtt_timestamp(end)}")
        lines.append(_vtt_escape(text))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_captions_srt(
    events: "Iterable[MoveEvent]", *, cue_seconds: float = CAPTION_CUE_SECONDS
) -> str:
    """Render a graded game's caster captions as an SRT (SubRip) subtitle track.

    Mirrors :func:`build_captions_vtt` cue-for-cue — same :func:`_caption_cues` timeline,
    same 1-based numbering, same boundaries — so the caster narration drops into any
    non-web editor (Premiere/Resolve/CapCut) that can't ingest WebVTT. Only the container
    differs: no ``WEBVTT`` header, ``HH:MM:SS,mmm`` comma-decimal timestamps, raw cue text
    (SRT, unlike WebVTT, does not reserve ``& < >``), and blank-line-separated blocks.
    """
    from chess_equity.reel import _srt_timestamp

    blocks = []
    for i, (start, end, text) in enumerate(_caption_cues(events, cue_seconds=cue_seconds), start=1):
        blocks.append(
            f"{i}\n"
            f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n"
            f"{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


# Human-readable labels for the equity model driving the bar (task 0222), keyed by the
# model's canonical ``SOURCE``. The badge exists so a viewer can tell the bar is a
# rating-conditioned HUMAN win-probability model, not a classic Stockfish eval. Unknown
# sources fall back to the raw ``SOURCE`` (or class name); a ``None`` model -> no badge.
_MODEL_LABELS: Dict[str, str] = {
    "maia2": "Maia-2",
    "wdl-a": "WDL-A",
    "wdl-net": "WDL-Net",
    "lichess-baseline": "baseline",
    "maia-rollout": "Maia rollout",
    "maia-search": "Maia search",
}


def model_label(model: Optional[object]) -> Optional[str]:
    """Human-readable label for the equity model driving the overlay bar (task 0222).

    Unwraps a :class:`~chess_equity.cache.CachingEquityModel` to its base, reads the
    model's ``SOURCE`` key, and maps it to a display label (e.g. ``"maia2"`` ->
    ``"Maia-2"``). Unknown sources fall back to the raw ``SOURCE`` (or the class name);
    a ``None`` model returns ``None`` so the overlay shows no badge.
    """
    if model is None:
        return None
    base = getattr(model, "base", model)  # unwrap CachingEquityModel
    source = getattr(base, "SOURCE", type(base).__name__)
    return _MODEL_LABELS.get(source, source)


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
    # Human-readable label for the equity model driving the bar (task 0222), e.g.
    # "Maia-2" / "WDL-A" / "baseline". ``None`` when unknown — the overlay then shows
    # no model badge, so a viewer never sees a misleading label.
    model: Optional[str] = None

    def to_overlay(self) -> Dict[str, object]:
        """Render as the overlay's ``{type: "game", players: {...}}`` event (see
        overlay/README.md). ``name``/``rating`` may be ``null``; overlay.js falls back
        to "White"/"Black" and a blank rating. ``board`` is the 0-based index in a
        multi-game round (omitted when single-game). ``model`` is the human-readable
        bar-model badge (task 0222), omitted when unknown so no badge renders."""
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
        if self.model is not None:
            event["model"] = self.model
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
    model: Optional[object] = None,
) -> GameEvent:
    """Build the one-time :class:`GameEvent` for a game from its PGN headers.

    An explicit ``white_elo``/``black_elo`` (the ingestor's override) wins over the
    header so the announced ratings match the ones the trackers actually evaluate at.
    ``board`` is the 0-based index of this game in a multi-game round (task 0185), or
    ``None`` for a single-game feed. ``model`` is the equity model driving the bar; its
    human-readable label is announced for the overlay's model badge (task 0222).
    """
    return GameEvent(
        game_id=game_id,
        white_name=_player_name(headers, "White"),
        black_name=_player_name(headers, "Black"),
        white_elo=white_elo if white_elo is not None else _parse_elo(headers, "WhiteElo"),
        black_elo=black_elo if black_elo is not None else _parse_elo(headers, "BlackElo"),
        board=board,
        model=model_label(model),
    )


# The three terminal PGN results; "*" means the game is still in progress.
_TERMINAL_RESULTS = frozenset({"1-0", "0-1", "1/2-1/2"})


def terminal_result(headers: chess.pgn.Headers) -> Optional[str]:
    """Return the game's final result (``"1-0"``/``"0-1"``/``"1/2-1/2"``) once the PGN
    reaches one, else ``None`` while it's still ``"*"`` (in progress).

    The ``Result`` header is the canonical broadcast end-signal — a Lichess/operator
    feed stamps it the moment a game ends (checkmate, resignation, flag, draw). Catching
    a game-over from the moves alone (checkmate with the header still ``"*"``) is left to
    a follow-up; the header covers the real broadcast case.
    """
    result = headers.get("Result", "*").strip()
    return result if result in _TERMINAL_RESULTS else None


@dataclass(frozen=True)
class ResultEvent:
    """A game-end signal for a board in a multi-game round (task 0189).

    Emitted once, the moment a board's PGN reaches a terminal :class:`Result`, so the
    overlay can advance focus off the finished board to a still-live one (an idle caster
    isn't stranded on an ended game). ``board`` is the 0-based round index — single-game
    feeds don't emit results, since there's nothing to advance to.
    """

    game_id: str
    board: int
    result: str  # one of "1-0" / "0-1" / "1/2-1/2"

    def to_overlay(self) -> Dict[str, object]:
        """Render as the overlay's ``{type: "result", board, game_id, result}`` event —
        routing metadata the board router consumes (never drawn on the bar)."""
        return {
            "type": "result",
            "board": self.board,
            "game_id": self.game_id,
            "result": self.result,
        }


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
        # Running per-side ok-or-better tallies for the live accuracy readout (task 0245),
        # keyed by mover_white -> [accurate, total]. Carried across append-only ingests;
        # reset together with ``emitted_ply`` on a resync, since the moves they counted are
        # re-emitted from scratch (so the figure is rebuilt, never double-counted).
        self._acc_accurate = {True: 0, False: 0}
        self._acc_total = {True: 0, False: 0}
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
            self._acc_accurate = {True: 0, False: 0}
            self._acc_total = {True: 0, False: 0}
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

            # Fold this move into the mover's running accuracy tally before publishing,
            # so the event carries the figure *through* this move (task 0245). Both sides'
            # current accuracy ride every event; a side yet to move stays None.
            grade = grade_delta(delta)
            if grade is not None:
                self._acc_total[mover_white] += 1
                if grade in ACCURATE_LABELS:
                    self._acc_accurate[mover_white] += 1

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
                    last_move_grade=grade,
                    source=self.model.__class__.__name__,
                    compute_ms=compute_ms,
                    cp=cp,
                    resync=resync,
                    cumulative_accuracy_white=_accuracy_pct(
                        self._acc_accurate[True], self._acc_total[True]
                    ),
                    cumulative_accuracy_black=_accuracy_pct(
                        self._acc_accurate[False], self._acc_total[False]
                    ),
                    flag_risk_white=self._flag_risk(white_clock),
                    flag_risk_black=self._flag_risk(black_clock),
                )
            )
            prev_equity_white = equity_white

        self.emitted_ply = len(nodes)
        self.emitted_ucis = new_ucis
        return events

    def _equity_white(self, fen: str, white_elo: int, black_elo: int) -> float:
        return self.model.evaluate(fen, white_elo, black_elo).equity_white

    def _flag_risk(self, clock_remaining: Optional[float]) -> Optional[float]:
        """A side's modelled flag risk from its remaining clock + the game's time control.

        Returns ``None`` (no alert) when the side is clock-blind (no recorded clock) or no
        time control has been read yet, so clock-blind feeds carry no flag-risk block.
        Otherwise :func:`chess_equity.clock.flag_risk` — correspondence games map to 0.0.
        """
        if clock_remaining is None or self.tc_bucket is None:
            return None
        return flag_risk(clock_remaining, self.tc_bucket)

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
        # Fired once per game, the first time its PGN reaches a terminal result, with a
        # :class:`ResultEvent` so the overlay can auto-advance off a finished board (task
        # 0189). Only multi-board rounds fire it (a single game has nowhere to advance).
        self.on_result: Optional[Callable[["ResultEvent"], None]] = None
        self._finished: set[str] = set()

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
                            model=self.model,
                        )
                    )
            new = self._tracker_for(gid).ingest(game_pgn)
            events.extend(new)
            # Auto-advance signal (task 0189): once a followed board's game ends, the
            # overlay should move focus to a still-live board. Announce the terminal
            # result once per game. Multi-board only — a single-game feed has no other
            # board to advance to, and we keep its event stream byte-identical.
            if multi_board and gid not in self._finished:
                result = terminal_result(headers)
                if result is not None:
                    self._finished.add(gid)
                    if self.on_result is not None:
                        self.on_result(
                            ResultEvent(game_id=gid, board=index, result=result)
                        )
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


# --------------------------------------------------------------------------- #
# Server-side drama auto-follow for `broadcast --board auto` (task 0256)
# --------------------------------------------------------------------------- #

# Light hysteresis: a rival board must out-drama the current focus by this much (on the
# 0..1 drama-magnitude scale) before it can steal the cut, so a hair-bigger blip on a
# quiet board doesn't thrash the focus every ply. Manual+auto blending is still deferred.
FOCUS_MARGIN = 0.15

# Geometric recency decay applied to every board's standing score on each ``note()``
# tick (task 0257). A board's drama fades by this factor per ply it stays quiet, so
# "recent" actually means recent: a board whose peak was its *final* move loses that
# score over the next few plies instead of holding focus forever. 0.85/ply gives a
# ~4-ply (≈2-move) half-life — gentle enough not to thrash, fast enough that a stale
# peak yields to a currently-active rival.
FOCUS_DECAY = 0.85


class FocusDirector:
    """Pick which board ``broadcast --board auto`` should focus, from per-board drama.

    A multi-board round streams every board down one feed; this watches each board's
    most-recent drama magnitude (0 when a move isn't highlight-worthy) and follows the
    most dramatic board, so the overlay auto-cuts to "the most exciting game right now"
    (task 0256). The first board seen is adopted silently — the overlay already defaults
    its router to board 0, so an opening focus event would be redundant. Thereafter a
    rival steals focus only when its swing out-dramas the current focus's most-recent
    swing by :data:`FOCUS_MARGIN` (light anti-thrash hysteresis).

    Pure + state-only (no model, no IO) so it unit-tests directly. :meth:`note` returns
    the board index when the focus CHANGES (so the caller emits one ``focus`` event),
    else ``None``.

    Recency window (task 0257): every board's standing score decays by ``decay`` on each
    :meth:`note` tick before the current board's fresh magnitude is recorded, so a board
    that goes quiet fades out of contention rather than holding focus on a historical
    peak. A board whose drama peak was its *final* move loses that score over the next
    few plies and yields to a currently-active rival. (Advancing focus off a *finished*
    board is still the overlay router's job, driven by the ``result`` event, task 0189.)

    Caster pin (task 0259): :meth:`pin` lets a caster hold focus on a board for a fixed
    number of ``note`` ticks (plies), suppressing every auto-cut regardless of how
    dramatic a rival gets; when the pin expires drama-following resumes automatically.
    A pin also clears the moment the pinned board's game ends (:meth:`result`), so a
    caster pinned to a finished game isn't stranded. ``recent`` keeps decaying/updating
    during a pin, so the director is current the instant the pin lifts.
    """

    def __init__(
        self, margin: float = FOCUS_MARGIN, decay: float = FOCUS_DECAY
    ) -> None:
        self.margin = margin
        self.decay = decay
        self.focus: Optional[int] = None
        self.recent: Dict[int, float] = {}
        self.pinned: Optional[int] = None  # board a caster has pinned, else None
        self.pin_remaining: int = 0  # note() ticks the pin still holds

    def pin(self, board: int, plies: int) -> Optional[int]:
        """Caster directive: hold focus on ``board`` for ``plies`` :meth:`note` ticks,
        suppressing auto-cuts, then auto-resume drama-following.

        Returns ``board`` when the pin moves the live cut (so the caller emits one
        ``focus`` event), else ``None`` (the board was already focused). A ``plies`` of
        0 or less is a no-op hold but still cuts to ``board`` if it differs from focus.
        """
        self.pinned = board
        self.pin_remaining = max(0, int(plies))
        if self.focus != board:
            self.focus = board
            return board
        return None

    def clear_pin(self) -> None:
        """Drop any active pin; drama-following resumes on the next :meth:`note`."""
        self.pinned = None
        self.pin_remaining = 0

    def result(self, board: int) -> None:
        """A board's game ended (task 0189). If it was the pinned board, clear the pin
        so focus can auto-resume off the finished game."""
        if self.pinned == board:
            self.clear_pin()

    def note(self, board: int, magnitude: float) -> Optional[int]:
        """Record ``board``'s latest drama ``magnitude``; return the new focus board if
        the cut changes (the current ``board`` stole focus), else ``None``.

        Each call is one ply tick: every board's recency score is decayed by
        :attr:`decay` first, then ``board``'s score is refreshed to ``magnitude`` — so
        the focus comparison weighs a rival against the *decayed* (recent) score of the
        current focus, not its all-time peak."""
        for b in self.recent:
            self.recent[b] *= self.decay
        self.recent[board] = magnitude
        if self.focus is None:
            self.focus = board  # adopt the first board silently
            return None
        if self.pin_remaining > 0:
            # A caster pin holds focus regardless of rival magnitude. Tick it down;
            # when it reaches zero the pin lifts and drama-following resumes next note.
            self.pin_remaining -= 1
            if self.pin_remaining == 0:
                self.pinned = None
            return None
        if board == self.focus:
            return None
        if magnitude - self.recent.get(self.focus, 0.0) >= self.margin:
            self.focus = board
            return board
        return None


def overlay_events(
    ingestor: "BroadcastIngestor", *, auto_follow: bool = False, **stream_kwargs
) -> Iterator[object]:
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

    def on_result(res: "ResultEvent") -> None:
        # A board's game ended: queue a routing-only `result` event so the overlay's
        # board router can advance focus off the finished board (task 0189). If a caster
        # had pinned this board, lift the pin so drama-following auto-resumes (task 0259).
        if director is not None:
            director.result(res.board)
        queued.append(res.to_overlay())

    # `--board auto` (task 0256): track each board's recent drama and emit a `focus`
    # routing event the moment the most-dramatic board changes, so the overlay auto-cuts.
    director = FocusDirector() if auto_follow else None

    ingestor.on_game = on_game
    ingestor.on_result = on_result
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
        # Drama auto-follow: if this move makes its board the liveliest, emit the `focus`
        # cut BEFORE the move so the dramatic move itself renders on the now-focused board.
        if director is not None and board is not None:
            drama = event.get("drama")
            mag = drama.get("magnitude", 0.0) if isinstance(drama, dict) else 0.0
            changed = director.note(board, mag or 0.0)
            if changed is not None:
                yield {"type": "focus", "board": changed, "game_id": move_event.game_id}
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
