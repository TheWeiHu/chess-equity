"""Auto-highlight reel export — the caster-facing drama artifact (task 0168).

Task 0020 ([[product-wedge-streaming]]) built the per-move drama *metrics*
(``clutch`` / ``missed_win`` / ``escape`` / ``scramble`` in :mod:`chess_equity.drama`)
but stopped at the metric. The streaming wedge's promise is an *auto-highlight reel*:
after a game (or a replayed broadcast) a caster wants a ranked list of the moments
worth replaying, in a form they can paste into a stream description or hand to an
editor. This module turns the ranked :class:`~chess_equity.drama.DramaEvent` stream
into two committed-artifact renderings:

- **JSON** — a structured payload (``moments`` ranked, plus ``count`` and a per-kind
  ``by_kind`` tally) for downstream tooling / the overlay.
- **Markdown** — a caster-facing reel: a numbered top-moments list followed by a
  by-drama-type breakdown, each line a one-glance headline + magnitude.

Ranking is **by drama type + magnitude**: primary key is the 0..1 drama magnitude
(bigger swings first); ties break by a fixed drama-kind priority so the order is
deterministic across runs. This composes directly over the broadcast pipeline — feed
it the events from ``BroadcastIngestor.ingest_snapshot`` (a ``LocalPgnFeed`` replay of
a committed PGN) and it emits the reel with no extra model calls.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Tuple

from chess_equity.drama import DramaEvent, detect

# Tie-break order when two moments share a magnitude: the bigger *story* first.
# Mirrors the priority :func:`chess_equity.drama.score_event` itself checks in.
_KIND_PRIORITY = {"missed_win": 0, "escape": 1, "clutch": 2, "scramble": 3}

# Caster-facing label + emoji per drama kind (markdown + caption lower-thirds).
_KIND_LABEL = {
    "clutch": ("🎯", "Clutch"),
    "missed_win": ("💥", "Missed win"),
    "escape": ("🛟", "Escape"),
    "scramble": ("⏱", "Scramble"),
}

# On-stream lower-third dwell time, seconds. Bigger swings linger longer (sized by the
# 0..1 drama magnitude) so a clutch p90 swing flashes briefly while a missed win holds.
_CAPTION_MIN_S = 3.0
_CAPTION_MAX_S = 6.0


def _rank_key(d: DramaEvent) -> Tuple[float, int, int]:
    # magnitude desc (negated), then kind priority asc, then ply asc — fully deterministic.
    return (-d.magnitude, _KIND_PRIORITY.get(d.kind, 99), d.ply)


def rank(events: Iterable[DramaEvent], *, top: Optional[int] = None) -> List[DramaEvent]:
    """Rank drama events into a highlight reel (by magnitude, then drama type)."""
    ranked = sorted(events, key=_rank_key)
    return ranked[:top] if top is not None else ranked


def build_reel(
    move_events: Iterable, *, top: Optional[int] = None
) -> List[DramaEvent]:
    """Detect drama over a stream of ``MoveEvent``s and rank it into a reel."""
    return rank(detect(move_events), top=top)


def by_kind(reel: Iterable[DramaEvent]) -> Dict[str, int]:
    """Count moments per drama kind (every kind that fired, others omitted)."""
    tally: Dict[str, int] = {}
    for d in reel:
        tally[d.kind] = tally.get(d.kind, 0) + 1
    return tally


def reel_payload(reel: List[DramaEvent], *, title: str = "Highlight reel") -> Dict[str, object]:
    """The structured JSON-ready payload for a ranked reel."""
    return {
        "title": title,
        "count": len(reel),
        "by_kind": by_kind(reel),
        "moments": [d.to_dict() for d in reel],
    }


def _caption_duration(magnitude: float) -> float:
    """Lower-third dwell time (s), scaled by 0..1 drama magnitude (saturating)."""
    span = _CAPTION_MAX_S - _CAPTION_MIN_S
    return round(_CAPTION_MIN_S + max(0.0, min(1.0, magnitude)) * span, 1)


def caption(d: DramaEvent) -> Dict[str, object]:
    """One OBS-ready lower-third caption for a drama moment.

    ``text`` is a compact on-stream headline built from the shared ``_KIND_LABEL``
    (emoji + caster label) plus the side and signed Δequity — e.g.
    ``💥 Missed win — White (-20 pts)``. ``kind`` and ``ply`` let the overlay sync the
    caption to the reel; ``duration_s`` is how long to hold it on screen.
    """
    emoji, label = _KIND_LABEL.get(d.kind, ("", d.kind))
    side = "White" if d.mover_white else "Black"
    text = f"{emoji} {label} — {side} ({d.delta_equity:+.0f} pts)"
    return {
        "text": text,
        "kind": d.kind,
        "ply": d.ply,
        "duration_s": _caption_duration(d.magnitude),
    }


def caption_payload(
    reel: List[DramaEvent], *, title: str = "Highlight reel"
) -> Dict[str, object]:
    """Structured caption payload: ranked lower-thirds an OBS source can drive."""
    return {
        "title": title,
        "count": len(reel),
        "captions": [caption(d) for d in reel],
    }


def render_captions(
    reel: List[DramaEvent], *, title: str = "Highlight reel", indent: int = 2
) -> str:
    """Render the reel's lower-third captions as a JSON string for an OBS source."""
    return json.dumps(caption_payload(reel, title=title), indent=indent)


def render_json(reel: List[DramaEvent], *, title: str = "Highlight reel", indent: int = 2) -> str:
    """Render the reel as a JSON string (structured payload)."""
    return json.dumps(reel_payload(reel, title=title), indent=indent)


def render_markdown(reel: List[DramaEvent], *, title: str = "Highlight reel") -> str:
    """Render the reel as caster-facing markdown.

    A numbered top-moments list (ranked) followed by a by-drama-type breakdown.
    Stays graceful on an empty reel (a quiet game) rather than emitting an empty doc.
    """
    lines: List[str] = [f"# {title}", ""]
    if not reel:
        lines.append(
            "_No highlight-worthy moments detected — a quiet game, or muted swings "
            "on the baseline model._"
        )
        return "\n".join(lines) + "\n"

    tally = by_kind(reel)
    summary = ", ".join(f"{n} {kind}" for kind, n in sorted(tally.items()))
    lines.append(f"> {len(reel)} moment(s), ranked by drama magnitude · {summary}")
    lines.append("")

    lines.append("## Top moments")
    lines.append("")
    for i, d in enumerate(reel, start=1):
        emoji, _ = _KIND_LABEL.get(d.kind, ("", d.kind))
        lines.append(
            f"{i}. {emoji} **{d.kind}** · `{d.magnitude:.2f}` — {d.headline} "
            f"_(game {d.game_id}, ply {d.ply})_"
        )
    lines.append("")

    lines.append("## By drama type")
    lines.append("")
    # Group in the same priority order the detector uses, skipping kinds that never fired.
    for kind in sorted(tally, key=lambda k: _KIND_PRIORITY.get(k, 99)):
        emoji, label = _KIND_LABEL.get(kind, ("", kind))
        members = [d for d in reel if d.kind == kind]
        lines.append(f"### {emoji} {label} ({len(members)})")
        for d in members:
            lines.append(f"- `{d.magnitude:.2f}` ply {d.ply}: {d.headline}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
