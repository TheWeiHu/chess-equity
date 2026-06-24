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

import base64
import html
import io
import json
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

from chess_equity.drama import DramaEvent, detect
from chess_equity.types import lichess_win_percent

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

# Verb that reads naturally for the one-line shareable caption, per drama kind —
# e.g. "Carlsen *finds* Qxf7#" / "White *lets a win slip on* Rd1".
_KIND_VERB = {
    "clutch": "finds",
    "missed_win": "lets a win slip on",
    "escape": "claws back with",
    "scramble": "swings the bar on",
}

# On-stream lower-third dwell time, seconds. Bigger swings linger longer (sized by the
# 0..1 drama magnitude) so a clutch p90 swing flashes briefly while a missed win holds.
_CAPTION_MIN_S = 3.0
_CAPTION_MAX_S = 6.0


# --- Cross-game round recap (task 0198) --------------------------------------
#
# A tournament round PGN holds many games; a caster wants the round's biggest swings
# pooled across ALL boards, each moment naming its source game. The pooling itself is
# free — ``BroadcastIngestor.ingest_snapshot`` already tags every ``MoveEvent`` with
# its ``game_id`` and ``drama.score_event`` is stateless, so ``detect()`` over a
# multi-game event list already ranks correctly across games. What a round recap adds
# is *labeling*: map each ``game_id`` back to its board number + players so a pooled
# moment reads "Board 2 · carol vs dave", not a bare game id. The label threads through
# the existing renderers as an optional ``sources`` map (``None`` ⇒ single-game output
# is byte-identical to before).


@dataclass(frozen=True)
class GameSource:
    """Where a pooled moment came from: its 1-based board # and the pairing."""

    game_id: str
    board: int
    white: str
    black: str

    @property
    def label(self) -> str:
        return f"Board {self.board} · {self.white} vs {self.black}"


def game_sources(pgn_text: str) -> Dict[str, GameSource]:
    """Map each game's ``game_id`` to its round source (board #, players).

    Reuses the same ``split_games`` + ``_game_id`` the broadcast ingestor uses, so the
    keys line up exactly with the ``game_id`` carried on every ``MoveEvent``/``DramaEvent``.
    Board numbers are 1-based in PGN order.
    """
    import chess.pgn

    from chess_equity.broadcast import _game_id, split_games

    sources: Dict[str, GameSource] = {}
    for index, game_pgn in enumerate(split_games(pgn_text)):
        headers = chess.pgn.read_headers(io.StringIO(game_pgn))
        if headers is None:
            continue
        gid = _game_id(headers, index)
        sources[gid] = GameSource(
            game_id=gid,
            board=index + 1,
            white=headers.get("White", "?"),
            black=headers.get("Black", "?"),
        )
    return sources


class _Sourced(Protocol):
    """The two fields the source/mover label helpers read — shared by the drama
    ``DramaEvent`` and the divergence ``DivergenceMoment`` so both reuse the labellers."""

    @property
    def game_id(self) -> str: ...
    @property
    def mover_white(self) -> bool: ...


def _source_text(d: _Sourced, sources: Optional[Dict[str, GameSource]]) -> str:
    """Human-readable source label for one moment ('Board N · W vs B' or 'game <id>')."""
    if sources and d.game_id in sources:
        return sources[d.game_id].label
    return f"game {d.game_id}"


def _mover_name(d: _Sourced, sources: Optional[Dict[str, GameSource]]) -> str:
    """The mover's display name — the actual player in a round recap, else the side.

    Only a round recap's ``sources`` map carries player names, so a single-game reel
    falls back to ``White``/``Black`` (which is all the event itself knows).
    """
    if sources and d.game_id in sources:
        src = sources[d.game_id]
        return src.white if d.mover_white else src.black
    return "White" if d.mover_white else "Black"


def social_caption(
    d: DramaEvent, sources: Optional[Dict[str, GameSource]] = None
) -> str:
    """One human, ready-to-post line summarising a moment for a social caption/title.

    Composes the shareable headline from the pieces a viewer cares about: the source
    board/pairing (round recap only), the mover, the move, the grade label, and the
    signed practical-equity swing — e.g.
    ``Board 3 — Carlsen finds Qxf7#, clutch (+48 vs peers)``. Without ``sources`` the
    mover is the bare side and no board prefix is shown
    (``White lets a win slip on Rd1, missed win (-20 vs peers)``).
    """
    _, label = _KIND_LABEL.get(d.kind, ("", d.kind))
    verb = _KIND_VERB.get(d.kind, "plays")
    mover = _mover_name(d, sources)
    prefix = ""
    if sources and d.game_id in sources:
        prefix = f"Board {sources[d.game_id].board} — "
    return (
        f"{prefix}{mover} {verb} {d.san}, "
        f"{label.lower()} ({d.delta_equity:+.0f} vs peers)"
    )


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


def drop_below_magnitude(
    events: Iterable[DramaEvent], floor: float
) -> Tuple[List[DramaEvent], int]:
    """Drop moments whose 0..1 drama magnitude is below ``floor``.

    A quiet game still surfaces low-magnitude noise; this trims it before
    ranking/rendering. Returns the kept events (input order preserved) and the count
    dropped, so the caller can log how many were cut rather than truncating silently.
    A ``floor`` of 0 keeps everything.
    """
    kept: List[DramaEvent] = []
    dropped = 0
    for d in events:
        if d.magnitude >= floor:
            kept.append(d)
        else:
            dropped += 1
    return kept, dropped


# --- Human-vs-engine DIVERGENCE category (task 0272) -------------------------
#
# The reel above ranks DRAMA — big *swings* in the practical-equity bar. But the
# project's signature is human-vs-engine DIVERGENCE: moves where the rating-conditioned
# equity bar most *disagrees* with the classic Stockfish (cp-implied) bar. A divergence
# need NOT be a swing — a quiet move can sit far from the engine bar (the human position
# is much easier/harder to play than the eval admits) yet move the bar barely at all, so
# the drama detector never surfaces it. This is a SEPARATE ranked category, computed over
# the raw ``MoveEvent`` stream (drama events drop ``cp``, which divergence needs).
#
# Both bars are White-POV in percentage points: ``equity`` is the rating-conditioned bar
# the broadcast publishes; the engine bar is ``lichess_win_percent(cp)`` — Lichess's
# rating-blind Win% of the White-POV centipawn eval (:mod:`chess_equity.types`), the same
# rating-blind baseline the overlay's human-edge divergence badge draws against (task
# 0103). The magnitude ranked is ``|equity − engine|``.


@dataclass(frozen=True)
class DivergenceMoment:
    """One human-vs-engine bar disagreement: how far the rating-conditioned equity
    bar sits from the rating-blind cp-implied bar at a single move.

    All percentages are White-POV in [0, 100]. ``cp`` is the White-POV centipawn eval;
    ``cp_win`` is its Lichess rating-blind Win%; ``divergence`` is ``|equity − cp_win|``.
    ``signed_gap`` (a property) is ``equity − cp_win`` — positive when the human bar reads
    *more* White-favorable than the engine, negative when *less*.
    """

    game_id: str
    ply: int
    san: str
    mover_white: bool
    equity: float
    cp: float
    cp_win: float
    divergence: float
    fen: Optional[str] = None

    @property
    def signed_gap(self) -> float:
        return self.equity - self.cp_win

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["signed_gap"] = self.signed_gap
        return d


def _divergence_rank_key(m: DivergenceMoment) -> Tuple[float, int]:
    # magnitude desc (negated), then ply asc — fully deterministic across runs.
    return (-m.divergence, m.ply)


def detect_divergence(
    move_events: Iterable, *, top: Optional[int] = None
) -> List[DivergenceMoment]:
    """Rank a move stream by human-vs-engine bar divergence (``|equity − cp_win|``).

    One moment per move that carries an engine ``cp`` (moves with ``cp is None`` — mates,
    or a clock-blind/cp-less feed — are skipped: there is no engine bar to diverge from).
    Ranked by divergence magnitude descending, ties broken by ply, optionally capped to
    the ``top`` N. Independent of the drama ranking: a quiet move with a big divergence
    ranks here even though it never registers as drama.
    """
    moments: List[DivergenceMoment] = []
    for e in move_events:
        if e.cp is None:
            continue
        cp_win = lichess_win_percent(e.cp)
        moments.append(
            DivergenceMoment(
                game_id=e.game_id,
                ply=e.ply,
                san=e.san,
                mover_white=not e.white_to_move,
                equity=e.equity,
                cp=e.cp,
                cp_win=cp_win,
                divergence=abs(e.equity - cp_win),
                fen=e.fen,
            )
        )
    moments.sort(key=_divergence_rank_key)
    return moments[:top] if top is not None else moments


def divergence_caption(
    m: DivergenceMoment, sources: Optional[Dict[str, GameSource]] = None
) -> str:
    """One ready-to-post line summarising a divergence moment for a caster/social card.

    Names the source board/pairing (round recap only), the mover + move, both bars, and
    the signed gap with the side it leans toward — e.g.
    ``Board 3 — Carlsen Nf3: human bar 65% vs engine 50% (+15 toward White)``.
    """
    mover = _mover_name(m, sources)
    prefix = ""
    if sources and m.game_id in sources:
        prefix = f"Board {sources[m.game_id].board} — "
    toward = "White" if m.signed_gap >= 0 else "Black"
    return (
        f"{prefix}{mover} {m.san}: human bar {m.equity:.0f}% vs engine "
        f"{m.cp_win:.0f}% ({m.signed_gap:+.0f} toward {toward})"
    )


def divergence_payload(
    moments: List[DivergenceMoment],
    *,
    title: str = "Human-vs-engine divergence",
    sources: Optional[Dict[str, GameSource]] = None,
) -> Dict[str, object]:
    """The structured JSON-ready payload for the ranked divergence category.

    Each moment carries a ready-to-post ``caption`` (see :func:`divergence_caption`); a
    round recap (``sources`` given) additionally tags each with its ``source`` label +
    ``board`` number and a top-level ``games`` count.
    """
    out: List[Dict[str, object]] = []
    for m in moments:
        d = m.to_dict()
        if sources is not None:
            src = sources.get(m.game_id)
            d["source"] = _source_text(m, sources)
            d["board"] = src.board if src is not None else None
        d["caption"] = divergence_caption(m, sources)
        out.append(d)
    payload: Dict[str, object] = {
        "title": title,
        "count": len(moments),
        "moments": out,
    }
    if sources is not None:
        payload["games"] = len({m.game_id for m in moments})
    return payload


def render_divergence_markdown(
    moments: List[DivergenceMoment],
    *,
    title: str = "Human-vs-engine divergence",
    sources: Optional[Dict[str, GameSource]] = None,
) -> str:
    """Render the ranked divergence category as caster-facing markdown.

    A numbered list of the moments where the rating-aware bar most disagrees with the
    engine bar, each line naming both bars + the signed gap. Stays graceful on an empty
    list (no cp-bearing moves) rather than emitting an empty doc.
    """
    lines: List[str] = [f"# {title}", ""]
    if not moments:
        lines.append(
            "_No human-vs-engine divergence to show — no moves carried an engine "
            "eval to compare against (a cp-less or mate-only feed)._"
        )
        return "\n".join(lines) + "\n"

    span = ""
    if sources is not None:
        span = f"across {len({m.game_id for m in moments})} board(s) "
    lines.append(
        f"> {len(moments)} moment(s) {span}where the rating-aware bar most "
        "disagrees with the engine bar, ranked by |gap|"
    )
    lines.append("")
    for i, m in enumerate(moments, start=1):
        loc = f"{_source_text(m, sources)}, ply {m.ply}"
        lines.append(
            f"{i}. ⚖️ `{m.divergence:.0f} pts` — {divergence_caption(m, sources)} "
            f"_({loc})_"
        )
    return "\n".join(lines).rstrip() + "\n"


def by_kind(reel: Iterable[DramaEvent]) -> Dict[str, int]:
    """Count moments per drama kind (every kind that fired, others omitted)."""
    tally: Dict[str, int] = {}
    for d in reel:
        tally[d.kind] = tally.get(d.kind, 0) + 1
    return tally


def reel_payload(
    reel: List[DramaEvent],
    *,
    title: str = "Highlight reel",
    sources: Optional[Dict[str, GameSource]] = None,
    divergence: Optional[List[DivergenceMoment]] = None,
) -> Dict[str, object]:
    """The structured JSON-ready payload for a ranked reel.

    Every moment carries a ``caption`` — a ready-to-post one-line social headline
    (see :func:`social_caption`). When ``sources`` is given (a round recap), each moment
    additionally gains a ``source`` label and ``board`` number, its caption names the
    source board + player, and a top-level ``games`` count is added so downstream tooling
    can see the pool spanned multiple boards.

    When ``divergence`` is given, a distinct ``"divergence"`` block (the ranked
    human-vs-engine category, see :func:`divergence_payload`) rides alongside the drama
    ``moments`` so a caster sees both talking-point lists in one artifact. Omitted
    entirely when ``None`` so the drama-only contract is byte-identical to before.
    """
    moments: List[Dict[str, object]] = []
    for d in reel:
        m = d.to_dict()
        if sources is not None:
            src = sources.get(d.game_id)
            m["source"] = _source_text(d, sources)
            m["board"] = src.board if src is not None else None
        # One ready-to-post shareable line per moment (names the board/player on a
        # round recap, the bare side single-game). Carried on every moment.
        m["caption"] = social_caption(d, sources)
        moments.append(m)
    payload: Dict[str, object] = {
        "title": title,
        "count": len(reel),
        "by_kind": by_kind(reel),
        "moments": moments,
    }
    if sources is not None:
        # How many distinct boards actually contributed a moment to the pool.
        payload["games"] = len({d.game_id for d in reel})
    if divergence is not None:
        payload["divergence"] = divergence_payload(divergence, sources=sources)
    return payload


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


def render_json(
    reel: List[DramaEvent],
    *,
    title: str = "Highlight reel",
    indent: int = 2,
    sources: Optional[Dict[str, GameSource]] = None,
    divergence: Optional[List[DivergenceMoment]] = None,
) -> str:
    """Render the reel as a JSON string (structured payload).

    When ``divergence`` is given, the human-vs-engine category rides alongside the drama
    moments under a ``"divergence"`` key (see :func:`reel_payload`).
    """
    return json.dumps(
        reel_payload(reel, title=title, sources=sources, divergence=divergence),
        indent=indent,
    )


def _divergence_md_section(
    divergence: Optional[List[DivergenceMoment]],
    sources: Optional[Dict[str, GameSource]],
) -> List[str]:
    """The trailing '## Human-vs-engine divergence' block, or [] when not requested.

    Rendered as an ``##`` section appended under the drama reel so both talking-point
    lists live in one document. ``None`` (divergence not requested) yields nothing.
    """
    if divergence is None:
        return []
    out: List[str] = ["", "## Human-vs-engine divergence", ""]
    if not divergence:
        out.append(
            "_No human-vs-engine divergence to show — no moves carried an engine "
            "eval to compare against._"
        )
        return out
    out.append(
        "_Moves where the rating-aware bar most disagrees with the engine bar "
        "(ranked by |gap|) — quiet moves count too._"
    )
    out.append("")
    for i, m in enumerate(divergence, start=1):
        loc = f"{_source_text(m, sources)}, ply {m.ply}"
        out.append(
            f"{i}. ⚖️ `{m.divergence:.0f} pts` — {divergence_caption(m, sources)} "
            f"_({loc})_"
        )
    return out


def render_markdown(
    reel: List[DramaEvent],
    *,
    title: str = "Highlight reel",
    sources: Optional[Dict[str, GameSource]] = None,
    divergence: Optional[List[DivergenceMoment]] = None,
) -> str:
    """Render the reel as caster-facing markdown.

    A numbered top-moments list (ranked) followed by a by-drama-type breakdown.
    Stays graceful on an empty reel (a quiet game) rather than emitting an empty doc.
    When ``sources`` is given (a round recap), each top moment names its source board +
    pairing and the summary line reports how many boards the pool spans. When
    ``divergence`` is given, a distinct '## Human-vs-engine divergence' section is
    appended below the drama reel.
    """
    lines: List[str] = [f"# {title}", ""]
    if not reel:
        lines.append(
            "_No highlight-worthy moments detected — a quiet game, or muted swings "
            "on the baseline model._"
        )
        lines.extend(_divergence_md_section(divergence, sources))
        return "\n".join(lines).rstrip() + "\n"

    tally = by_kind(reel)
    summary = ", ".join(f"{n} {kind}" for kind, n in sorted(tally.items()))
    headline = f"> {len(reel)} moment(s), ranked by drama magnitude · {summary}"
    if sources is not None:
        n_games = len({d.game_id for d in reel})
        headline = (
            f"> {len(reel)} moment(s) across {n_games} board(s), "
            f"ranked by drama magnitude · {summary}"
        )
    lines.append(headline)
    lines.append("")

    lines.append("## Top moments")
    lines.append("")
    for i, d in enumerate(reel, start=1):
        emoji, _ = _KIND_LABEL.get(d.kind, ("", d.kind))
        loc = f"{_source_text(d, sources)}, ply {d.ply}"
        lines.append(
            f"{i}. {emoji} **{d.kind}** · `{d.magnitude:.2f}` — {d.headline} "
            f"_({loc})_"
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

    lines.extend(_divergence_md_section(divergence, sources))
    return "\n".join(lines).rstrip() + "\n"


# --- Self-contained HTML clip player (task 0184) -----------------------------
#
# A single shareable file casters can open offline to review/clip the drama
# moments after a stream. No external deps, no CDN, no JS — pure server-side
# string-gen: ranked cards, each with a Unicode board rendered from the FEN, the
# caster caption (reused from :func:`caption`), drama kind/emoji and the equity
# swing. The board uses figurine glyphs so it renders with the system font alone.

# Unicode chess figurines, keyed by FEN piece letter (upper = White, lower = Black).
_PIECE_GLYPH = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}


def _board_html(fen: Optional[str]) -> str:
    """Render the FEN's placement field as an 8x8 HTML board (inline-styled).

    Returns a small placeholder when no FEN is carried (synthetic events).
    """
    if not fen:
        return '<div class="board board--empty">no board snapshot</div>'
    placement = fen.split(" ", 1)[0]
    cells: List[str] = []
    for rank_idx, row in enumerate(placement.split("/")):
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                for _ in range(int(ch)):
                    light = (rank_idx + file_idx) % 2 == 0
                    cells.append(f'<span class="sq {"l" if light else "d"}"></span>')
                    file_idx += 1
            else:
                light = (rank_idx + file_idx) % 2 == 0
                glyph = _PIECE_GLYPH.get(ch, "")
                cells.append(f'<span class="sq {"l" if light else "d"}">{glyph}</span>')
                file_idx += 1
    return '<div class="board" role="img" aria-label="board position">' + "".join(cells) + "</div>"


def _moment_card_html(
    index: int, d: DramaEvent, sources: Optional[Dict[str, GameSource]] = None
) -> str:
    """One ranked-moment card: board + kind/emoji + caption + equity swing.

    With ``sources`` (a round recap), the location line names the source board + pairing
    instead of a bare game id.
    """
    emoji, label = _KIND_LABEL.get(d.kind, ("", d.kind))
    cap = str(caption(d)["text"])
    share = social_caption(d, sources)
    side = "White" if d.mover_white else "Black"
    swing = (
        f"{d.delta_equity:+.0f} pts → {d.equity:.0f}% (White POV)"
    )
    loc = f"{_source_text(d, sources)}, ply {d.ply}"
    return (
        '<article class="moment">'
        f'<div class="rank">#{index}</div>'
        f"{_board_html(d.fen)}"
        '<div class="meta">'
        f'<div class="kind"><span class="emoji">{emoji}</span>'
        f'<span class="label">{html.escape(label)}</span>'
        f'<span class="mag">magnitude {d.magnitude:.2f}</span></div>'
        f'<div class="caption">{html.escape(cap)}</div>'
        f'<div class="share" aria-label="shareable caption">'
        f'📋 {html.escape(share)}</div>'
        f'<div class="headline">{html.escape(d.headline)}</div>'
        f'<div class="swing">{html.escape(side)} · {html.escape(swing)} '
        f'<span class="loc">{html.escape(loc)}</span></div>'
        "</div>"
        "</article>"
    )


_HTML_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; font: 16px/1.45 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  background: #14151a; color: #e8e8ec; padding: 24px; }
h1 { margin: 0 0 4px; font-size: 22px; }
.sub { color: #9aa0aa; margin: 0 0 20px; font-size: 14px; }
.empty { color: #9aa0aa; font-style: italic; }
.moment { display: grid; grid-template-columns: auto 1fr; gap: 16px; align-items: center;
  background: #1d1f27; border: 1px solid #2a2d38; border-radius: 12px;
  padding: 14px; margin: 0 0 14px; position: relative; }
.rank { position: absolute; top: 8px; right: 12px; color: #6b7280; font-weight: 700; }
.board { display: grid; grid-template-columns: repeat(8, 26px); grid-template-rows: repeat(8, 26px);
  border: 2px solid #2a2d38; border-radius: 6px; overflow: hidden; }
.board--empty { display: flex; align-items: center; justify-content: center; width: 212px;
  height: 212px; color: #6b7280; border-radius: 6px; }
.sq { display: flex; align-items: center; justify-content: center; font-size: 20px; line-height: 1; }
.sq.l { background: #e9edf2; color: #14151a; }
.sq.d { background: #6f7b8a; color: #14151a; }
.meta { min-width: 0; }
.kind { display: flex; align-items: baseline; gap: 8px; margin-bottom: 6px; }
.emoji { font-size: 22px; }
.label { font-weight: 700; font-size: 18px; }
.mag { color: #9aa0aa; font-size: 12px; }
.caption { font-size: 15px; margin-bottom: 4px; }
.share { font-size: 13px; color: #9fd0ff; margin-bottom: 4px; user-select: all; }
.headline { color: #c3c8d1; font-size: 14px; margin-bottom: 6px; }
.swing { font-size: 13px; color: #ffd479; }
.loc { color: #6b7280; }
.reel-clips { width: 100%; max-width: 360px; margin: 0 0 6px; background: #000;
  border: 1px solid #2a2d38; border-radius: 8px; }
.clips-note { color: #9aa0aa; font-size: 13px; margin: 0 0 18px; }
""".strip()


# --- WebVTT narration track (task 0205) --------------------------------------
#
# The clip player plays each ranked moment as a back-to-back "clip"; a <track
# kind="captions"> carries one narration cue per clip so the export is accessible
# and social-ready. The cue text reuses the caster caption (move-grade + signed
# swing); the cue timings line up with the clip boundaries (each clip dwells for
# its caption duration). The track is embedded as an inline data: URI so the HTML
# stays a single self-contained file (no sidecar .vtt).


def _vtt_timestamp(seconds: float) -> str:
    """Format a second offset as a WebVTT cue timestamp (``HH:MM:SS.mmm``)."""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _vtt_escape(text: str) -> str:
    """Escape the three characters WebVTT cue payloads reserve (``&``, ``<``, ``>``)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def clip_durations(reel: List[DramaEvent]) -> List[float]:
    """Per-clip dwell time (s) — the player holds each moment for its caption duration."""
    return [_caption_duration(d.magnitude) for d in reel]


def build_webvtt(reel: List[DramaEvent]) -> str:
    """Render the reel as a WebVTT caption track — one cue per clip.

    Clips play back-to-back, so cue *i* spans ``[start_i, start_i + duration_i)``
    where ``start_i`` is the summed duration of every earlier clip and
    ``duration_i`` is that moment's caption dwell time (:func:`caption`). Each cue
    narrates the move-grade + signed equity swing, reusing the caster caption text
    verbatim, so the cue count equals the clip count and the timings line up with
    the clip boundaries.
    """
    lines = ["WEBVTT", ""]
    start = 0.0
    for i, d in enumerate(reel, start=1):
        end = start + _caption_duration(d.magnitude)
        lines.append(str(i))
        lines.append(f"{_vtt_timestamp(start)} --> {_vtt_timestamp(end)}")
        lines.append(_vtt_escape(str(caption(d)["text"])))
        lines.append("")
        start = end
    return "\n".join(lines).rstrip() + "\n"


def _srt_timestamp(seconds: float) -> str:
    """Format a second offset as an SRT cue timestamp (``HH:MM:SS,mmm``).

    Identical to :func:`_vtt_timestamp` but with the SubRip comma decimal separator
    (``,`` not ``.``) that non-web editors (Premiere/Resolve/CapCut) expect.
    """
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(reel: List[DramaEvent]) -> str:
    """Render the reel as an SRT (SubRip) subtitle track — one cue per clip.

    SRT mirrors :func:`build_webvtt` exactly — same one-cue-per-clip cadence, same
    contiguous ``[start_i, start_i + duration_i)`` timings, same caster caption text —
    so the narration drops into any non-web editor that can't ingest WebVTT. Only the
    container differs: no ``WEBVTT`` header, ``HH:MM:SS,mmm`` comma-decimal timestamps,
    and raw cue text (SRT, unlike WebVTT, does not reserve ``& < >``). The 1-based
    sequential indices and cue boundaries match the WebVTT track cue-for-cue.
    """
    blocks = []
    start = 0.0
    for i, d in enumerate(reel, start=1):
        end = start + _caption_duration(d.magnitude)
        blocks.append(
            f"{i}\n"
            f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n"
            f"{caption(d)['text']}"
        )
        start = end
    return "\n\n".join(blocks) + ("\n" if blocks else "")


# --- VOD chapter markers (task 0237) -----------------------------------------
#
# YouTube/Twitch VODs let a caster paste "HH:MM:SS Title" lines into the video
# description to seed a chapter list the viewer can jump between. The reel is
# already a ranked clip timeline (each moment dwells for its caption duration,
# clips back-to-back — see :func:`build_webvtt`), so a chapter export is the same
# cumulative timeline rendered as bare ``HH:MM:SS`` markers instead of cue ranges.
# The first chapter is pinned to ``00:00:00`` (YouTube requires the opening
# chapter to start at zero), and each line names the drama kind, the move, and the
# signed equity swing so the description reads as a self-describing jump list.


def _chapter_timestamp(seconds: float) -> str:
    """Format a second offset as a bare ``HH:MM:SS`` VOD chapter marker (no ms).

    VOD chapter lists (YouTube/Twitch) key off whole-second ``HH:MM:SS`` stamps, so
    this drops the millisecond field :func:`_vtt_timestamp` carries. Offsets are
    floored to the second the marker lands on (a chapter jumps to the start of its clip).
    """
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_chapters(reel: List[DramaEvent]) -> str:
    """Render the reel as VOD chapter markers — one ``HH:MM:SS Title`` line per moment.

    Reuses the same clip timeline as :func:`build_webvtt`/:func:`build_srt`: clips play
    back-to-back, each dwelling for its caption duration (:func:`clip_durations`), so
    chapter *i* starts at the summed duration of every earlier clip. The first chapter is
    pinned to ``00:00:00`` (YouTube requires the opening chapter at zero) and the stamps
    increase monotonically. Each line reads ``HH:MM:SS <kind>: <san> (<signed swing>)`` —
    e.g. ``00:00:00 missed_win: Rd1 (-20)`` — so a caster pastes the block straight into a
    VOD description. An empty reel (a quiet game) yields an empty string.
    """
    lines: List[str] = []
    start = 0.0
    for d in reel:
        swing = f"{d.delta_equity:+.0f}"
        lines.append(f"{_chapter_timestamp(start)} {d.kind}: {d.san} ({swing})")
        start += _caption_duration(d.magnitude)
    return "\n".join(lines) + ("\n" if lines else "")


def _webvtt_track_html(reel: List[DramaEvent]) -> str:
    """A <video> clip timeline carrying the reel's narration as an inline WebVTT track.

    The track's ``src`` is a base64 ``data:`` URI, so the document stays one
    self-contained file (no sidecar .vtt). Empty reel → no track.
    """
    if not reel:
        return ""
    vtt = build_webvtt(reel)
    b64 = base64.b64encode(vtt.encode("utf-8")).decode("ascii")
    uri = f"data:text/vtt;base64,{b64}"
    return (
        '<video class="reel-clips" controls preload="none" '
        'aria-label="highlight clip timeline">'
        f'<track kind="captions" srclang="en" label="Narration" default src="{uri}">'
        "</video>\n"
        '<p class="clips-note">Narration captions (one cue per clip) are embedded '
        "as a WebVTT track for accessible, social-ready export.</p>"
    )


# --- Static SVG poster frame per moment (task 0218) --------------------------
#
# The HTML clip player above is a single multi-moment document; a *poster* is one
# still SVG per ranked moment — a shareable social card (tweet image, article still,
# thumbnail). Pure server-side string-gen, no external deps: a White-POV equity bar,
# the board rendered from the moment FEN (reusing ``_PIECE_GLYPH``), the caster
# kind/emoji label, and the ready-to-post social caption. This mirrors the
# ``eval --svg`` bar renderer (task 0217); when that isn't merged yet the minimal bar
# below stands alone. The bar is White-POV like every other equity bar in the project
# (full = White winning, see :func:`chess_equity.bar.render_bar`): ``DramaEvent.equity``
# is already the White-POV percentage after the move, so the white fill rises from the
# bottom to that fraction.

_POSTER_SQ = 40  # board square edge, px
_POSTER_PAD = 16
_POSTER_BAR_W = 26
_POSTER_GAP = 12
_POSTER_HEADER_H = 46
_POSTER_LINE_H = 22
_POSTER_CAP_CHARS = 50  # caption wrap width (chars) tuned to the poster inner width

# Board square fills (light/dark) and piece colours — match the HTML player's palette.
_POSTER_LIGHT = "#e9edf2"
_POSTER_DARK = "#6f7b8a"


def _svg_escape(text: str) -> str:
    """Escape text for an SVG/XML text node (``&``, ``<``, ``>`` and both quotes)."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _wrap(text: str, width: int) -> List[str]:
    """Greedy word-wrap to ``width`` chars; never splits a word, never returns []."""
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= width:
            cur = f"{cur} {w}"
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _poster_board_svg(fen: Optional[str], x: float, y: float) -> str:
    """Render the FEN placement field as an 8x8 SVG board (rects + figurine glyphs)."""
    side = _POSTER_SQ * 8
    parts = [
        f'<rect x="{x}" y="{y}" width="{side}" height="{side}" '
        f'fill="{_POSTER_LIGHT}" stroke="#2a2d38" stroke-width="2"/>'
    ]
    if not fen:
        parts.append(
            f'<text x="{x + side / 2}" y="{y + side / 2}" text-anchor="middle" '
            'dominant-baseline="central" font-size="14" fill="#6b7280">'
            "no board snapshot</text>"
        )
        return "".join(parts)
    placement = fen.split(" ", 1)[0]
    for rank_idx, row in enumerate(placement.split("/")):
        file_idx = 0
        for ch in row:
            span = int(ch) if ch.isdigit() else 1
            for _ in range(span):
                cx = x + file_idx * _POSTER_SQ
                cy = y + rank_idx * _POSTER_SQ
                light = (rank_idx + file_idx) % 2 == 0
                if not light:
                    parts.append(
                        f'<rect x="{cx}" y="{cy}" width="{_POSTER_SQ}" '
                        f'height="{_POSTER_SQ}" fill="{_POSTER_DARK}"/>'
                    )
                if not ch.isdigit():
                    glyph = _PIECE_GLYPH.get(ch, "")
                    if glyph:
                        parts.append(
                            f'<text x="{cx + _POSTER_SQ / 2}" '
                            f'y="{cy + _POSTER_SQ / 2}" text-anchor="middle" '
                            'dominant-baseline="central" '
                            f'font-size="{int(_POSTER_SQ * 0.74)}" fill="#14151a">'
                            f"{glyph}</text>"
                        )
                file_idx += 1
    return "".join(parts)


def _poster_bar_svg(equity_white: float, x: float, y: float, h: float) -> str:
    """A White-POV vertical equity bar: white fills from the bottom to ``equity_white%``.

    Full bar (white) = White winning, empty = Black — the project's one bar convention
    (see :func:`chess_equity.bar.render_bar`). A midline marks the 50% reference.
    """
    pct = max(0.0, min(100.0, equity_white))
    fill_h = h * pct / 100.0
    mid_y = y + h / 2.0
    return (
        f'<rect x="{x}" y="{y}" width="{_POSTER_BAR_W}" height="{h}" '
        'fill="#14151a" stroke="#2a2d38" stroke-width="2"/>'
        f'<rect x="{x}" y="{y + h - fill_h}" width="{_POSTER_BAR_W}" '
        f'height="{fill_h}" fill="#f2f3f5"/>'
        f'<line x1="{x}" y1="{mid_y}" x2="{x + _POSTER_BAR_W}" y2="{mid_y}" '
        'stroke="#9aa0aa" stroke-width="1" stroke-dasharray="3 3"/>'
    )


def build_poster_svg(
    d: DramaEvent,
    index: int,
    *,
    sources: Optional[Dict[str, GameSource]] = None,
) -> str:
    """Render ONE ranked moment as a self-contained static SVG poster card.

    Layout (White-POV throughout): a vertical equity bar on the left, the board from
    the moment FEN beside it, a header naming the rank + drama kind/emoji + White-POV
    equity, and the ready-to-post social caption wrapped beneath. No external deps,
    fonts, or scripts — the SVG opens offline straight from disk. ``index`` is the
    1-based reel rank (shown as ``#N``).
    """
    emoji, label = _KIND_LABEL.get(d.kind, ("", d.kind))
    board_side = _POSTER_SQ * 8
    width = _POSTER_PAD + _POSTER_BAR_W + _POSTER_GAP + board_side + _POSTER_PAD
    board_x = _POSTER_PAD + _POSTER_BAR_W + _POSTER_GAP
    board_y = _POSTER_HEADER_H
    cap_lines = _wrap(social_caption(d, sources), _POSTER_CAP_CHARS)
    cap_top = board_y + board_side + 20
    height = cap_top + len(cap_lines) * _POSTER_LINE_H + _POSTER_PAD - _POSTER_LINE_H

    header = (
        f'<text x="{_POSTER_PAD}" y="22" font-size="18" font-weight="700" '
        f'fill="#14151a">#{index} {_svg_escape(emoji)} {_svg_escape(label)}</text>'
        f'<text x="{_POSTER_PAD}" y="40" font-size="13" fill="#6b7280">'
        f"magnitude {d.magnitude:.2f} · White {d.equity:.0f}% "
        f"({d.delta_equity:+.0f} pts)</text>"
    )
    caps = "".join(
        f'<text x="{_POSTER_PAD}" y="{cap_top + i * _POSTER_LINE_H}" '
        f'font-size="15" fill="#1d1f27">{_svg_escape(line)}</text>'
        for i, line in enumerate(cap_lines)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height:.0f}" viewBox="0 0 {width} {height:.0f}" '
        'font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif">\n'
        f'<rect width="{width}" height="{height:.0f}" fill="#ffffff"/>\n'
        f"{header}\n"
        f"{_poster_bar_svg(d.equity, _POSTER_PAD, board_y, board_side)}\n"
        f"{_poster_board_svg(d.fen, board_x, board_y)}\n"
        f"{caps}\n"
        "</svg>\n"
    )


def _poster_filename(index: int, d: DramaEvent) -> str:
    """Stable per-moment filename: zero-padded rank + drama kind (``poster-01-clutch.svg``)."""
    return f"poster-{index:02d}-{d.kind}.svg"


def write_posters(
    reel: List[DramaEvent],
    out_dir: str,
    *,
    sources: Optional[Dict[str, GameSource]] = None,
) -> List[str]:
    """Write one SVG poster per ranked moment into ``out_dir``; return the paths written.

    Files are named ``poster-NN-<kind>.svg`` in reel-rank order. An empty reel writes
    nothing and returns ``[]`` (a quiet game produces no posters). The directory is
    created if needed.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    for i, d in enumerate(reel, start=1):
        path = os.path.join(out_dir, _poster_filename(i, d))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(build_poster_svg(d, i, sources=sources))
        written.append(path)
    return written


def render_html(
    reel: List[DramaEvent],
    *,
    title: str = "Highlight reel",
    sources: Optional[Dict[str, GameSource]] = None,
) -> str:
    """Render the ranked reel as ONE self-contained HTML clip player.

    No external dependencies, CDNs, or scripts — opens offline straight from disk.
    Each ranked moment is a card with a Unicode board (from the FEN), the drama
    kind/emoji, the caster caption, and the equity swing. A <track kind="captions">
    (inline WebVTT data URI, one cue per clip — see :func:`build_webvtt`) rides
    along so the export is accessible and social-ready. Stays graceful on an empty
    reel (a quiet game). With ``sources`` (a round recap), each card names its
    source board + pairing and the subtitle reports how many boards the pool spans.
    """
    esc_title = html.escape(title)
    if not reel:
        body = (
            '<p class="empty">No highlight-worthy moments detected '
            "— a quiet game, or muted swings on the baseline model.</p>"
        )
        sub = ""
    else:
        tally = by_kind(reel)
        summary = ", ".join(f"{n} {kind}" for kind, n in sorted(tally.items()))
        span = ""
        if sources is not None:
            span = f"across {len({d.game_id for d in reel})} board(s), "
        sub = (
            f'<p class="sub">{len(reel)} moment(s), {span}ranked by drama magnitude '
            f"· {html.escape(summary)}</p>"
        )
        cards = "\n".join(
            _moment_card_html(i, d, sources) for i, d in enumerate(reel, start=1)
        )
        body = f"{_webvtt_track_html(reel)}\n{cards}"
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc_title}</title>\n"
        f"<style>\n{_HTML_STYLE}\n</style>\n</head>\n<body>\n"
        f"<h1>{esc_title}</h1>\n{sub}\n{body}\n</body>\n</html>\n"
    )
