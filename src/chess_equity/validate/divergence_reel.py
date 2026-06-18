"""Divergence-reel proof artifact: SHOW the practical bar disagreeing with the engine (task 0139).

The product wedge is "the practical equity bar disagrees with the objective engine" —
the human-edge divergence badge (task 0048). The broadcast/overlay path now threads an
objective centipawn eval (tasks 0052/0103/0110) and the badge *can* fire, but until now
there was no committed, demonstrable EXAMPLE of it firing: the wedge's money shot lived
only in code.

This module is that money shot, made into a committed, CI-checkable artifact. It replays
one committed bullet game (``data/sample/divergence_reel.pgn``) through the exact
broadcast pipeline ``chess-equity broadcast --pgn`` uses, finds the plies where the
practical (clock-aware) bar diverges most from the objective cp, and writes:

* ``reports/divergence_reel.json`` — the overlay ``position`` events for the top
  divergence plies, each annotated with the human-edge badge that would fire; and
* ``reports/divergence_reel.md`` — a short, human-readable reel of those moments.

The committed game is a bullet scramble where White wins a clean knight (the objective
engine says White is winning, ~+3.0) but White's clock crashes to seconds. The practical
bar — which warps equity by the side-to-move's time pressure (task 0097) — flips to favor
Black: a winning position you can't convert before the flag falls is not a winning
position. That gap *is* the wedge.

UNATTENDED-OK / null-degrade: the artifact pins :class:`MaterialEngine` for both the
equity model and the cp fallback, so it is torch-free, Stockfish-free, and byte-for-byte
reproducible — CI is green without any engine binary. ``test_divergence_reel`` asserts
both that at least one ply fires the badge and that the committed artifacts stay in sync
with the generator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chess_equity.broadcast import BroadcastIngestor, GameEvent, LocalPgnFeed, MoveEvent
from chess_equity.models import LichessBaselineModel, MaterialEngine
from chess_equity.types import lichess_win_percent

# The committed bullet game whose scramble drives the divergence (one fixed input so the
# artifact is a stable regression target). src/chess_equity/validate/ -> parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
PGN_PATH = _REPO_ROOT / "data" / "sample" / "divergence_reel.pgn"
JSON_ARTIFACT_PATH = _REPO_ROOT / "reports" / "divergence_reel.json"
MD_ARTIFACT_PATH = _REPO_ROOT / "reports" / "divergence_reel.md"

# The human-edge divergence threshold, mirroring overlay/overlay.js (15 win-prob points).
# Keep in lockstep with the overlay badge: if one changes, change both.
THRESHOLD = 0.15
# How many of the largest-divergence plies to feature in the reel.
TOP_N = 5
# Round every float in the JSON artifact to this many places so the committed file is
# byte-for-byte reproducible (raw IEEE tails would make the drift guard platform-fragile).
_ROUND = 6


@dataclass(frozen=True)
class Divergence:
    """One ply's gap between the practical bar and the objective cp-implied win prob."""

    event: Dict[str, Any]  # the overlay ``position`` event (White-POV, equity in [0,1])
    cp_implied: float  # the rating-blind logistic of the engine cp, in [0, 1]
    gap: float  # equity_white - cp_implied (signed; >0 => White practically favored)


def cp_to_white_prob(cp: float) -> float:
    """Rating-blind win prob (White POV) for a White-POV centipawn eval, in [0, 1].

    The same logistic the overlay's cp ghost tick uses (``lichess_win_percent`` / 100),
    so the gap we measure here is exactly the one the overlay badge would compute.
    """
    return lichess_win_percent(cp) / 100.0


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def human_edge(event: Dict[str, Any], threshold: float = THRESHOLD) -> Optional[Dict[str, Any]]:
    """The human-edge badge for an overlay event, or ``None`` when it would not fire.

    Mirrors ``edge()`` / ``edgeLabel()`` in ``overlay/overlay.js`` exactly: the badge
    fires when the practical bar and the cp-implied bar disagree by ``>= threshold``
    win-prob points, and names which side holds the practical edge.
    """
    cp = event.get("cp")
    if cp is None:  # mate / no engine cp -> overlay hides the tick, no badge
        return None
    gap = _clamp01(float(event["equity"])) - cp_to_white_prob(float(cp))
    if abs(gap) < threshold:
        return None
    side = "white" if gap > 0 else "black"
    pts = round(abs(gap) * 100)
    return {
        "fires": True,
        "side": side,
        "gap": gap,
        "magnitude": min(1.0, abs(gap) / 0.5),
        "label": f"human edge · {side} +{pts} vs engine",
    }


def _model() -> LichessBaselineModel:
    """The pinned, torch-free, Stockfish-free model: rating-blind logistic over material.

    Material cp means equity == cp-implied *before* the clock warp, so every bit of the
    divergence we surface comes from the practical clock dimension — the cleanest possible
    demonstration of the wedge, and fully reproducible without any engine binary.
    """
    return LichessBaselineModel(MaterialEngine())


def collect_events(pgn_text: str) -> Tuple[Optional[GameEvent], List[MoveEvent]]:
    """Replay ``pgn_text`` through the broadcast pipeline -> (game event, move events).

    Uses the same :class:`BroadcastIngestor` wiring as ``chess-equity broadcast --pgn``
    (clock-aware, MaterialEngine cp fallback), so the events are what the live overlay
    would consume.
    """
    feed = LocalPgnFeed(pgn_text)
    ingestor = BroadcastIngestor(feed, _model(), clock_aware=True, engine=MaterialEngine())
    captured: List[GameEvent] = []
    ingestor.on_game = captured.append
    events = ingestor.ingest_snapshot(pgn_text)
    return (captured[0] if captured else None), events


def divergences(events: List[MoveEvent]) -> List[Divergence]:
    """All plies with a defined cp, sorted by descending |gap| (largest divergence first)."""
    out: List[Divergence] = []
    for ev in events:
        overlay: Dict[str, Any] = ev.to_overlay_event()
        cp = overlay.get("cp")
        if cp is None:
            continue
        cp_implied = cp_to_white_prob(float(cp))
        gap = _clamp01(float(overlay["equity"])) - cp_implied
        out.append(Divergence(event=overlay, cp_implied=cp_implied, gap=gap))
    out.sort(key=lambda d: abs(d.gap), reverse=True)
    return out


def top_divergences(pgn_text: str, top_n: int = TOP_N) -> Tuple[Optional[GameEvent], List[Divergence]]:
    """The ``top_n`` largest-divergence plies of the replayed game."""
    game, events = collect_events(pgn_text)
    return game, divergences(events)[:top_n]


def _round_floats(value: object) -> object:
    """Recursively round floats so the committed JSON is byte-for-byte reproducible."""
    if isinstance(value, float):
        return round(value, _ROUND)
    if isinstance(value, dict):
        return {k: _round_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_floats(v) for v in value]
    return value


def _read_pgn() -> str:
    return PGN_PATH.read_text(encoding="utf-8")


def generate_json_artifact(pgn_text: Optional[str] = None) -> str:
    """Build the committed ``reports/divergence_reel.json`` content.

    A JSON object carrying the game metadata and the top-``TOP_N`` overlay ``position``
    events where the practical bar diverges most from the objective cp, each annotated
    with the human-edge badge that fires.
    """
    if pgn_text is None:
        pgn_text = _read_pgn()
    game, top = top_divergences(pgn_text)
    payload = {
        "source": (
            "chess-equity broadcast --pgn data/sample/divergence_reel.pgn "
            "(clock-aware; rating-blind material cp fallback)"
        ),
        "threshold": THRESHOLD,
        "game": game.to_overlay() if game is not None else None,
        "divergences": [
            {
                **d.event,
                "cp_implied": d.cp_implied,
                "divergence": {
                    "gap": d.gap,
                    "abs_gap": abs(d.gap),
                    **(human_edge(d.event) or {"fires": False}),
                },
            }
            for d in top
        ],
    }
    return json.dumps(_round_floats(payload), indent=2, ensure_ascii=False) + "\n"


def generate_md_artifact(pgn_text: Optional[str] = None) -> str:
    """Build the committed ``reports/divergence_reel.md`` — a short human-readable reel."""
    if pgn_text is None:
        pgn_text = _read_pgn()
    game, top = top_divergences(pgn_text)
    white = black = None
    if game is not None:
        white, black = game.white_name, game.black_name
    lines = [
        "# Divergence reel — where the practical bar disagrees with the engine",
        "",
        "_Generated by `python -m chess_equity.validate.divergence_reel` from "
        "`data/sample/divergence_reel.pgn` (a test asserts this file stays in sync)._",
        "",
        "The product wedge: **the practical equity bar disagrees with the objective "
        "engine** (the human-edge divergence badge, task 0048). Below is it firing on a "
        "real game.",
        "",
        f"The committed bullet game ({white or 'White'} vs {black or 'Black'}, 60+0): "
        "White wins a clean knight, so the objective engine sees White clearly winning "
        "(~+3.0). But White's clock crashes to seconds. The **clock-aware** practical bar "
        "(task 0097) knows a won position you can't convert before the flag falls is not "
        "won — so it flips toward Black. That gap is the wedge.",
        "",
        f"_Divergence badge fires when |practical equity − cp-implied win prob| "
        f"≥ {THRESHOLD:.2f} (15 win-prob points). Top {len(top)} moments:_",
        "",
        "| Ply | Move | Clock (W/B) | Engine cp | cp-implied | Practical bar | Gap | Badge |",
        "|--:|:--|:--|--:|--:|--:|--:|:--|",
    ]
    for d in top:
        e = d.event
        clk = e.get("clock") or {}
        cw, cb = clk.get("white"), clk.get("black")
        cp = e.get("cp")
        edge = human_edge(e)
        badge = edge["label"] if edge else "—"
        lines.append(
            f"| {e['ply']} | {e['move']['san']} "
            f"| {_fmt_secs(cw)}/{_fmt_secs(cb)} "
            f"| {cp:+.0f} | {d.cp_implied * 100:.0f}% "
            f"| {float(e['equity']) * 100:.0f}% | {d.gap * 100:+.0f} pts | {badge} |"
        )
    lines += [
        "",
        "The engine column never moves off the won evaluation; the practical bar collapses "
        "as the flag approaches. **Objective truth and practical reality disagree — and the "
        "overlay shows it.** That is the human edge the product sells.",
        "",
    ]
    return "\n".join(lines)


def _fmt_secs(secs: Optional[float]) -> str:
    if secs is None:
        return "–"
    if secs == int(secs):
        return f"{int(secs)}s"
    return f"{secs:g}s"


def write_artifacts() -> List[Path]:
    """Regenerate both committed artifacts on disk; returns the paths written."""
    pgn_text = _read_pgn()
    JSON_ARTIFACT_PATH.write_text(generate_json_artifact(pgn_text), encoding="utf-8")
    MD_ARTIFACT_PATH.write_text(generate_md_artifact(pgn_text), encoding="utf-8")
    return [JSON_ARTIFACT_PATH, MD_ARTIFACT_PATH]


if __name__ == "__main__":  # pragma: no cover - manual regeneration entry point
    for written in write_artifacts():
        print(f"wrote divergence-reel artifact to {written}")
