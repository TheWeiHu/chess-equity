"""Real-time latency guard on the broadcast -> overlay per-move path (task 0169).

The OBS overlay is the live deliverable: it has to keep up with a real broadcast
feed. Task 0012 covered general caching/precompute, but nothing today *asserts* that
the per-move "ingest one half-move -> compute equity (clock-warped) -> build the
MoveEvent -> serialize the overlay event" path stays inside a real-time budget. A
silent regression (e.g. an accidental full re-parse per ply, a synchronous engine
call, an O(n^2) diff) would still pass every functional test while quietly making the
overlay lag behind the board. This test is that guard.

It replays a finished game one half-move per poll through the very objects the live
bridge runs (:class:`LocalPgnFeed` -> :class:`BroadcastIngestor.ingest_snapshot` ->
:meth:`MoveEvent.to_overlay_event`), times each per-move turnaround, and asserts the
**p95 per-move latency stays under a documented budget**.

Budget: **PER_MOVE_BUDGET_MS = 50 ms (p95)**. Rationale: a real broadcast emits a
move every few *seconds*, so anything in the tens of ms keeps the overlay effectively
instantaneous; 50 ms is a generous ceiling that absorbs slow/contended CI while still
catching a real (~40x) regression.

Measured on the baseline model (LichessBaselineModel, Apple-silicon laptop,
2026-06-24, n=280 over 40 replays of the 7-ply sample game):
    p50 ~ 0.7 ms, p95 ~ 1.2 ms, p99 ~ 1.4 ms, max ~ 1.4 ms.
i.e. ~40x of headroom under the 50 ms budget.

Fixture: ``data/sample/sample_games.pgn`` — the sanctioned offline-smoke fixture
(illustrative, not evidence; see project CLAUDE.md / ``reports/validation_sample.md``).
No torch, no network, no browser — unattended-safe.
"""
import io
import os
import time

import chess.pgn

from chess_equity.broadcast import BroadcastIngestor, LocalPgnFeed
from chess_equity.models import LichessBaselineModel

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PGN = os.path.join(HERE, "..", "data", "sample", "sample_games.pgn")

# Documented real-time budget for the per-move overlay path (see module docstring).
PER_MOVE_BUDGET_MS = 50.0
# Repeat the replay to accumulate a stable p95 from the short sample game.
REPLAYS = 40


def _percentile(values, pct):
    """Nearest-rank percentile of an already-collected latency list (pct in 0..100)."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    import math

    rank = max(1, int(math.ceil(pct / 100.0 * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]


def _first_game_pgn():
    with open(SAMPLE_PGN, "r", encoding="utf-8") as fh:
        text = fh.read()
    game = chess.pgn.read_game(io.StringIO(text))
    assert game is not None, "sample PGN must contain at least one game"
    assert list(game.mainline_moves()), "first sample game must have moves"
    return text


def _measure_per_move_latencies(pgn_text):
    """Replay the game move-by-move and return (per-move latency ms, event count)."""
    latencies = []
    events_seen = 0
    for _ in range(REPLAYS):
        feed = LocalPgnFeed(pgn_text)
        ingestor = BroadcastIngestor(
            feed, LichessBaselineModel(), white_elo=1500, black_elo=1500
        )
        while True:
            snapshot = feed.poll()
            if snapshot is None:
                break
            # Time the full per-move path: equity compute + MoveEvent build + the
            # overlay serialization the SSE bridge actually emits.
            t0 = time.perf_counter()
            events = ingestor.ingest_snapshot(snapshot)
            for ev in events:
                ev.to_overlay_event()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            # Attribute the snapshot's latency to each move it produced (one, here).
            latencies.extend([dt_ms] * max(1, len(events)))
            events_seen += len(events)
    return latencies, events_seen


def test_per_move_latency_p95_under_budget():
    pgn_text = _first_game_pgn()
    latencies, events_seen = _measure_per_move_latencies(pgn_text)

    assert events_seen > 0, "replay produced no move events"
    assert len(latencies) >= 100, (
        f"need a meaningful sample for p95; got {len(latencies)} timings"
    )

    p95 = _percentile(latencies, 95)
    worst = max(latencies)
    assert p95 < PER_MOVE_BUDGET_MS, (
        f"p95 per-move latency {p95:.2f} ms exceeds real-time budget "
        f"{PER_MOVE_BUDGET_MS:.0f} ms (worst {worst:.2f} ms over {len(latencies)} moves)"
    )
