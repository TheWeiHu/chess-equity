#!/usr/bin/env python3
"""Broadcast bridge throughput benchmark + documented real-time floor (task 0191).

The streaming OBS overlay is the live deliverable (the 0012 real-time requirement):
the bridge has to ingest a broadcast feed faster than the board produces moves. The
companion ``tests/test_broadcast_latency_guard.py`` already guards the *p95 per-move
latency* (< 50 ms) on the hot CI path. This script is the *throughput* view — how many
moves/sec the bridge sustains end-to-end — and a documented floor so a regression that
quietly breaks real-time (an accidental full re-parse per ply, a synchronous engine
call, an O(n^2) diff) is caught.

It replays a cached PGN through the very objects the live bridge runs
(:class:`LocalPgnFeed` -> :class:`BroadcastIngestor.ingest_snapshot` ->
:meth:`MoveEvent.to_overlay_event`) with the baseline model and drama ON (the default),
and reports moves/sec plus p50/p95/p99 per-move latency.

This is an OPT-IN check, not a pytest test: it lives in ``scripts/`` (outside
``testpaths = ["tests"]``) so it never runs on the hot CI path. Run it by hand, or wire
``--check`` into a perf job; ``--check`` exits non-zero when sustained throughput falls
below the floor.

    # human-readable report
    python scripts/bench-broadcast.py

    # regression gate (exit 1 if below floor)
    python scripts/bench-broadcast.py --check

Floor: ``MIN_MOVES_PER_SEC = 200``. Rationale: a real broadcast emits a move every few
*seconds*, so 200 moves/sec is already ~1000x of real-time headroom; it sits ~6.5x under
the baseline-model throughput measured below, so it absorbs slow/contended CI while still
catching a real (>6x) regression.

Measured on the baseline model (LichessBaselineModel, Apple-silicon laptop, 2026-06-24,
n=1400 over 200 replays of the 7-ply sample game):
    throughput ~ 1300 moves/sec; per-move p50 ~ 0.8 ms, p95 ~ 1.3 ms, p99 ~ 1.5 ms.
i.e. ~6.5x of headroom over the 200 moves/sec floor.

Fixture: ``data/sample/sample_games.pgn`` — the sanctioned offline-smoke fixture
(illustrative, not evidence; see project CLAUDE.md / ``reports/validation_sample.md``).
No torch, no network, no browser — unattended-safe.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time

import chess.pgn

from chess_equity.broadcast import BroadcastIngestor, LocalPgnFeed
from chess_equity.models import LichessBaselineModel

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PGN = os.path.join(HERE, "..", "data", "sample", "sample_games.pgn")

# Documented real-time throughput floor for the broadcast bridge (see module docstring).
MIN_MOVES_PER_SEC = 200.0
# Replay the short sample game this many times to accumulate a stable sample.
DEFAULT_REPLAYS = 200


def _percentile(values, pct):
    """Nearest-rank percentile of a latency list (pct in 0..100)."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(1, int(math.ceil(pct / 100.0 * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]


def _load_pgn(path):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None or not list(game.mainline_moves()):
        raise SystemExit(f"PGN {path!r} has no playable game with moves")
    return text


def benchmark(pgn_text, replays):
    """Replay the feed `replays` times; return (moves_per_sec, latencies_ms, events)."""
    latencies = []
    events_seen = 0
    # Time only the per-move compute path (ingest + overlay serialize), excluding feed
    # construction, so the moves/sec figure reflects what the bridge does per move.
    compute_s = 0.0
    for _ in range(replays):
        feed = LocalPgnFeed(pgn_text)
        ingestor = BroadcastIngestor(
            feed, LichessBaselineModel(), white_elo=1500, black_elo=1500
        )
        while True:
            snapshot = feed.poll()
            if snapshot is None:
                break
            t0 = time.perf_counter()
            events = ingestor.ingest_snapshot(snapshot)
            for ev in events:
                ev.to_overlay_event()  # the serialization the SSE bridge actually emits
            dt = time.perf_counter() - t0
            compute_s += dt
            latencies.extend([dt * 1000.0] * max(1, len(events)))
            events_seen += len(events)
    moves_per_sec = events_seen / compute_s if compute_s > 0 else 0.0
    return moves_per_sec, latencies, events_seen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn", default=DEFAULT_PGN, help="PGN fixture to replay")
    ap.add_argument("--replays", type=int, default=DEFAULT_REPLAYS,
                    help="how many times to replay the game (sample size)")
    ap.add_argument("--floor", type=float, default=MIN_MOVES_PER_SEC,
                    help="documented moves/sec floor for --check")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if sustained moves/sec is below the floor")
    args = ap.parse_args(argv)

    pgn_text = _load_pgn(args.pgn)
    # One warm-up replay so first-call import/JIT costs don't skew the measured run.
    benchmark(pgn_text, 1)
    moves_per_sec, latencies, events = benchmark(pgn_text, args.replays)

    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)
    worst = max(latencies) if latencies else 0.0

    print(f"broadcast bridge throughput benchmark (baseline model, drama on)")
    print(f"  fixture     : {os.path.relpath(args.pgn)}")
    print(f"  replays     : {args.replays}  ({events} move events)")
    print(f"  throughput  : {moves_per_sec:,.0f} moves/sec")
    print(f"  per-move ms : p50 {p50:.3f}  p95 {p95:.3f}  p99 {p99:.3f}  max {worst:.3f}")
    print(f"  floor       : {args.floor:,.0f} moves/sec")

    if args.check:
        if moves_per_sec < args.floor:
            print(f"FAIL: {moves_per_sec:,.0f} moves/sec is below the {args.floor:,.0f} "
                  f"moves/sec real-time floor", file=sys.stderr)
            return 1
        print(f"OK: {moves_per_sec / args.floor:.1f}x of headroom over the floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
