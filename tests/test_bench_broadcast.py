"""Liveness smoke for the broadcast throughput bench (task 0191).

``scripts/bench-broadcast.py`` is the *opt-in* throughput gate — it is deliberately
outside ``testpaths`` so the moves/sec *floor* never runs on the hot CI path (a perf
assertion would flake on slow/contended runners). But the bench is only useful if it
still *runs*: an API drift in :class:`BroadcastIngestor` or the feed would otherwise
silently rot it. This smoke runs the bench at a tiny replay count and asserts only that
it produced move events and a positive throughput — no floor, so it stays fast and
non-flaky while keeping the bench wired to reality.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(HERE, "..", "scripts", "bench-broadcast.py")


def _load_bench():
    spec = importlib.util.spec_from_file_location("bench_broadcast", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_runs_and_reports_throughput():
    bench = _load_bench()
    pgn_text = bench._load_pgn(bench.DEFAULT_PGN)
    moves_per_sec, latencies, events = bench.benchmark(pgn_text, replays=2)

    assert events > 0, "bench replay produced no move events"
    assert moves_per_sec > 0, "bench reported non-positive throughput"
    assert len(latencies) == events, "one latency sample per emitted move event"


def test_bench_check_passes_at_trivial_floor():
    bench = _load_bench()
    # --check with a trivially-low floor must exit 0 (the floor gate is wired correctly).
    assert bench.main(["--replays", "2", "--check", "--floor", "1"]) == 0
