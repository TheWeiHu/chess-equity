"""Tests for the wdl-a per-move latency micro-benchmark (task 0145)."""

from __future__ import annotations

from chess_equity import bench_latency


def test_load_sample_positions_nonempty():
    positions = bench_latency.load_sample_positions()
    assert positions, "expected real positions from the committed sample dataset"
    fen, white_elo, black_elo = positions[0]
    assert isinstance(fen, str) and " " in fen  # a FEN, not an empty cell
    assert white_elo > 0 and black_elo > 0


def test_percentile_nearest_rank():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert bench_latency._percentile(data, 0) == 1.0
    assert bench_latency._percentile(data, 100) == 5.0
    assert bench_latency._percentile(data, 50) == 3.0


def test_benchmark_stats_well_formed():
    # Small n keeps the test in the light path; the point is shape, not timing.
    stats = bench_latency.run(n=40, warmup=5)
    assert stats.n == 40
    assert stats.positions == len(bench_latency.load_sample_positions())
    assert stats.min_ms > 0
    assert stats.median_ms > 0
    assert stats.mean_ms > 0
    # Distribution ordering invariants.
    assert stats.min_ms <= stats.median_ms <= stats.max_ms
    assert stats.p95_ms >= stats.median_ms
    assert stats.p95_ms <= stats.max_ms
    assert stats.throughput_per_s > 0


def test_render_and_write_artifact(tmp_path):
    stats = bench_latency.run(n=20, warmup=2)
    text = bench_latency.render_text(stats)
    assert "median" in text and "p95" in text
    md = bench_latency.render_markdown(stats)
    assert md.startswith("# wdl-a per-move latency")
    assert "| median |" in md and "| p95 |" in md
    out = tmp_path / "latency_wdl_a.md"
    written = bench_latency.write_artifact(stats, out)
    assert written == out
    assert out.read_text(encoding="utf-8") == md
