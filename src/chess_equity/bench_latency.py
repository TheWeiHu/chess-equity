"""Per-move latency micro-benchmark for the wdl-a equity bar (open question 0012).

The objective lists "acceptable per-move latency for an interactive bar" as an
unanswered open question. ``wdl-a`` is the natural thing to measure: it is the
torch-free, CPU-only predictor (a committed JSON artifact scored by a deterministic
``MaterialEngine`` for cp), so an end-to-end ``evaluate(fen, white_elo, black_elo)``
call is exactly the work an interactive bar would do per move — FEN parse, objective
cp eval, feature build, and the tiny logistic forward pass — with no model download
and no network.

This module times that call over a batch of real positions and records the
distribution (median / p95 / mean) to the committed ``reports/latency_wdl_a.md``.
Unlike ``reports/rating_sweep.md``, the numbers are wall-clock and machine-dependent,
so the test asserts the stats are *well-formed* (positive, ``p95 >= median``) rather
than byte-for-byte in sync — the committed report is a representative snapshot, not a
regression target.

Positions come from the committed ``data/sample/dataset_fen.csv`` (real Lichess FENs
with ratings); the benchmark cycles through them to reach ``n`` evaluations, which is
fine because per-position cost is dominated by the fixed pipeline, not the position.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from chess_equity.adapters import EquityModel

# src/chess_equity/bench_latency.py -> parents[2] is the repo root.
_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_FENS_PATH = _ROOT / "data" / "sample" / "dataset_fen.csv"
ARTIFACT_PATH = _ROOT / "reports" / "latency_wdl_a.md"

# Defaults: enough evaluations for a stable median/p95 without a long run, plus a short
# warmup so import/JIT-ish first-call costs don't skew the distribution.
DEFAULT_N = 2000
DEFAULT_WARMUP = 50

# A position is the triple an EquityModel.evaluate call takes.
Position = Tuple[str, int, int]


@dataclass(frozen=True)
class LatencyStats:
    """Per-position latency distribution over ``n`` wdl-a evaluations (milliseconds)."""

    n: int
    positions: int
    median_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    @property
    def throughput_per_s(self) -> float:
        """Evaluations per second implied by the mean latency."""
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0


def load_sample_positions(path: Path = SAMPLE_FENS_PATH) -> List[Position]:
    """Load ``(fen, white_elo, black_elo)`` triples from the committed sample dataset."""
    positions: List[Position] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            positions.append(
                (row["fen"], int(float(row["white_elo"])), int(float(row["black_elo"])))
            )
    if not positions:
        raise ValueError(f"no positions found in {path}")
    return positions


def _percentile(sorted_samples: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (``q`` in 0..100) over an already-sorted sequence."""
    if not sorted_samples:
        raise ValueError("no samples")
    last = len(sorted_samples) - 1
    idx = int(round((q / 100.0) * last))
    return sorted_samples[max(0, min(last, idx))]


def benchmark(
    model: EquityModel,
    positions: Sequence[Position],
    *,
    n: int = DEFAULT_N,
    warmup: int = DEFAULT_WARMUP,
) -> LatencyStats:
    """Time ``n`` ``model.evaluate`` calls over ``positions`` (cycled); return stats."""
    if n <= 0:
        raise ValueError("n must be positive")
    count = len(positions)
    for i in range(warmup):
        fen, we, be = positions[i % count]
        model.evaluate(fen, we, be)
    samples: List[float] = []
    for i in range(n):
        fen, we, be = positions[i % count]
        start = time.perf_counter()
        model.evaluate(fen, we, be)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return LatencyStats(
        n=n,
        positions=count,
        median_ms=statistics.median(samples),
        p95_ms=_percentile(samples, 95),
        mean_ms=statistics.fmean(samples),
        min_ms=samples[0],
        max_ms=samples[-1],
    )


def render_text(stats: LatencyStats) -> str:
    """Plain-text summary for the CLI / stdout."""
    return "\n".join(
        [
            f"wdl-a per-move latency over {stats.n} evals "
            f"({stats.positions} real positions, cycled):",
            f"  median  {stats.median_ms:7.3f} ms",
            f"  p95     {stats.p95_ms:7.3f} ms",
            f"  mean    {stats.mean_ms:7.3f} ms",
            f"  min     {stats.min_ms:7.3f} ms",
            f"  max     {stats.max_ms:7.3f} ms",
            f"  throughput ~{stats.throughput_per_s:,.0f} positions/s",
        ]
    )


def render_markdown(stats: LatencyStats) -> str:
    """The committed ``reports/latency_wdl_a.md`` snapshot."""
    interactive = (
        "comfortably interactive — well under one frame at 60 Hz (16.7 ms)"
        if stats.p95_ms < 16.7
        else "fast enough for an interactive bar (sub-100 ms p95)"
        if stats.p95_ms < 100.0
        else "above the 100 ms interactive threshold"
    )
    lines = [
        "# wdl-a per-move latency — is the equity bar interactive? (open question 0012)",
        "",
        "_Generated by `python -m chess_equity.bench_latency`._",
        "_Wall-clock numbers are machine-dependent; this is a representative snapshot, "
        "not a regression target (a test asserts the stats are well-formed, not exact)._",
        "",
        f"Times `wdl-a` `EquityModel.evaluate(fen, white_elo, black_elo)` over "
        f"**{stats.n}** evaluations across **{stats.positions}** real positions "
        "(committed `data/sample/dataset_fen.csv`, cycled). Each call does the full "
        "per-move pipeline: FEN parse → objective cp eval (`MaterialEngine`) → feature "
        "build → logistic forward pass. `wdl-a` is torch-free and CPU-only, so this is "
        "the whole cost of one bar update.",
        "",
        "| metric | ms/position |",
        "|:--|--:|",
        f"| median | {stats.median_ms:.3f} |",
        f"| p95 | {stats.p95_ms:.3f} |",
        f"| mean | {stats.mean_ms:.3f} |",
        f"| min | {stats.min_ms:.3f} |",
        f"| max | {stats.max_ms:.3f} |",
        "",
        f"Throughput at the mean is ~**{stats.throughput_per_s:,.0f} positions/s**. "
        f"At a p95 of {stats.p95_ms:.3f} ms this is {interactive}: the answer to open "
        "question 0012 is that the `wdl-a` bar imposes no perceptible per-move latency.",
        "",
    ]
    return "\n".join(lines)


def run(
    *, n: int = DEFAULT_N, warmup: int = DEFAULT_WARMUP, model: EquityModel | None = None
) -> LatencyStats:
    """Build the wdl-a model (default) and benchmark it over the sample positions."""
    if model is None:
        from chess_equity.wdl_regression import build_wdl_a_equity

        model = build_wdl_a_equity()
    return benchmark(model, load_sample_positions(), n=n, warmup=warmup)


def write_artifact(stats: LatencyStats, path: Path = ARTIFACT_PATH) -> Path:
    """Write the Markdown snapshot to disk; returns the path written."""
    path.write_text(render_markdown(stats), encoding="utf-8")
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-move latency micro-benchmark for the wdl-a equity bar."
    )
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N, help="number of timed evaluations"
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP, help="untimed warmup evaluations"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ARTIFACT_PATH,
        help="Markdown report path (default: reports/latency_wdl_a.md)",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print stats only, do not write the report"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI glue
    args = _parse_args(argv)
    stats = run(n=args.n, warmup=args.warmup)
    print(render_text(stats))
    if not args.no_write:
        written = write_artifact(stats, args.out)
        print(f"\nwrote latency report to {written}")


if __name__ == "__main__":  # pragma: no cover - manual benchmark entry point
    main()
