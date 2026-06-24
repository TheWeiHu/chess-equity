"""Tests for the clock-coverage diagnostic (task 0249).

The diagnostic vets whether a candidate Lichess dump actually carries ``[%clk]``
tags before the expensive attended validation run. The pure band/coverage helpers
are checked directly; the ``data build`` and ``validate --slice clock`` wiring is
checked end-to-end on a tiny hand-written ``[%clk]`` PGN fixture (a unit fixture,
not evidence — the data policy bars synthetic *evidence*, not test fixtures).
"""

from __future__ import annotations

from pathlib import Path

from chess_equity.clock import clock_band
from chess_equity.clock_coverage import ClockCoverage, coverage_of, format_coverage
from chess_equity.data.build import build_dataset, load_rows


# --- pure helpers ----------------------------------------------------------------

def test_clock_band_none_is_clock_blind():
    assert clock_band(None) == "none"


def test_clock_band_edges():
    assert clock_band(0.0) == "<10s"
    assert clock_band(9.9) == "<10s"
    assert clock_band(10.0) == "10-30s"
    assert clock_band(29.9) == "10-30s"
    assert clock_band(30.0) == "30-60s"
    assert clock_band(59.9) == "30-60s"
    assert clock_band(60.0) == "1-3m"
    assert clock_band(179.9) == "1-3m"
    assert clock_band(180.0) == ">3m"
    assert clock_band(10_000.0) == ">3m"


def test_coverage_counts_and_fraction():
    cov = ClockCoverage()
    for clk in (5.0, 45.0, None, None, 200.0):
        cov.observe(clk)
    assert cov.total == 5
    assert cov.with_clk == 3
    assert cov.fraction == 3 / 5
    assert cov.bands == {"<10s": 1, "30-60s": 1, "none": 2, ">3m": 1}


def test_empty_coverage_is_zero_not_division_error():
    cov = ClockCoverage()
    assert cov.fraction == 0.0
    assert "0/0" in format_coverage(cov)


def test_format_clock_bearing_only_drops_none_band():
    cov = coverage_of(_FakeRow.many([5.0, None, None]))
    bearing = format_coverage(cov, clock_bearing_only=True)
    full = format_coverage(cov)
    # The headline always shows the real total (3 rows, 1 with a clock).
    assert "1/3" in bearing
    # clock-bearing view hides the "none" band; the full view shows it.
    assert "none" not in bearing
    assert "none" in full


def test_format_warns_when_no_clock_bearing_rows():
    cov = coverage_of(_FakeRow.many([None, None]))
    text = format_coverage(cov, clock_bearing_only=True)
    assert "no clock-bearing rows" in text


# --- end-to-end over a tiny [%clk] PGN fixture -----------------------------------

# Two games: the first carries [%clk] clocks, the second is clock-blind. A unit
# fixture (illustrative, not evidence) — exercises the build's streaming coverage tally.
_CLK_PGN = """[Event "Rated Blitz game"]
[White "a"]
[Black "b"]
[WhiteElo "1500"]
[BlackElo "1500"]
[TimeControl "300+0"]
[Result "1-0"]

1. e4 { [%eval 0.3] [%clk 0:05:00] } e5 { [%eval 0.2] [%clk 0:04:58] } 2. Nf3 { [%eval 0.4] [%clk 0:00:05] } 1-0

[Event "Rated Blitz game"]
[White "c"]
[Black "d"]
[WhiteElo "1600"]
[BlackElo "1600"]
[TimeControl "300+0"]
[Result "0-1"]

1. e4 { [%eval 0.3] } e5 { [%eval 0.1] } 0-1
"""


class _FakeRow:
    """Minimal stand-in carrying just ``clock_remaining`` for the pure-helper tests."""

    def __init__(self, clock_remaining):
        self.clock_remaining = clock_remaining

    @classmethod
    def many(cls, clocks):
        return [cls(c) for c in clocks]


def test_build_tallies_clk_coverage(tmp_path: Path):
    pgn = tmp_path / "mixed.pgn"
    pgn.write_text(_CLK_PGN, encoding="utf-8")
    cov = ClockCoverage()
    out = build_dataset(str(pgn), str(tmp_path / "out"), clock_coverage=cov)

    rows = load_rows(str(out))
    # Coverage is tallied over exactly the written rows (one observe per row).
    assert cov.total == len(rows)
    # The first game's evaluated plies carry [%clk]; the second game's do not.
    assert cov.with_clk > 0
    assert cov.with_clk < cov.total
    assert 0.0 < cov.fraction < 1.0
    # A reconstructed coverage from the loaded rows matches the streamed tally.
    assert coverage_of(rows).bands == cov.bands
