"""Clock-coverage diagnostic (task 0249).

Before paying for a newer Lichess dump to *prove* the clock dimension (clock.py,
task 0015), we need to know whether a candidate dump actually carries ``[%clk]``
tags at all — the cached 2016-05 dump predates them, so it can't validate the
flag-risk model no matter how it's sliced.

This module is the cheap, unattended-safe vetting scaffold: it tallies the
fraction of parsed rows carrying a side-to-move clock (``clock_remaining is not
None``) and their distribution over :func:`chess_equity.clock.clock_band`. ``data
build`` feeds rows through :class:`ClockCoverage` in its single streaming pass (so
a multi-GB dump is never re-read), and ``validate --slice clock`` re-derives the
same summary from a built dataset over clock-bearing rows only.

Pure and dependency-free (stdlib + :func:`~chess_equity.clock.clock_band`), so it
is trivially testable on a tiny ``[%clk]`` fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from chess_equity.clock import CLOCK_BANDS, clock_band


@dataclass
class ClockCoverage:
    """Running tally of how many parsed rows carry a side-to-move ``[%clk]`` clock.

    ``observe`` is fed one row's ``clock_remaining`` at a time (``None`` when that row
    carried no clock), so it composes with the build's streaming write — no need to
    hold the whole dataset in memory.
    """

    total: int = 0
    with_clk: int = 0
    bands: Dict[str, int] = field(default_factory=dict)

    def observe(self, clock_remaining: Optional[float]) -> None:
        """Record one row's side-to-move clock (``None`` == no ``[%clk]``)."""
        self.total += 1
        if clock_remaining is not None:
            self.with_clk += 1
        label = clock_band(clock_remaining)
        self.bands[label] = self.bands.get(label, 0) + 1

    @property
    def fraction(self) -> float:
        """Fraction of observed rows carrying a clock (``0.0`` when nothing seen)."""
        return self.with_clk / self.total if self.total else 0.0


def coverage_of(rows: Iterable) -> ClockCoverage:
    """Build a :class:`ClockCoverage` from an iterable of :class:`PositionRow`."""
    cov = ClockCoverage()
    for row in rows:
        cov.observe(row.clock_remaining)
    return cov


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def format_coverage(cov: ClockCoverage, *, clock_bearing_only: bool = False) -> str:
    """Render a coverage summary as a short, diff-friendly text block.

    The headline is ``<with_clk>/<total>`` rows carrying ``[%clk]`` and the percentage,
    followed by a per-:func:`~chess_equity.clock.clock_band` table. With
    ``clock_bearing_only`` the ``"none"`` band is dropped and each band's share is taken
    over clock-bearing rows (what ``validate --slice clock`` reports); otherwise the
    share is over all rows (what ``data build`` reports).
    """
    lines = [
        f"clock coverage: {cov.with_clk}/{cov.total} rows carry [%clk] "
        f"({cov.fraction * 100:.1f}%)"
    ]
    denom = cov.with_clk if clock_bearing_only else cov.total
    for band in CLOCK_BANDS:
        if clock_bearing_only and band == "none":
            continue
        n = cov.bands.get(band, 0)
        lines.append(f"  {band:>7}: {n:>8}  ({_pct(n, denom):5.1f}%)")
    if clock_bearing_only and cov.with_clk == 0:
        lines.append("  (no clock-bearing rows — this dump predates [%clk] or has none)")
    return "\n".join(lines)
