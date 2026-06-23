"""N-aware shrinkage of the ``wdl-a`` predictor toward the rating-blind baseline (task 0163).

The rating-conditioned ``wdl-a`` regression over-predicts in sparse high-rating cp×rating
cells — e.g. ``failure_modes_real.md`` shows it reading ``<=-1000cp × 2000-2399`` (n=35)
as 0.512 when the measured White score is 0.029. That blowup drives the ``2000-2399`` band's
ECE (~0.30) — the one gate slice where equity loses to the rating-blind baseline. The cause
is small-n, not the thesis: with a handful of games behind a cell the regression's
``cp × skill`` extrapolation has nothing to anchor it.

The fix here is a textbook empirical-Bayes shrinkage: blend each prediction toward the
rating-blind baseline with a per-cell weight ``w = n / (n + k)``. Well-populated cells keep
their ``wdl-a`` value (``w → 1``); sparse cells fall back toward the baseline (``w → 0``).
``k`` is the shrinkage strength (the pseudo-count at which a cell is trusted half-and-half).

``k = 0`` is an exact no-op (``w = 1`` for every seen cell), so the knob is **off by
default** and the committed report numbers do not move unless an operator opts in with
``chess-equity validate --shrink-wdl-a-k K``.

Pure computation over :class:`~chess_equity.data.schema.PositionRow` lists (no I/O, no
numpy), so it stays in the light test path. The cell binning is shared with
:mod:`chess_equity.validate.binned_outcomes` (same cp bins) and
:func:`chess_equity.validate.harness.band_for_avg` (same rating bands), so the cells the
shrinkage reads are exactly the cells ``failure_modes_real.md`` reports.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from chess_equity.data.schema import PositionRow
from chess_equity.validate.binned_outcomes import _cp_bin_index
from chess_equity.validate.harness import (
    Predictor,
    band_for_avg,
    baseline_cp,
    wdl_a,
)

# A shrinkage cell key: (cp-bin index, rating band) — identical to the cells in
# binned_outcomes / failure_modes_real.md so "low-n cell" means the same thing here.
CellKey = Tuple[int, str]


def cell_key(row: PositionRow) -> CellKey:
    """The (cp-bin index, rating band) cell ``row`` falls in."""
    avg = (row.white_elo + row.black_elo) / 2.0
    return (_cp_bin_index(row.cp_eval), band_for_avg(avg))


def cell_counts(rows: Sequence[PositionRow]) -> Dict[CellKey, int]:
    """Per-cell row counts over ``rows`` — the support the shrinkage weight reads."""
    counts: Dict[CellKey, int] = {}
    for row in rows:
        key = cell_key(row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def shrinkage_weight(n: int, k: float) -> float:
    """``w = n / (n + k)`` — the weight on the model vs the baseline for a cell of size ``n``.

    ``k == 0`` returns ``1.0`` (the model unchanged, even for an unseen ``n == 0`` cell), so
    the knob is an exact no-op when off. ``k > 0`` and ``n == 0`` returns ``0.0`` (a cell with
    no support is fully baseline).
    """
    if k < 0:
        raise ValueError(f"shrinkage k must be >= 0, got {k}")
    if k == 0:
        return 1.0
    denom = n + k
    return n / denom if denom > 0 else 0.0


def make_shrunk_predictor(
    rows: Sequence[PositionRow],
    k: float,
    *,
    model: Predictor = wdl_a,
    baseline: Predictor = baseline_cp,
) -> Predictor:
    """A predictor blending ``model`` toward ``baseline`` by per-cell weight ``n / (n + k)``.

    ``rows`` supplies the per-cell counts (the support the shrinkage reads — pass the full
    evaluation dataset). ``k`` is the shrinkage strength; ``k == 0`` returns ``model``
    unchanged. A cell unseen in ``rows`` (``n == 0``) is fully ``baseline`` when ``k > 0``.
    """
    if k < 0:
        raise ValueError(f"shrinkage k must be >= 0, got {k}")
    counts = cell_counts(rows)

    def predict(row: PositionRow) -> float:
        n = counts.get(cell_key(row), 0)
        w = shrinkage_weight(n, k)
        if w >= 1.0:
            return model(row)
        if w <= 0.0:
            return baseline(row)
        return w * model(row) + (1.0 - w) * baseline(row)

    return predict


__all__ = [
    "CellKey",
    "cell_key",
    "cell_counts",
    "shrinkage_weight",
    "make_shrunk_predictor",
]
