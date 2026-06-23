"""Binned REAL outcomes — the two named failure modes, measured not asserted (task 0151).

This is the *direct* evidence behind the headline claim, on actual Lichess results: bin
the dataset by ``cp_eval`` × rating band and, in each cell, report the **measured** White
score (the mean of the real game outcomes, with its ``n``) beside what the rating-blind
``baseline`` and the rating-conditioned ``wdl-a`` predictor *say* the White score is. The
gap between "measured" and "baseline" is exactly the failure the thesis names; whether
``wdl-a`` closes that gap is exactly the thesis.

Two regions of the table carry the named failure modes (objective 0003), so the report
calls them out explicitly with a data-driven verdict (which predictor's mean is closer to
the measured rate — no hand-asserted "practical" numbers):

- **hard-0.00 isn't 50/50** — the ``|cp| <= 75`` dead-draw band (the same ±75cp window the
  failure-mode slicer and calibration code use). The rating-blind baseline reads cp≈0 as
  ≈0.50 by construction; the measured rate need not be, and it can drift by rating band.
- **good moves read as good** — the ``|cp| >= 1000`` decisive band (the absurd-refutation
  anchor magnitude). The engine reads these as near-won; whether the *result* is actually
  near-won — and how much that depends on who is playing — is the measured question.

Pure and dependency-free: it takes already-loaded :class:`PositionRow`s and the two
predictors and returns plain dataclasses, so it is unit-testable on a tiny fixture without
touching a dump. The *evidence* run (``scripts/failure_modes_real.py``) feeds it a real
dump; a fixture run is a smoke test of the aggregation, never evidence (CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from chess_equity.data.schema import PositionRow
from chess_equity.validate.failure_modes import CP_WINDOW
from chess_equity.validate.harness import baseline_cp, band_for_avg, wdl_a

# A predictor maps a row to a predicted White expected-score in [0, 1] (same contract as
# ``harness.Predictor``). The report compares each named predictor against the measured rate.
Predictor = Callable[[PositionRow], float]

# Signed cp_eval bin edges (White POV), in centipawns. ``-inf``/``+inf`` cap the ends so a
# mate sentinel lands in the decisive bin. The central ``(-CP_WINDOW, +CP_WINDOW]`` bin IS
# the "hard-0.00" dead-draw band; the outer ``|cp| > 1000`` bins are the decisive band.
_CP_EDGES: Tuple[float, ...] = (
    float("-inf"),
    -1000.0,
    -500.0,
    -200.0,
    -CP_WINDOW,
    CP_WINDOW,
    200.0,
    500.0,
    1000.0,
    float("inf"),
)

# Magnitude thresholds for the two named failure-mode regions (kept in lockstep with the
# bin edges above so a region is always a union of whole bins).
HARD_DRAW_MAX = CP_WINDOW       # |cp| <= 75 -> dead-draw / "hard 0.00"
DECISIVE_MIN = 1000.0           # |cp| >= 1000 -> "absurd-refutation" / decisive


def _cp_bin_label(cp: float) -> str:
    """Human-readable label for the signed-cp bin ``cp`` falls in (e.g. ``"(-75, 75]"``)."""
    for lo, hi in zip(_CP_EDGES, _CP_EDGES[1:]):
        if lo < cp <= hi:
            if lo == float("-inf"):
                return f"<= {int(hi)}"
            if hi == float("inf"):
                return f"> {int(lo)}"
            return f"({int(lo)}, {int(hi)}]"
    # cp == -inf can't occur for a real eval; fall through to the lowest bin defensively.
    return f"<= {int(_CP_EDGES[1])}"


def _cp_bin_index(cp: float) -> int:
    """Index into :data:`_CP_EDGES` pairs for sorting bins low→high cp."""
    for i, (lo, hi) in enumerate(zip(_CP_EDGES, _CP_EDGES[1:])):
        if lo < cp <= hi:
            return i
    return 0


@dataclass(frozen=True)
class Cell:
    """One (cp bin × rating band) cell of the measured-vs-predicted table.

    ``measured`` is the mean of the real White results in the cell — the ground truth the
    predictors are judged against. ``predicted`` maps each predictor name to its mean
    prediction over the same rows. ``n`` is the cell's row count (always reported, per the
    task) so under-powered cells are visible rather than hidden.
    """

    cp_bin: str
    band: str
    n: int
    measured: float
    predicted: dict
    _cp_index: int = 0

    def abs_error(self, predictor: str) -> float:
        """``|mean prediction − measured rate|`` for ``predictor`` in this cell."""
        return abs(self.predicted[predictor] - self.measured)


def bin_outcomes(
    rows: Sequence[PositionRow],
    predictors: Sequence[Tuple[str, Predictor]] = (
        ("baseline", baseline_cp),
        ("wdl-a", wdl_a),
    ),
    *,
    min_n: int = 1,
) -> List[Cell]:
    """Aggregate ``rows`` into (cp bin × rating band) cells of measured vs predicted score.

    For each cell: the measured White score (mean real ``result``), ``n``, and the mean of
    each predictor over the cell's rows. Cells with fewer than ``min_n`` rows are dropped
    (default keeps everything — the caller decides the floor). Returned sorted by cp bin
    (low→high) then rating band so the table reads top-to-bottom as Black-winning →
    White-winning within each band grouping.
    """
    buckets: dict = {}
    for row in rows:
        cp_idx = _cp_bin_index(row.cp_eval)
        key = (cp_idx, band_for_avg((row.white_elo + row.black_elo) / 2.0))
        buckets.setdefault(key, []).append(row)

    cells: List[Cell] = []
    for (cp_idx, band), cell_rows in buckets.items():
        n = len(cell_rows)
        if n < min_n:
            continue
        measured = sum(r.result for r in cell_rows) / n
        predicted = {
            name: sum(pred(r) for r in cell_rows) / n for name, pred in predictors
        }
        cells.append(
            Cell(
                cp_bin=_cp_bin_label(cell_rows[0].cp_eval),
                band=band,
                n=n,
                measured=measured,
                predicted=predicted,
                _cp_index=cp_idx,
            )
        )
    cells.sort(key=lambda c: (c._cp_index, c.band))
    return cells


def _region_cells(cells: Sequence[Cell], *, decisive: bool) -> List[Cell]:
    """Cells in a named failure-mode region: the decisive band, or the hard-draw band."""
    out = []
    for c in cells:
        # Recover the bin's magnitude from its index → edges (a bin is one whole edge pair).
        lo, hi = _CP_EDGES[c._cp_index], _CP_EDGES[c._cp_index + 1]
        is_hard = lo >= -HARD_DRAW_MAX and hi <= HARD_DRAW_MAX
        is_decisive = hi <= -DECISIVE_MIN or lo >= DECISIVE_MIN
        if (decisive and is_decisive) or (not decisive and is_hard):
            out.append(c)
    return out


def _winner(cells: Sequence[Cell], predictors: Sequence[str]) -> Tuple[dict, str]:
    """n-weighted mean |prediction − measured| per predictor over ``cells``, plus winner.

    The verdict is data-driven: whichever predictor's *mean prediction* sits closer to the
    *measured* rate (weighted by cell ``n`` so big cells count more) "tracks" the region.
    """
    total_n = sum(c.n for c in cells) or 1
    mae = {
        p: sum(c.abs_error(p) * c.n for c in cells) / total_n for p in predictors
    }
    winner = min(mae, key=lambda p: mae[p]) if mae else ""
    return mae, winner


def _table(cells: Sequence[Cell], predictors: Sequence[str]) -> List[str]:
    """A markdown table: one row per cell, measured beside each predictor's mean."""
    head = "| cp bin | rating | n | measured | " + " | ".join(predictors) + " |"
    rule = "|---|---|--:|--:|" + "--:|" * len(predictors)
    out = [head, rule]
    for c in cells:
        preds = " | ".join(f"{c.predicted[p]:.3f}" for p in predictors)
        out.append(
            f"| {c.cp_bin} | {c.band} | {c.n} | {c.measured:.3f} | {preds} |"
        )
    return out


def format_report(
    cells: Sequence[Cell],
    *,
    dump: str,
    n: int,
    seed: int = 0,
    predictors: Sequence[str] = ("baseline", "wdl-a"),
) -> str:
    """Render the binned-outcomes report as markdown.

    ``dump``/``n`` go in the header so the artifact states its real provenance (CLAUDE.md:
    no number without a named real dump). The body is the full cp×band table, then the two
    named failure-mode call-outs (hard-0.00, decisive) each with a data-driven verdict on
    which predictor tracks the measured rate.
    """
    lines: List[str] = []
    lines.append(
        f"# Failure modes on REAL binned Lichess outcomes — {dump}, n={n} (seed {seed})"
    )
    lines.append("")
    lines.append(
        "Each cell is **measured** from real game results: `measured` is the mean White "
        "score (1=win, 0.5=draw, 0=loss) of the games in that cp×rating cell, with its `n`. "
        "`baseline` is Lichess's rating-blind cp→Win%; `wdl-a` is the rating-conditioned "
        "predictor. The question is which prediction sits closer to what actually happened."
    )
    lines.append("")
    lines.append("## All cells (cp bin × rating band)")
    lines.append("")
    lines.extend(_table(cells, predictors))
    lines.append("")

    # --- Failure mode 1: hard 0.00 isn't 50/50 -------------------------------------
    hard = _region_cells(cells, decisive=False)
    lines.append(f"## Failure mode — \"hard 0.00 isn't 50/50\" (|cp| ≤ {int(HARD_DRAW_MAX)})")
    lines.append("")
    lines.append(
        "The rating-blind baseline reads a ≈0.00 eval as ≈0.50 by construction. These are "
        "the cells where that's tested against the real result."
    )
    lines.append("")
    if hard:
        lines.extend(_table(hard, predictors))
        lines.append("")
        mae, winner = _winner(hard, predictors)
        mae_str = ", ".join(f"{p} {mae[p]:.4f}" for p in predictors)
        lines.append(
            f"n-weighted mean |prediction − measured|: {mae_str} → **{winner} tracks the "
            f"measured rate closer** in the dead-draw band."
        )
    else:
        lines.append("_No rows in the |cp| ≤ 75 band in this dataset._")
    lines.append("")

    # --- Failure mode 2: good moves read as good -----------------------------------
    dec = _region_cells(cells, decisive=True)
    lines.append(f"## Failure mode — \"good moves read as good\" (|cp| ≥ {int(DECISIVE_MIN)})")
    lines.append("")
    lines.append(
        "Decisive positions the engine reads as near-won. Whether the *result* is near-won "
        "— and how much that hinges on who is playing — is the measured question."
    )
    lines.append("")
    if dec:
        lines.extend(_table(dec, predictors))
        lines.append("")
        mae, winner = _winner(dec, predictors)
        mae_str = ", ".join(f"{p} {mae[p]:.4f}" for p in predictors)
        lines.append(
            f"n-weighted mean |prediction − measured|: {mae_str} → **{winner} tracks the "
            f"measured rate closer** in the decisive band."
        )
    else:
        lines.append("_No rows in the |cp| ≥ 1000 band in this dataset._")
    lines.append("")

    return "\n".join(lines)
