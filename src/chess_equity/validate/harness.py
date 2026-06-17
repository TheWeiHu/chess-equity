"""The validation gate: does a rating-aware predictor beat the rating-blind centipawn
baseline at predicting *actual* Lichess outcomes? (task 0009)

A **predictor** maps a dataset :class:`~chess_equity.data.schema.PositionRow` to a
predicted White expected-score in [0, 1]. That signature is deliberately the natural
fit for a model that conditions on ``(cp_eval, white_elo, black_elo)`` — exactly
Approach A (task 0004) — so it drops in as a registry entry with no harness change.

The one predictor shipped today is :func:`baseline_cp` — Lichess's rating-blind
Win% over the row's centipawn eval. It IS the thing to beat (the whole thesis). The
harness scores every registered predictor with :mod:`chess_equity.validate.metrics`,
overall and sliced by rating band and game phase, so a model that only wins in the
off-2300 bands (Wei's claim) shows up even when the global number is a wash.

Models that need the full board (Maia, 0005) can't be scored here yet: the 0002
dataset stores ``cp_eval`` but not the FEN. See the follow-up to add positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from chess_equity.clock import clock_adjusted_white_equity
from chess_equity.data.schema import PositionRow
from chess_equity.types import lichess_win_percent
from chess_equity.validate.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
)

# A predictor: a position row -> predicted White expected-score in [0, 1].
Predictor = Callable[[PositionRow], float]


def baseline_cp(row: PositionRow) -> float:
    """Lichess's rating-blind Win% for the row's centipawn eval, as White expected-score.

    ``cp_eval`` is already White-POV, so the Win% maps straight to a White prediction.
    Rating-blind by construction — the baseline every rating-conditioned model must beat.
    """
    return lichess_win_percent(row.cp_eval) / 100.0


def baseline_cp_clock(row: PositionRow) -> float:
    """The rating-blind baseline, then warped by the side-to-move's time pressure (0015).

    Same centipawn signal as :func:`baseline_cp`, but a winning position with seconds
    left reads as less safe — the practical-result effect objective eval (and Maia-2,
    which has no clock input) misses entirely. A no-op on clock-blind rows, so it can
    only help where ``[%clk]`` data exists.
    """
    return clock_adjusted_white_equity(
        baseline_cp(row),
        row.clock_remaining,
        row.tc_bucket,
        white_to_move=row.side_to_move == "white",
    )


_WDL_A_MODEL = None


def wdl_a(row: PositionRow) -> float:
    """Approach A — the rating-conditioned WDL regression (task 0004).

    The natural drop-in for this gate: it reads ``(cp_eval, white_elo, black_elo, ply,
    tc_bucket)`` straight off the row, so it sits beside the rating-blind baseline with
    no harness change. The fitted artifact is loaded lazily and cached, so importing
    this module stays free of the model file (and a missing artifact only bites the
    callers that actually ask for ``wdl-a``).
    """
    global _WDL_A_MODEL
    if _WDL_A_MODEL is None:
        from chess_equity.wdl_regression import load_wdl_a_model

        _WDL_A_MODEL = load_wdl_a_model()
    return _WDL_A_MODEL.predict_white_equity(
        row.cp_eval, row.white_elo, row.black_elo, row.ply, row.tc_bucket
    )


# The registry the CLI selects from. New approaches register here.
PREDICTORS: Dict[str, Predictor] = {
    "baseline": baseline_cp,
    "baseline+clock": baseline_cp_clock,
    "wdl-a": wdl_a,
}


def rating_band(row: PositionRow) -> str:
    """Coarse band on the average of the two ratings (the relevant joint skill level)."""
    avg = (row.white_elo + row.black_elo) / 2.0
    if avg < 1200:
        return "<1200"
    if avg < 1600:
        return "1200-1599"
    if avg < 2000:
        return "1600-1999"
    if avg < 2400:
        return "2000-2399"
    return "2400+"


def high_rating_band(row: PositionRow) -> str:
    """Fine bands at the *top* of the rating range (task 0016).

    Maia-2's highest skill embedding is a single coarse ``">2000"`` bin — it cannot
    tell 2200 from 2800, and :func:`rating_band` likewise lumps everyone above 2400
    together. This slicer keeps low play as one ``"<2000"`` bucket and instead spends
    its resolution where the streaming wedge needs it (titled / super-GM play), so
    the 0009 calibration report quantifies how mis-calibrated the equity is per
    high-rating band *before* anyone trains a finer model.
    """
    avg = (row.white_elo + row.black_elo) / 2.0
    if avg < 2000:
        return "<2000"
    if avg < 2200:
        return "2000-2199"
    if avg < 2400:
        return "2200-2399"
    if avg < 2600:
        return "2400-2599"
    return "2600+"


def clock_band(row: PositionRow) -> str:
    """Coarse band on the side-to-move's remaining clock — where the clock model bites.

    Mirrors :func:`chess_equity.clock.time_pressure`'s scale: the gain from clock
    awareness concentrates in "scramble"/"low", and is ~nil once minutes remain.
    """
    clk = row.clock_remaining
    if clk is None:
        return "no-clock"
    if clk < 15.0:
        return "scramble(<15s)"
    if clk < 60.0:
        return "low(<60s)"
    return "comfortable(60s+)"


# The slicings reported alongside the overall number.
SLICERS: Dict[str, Callable[[PositionRow], str]] = {
    "rating": rating_band,
    "high_rating": high_rating_band,
    "phase": lambda row: row.phase,
    "clock": clock_band,
}


@dataclass(frozen=True)
class Scores:
    """The three scores for one predictor over one set of rows."""

    n: int
    log_loss: float
    brier: float
    ece: float


def _score(preds: Sequence[float], labels: Sequence[float]) -> Scores:
    return Scores(
        n=len(preds),
        log_loss=log_loss(preds, labels),
        brier=brier_score(preds, labels),
        ece=expected_calibration_error(preds, labels),
    )


@dataclass(frozen=True)
class PredictorReport:
    """A predictor's overall scores plus per-slice breakdowns."""

    name: str
    overall: Scores
    slices: Dict[str, Dict[str, Scores]]  # slicer name -> slice value -> scores


def evaluate(
    rows: Sequence[PositionRow],
    predictors: Dict[str, Predictor],
    *,
    slicers: Dict[str, Callable[[PositionRow], str]] = SLICERS,
) -> List[PredictorReport]:
    """Score each predictor over ``rows``, overall and per slice.

    Pure computation — no I/O. The caller loads rows (e.g. via
    :func:`chess_equity.data.load_rows`) and renders the returned reports.
    """
    rows = list(rows)
    labels = [r.result for r in rows]
    reports: List[PredictorReport] = []
    for name, predictor in predictors.items():
        preds = [predictor(r) for r in rows]
        slices: Dict[str, Dict[str, Scores]] = {}
        for slicer_name, slicer in slicers.items():
            grouped: Dict[str, List[int]] = {}
            for i, row in enumerate(rows):
                grouped.setdefault(slicer(row), []).append(i)
            slices[slicer_name] = {
                value: _score([preds[i] for i in idxs], [labels[i] for i in idxs])
                for value, idxs in sorted(grouped.items())
            }
        reports.append(PredictorReport(name=name, overall=_score(preds, labels), slices=slices))
    return reports


# The threshold above which we care about per-band resolution (Maia-2's coarse bin).
HIGH_RATING_MIN = 2000.0


def high_rating_calibration(
    rows: Sequence[PositionRow],
    predictors: Dict[str, Predictor] = PREDICTORS,
    *,
    min_avg_rating: float = HIGH_RATING_MIN,
) -> List[PredictorReport]:
    """Calibration of each predictor on high-rated play only, sliced by fine top bands.

    The acceptance artifact for task 0016's "first, measure the gap" step: keep only
    rows whose average rating is at least ``min_avg_rating`` and score them sliced by
    :func:`high_rating_band`, so the report reads as "how (mis)calibrated is the
    equity at 2200 / 2400 / 2600+?" — model-agnostic, so a finer-tuned model registered
    later shows up beside today's stock predictor as the before/after comparison.

    Returns an empty list when no row clears the bar (the committed sample barely
    reaches 2000), which the caller should surface rather than treat as "all good".
    """
    high = [r for r in rows if (r.white_elo + r.black_elo) / 2.0 >= min_avg_rating]
    if not high:
        return []
    return evaluate(high, predictors, slicers={"high_rating": high_rating_band})


def _scores_row(label: str, s: Scores) -> str:
    return f"| {label} | {s.n} | {s.log_loss:.4f} | {s.brier:.4f} | {s.ece:.4f} |"


def format_report(reports: Sequence[PredictorReport], *, title: str = "Validation report") -> str:
    """Render reports as a Markdown document (lower log-loss / Brier / ECE is better)."""
    out: List[str] = [f"# {title}", ""]
    out.append("Metric = predicting White expected-score (P(win)+0.5·P(draw)) vs actual result.")
    out.append("**Lower is better** for all three (log-loss, Brier, ECE).")
    out.append("")
    out.append("## Overall")
    out.append("")
    out.append("| predictor | n | log-loss | Brier | ECE |")
    out.append("|---|--:|--:|--:|--:|")
    for r in reports:
        out.append(_scores_row(r.name, r.overall))
    for slicer_name in (reports[0].slices if reports else {}):
        out.append("")
        out.append(f"## By {slicer_name}")
        out.append("")
        out.append(f"| predictor | {slicer_name} | n | log-loss | Brier | ECE |")
        out.append("|---|---|--:|--:|--:|--:|")
        for r in reports:
            for value, s in r.slices[slicer_name].items():
                out.append(f"| {r.name} | {value} " + _scores_row("", s)[2:])
    out.append("")
    return "\n".join(out)
