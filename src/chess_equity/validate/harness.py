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

Models that need the full board (Maia, 0005) are scored via :func:`model_predictor`,
which reads each row's ``fen`` — present only when the dataset was built with
``include_fen`` (see :func:`chess_equity.data.build.build_dataset`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from chess_equity.adapters import EquityModel
from chess_equity.clock import clock_adjusted_white_equity
from chess_equity.data.schema import PositionRow
from chess_equity.types import lichess_win_percent
from chess_equity.validate.bootstrap import (
    DeltaCI,
    EceCI,
    compare_predictions,
    ece_bootstrap_ci,
)
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


def model_predictor(model: EquityModel) -> Predictor:
    """Adapt a board-based :class:`EquityModel` into a row :data:`Predictor`.

    This is what task 0029 unblocks: a Maia-style model that needs the full position
    (not just ``cp_eval``) can now be scored in 0009, because the row carries its
    ``fen``. The model's White-POV equity (0–100) maps straight to the predicted White
    expected-score in [0, 1]. Raises ``ValueError`` on a row with no FEN — i.e. a
    dataset built without ``include_fen`` — so the gap surfaces loudly instead of
    silently scoring garbage.
    """

    def predict(row: PositionRow) -> float:
        if row.fen is None:
            raise ValueError(
                "model_predictor needs row.fen; rebuild the dataset with include_fen=True"
            )
        return model.evaluate(row.fen, row.white_elo, row.black_elo).equity_white / 100.0

    return predict


def band_for_avg(avg: float) -> str:
    """Coarse rating band for an average rating — the single source of band thresholds."""
    if avg < 1200:
        return "<1200"
    if avg < 1600:
        return "1200-1599"
    if avg < 2000:
        return "1600-1999"
    if avg < 2400:
        return "2000-2399"
    return "2400+"


# Board-based models that condition on the full position, keyed by the name the CLI
# accepts in ``--models``. Each value is a zero-arg factory so construction stays lazy
# (the Maia-2 factory does NOT load torch until a row is actually scored). These are
# scored via :func:`model_predictor`, so they require a ``--with-fen`` dataset (0029).
def _build_maia2() -> EquityModel:
    from chess_equity.maia2 import build_maia2_equity

    return build_maia2_equity()


def _build_maia_search() -> EquityModel:
    # The Maia-weighted expectimax (task 0006), scored as a board predictor so the
    # 0009 gate can ask: does explicit look-ahead beat Maia-2's implicit value head
    # (``maia2``)? Defaults (depth=2, k=4); the comparison run needs Maia weights.
    from chess_equity.search import build_maia_search

    return build_maia_search()


BOARD_MODELS: Dict[str, Callable[[], EquityModel]] = {
    "maia2": _build_maia2,
    "maia-search": _build_maia_search,
}


def build_predictors(names: Sequence[str]) -> Dict[str, Predictor]:
    """Resolve ``--models`` names to predictors, mixing row and board models.

    A name is either a row predictor in :data:`PREDICTORS` (reads ``cp_eval`` etc.) or
    a board model in :data:`BOARD_MODELS` (built once, wrapped with
    :func:`model_predictor` so it reads ``row.fen``). This is the seam task 0031 wires:
    ``--models baseline,maia2`` now scores Maia-2's rating-conditioned ``win_prob``
    beside the centipawn baseline. Raises ``KeyError`` listing any unknown name.
    """
    unknown = [n for n in names if n not in PREDICTORS and n not in BOARD_MODELS]
    if unknown:
        available = sorted(set(PREDICTORS) | set(BOARD_MODELS))
        raise KeyError(f"unknown model(s) {unknown}; available: {available}")
    predictors: Dict[str, Predictor] = {}
    for name in names:
        if name in PREDICTORS:
            predictors[name] = PREDICTORS[name]
        else:
            predictors[name] = model_predictor(BOARD_MODELS[name]())
    return predictors


def rating_band(row: PositionRow) -> str:
    """Coarse band on the average of the two ratings (the relevant joint skill level)."""
    return band_for_avg((row.white_elo + row.black_elo) / 2.0)


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


def _score(preds: Sequence[float], labels: Sequence[float], *, bins: int = 10) -> Scores:
    return Scores(
        n=len(preds),
        log_loss=log_loss(preds, labels),
        brier=brier_score(preds, labels),
        ece=expected_calibration_error(preds, labels, bins=bins),
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
    bins: int = 10,
) -> List[PredictorReport]:
    """Score each predictor over ``rows``, overall and per slice.

    Pure computation — no I/O. The caller loads rows (e.g. via
    :func:`chess_equity.data.load_rows`) and renders the returned reports. ``bins`` is
    the reliability-bin count for ECE (default 10, the metrics default — unchanged when
    omitted); the validate CLI exposes it as ``--ece-bins`` for sensitivity checks.
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
                value: _score([preds[i] for i in idxs], [labels[i] for i in idxs], bins=bins)
                for value, idxs in sorted(grouped.items())
            }
        reports.append(
            PredictorReport(name=name, overall=_score(preds, labels, bins=bins), slices=slices)
        )
    return reports


@dataclass(frozen=True)
class BaselineComparison:
    """One model's paired-bootstrap metric deltas vs the baseline (task 0060)."""

    name: str  # the model predictor's name
    baseline: str  # the baseline predictor's name
    cis: List[DeltaCI]  # one per bootstrapped metric (log-loss, Brier)


def compare_to_baseline(
    rows: Sequence[PositionRow],
    predictors: Dict[str, Predictor],
    *,
    baseline: str = "baseline",
    metrics: Sequence[str] = ("log_loss", "brier"),
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> List[BaselineComparison]:
    """Paired-bootstrap CIs on each non-baseline predictor's metric delta vs ``baseline``.

    Turns the side-by-side scores into a *significance* statement: for every other
    predictor, resample the held-out rows and put a ``confidence`` CI on its log-loss /
    Brier delta against the baseline (negative = the model wins). Returns an empty list
    when ``baseline`` is the only predictor — there is nothing to compare it to.

    Pure computation; ``seed`` makes the CIs byte-reproducible. Raises ``KeyError`` if
    ``baseline`` is not among ``predictors``.
    """
    if baseline not in predictors:
        raise KeyError(f"baseline {baseline!r} not in predictors {sorted(predictors)}")
    rows = list(rows)
    labels = [r.result for r in rows]
    base_preds = [predictors[baseline](r) for r in rows]
    comparisons: List[BaselineComparison] = []
    for name, predictor in predictors.items():
        if name == baseline:
            continue
        model_preds = [predictor(r) for r in rows]
        cis = compare_predictions(
            model_preds,
            base_preds,
            labels,
            metrics=metrics,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
        comparisons.append(BaselineComparison(name=name, baseline=baseline, cis=cis))
    return comparisons


def _verdict(ci: DeltaCI) -> str:
    """A one-word read on a delta CI: does the model significantly beat the baseline?"""
    if ci.beats_baseline:
        return "beats"
    if ci.worse_than_baseline:
        return "worse"
    return "inconclusive"


def format_baseline_comparison(
    comparisons: Sequence[BaselineComparison], *, title: str = "Significance vs baseline"
) -> str:
    """Render the paired-bootstrap deltas + CIs as a Markdown section (task 0060).

    One row per (model, metric): the delta (model - baseline; negative = better), the
    confidence interval, and a verdict. ``beats`` only when the whole CI clears zero, so
    a real win is distinguished from noise at a glance.
    """
    if not comparisons:
        return ""
    conf_pct = round(comparisons[0].cis[0].confidence * 100) if comparisons[0].cis else 95
    out: List[str] = [f"## {title}", ""]
    out.append(
        f"Paired bootstrap ({comparisons[0].cis[0].n_resamples if comparisons[0].cis else 0} "
        f"resamples) on the per-row metric delta vs `{comparisons[0].baseline}`. "
        "**Negative delta = the model is better** (lower loss); a verdict of `beats` "
        f"means the whole {conf_pct}% CI sits below zero."
    )
    out.append("")
    out.append(f"| model | metric | delta | {conf_pct}% CI | verdict |")
    out.append("|---|---|--:|:--:|:--:|")
    for c in comparisons:
        for ci in c.cis:
            out.append(
                f"| {c.name} | {ci.metric} | {ci.delta:+.4f} | "
                f"[{ci.lo:+.4f}, {ci.hi:+.4f}] | {_verdict(ci)} |"
            )
    out.append("")
    return "\n".join(out)


def compare_ece_to_baseline(
    rows: Sequence[PositionRow],
    predictors: Dict[str, Predictor],
    *,
    baseline: str = "baseline",
    bins: int = 10,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> List[EceCI]:
    """Bin-resampling bootstrap CI on each predictor's ECE, with the ECE delta vs baseline.

    Companion to :func:`compare_to_baseline` for calibration (task 0072): 0060's paired
    row-bootstrap can't touch ECE (no per-row term), so here every predictor's ECE gets
    its own bin-resampling CI, and each non-baseline predictor also gets a paired CI on
    its ECE delta vs ``baseline`` (negative = better calibrated). Returns one
    :class:`~chess_equity.validate.bootstrap.EceCI` per predictor in registry order; the
    baseline's own entry carries no delta. ``seed`` makes the CIs byte-reproducible.

    Raises ``KeyError`` if ``baseline`` is not among ``predictors``.
    """
    if baseline not in predictors:
        raise KeyError(f"baseline {baseline!r} not in predictors {sorted(predictors)}")
    rows = list(rows)
    labels = [r.result for r in rows]
    base_preds = [predictors[baseline](r) for r in rows]
    out: List[EceCI] = []
    for i, (name, predictor) in enumerate(predictors.items()):
        preds = [predictor(r) for r in rows]
        out.append(
            ece_bootstrap_ci(
                preds,
                labels,
                predictor=name,
                baseline_preds=None if name == baseline else base_preds,
                bins=bins,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed + i,
            )
        )
    return out


def _ece_verdict(ci: EceCI) -> str:
    """One-word read on an ECE delta CI: better/worse calibrated than the baseline?"""
    if ci.delta is None:
        return "—"
    if ci.beats_baseline:
        return "beats"
    if ci.worse_than_baseline:
        return "worse"
    return "inconclusive"


def format_ece_comparison(
    ece_cis: Sequence[EceCI], *, title: str = "Calibration (ECE) confidence intervals"
) -> str:
    """Render per-predictor ECE CIs + the ECE delta vs baseline as Markdown (task 0072).

    One row per predictor: its ECE point estimate and CI, then (for non-baseline
    predictors) the ECE delta vs baseline, that delta's CI, and a verdict. ``beats`` only
    when the whole delta CI clears zero below, so a real calibration win is distinguished
    from noise at a glance. Lower ECE = better calibrated.
    """
    if not ece_cis:
        return ""
    conf_pct = round(ece_cis[0].confidence * 100)
    out: List[str] = [f"## {title}", ""]
    out.append(
        f"Bin-resampling bootstrap ({ece_cis[0].n_resamples} resamples) on ECE "
        "(**lower = better calibrated**); ECE has no per-row term, so rows are resampled "
        "and the binning recomputed each draw. A `beats` verdict means the whole "
        f"ECE-delta {conf_pct}% CI vs baseline sits below zero."
    )
    out.append("")
    out.append(
        f"| predictor | ECE | {conf_pct}% CI | Δ vs baseline | Δ {conf_pct}% CI | verdict |"
    )
    out.append("|---|--:|:--:|--:|:--:|:--:|")
    for c in ece_cis:
        ci_str = f"[{c.lo:.4f}, {c.hi:.4f}]"
        if c.delta is None:
            out.append(f"| {c.predictor} | {c.ece:.4f} | {ci_str} | — | — | — |")
        else:
            out.append(
                f"| {c.predictor} | {c.ece:.4f} | {ci_str} | {c.delta:+.4f} | "
                f"[{c.delta_lo:+.4f}, {c.delta_hi:+.4f}] | {_ece_verdict(c)} |"
            )
    out.append("")
    return "\n".join(out)


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


# The rating-blind centipawn predictor every rating-conditioned model must beat (0009).
BASELINE_NAME = "baseline"


@dataclass(frozen=True)
class Verdict:
    """One rating-conditioned predictor's gate result vs the baseline (task 0058).

    ``log_loss_delta`` / ``brier_delta`` are ``model - baseline`` on the held-out
    overall scores, so a *negative* delta is an improvement (lower is better).
    """

    name: str
    log_loss_delta: float
    brier_delta: float
    passed: bool


def gate_verdicts(
    reports: Sequence[PredictorReport], *, baseline_name: str = BASELINE_NAME
) -> List[Verdict]:
    """The thesis gate (0009): does each non-baseline predictor beat the centipawn baseline?

    For every predictor that is not ``baseline_name``, compute the overall log-loss and
    Brier deltas against the baseline. PASS iff the predictor is strictly lower on
    **both** (a model that only wins on one metric is not an unambiguous win). Returns
    an empty list if the run has no baseline predictor — there is nothing to gate against.
    """
    by_name = {r.name: r for r in reports}
    baseline = by_name.get(baseline_name)
    if baseline is None:
        return []
    verdicts: List[Verdict] = []
    for r in reports:
        if r.name == baseline_name:
            continue
        ll = r.overall.log_loss - baseline.overall.log_loss
        br = r.overall.brier - baseline.overall.brier
        verdicts.append(Verdict(r.name, ll, br, passed=ll < 0 and br < 0))
    return verdicts


def format_verdict(verdicts: Sequence[Verdict], *, baseline_name: str = BASELINE_NAME) -> List[str]:
    """Render the top-line PASS/FAIL gate block as Markdown lines (task 0058)."""
    out: List[str] = ["## Gate verdict", ""]
    if not verdicts:
        out.append(
            f"_No `{baseline_name}` predictor in this run — cannot compute a gate verdict._"
        )
        out.append("")
        return out
    out.append(
        f"Does each rating-conditioned predictor beat the rating-blind `{baseline_name}` "
        "on held-out outcomes? **PASS** = strictly lower log-loss *and* Brier (deltas are "
        "model − baseline; negative is better)."
    )
    out.append("")
    for v in verdicts:
        status = "PASS" if v.passed else "FAIL"
        out.append(
            f"- **{v.name}** beats {baseline_name}: "
            f"logloss {v.log_loss_delta:+.4f}, brier {v.brier_delta:+.4f} -> **{status}**"
        )
    out.append("")
    return out


def _scores_row(label: str, s: Scores) -> str:
    return f"| {label} | {s.n} | {s.log_loss:.4f} | {s.brier:.4f} | {s.ece:.4f} |"


def format_report(
    reports: Sequence[PredictorReport],
    *,
    title: str = "Validation report",
    head_to_head: Optional["HeadToHead"] = None,
) -> str:
    """Render reports as a Markdown document (lower log-loss / Brier / ECE is better).

    The report opens with a PASS/FAIL gate verdict (task 0058) so the attended proof run
    yields an unambiguous answer instead of a table to eyeball, then the full metrics.

    ``head_to_head`` lets the caller pass a pre-built head-to-head — e.g. the CI-enriched
    one from :func:`head_to_head_with_cis` (task 0068), which needs the rows the report
    object doesn't carry. When omitted, the point-estimate head-to-head is derived from
    ``reports`` as before.
    """
    out: List[str] = [f"# {title}", ""]
    out.append("Metric = predicting White expected-score (P(win)+0.5·P(draw)) vs actual result.")
    out.append("**Lower is better** for all three (log-loss, Brier, ECE).")
    out.append("")
    out.extend(format_verdict(gate_verdicts(reports)))
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

    h2h = head_to_head if head_to_head is not None else head_to_head_deltas(reports)
    if h2h is not None:
        out.append("")
        out.append(format_head_to_head(h2h))
    out.append("")
    return "\n".join(out)


# --- head-to-head: where does the rating-conditioned model beat the baseline? ----
#
# The thesis (see roadmap / product-wedge-streaming) is not "equity beats centipawns
# everywhere" — it's that equity wins *most* exactly where the rating-blind bar is most
# wrong: low/high rating bands and under time pressure. The per-predictor tables above
# show each model's scores by slice, but the reader has to subtract them by eye. This
# section does that subtraction once: for every slice, baseline log-loss minus the best
# rating-conditioned model's log-loss, ranked so the slices where equity wins most sit
# at the top. It reuses the already-computed per-slice scores (no re-evaluation, no new
# deps) — the same slicings the calibration/holdout path emits.


@dataclass(frozen=True)
class SliceDelta:
    """One slice's head-to-head gap: baseline log-loss minus the model's, on the same rows.

    ``delta > 0`` means the rating-conditioned model has the *lower* log-loss there — i.e.
    equity wins in that slice. ``delta < 0`` means the rating-blind baseline is better.

    When the head-to-head was built with per-slice bootstrap (task 0068), ``ci_lo``/``ci_hi``
    are the ``confidence``-level bounds on ``delta`` *in this same orientation* (baseline −
    model, so > 0 = equity wins), and ``verdict`` reads ``"beats"`` (whole CI clears zero
    above), ``"worse"`` (whole CI below zero), or ``"inconclusive"`` (CI straddles zero, or
    the slice fell below the small-n floor). They stay ``None`` for the point-estimate-only
    path (:func:`head_to_head_deltas`).
    """

    slicer: str
    value: str
    n: int
    baseline_log_loss: float
    model_log_loss: float
    delta: float
    ci_lo: Optional[float] = None
    ci_hi: Optional[float] = None
    verdict: Optional[str] = None


@dataclass(frozen=True)
class HeadToHead:
    """Baseline-vs-best-model log-loss deltas, overall and per slice (sorted best-first).

    ``overall_ci_lo``/``overall_ci_hi``/``overall_verdict`` mirror :class:`SliceDelta`'s
    CI fields for the all-rows delta; they are filled only on the bootstrap path
    (:func:`head_to_head_with_cis`) and stay ``None`` otherwise.
    """

    baseline: str
    model: str
    overall_delta: float
    slices: List[SliceDelta]  # every (slicer, value) pair, sorted by delta descending
    overall_ci_lo: Optional[float] = None
    overall_ci_hi: Optional[float] = None
    overall_verdict: Optional[str] = None


def head_to_head_deltas(
    reports: Sequence[PredictorReport], *, baseline_name: str = "baseline"
) -> Optional[HeadToHead]:
    """Rank slices by how much the best rating-conditioned model beats the baseline.

    Picks ``baseline_name`` as the rating-blind reference and the non-baseline predictor
    with the lowest *overall* log-loss as its challenger ("the best rating-conditioned
    model"), then for every slice computes ``baseline_log_loss - model_log_loss`` on the
    same rows. Positive = equity wins. Returns the deltas sorted descending, so the
    report directly answers "where does the thesis hold". ``None`` when there is no
    baseline or no challenger to compare against (e.g. a single-predictor run).
    """
    by_name = {r.name: r for r in reports}
    base = by_name.get(baseline_name)
    if base is None:
        return None
    challengers = [r for r in reports if r.name != baseline_name]
    if not challengers:
        return None
    best = min(challengers, key=lambda r: r.overall.log_loss)

    deltas: List[SliceDelta] = []
    for slicer_name, base_slices in base.slices.items():
        model_slices = best.slices.get(slicer_name, {})
        for value, base_scores in base_slices.items():
            model_scores = model_slices.get(value)
            if model_scores is None:
                continue
            deltas.append(
                SliceDelta(
                    slicer=slicer_name,
                    value=value,
                    n=base_scores.n,
                    baseline_log_loss=base_scores.log_loss,
                    model_log_loss=model_scores.log_loss,
                    delta=base_scores.log_loss - model_scores.log_loss,
                )
            )
    deltas.sort(key=lambda d: d.delta, reverse=True)
    return HeadToHead(
        baseline=baseline_name,
        model=best.name,
        overall_delta=base.overall.log_loss - best.overall.log_loss,
        slices=deltas,
    )


# Slices below this many rows are too small to bootstrap a meaningful CI: a handful of
# rows would let one lucky row flip the verdict. We still SHOW the slice (the point delta
# is informative), but label it inconclusive rather than letting tiny-n noise read as a
# significant win. Mirrors the spirit of the gate's "CI must clear zero" rule (task 0068).
MIN_SLICE_N = 30


def _delta_ci_verdict(ci: DeltaCI) -> str:
    """Verdict for a per-slice CI, expressed in the head-to-head (baseline − model) sign.

    :func:`paired_bootstrap_ci` uses ``delta = model − baseline`` (negative = model wins),
    so its ``beats_baseline``/``worse_than_baseline`` already answer the right question;
    we only relabel them for the equity-wins-positive head-to-head convention.
    """
    if ci.beats_baseline:
        return "beats"
    if ci.worse_than_baseline:
        return "worse"
    return "inconclusive"


def head_to_head_with_cis(
    rows: Sequence[PositionRow],
    predictors: Dict[str, Predictor],
    *,
    baseline_name: str = "baseline",
    slicers: Dict[str, Callable[[PositionRow], str]] = SLICERS,
    metric: str = "log_loss",
    min_slice_n: int = MIN_SLICE_N,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Optional[HeadToHead]:
    """Head-to-head deltas with a paired-bootstrap CI + verdict on each per-slice delta.

    The point-estimate :func:`head_to_head_deltas` answers *where* equity wins; this answers
    *whether each band-level win is real or small-n noise* — the wedge's actual thesis is
    "equity wins in the off-2300 bands and under time pressure", and a bare per-slice delta
    can't tell a genuine band win from a few lucky rows. For every slice it resamples the
    rows in that slice (paired: the same indices score both predictors, via
    :func:`~chess_equity.validate.bootstrap.paired_bootstrap_ci` over the per-row metric
    terms) and attaches a ``confidence`` CI + ``beats``/``worse``/``inconclusive`` verdict.
    The CI is stored in the head-to-head orientation (baseline − model, so ``> 0`` = equity
    wins). Slices below ``min_slice_n`` rows are shown but flagged ``inconclusive`` with no
    CI rather than risking a spurious "significant" read on a handful of rows.

    Picks the challenger the same way as :func:`head_to_head_deltas` (the non-baseline
    predictor with the lowest *overall* log-loss). Returns ``None`` when there is no
    baseline or no challenger. Each slice gets its own seed offset so the CIs are
    independent across slices yet byte-reproducible under ``seed``.
    """
    from chess_equity.validate.bootstrap import METRIC_TERMS, paired_bootstrap_ci

    rows = list(rows)
    labels = [r.result for r in rows]
    if baseline_name not in predictors:
        return None
    challengers = [name for name in predictors if name != baseline_name]
    if not challengers:
        return None

    preds = {name: [predictors[name](r) for r in rows] for name in predictors}
    base_preds = preds[baseline_name]
    # Best challenger = lowest overall log-loss, matching head_to_head_deltas.
    model_name = min(challengers, key=lambda name: log_loss(preds[name], labels))
    model_preds = preds[model_name]

    terms = METRIC_TERMS[metric]

    def _ci_for(idxs: Sequence[int], offset: int) -> tuple:
        """(baseline_loss, model_loss, ci_lo, ci_hi, verdict) for the rows at ``idxs``."""
        m = [model_preds[i] for i in idxs]
        b = [base_preds[i] for i in idxs]
        ls = [labels[i] for i in idxs]
        model_loss = log_loss(m, ls)
        base_loss = log_loss(b, ls)
        if len(idxs) < min_slice_n:
            return base_loss, model_loss, None, None, "inconclusive"
        ci = paired_bootstrap_ci(
            terms(m, ls),
            terms(b, ls),
            metric,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed + offset,
        )
        # ci.delta is model − baseline; flip to baseline − model so > 0 = equity wins.
        return base_loss, model_loss, -ci.hi, -ci.lo, _delta_ci_verdict(ci)

    all_idx = list(range(len(rows)))
    o_base, o_model, o_lo, o_hi, o_verdict = _ci_for(all_idx, 0)

    deltas: List[SliceDelta] = []
    offset = 1
    for slicer_name, slicer in slicers.items():
        grouped: Dict[str, List[int]] = {}
        for i, row in enumerate(rows):
            grouped.setdefault(slicer(row), []).append(i)
        for value, idxs in sorted(grouped.items()):
            base_loss, model_loss, lo, hi, verdict = _ci_for(idxs, offset)
            offset += 1
            deltas.append(
                SliceDelta(
                    slicer=slicer_name,
                    value=value,
                    n=len(idxs),
                    baseline_log_loss=base_loss,
                    model_log_loss=model_loss,
                    delta=base_loss - model_loss,
                    ci_lo=lo,
                    ci_hi=hi,
                    verdict=verdict,
                )
            )
    deltas.sort(key=lambda d: d.delta, reverse=True)
    return HeadToHead(
        baseline=baseline_name,
        model=model_name,
        overall_delta=o_base - o_model,
        slices=deltas,
        overall_ci_lo=o_lo,
        overall_ci_hi=o_hi,
        overall_verdict=o_verdict,
    )


def worst_slice_verdict(h2h: HeadToHead) -> str:
    """A one-line read on the head-to-head's *worst* slice (task 0121).

    The head-to-head table is sorted equity-wins-first, so the single worst slice — the
    one most favouring the rating-blind baseline — is the last entry. A buyer of the
    thesis wants that surfaced directly: is there *any* rating × time-control slice where
    the rating-conditioned model actually LOSES to the baseline? This states the win/total
    slice count and names that worst slice (``baseline log-loss − model log-loss``; Δ < 0
    means the baseline is better there). Returns ``""`` when there are no comparable slices.
    """
    if not h2h.slices:
        return ""
    wins = sum(1 for d in h2h.slices if d.delta > 0)
    total = len(h2h.slices)
    worst = h2h.slices[-1]  # smallest Δ — most baseline-favouring
    where = (
        "the baseline wins here" if worst.delta < 0 else "equity still wins every slice"
    )
    return (
        f"**Worst slice:** `{worst.slicer}` `{worst.value}` (n={worst.n}) "
        f"Δ={worst.delta:+.4f} — {where}. "
        f"Equity wins on {wins}/{total} slices."
    )


def _ci_cell(d: SliceDelta) -> str:
    """The `95% CI` cell for a slice: the bounds when bootstrapped, else an em dash."""
    if d.ci_lo is None or d.ci_hi is None:
        return "—"
    return f"[{d.ci_lo:+.4f}, {d.ci_hi:+.4f}]"


def format_head_to_head(h2h: HeadToHead) -> str:
    """Render the head-to-head deltas as a compact Markdown table (equity-wins first).

    When the head-to-head carries per-slice CIs (built by :func:`head_to_head_with_cis`,
    task 0068) the table grows ``95% CI`` and ``verdict`` columns so a reader can tell a
    real band-level win from small-n noise; otherwise it renders the point-estimate table.
    """
    has_cis = any(d.verdict is not None for d in h2h.slices)
    out: List[str] = []
    out.append(f"## Head-to-head: where equity wins ({h2h.baseline} vs {h2h.model})")
    out.append("")
    out.append(
        f"Δ log-loss = `{h2h.baseline}` − `{h2h.model}` on the same rows; "
        "**Δ > 0 means equity wins** (lower model log-loss). Sorted by Δ, biggest win first."
    )
    overall = f"Overall Δ: {h2h.overall_delta:+.4f}"
    if h2h.overall_verdict is not None:
        if h2h.overall_ci_lo is not None and h2h.overall_ci_hi is not None:
            overall += f" (95% CI [{h2h.overall_ci_lo:+.4f}, {h2h.overall_ci_hi:+.4f}])"
        overall += f" — {h2h.overall_verdict}"
    out.append(overall)
    if has_cis:
        out.append(
            "Per-slice 95% CIs are paired bootstraps on the Δ; "
            f"slices under n={MIN_SLICE_N} are shown but read **inconclusive** "
            "(too few rows to bootstrap)."
        )
    verdict = worst_slice_verdict(h2h)
    if verdict:
        out.append(verdict)
    out.append("")
    if has_cis:
        out.append("| slice | value | n | baseline log-loss | model log-loss | Δ | 95% CI | verdict |")
        out.append("|---|---|--:|--:|--:|--:|---|---|")
        for d in h2h.slices:
            out.append(
                f"| {d.slicer} | {d.value} | {d.n} | "
                f"{d.baseline_log_loss:.4f} | {d.model_log_loss:.4f} | {d.delta:+.4f} | "
                f"{_ci_cell(d)} | {d.verdict or '—'} |"
            )
    else:
        out.append("| slice | value | n | baseline log-loss | model log-loss | Δ |")
        out.append("|---|---|--:|--:|--:|--:|")
        for d in h2h.slices:
            out.append(
                f"| {d.slicer} | {d.value} | {d.n} | "
                f"{d.baseline_log_loss:.4f} | {d.model_log_loss:.4f} | {d.delta:+.4f} |"
            )
    return "\n".join(out)
