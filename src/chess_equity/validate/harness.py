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

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from chess_equity.adapters import EquityModel
from chess_equity.clock import clock_adjusted_white_equity
from chess_equity.data.schema import PositionRow
from chess_equity.validate.failure_modes import failure_mode
from chess_equity.types import lichess_win_percent
from chess_equity.validate.bootstrap import (
    METRIC_TERMS,
    DeltaCI,
    EceCI,
    compare_predictions,
    ece_bootstrap_ci,
    paired_bootstrap_ci,
)
from chess_equity.validate.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_table,
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


def _load_wdl_a():
    """The fitted Approach-A model, loaded lazily and cached on first use."""
    global _WDL_A_MODEL
    if _WDL_A_MODEL is None:
        from chess_equity.wdl_regression import load_wdl_a_model

        _WDL_A_MODEL = load_wdl_a_model()
    return _WDL_A_MODEL


def wdl_a(row: PositionRow) -> float:
    """Approach A — the rating-conditioned WDL regression (task 0004).

    The natural drop-in for this gate: it reads ``(cp_eval, white_elo, black_elo, ply,
    tc_bucket)`` straight off the row, so it sits beside the rating-blind baseline with
    no harness change. The fitted artifact is loaded lazily and cached, so importing
    this module stays free of the model file (and a missing artifact only bites the
    callers that actually ask for ``wdl-a``).
    """
    return _load_wdl_a().predict_white_equity(
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


def _build_wdl_net() -> EquityModel:
    # Approach D (task 0013): the end-to-end board → rating-conditioned WDL net,
    # scored as a board predictor so the 0009 gate can ask the task's real question —
    # does predicting WDL straight from the board beat regression-on-Stockfish-eval
    # (``wdl-a``)? Loads the committed artifact; torch stays unimported until scored.
    from chess_equity.wdl_net import build_wdl_net_equity

    return build_wdl_net_equity()


BOARD_MODELS: Dict[str, Callable[[], EquityModel]] = {
    "maia2": _build_maia2,
    "maia-search": _build_maia_search,
    "wdl-net": _build_wdl_net,
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


def rating_gap_band(row: PositionRow) -> str:
    """Coarse band on the *absolute rating gap* between the two players (task 0144).

    The thesis conditions on BOTH ratings, so the rating-conditioned model's edge over
    the rating-blind centipawn baseline should be largest exactly where the two players
    are most mismatched: the cp-only Win% can't tell a 2400-vs-1500 game (the strong side
    converts) from a 1500-vs-1500 one. :func:`rating_band` slices by the *average* skill
    level; this slices the orthogonal axis — ``|white_elo - black_elo|`` — so the
    head-to-head section can show the win concentrating in the high-gap band. Bands mirror
    the other slicers' ``<lo`` / range / ``hi+`` shape.
    """
    gap = abs(row.white_elo - row.black_elo)
    if gap < 100:
        return "<100"
    if gap < 300:
        return "100-299"
    return "300+"


# The slicings reported alongside the overall number.
SLICERS: Dict[str, Callable[[PositionRow], str]] = {
    "rating": rating_band,
    "high_rating": high_rating_band,
    "phase": lambda row: row.phase,
    "clock": clock_band,
    "rating_gap": rating_gap_band,
    # The most direct read on the thesis: score ON the two named failure modes (task 0111),
    # tagged from baseline/failure_modes.json. See validate/failure_modes.py.
    "failure_mode": failure_mode,
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


# One reliability bucket: (bin_lo, mean_pred, mean_obs, count) — the binned empirical
# White win-rate vs the predicted win-rate that backs the scalar ECE (task 0118).
ReliabilityRow = Tuple[float, float, float, int]


@dataclass(frozen=True)
class PredictorReport:
    """A predictor's overall scores plus per-slice breakdowns.

    ``reliability`` is the overall reliability curve (task 0118): one
    ``(bin_lo, mean_pred, mean_obs, count)`` per non-empty prediction bin, the binned
    empirical win-rate that backs the scalar ``overall.ece`` — so the report can show
    *why* a bar reading 70% is (or isn't) honest, not just the one-number ECE. Defaults
    to empty so a hand-built report stays valid; :func:`evaluate` always fills it.
    """

    name: str
    overall: Scores
    slices: Dict[str, Dict[str, Scores]]  # slicer name -> slice value -> scores
    reliability: List[ReliabilityRow] = field(default_factory=list)


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
            PredictorReport(
                name=name,
                overall=_score(preds, labels, bins=bins),
                slices=slices,
                reliability=reliability_table(preds, labels, bins=bins),
            )
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


# The metric whose paired-bootstrap CI must clear zero for a *significant* gate PASS
# (task 0069). Log-loss is the thesis's headline metric (it drives the head-to-head
# ranking and the "where equity wins" story), so significance is gated on it.
HEADLINE_METRIC = "log_loss"


# Below this held-out n the gate refuses to call a PASS — it reads INCONCLUSIVE instead
# (task 0132). With only a handful of rows a lucky point win and a barely-non-straddling
# bootstrap CI can read green by chance, overstating the thesis; the committed 15-row
# `validation_sample.md` is far under this floor, while the real proof run is n=8000. Set
# at the n>=2000 size the synthetic PASS fixture (task 0131) is built to clear, so an
# honest PASS needs a sample with the statistical power to back it. Pass ``min_n=0`` to
# :func:`gate_verdicts` (``--min-n 0`` on the CLI) to disable the guard.
MIN_GATE_N = 2000


@dataclass(frozen=True)
class Verdict:
    """One rating-conditioned predictor's gate result vs the baseline (task 0058/0069/0132).

    ``log_loss_delta`` / ``brier_delta`` are ``model - baseline`` on the held-out
    overall scores, so a *negative* delta is an improvement (lower is better).

    ``significant`` records whether the headline-metric delta CI clears zero (task 0069),
    when paired-bootstrap ``comparisons`` were supplied to :func:`gate_verdicts`. It is
    ``None`` when no CIs were given — the point-only gate (pre-0069 behaviour). ``passed``
    then requires *both* a point win on log-loss and Brier **and** ``significant`` being
    true. ``headline_ci`` is the delta CI that drove the significance call, kept for the
    report to render inline.

    ``underpowered`` is the third, distinct state (task 0132): when the held-out
    ``held_out_n`` is below ``min_n`` the gate cannot trust *any* call, so the verdict
    reads INCONCLUSIVE rather than PASS/FAIL and ``passed`` is forced ``False`` — a
    tiny-n point win must not read green. ``held_out_n``/``min_n`` are kept so the report
    and exit-code paths can name the shortfall.

    ``baseline_log_loss`` / ``baseline_brier`` are the baseline's overall scores, kept so
    the report can express each delta as a percent reduction relative to the baseline
    (task 0133) — the one human-legible number that sells the thesis.
    """

    name: str
    log_loss_delta: float
    brier_delta: float
    passed: bool
    significant: Optional[bool] = None
    headline_metric: Optional[str] = None
    headline_ci: Optional[DeltaCI] = None
    underpowered: bool = False
    held_out_n: Optional[int] = None
    min_n: Optional[int] = None
    baseline_log_loss: Optional[float] = None
    baseline_brier: Optional[float] = None


def gate_verdicts(
    reports: Sequence[PredictorReport],
    *,
    baseline_name: str = BASELINE_NAME,
    comparisons: Optional[Sequence[BaselineComparison]] = None,
    headline_metric: str = HEADLINE_METRIC,
    min_n: int = 0,
) -> List[Verdict]:
    """The thesis gate (0009): does each non-baseline predictor beat the centipawn baseline?

    For every predictor that is not ``baseline_name``, compute the overall log-loss and
    Brier deltas against the baseline. The point requirement is strictly lower on
    **both** (a model that only wins on one metric is not an unambiguous win).

    When ``comparisons`` (the paired-bootstrap delta CIs from :func:`compare_to_baseline`)
    are supplied, the gate is *significance-aware* (task 0069): PASS additionally requires
    the headline-metric (``headline_metric``, default log-loss) delta CI to clear zero —
    a point delta whose CI straddles zero is not proof, so it reads FAIL. With no
    ``comparisons`` the gate stays point-only (the pre-0069 behaviour), so callers that
    can't afford a bootstrap degrade gracefully rather than silently passing on noise.

    When the held-out sample is smaller than ``min_n`` the gate is *underpowered* (task
    0132): every verdict reads INCONCLUSIVE and ``passed`` is forced ``False`` — a lucky
    tiny-n point win must not read green. The check takes precedence over PASS/FAIL because
    at tiny n neither direction is trustworthy. ``min_n`` defaults to ``0`` (guard off) so
    direct callers and the logic tests keep the pre-0132 behaviour; the machine-checkable
    gate entry point applies the real floor — the ``validate --gate`` CLI defaults
    ``--min-n`` to :data:`MIN_GATE_N`.

    Returns an empty list if the run has no baseline predictor — nothing to gate against.
    """
    by_name = {r.name: r for r in reports}
    baseline = by_name.get(baseline_name)
    if baseline is None:
        return []
    # The held-out n every predictor was scored on (same row set), so one number gates
    # the whole run (task 0132).
    held_out_n = baseline.overall.n
    underpowered = min_n > 0 and held_out_n < min_n
    # model name -> {metric -> DeltaCI} for the supplied comparisons (if any).
    ci_by_name: Dict[str, Dict[str, DeltaCI]] = {}
    if comparisons is not None:
        ci_by_name = {c.name: {ci.metric: ci for ci in c.cis} for c in comparisons}
    verdicts: List[Verdict] = []
    for r in reports:
        if r.name == baseline_name:
            continue
        ll = r.overall.log_loss - baseline.overall.log_loss
        br = r.overall.brier - baseline.overall.brier
        point_win = ll < 0 and br < 0
        significant: Optional[bool] = None
        headline_ci: Optional[DeltaCI] = None
        if comparisons is not None:
            headline_ci = ci_by_name.get(r.name, {}).get(headline_metric)
            # No CI for the headline metric (shouldn't happen on a normal run) reads as
            # not-significant — fail closed rather than pass on missing evidence.
            significant = bool(headline_ci is not None and headline_ci.beats_baseline)
        # Underpowered can't be a PASS — neither a tiny-n win nor a tiny-n loss is proof.
        passed = (not underpowered) and point_win and (significant is None or significant)
        verdicts.append(
            Verdict(
                r.name,
                ll,
                br,
                passed=passed,
                significant=significant,
                headline_metric=headline_metric if comparisons is not None else None,
                headline_ci=headline_ci,
                underpowered=underpowered,
                held_out_n=held_out_n,
                min_n=min_n,
                baseline_log_loss=baseline.overall.log_loss,
                baseline_brier=baseline.overall.brier,
            )
        )
    return verdicts


def _percent_reduction(delta: Optional[float], baseline: Optional[float]) -> Optional[float]:
    """Percent reduction of a (model − baseline) ``delta`` relative to ``baseline``.

    A *negative* delta is an improvement (lower log-loss/Brier is better), so the
    reduction is ``-delta / baseline * 100`` — positive when the model beats the baseline.
    Returns ``None`` if either value is missing or the baseline is non-positive (no
    meaningful percentage to report).
    """
    if delta is None or baseline is None or baseline <= 0:
        return None
    return -delta / baseline * 100.0


def format_verdict(verdicts: Sequence[Verdict], *, baseline_name: str = BASELINE_NAME) -> List[str]:
    """Render the top-line PASS/FAIL gate block as Markdown lines (task 0058/0069).

    For each *passing* predictor, the line also states the percent reduction in log-loss
    (and Brier) relative to the baseline (task 0133) — the human-legible "equity cuts
    log-loss by X%" headline that sells the thesis, alongside the absolute deltas + CI.
    """
    out: List[str] = ["## Gate verdict", ""]
    if not verdicts:
        out.append(
            f"_No `{baseline_name}` predictor in this run — cannot compute a gate verdict._"
        )
        out.append("")
        return out
    # Underpowered run (task 0132): the held-out n is below the floor, so no verdict is
    # trustworthy. Say so up front and render every line as INCONCLUSIVE.
    if verdicts[0].underpowered:
        n = verdicts[0].held_out_n
        floor = verdicts[0].min_n
        out.append(
            f"**INCONCLUSIVE — underpowered.** Held-out n={n} is below the n>={floor} "
            "floor needed to trust a gate call (task 0132); a tiny-n point win can read "
            "green by chance, so the gate refuses to call PASS or FAIL."
        )
        out.append("")
        for v in verdicts:
            out.append(
                f"- **{v.name}** vs {baseline_name}: "
                f"logloss {v.log_loss_delta:+.4f}, brier {v.brier_delta:+.4f} "
                "-> **INCONCLUSIVE**"
            )
        out.append("")
        return out
    # Whether this run carried paired-bootstrap CIs (task 0069) — the criterion line and
    # each verdict line render the significance check only when it was actually applied.
    gated = verdicts[0].headline_metric is not None
    if gated:
        metric = verdicts[0].headline_metric
        out.append(
            f"Does each rating-conditioned predictor beat the rating-blind `{baseline_name}` "
            "on held-out outcomes? **PASS** = strictly lower log-loss *and* Brier (deltas "
            f"are model − baseline; negative is better) **and** the {metric} 95% CI clears "
            "zero — a delta whose CI straddles zero is not proof."
        )
    else:
        out.append(
            f"Does each rating-conditioned predictor beat the rating-blind `{baseline_name}` "
            "on held-out outcomes? **PASS** = strictly lower log-loss *and* Brier (deltas are "
            "model − baseline; negative is better)."
        )
    out.append("")
    for v in verdicts:
        status = "PASS" if v.passed else "FAIL"
        line = (
            f"- **{v.name}** beats {baseline_name}: "
            f"logloss {v.log_loss_delta:+.4f}, brier {v.brier_delta:+.4f}"
        )
        if v.headline_ci is not None:
            ci = v.headline_ci
            sig = "CI clears zero" if v.significant else "CI straddles zero"
            line += f"; {v.headline_metric} 95% CI [{ci.lo:+.4f}, {ci.hi:+.4f}] ({sig})"
        elif v.headline_metric is not None:
            # Gated run but no CI for this predictor — surface the missing evidence.
            line += f"; {v.headline_metric} CI unavailable"
        line += f" -> **{status}**"
        if v.passed:
            # The thesis-selling headline: how much does equity cut the loss, in percent?
            ll_pct = _percent_reduction(v.log_loss_delta, v.baseline_log_loss)
            br_pct = _percent_reduction(v.brier_delta, v.baseline_brier)
            if ll_pct is not None and br_pct is not None:
                line += f" — cuts log-loss {ll_pct:.1f}% (Brier {br_pct:.1f}%) vs {baseline_name}"
        out.append(line)
    out.append("")
    return out


def _scores_row(label: str, s: Scores) -> str:
    return f"| {label} | {s.n} | {s.log_loss:.4f} | {s.brier:.4f} | {s.ece:.4f} |"


def format_reliability(reports: Sequence[PredictorReport]) -> str:
    """Render each predictor's overall reliability curve as Markdown (task 0118).

    One table per predictor: for each prediction bin, the mean predicted White
    expected-score vs the **empirical** White win-rate observed in that bin, the per-bin
    count, and their gap. This is what makes the scalar ECE honest to a reader — a bar
    reading 70% is only a real P(win)+½P(draw) if the ``mean obs`` of the 0.70 bin is
    ~0.70. A well-calibrated predictor hugs the diagonal (``gap ≈ 0``) in every bin.
    """
    out: List[str] = ["## Reliability curve (is the equity bar an honest probability?)", ""]
    out.append(
        "For each predicted-probability bin: mean predicted vs **observed** White "
        "expected-score, the bin's row count, and the gap (obs − pred). A calibrated "
        "predictor has `gap ≈ 0` in every bin; the count-weighted mean `|gap|` is the ECE."
    )
    for r in reports:
        out.append("")
        out.append(f"### {r.name}  (n={r.overall.n}, ECE={r.overall.ece:.4f})")
        out.append("")
        out.append("| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |")
        out.append("|--:|--:|--:|--:|--:|")
        for bin_lo, mean_pred, mean_obs, count in r.reliability:
            out.append(
                f"| {bin_lo:.2f} | {mean_pred:.3f} | {mean_obs:.3f} | {count} "
                f"| {mean_obs - mean_pred:+.3f} |"
            )
    out.append("")
    return "\n".join(out)


def format_report(
    reports: Sequence[PredictorReport],
    *,
    title: str = "Validation report",
    comparisons: Optional[Sequence[BaselineComparison]] = None,
    head_to_head_cis: Optional["HeadToHeadCI"] = None,
) -> str:
    """Render reports as a Markdown document (lower log-loss / Brier / ECE is better).

    The report opens with a PASS/FAIL gate verdict (task 0058) so the attended proof run
    yields an unambiguous answer instead of a table to eyeball, then the full metrics.
    When ``comparisons`` (paired-bootstrap delta CIs) are supplied, the gate verdict is
    significance-aware (task 0069): PASS requires the headline-metric CI to clear zero, and
    each verdict line shows that CI inline. Omit them for the point-only gate.

    ``head_to_head_cis`` (per-slice CIs from :func:`head_to_head_slice_cis`) feeds the
    worst-slice line a clears-zero / straddles-zero caveat (task 0161); omit it for the
    point-only worst-slice read.
    """
    out: List[str] = [f"# {title}", ""]
    out.append("Metric = predicting White expected-score (P(win)+0.5·P(draw)) vs actual result.")
    out.append("**Lower is better** for all three (log-loss, Brier, ECE).")
    out.append("")
    out.extend(format_verdict(gate_verdicts(reports, comparisons=comparisons)))
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

    if reports and any(r.reliability for r in reports):
        out.append("")
        out.append(format_reliability(reports))

    h2h = head_to_head_deltas(reports)
    if h2h is not None:
        out.append("")
        out.append(format_head_to_head(h2h, cis=head_to_head_cis))
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
    """

    slicer: str
    value: str
    n: int
    baseline_log_loss: float
    model_log_loss: float
    delta: float


@dataclass(frozen=True)
class HeadToHead:
    """Baseline-vs-best-model log-loss deltas, overall and per slice (sorted best-first)."""

    baseline: str
    model: str
    overall_delta: float
    slices: List[SliceDelta]  # every (slicer, value) pair, sorted by delta descending


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


# Per-band sample floor for a head-to-head beats/loses claim (task 0146). A single
# rating / clock / phase band with fewer than this many rows is too small to trust its
# own win or loss against the baseline: the point delta (and even a bootstrap CI that
# happens to clear zero) is dominated by sampling noise. On real data the 2000-2399 band
# at n=415 reads as a wdl-a *loss* purely from small-n — a spurious thesis regression.
# Bands below this floor are marked ``underpowered (n=…)`` and excluded from any per-band
# beats/loses claim (the win count and the worst-slice verdict). Set 0 to disable.
H2H_UNDERPOWERED_N = 1000


def _worst_slice_ci_caveat(
    worst: SliceDelta, cis: Optional["HeadToHeadCI"]
) -> str:
    """A clause stating whether the worst slice's Δ CI clears or straddles zero (task 0161).

    The worst-slice line above reads a *point* Δ that may favour the baseline, but a point
    delta on a small band can't tell "equity is broken at this level" from "too few games to
    say". This looks up that slice's paired-bootstrap CI (the per-slice CIs from task 0068,
    same head-to-head convention: Δ = baseline − model, Δ > 0 = equity wins) and returns a
    one-clause caveat: the CI **clears zero** below (``hi < 0``) → a real, significant
    regression; the CI **straddles zero** → small-n noise, not a proven regression. Returns
    ``""`` when no matching CI is available (no ``cis``, no match, or the band was too small
    for a CI), so the caller falls back to the bare point read.
    """
    if cis is None:
        return ""
    match = next(
        (
            d
            for d in cis.slices
            if d.slicer == worst.slicer and d.value == worst.value and d.lo is not None
        ),
        None,
    )
    if match is None or match.lo is None or match.hi is None:
        return ""
    conf = round(cis.confidence * 100)
    ci = f"[{match.lo:+.4f}, {match.hi:+.4f}]"
    if match.hi < 0:  # whole CI below zero in baseline−model terms: baseline really wins
        return (
            f" — but note the {conf}% CI on that Δ is {ci}, which clears zero, so the "
            f"baseline win here is real and not small-n noise"
        )
    if match.lo > 0:  # whole CI above zero: equity actually wins significantly here
        return (
            f" — though the {conf}% CI on that Δ is {ci}, which clears zero the other way: "
            f"equity in fact wins this band significantly"
        )
    return (
        f" — but the {conf}% CI on that Δ is {ci}, which straddles zero, so at n={match.n} "
        f"this is small-n noise, not a proven regression at this level"
    )


def worst_slice_verdict(
    h2h: HeadToHead,
    *,
    underpowered_n: int = H2H_UNDERPOWERED_N,
    cis: Optional["HeadToHeadCI"] = None,
) -> str:
    """A one-line read on the head-to-head's *worst* slice (task 0121).

    The head-to-head table is sorted equity-wins-first, so the single worst slice — the
    one most favouring the rating-blind baseline — is the last entry. A buyer of the
    thesis wants that surfaced directly: is there *any* rating × time-control slice where
    the rating-conditioned model actually LOSES to the baseline? This states the win/total
    slice count and names that worst slice (``baseline log-loss − model log-loss``; Δ < 0
    means the baseline is better there). Returns ``""`` when there are no comparable slices.

    Bands with fewer than ``underpowered_n`` rows are *excluded* from the win/total count
    and from the worst-slice pick (task 0146): a single small band's loss is small-n noise,
    not a real regression, so it must not read as "the baseline wins here". The count of
    excluded bands is reported. Pass ``underpowered_n=0`` to disable the floor.

    When ``cis`` (the per-slice paired-bootstrap CIs from :func:`head_to_head_slice_cis`,
    task 0068) is supplied, the named worst slice gets a CI caveat appended (task 0161):
    whether the Δ CI clears zero (a real regression) or straddles it (small-n, inconclusive)
    — so a reader can tell "equity is broken at master level" from "too few games to say".
    """
    if not h2h.slices:
        return ""
    powered = [d for d in h2h.slices if d.n >= underpowered_n]
    weak = [d for d in h2h.slices if d.n < underpowered_n]
    wins = sum(1 for d in powered if d.delta > 0)
    total = len(powered)
    if powered:
        worst = powered[-1]  # smallest Δ among adequately-powered slices (sorted desc)
        where = (
            "the baseline wins here"
            if worst.delta < 0
            else "equity still wins every slice"
        )
        caveat = _worst_slice_ci_caveat(worst, cis) if worst.delta < 0 else ""
        head = (
            f"**Worst slice:** `{worst.slicer}` `{worst.value}` (n={worst.n}) "
            f"Δ={worst.delta:+.4f} — {where}{caveat}. "
        )
    else:
        head = "**Worst slice:** no adequately-powered band to judge. "
    label = "slices" if not weak else "adequately-powered slices"
    line = f"{head}Equity wins on {wins}/{total} {label}."
    if weak:
        line += f" {len(weak)} band(s) below n={underpowered_n} excluded as underpowered."
    return line


def format_head_to_head(
    h2h: HeadToHead,
    *,
    underpowered_n: int = H2H_UNDERPOWERED_N,
    cis: Optional["HeadToHeadCI"] = None,
) -> str:
    """Render the head-to-head deltas as a compact Markdown table (equity-wins first).

    Bands below ``underpowered_n`` rows are tagged ``(underpowered)`` in the Δ table and
    excluded from the worst-slice beats/loses verdict (task 0146) — their per-band Δ is
    small-n noise, not a thesis win or loss. Pass ``underpowered_n=0`` to disable.

    When ``cis`` (the per-slice CIs from :func:`head_to_head_slice_cis`) is supplied, the
    worst-slice line gets a CI caveat — clears zero (real regression) vs straddles it
    (small-n, inconclusive) — so the headline read on the losing slice is honest (task 0161).
    """
    out: List[str] = []
    out.append(f"## Head-to-head: where equity wins ({h2h.baseline} vs {h2h.model})")
    out.append("")
    out.append(
        f"Δ log-loss = `{h2h.baseline}` − `{h2h.model}` on the same rows; "
        "**Δ > 0 means equity wins** (lower model log-loss). Sorted by Δ, biggest win first."
    )
    out.append(f"Overall Δ: {h2h.overall_delta:+.4f}")
    verdict = worst_slice_verdict(h2h, underpowered_n=underpowered_n, cis=cis)
    if verdict:
        out.append(verdict)
    out.append("")
    out.append("| slice | value | n | baseline log-loss | model log-loss | Δ |")
    out.append("|---|---|--:|--:|--:|--:|")
    for d in h2h.slices:
        weak = " (underpowered)" if underpowered_n > 0 and d.n < underpowered_n else ""
        out.append(
            f"| {d.slicer} | {d.value} | {d.n} | "
            f"{d.baseline_log_loss:.4f} | {d.model_log_loss:.4f} | {d.delta:+.4f}{weak} |"
        )
    return "\n".join(out)


# --- per-slice significance: a confidence interval on each head-to-head delta -------
#
# The head-to-head table above ranks the slices by a *point* delta — but the wedge's
# actual thesis ("equity wins in the off-2300 bands and under time pressure") lives in
# those per-slice numbers, and a bare point estimate can't tell a real band-level win
# from small-n noise (task 0068). The overall significance section (task 0060) only puts
# a CI on the *aggregate* delta. This section closes the gap: it reuses the same paired
# bootstrap (:func:`~chess_equity.validate.bootstrap.paired_bootstrap_ci`) on the per-row
# metric terms *restricted to each slice*, so every rating / clock / phase band gets its
# own 95% CI and a `equity` / `baseline` / `inconclusive` verdict. Slices below a small-n
# floor are labelled `small-n` and get no CI, so a 3-row slice can't read as significant.

# Default minimum rows for a per-slice bootstrap CI: below this a resampled CI is too
# unstable to trust, so the slice is reported as `small-n` rather than significant.
H2H_SLICE_MIN_N = 30


@dataclass(frozen=True)
class SliceDeltaCI:
    """One slice's head-to-head delta with a paired-bootstrap confidence interval (0068).

    ``delta`` keeps the head-to-head sign convention — ``baseline`` metric minus ``model``
    metric on that slice's rows, so **Δ > 0 means equity wins**. ``lo``/``hi`` are the
    confidence bounds on that delta; they are ``None`` for a below-``min_n`` slice (fewer
    than the small-n floor's rows), where no CI is computed. ``verdict`` is one of
    ``equity`` (whole CI above zero — a significant equity win), ``baseline`` (whole CI
    below zero — baseline significantly better), ``inconclusive`` (CI straddles zero),
    or ``underpowered`` (the band has fewer than the underpowered floor's rows, task 0146 —
    a CI may still be shown but the band is excluded from any per-band beats/loses claim;
    a below-``min_n`` band is also underpowered and additionally has ``lo``/``hi`` ``None``).
    """

    slicer: str
    value: str
    n: int
    delta: float
    lo: Optional[float]
    hi: Optional[float]
    verdict: str


@dataclass(frozen=True)
class HeadToHeadCI:
    """Per-slice head-to-head deltas with CIs, baseline vs the best challenger (0068)."""

    baseline: str
    model: str
    metric: str
    min_n: int
    underpowered_n: int  # bands below this read `underpowered`, never a per-band win/loss
    confidence: float
    n_resamples: int
    slices: List[SliceDeltaCI]  # every comparable (slicer, value), sorted by delta desc


def head_to_head_slice_cis(
    rows: Sequence[PositionRow],
    predictors: Dict[str, Predictor],
    *,
    baseline_name: str = "baseline",
    metric: str = "log_loss",
    slicers: Dict[str, Callable[[PositionRow], str]] = SLICERS,
    min_n: int = H2H_SLICE_MIN_N,
    underpowered_n: int = H2H_UNDERPOWERED_N,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Optional[HeadToHeadCI]:
    """Paired-bootstrap CI on the head-to-head ``metric`` delta *within each slice* (0068).

    Picks ``baseline_name`` as the rating-blind reference and the non-baseline predictor
    with the lowest *overall* ``metric`` as its challenger — the same "best rating-
    conditioned model" :func:`head_to_head_deltas` ranks against — then, for every slice
    of every slicer, bootstraps a ``confidence`` CI on ``baseline − model`` using the
    per-row metric terms restricted to that slice (Δ > 0 = equity wins, matching the
    head-to-head table). Slices with fewer than ``min_n`` rows are reported as ``small-n``
    with no CI, so a tiny slice can't read as significant. Returns the slices sorted by Δ
    descending, or ``None`` when there is no baseline or no challenger to compare against.

    Pure computation; ``seed`` makes the CIs byte-reproducible (each slice gets its own
    seed offset so the resamples are independent across slices). Raises ``KeyError`` if
    ``metric`` is not a per-row term metric (only ``log_loss`` / ``brier``).
    """
    if baseline_name not in predictors:
        return None
    challengers = [n for n in predictors if n != baseline_name]
    if not challengers:
        return None
    rows = list(rows)
    labels = [r.result for r in rows]
    term_fn = METRIC_TERMS[metric]

    base_preds = [predictors[baseline_name](r) for r in rows]
    preds_by_name = {n: [predictors[n](r) for r in rows] for n in challengers}
    # The challenger with the lowest overall metric (mean of per-row terms); min() keeps
    # the first on a tie, matching head_to_head_deltas' predictor-order tie-break.
    def _overall(name: str) -> float:
        t = term_fn(preds_by_name[name], labels)
        return sum(t) / len(t)

    best = min(challengers, key=_overall)
    base_terms = term_fn(base_preds, labels)
    model_terms = term_fn(preds_by_name[best], labels)

    slice_cis: List[SliceDeltaCI] = []
    offset = 0
    for slicer_name, slicer in slicers.items():
        grouped: Dict[str, List[int]] = {}
        for i, row in enumerate(rows):
            grouped.setdefault(slicer(row), []).append(i)
        for value, idxs in sorted(grouped.items()):
            n = len(idxs)
            # baseline − model, in the head-to-head sign convention (Δ > 0 = equity wins).
            point = sum(base_terms[i] - model_terms[i] for i in idxs) / n
            if n < min_n:
                # Below the small-n floor a resampled CI is untrustworthy; report the point
                # delta but force no CI so a tiny slice can't read significant. Such a band
                # is also below the underpowered floor, so it reads `underpowered` (task
                # 0146); only with the floor disabled does it fall back to `inconclusive`.
                verdict = (
                    "underpowered"
                    if underpowered_n > 0 and n < underpowered_n
                    else "inconclusive"
                )
                slice_cis.append(
                    SliceDeltaCI(slicer_name, value, n, point, None, None, verdict)
                )
                offset += 1
                continue
            ci = paired_bootstrap_ci(
                [model_terms[i] for i in idxs],
                [base_terms[i] for i in idxs],
                metric,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed + offset,
            )
            offset += 1
            # paired_bootstrap_ci's delta is model − baseline (negative = model wins);
            # flip it (and swap the bounds) into the head-to-head's baseline − model.
            lo, hi = -ci.hi, -ci.lo
            if ci.beats_baseline:
                verdict = "equity"
            elif ci.worse_than_baseline:
                verdict = "baseline"
            else:
                verdict = "inconclusive"
            # A band below the underpowered floor can never read as a per-band win or loss,
            # however its CI lands — too few rows to trust (task 0146). Keep the CI bounds
            # (informative) but override the verdict to `underpowered`.
            if underpowered_n > 0 and n < underpowered_n:
                verdict = "underpowered"
            slice_cis.append(SliceDeltaCI(slicer_name, value, n, point, lo, hi, verdict))

    slice_cis.sort(key=lambda d: d.delta, reverse=True)
    return HeadToHeadCI(
        baseline=baseline_name,
        model=best,
        metric=metric,
        min_n=min_n,
        underpowered_n=underpowered_n,
        confidence=confidence,
        n_resamples=n_resamples,
        slices=slice_cis,
    )


def format_head_to_head_cis(h2h: HeadToHeadCI) -> str:
    """Render the per-slice head-to-head CIs as a Markdown table (equity-wins first, 0068)."""
    conf_pct = round(h2h.confidence * 100)
    metric_label = h2h.metric.replace("_", "-")
    out: List[str] = []
    out.append(
        f"## Head-to-head significance: per-slice CIs ({h2h.baseline} vs {h2h.model})"
    )
    out.append("")
    floor_note = (
        f" A band with fewer than n={h2h.underpowered_n} rows reads `underpowered` and is "
        "excluded from any per-band beats/loses claim — its own win or loss is small-n "
        "noise, not the thesis (e.g. a 2000-2399 band at n=415 can flip on a handful of "
        "games)."
        if h2h.underpowered_n > 0
        else ""
    )
    out.append(
        f"Paired bootstrap ({h2h.n_resamples} resamples) on the per-row {metric_label} "
        f"delta *within each slice*. Δ = `{h2h.baseline}` − `{h2h.model}` "
        f"(**Δ > 0 = equity wins**); `equity` means the whole {conf_pct}% CI clears zero, "
        f"so the band-level win is real and not small-n noise. Slices below n={h2h.min_n} "
        f"read `small-n` (too few rows for a trustworthy CI).{floor_note} "
        "Sorted by Δ, biggest win first."
    )
    out.append("")
    out.append(f"| slice | value | n | Δ {metric_label} | {conf_pct}% CI | verdict |")
    out.append("|---|---|--:|--:|:--:|:--:|")
    for d in h2h.slices:
        ci_str = f"n<{h2h.min_n}" if d.lo is None else f"[{d.lo:+.4f}, {d.hi:+.4f}]"
        verdict = (
            f"underpowered (n={d.n})" if d.verdict == "underpowered" else d.verdict
        )
        out.append(
            f"| {d.slicer} | {d.value} | {d.n} | {d.delta:+.4f} | {ci_str} | {verdict} |"
        )
    return "\n".join(out)


# --- per-time-control-bucket gate: does equity beat the baseline within each TC class? --
#
# The thesis' north star is the streaming / time-pressure wedge, but a clock-bearing dump
# is held (task 0153). This is the torch-free step toward it the *current* real dump already
# supports: slice the gate by the position's time-control class (bullet / blitz / rapid /
# classical), present on every row as ``row.tc_bucket`` (data/schema.py) — derived from the
# game's TimeControl header, so it is set even on a clock-blind dump where the per-move
# ``clock`` slicer is a no-op (e.g. 2013-01). For each bucket it asks the gate's core
# question — does the best rating-conditioned model beat the rating-blind centipawn baseline
# here? — on BOTH headline metrics (log-loss and Brier). Buckets below the head-to-head
# underpowered floor (:data:`H2H_UNDERPOWERED_N`, task 0146) are flagged, never silently
# passed: a small bucket's per-band win or loss is sampling noise, not the thesis.


@dataclass(frozen=True)
class TcBucketDelta:
    """One time-control bucket's head-to-head gate result (task 0155).

    ``log_loss_delta`` / ``brier_delta`` are ``model − baseline`` on that bucket's rows, so
    a *negative* delta means equity wins (lower loss). ``verdict`` is ``beats`` (both deltas
    negative — an unambiguous bucket-level win), ``worse`` (both positive), ``mixed`` (the
    two metrics disagree), or ``underpowered`` (fewer than the floor's rows — too few to
    trust either way, so excluded from any beats/loses claim).
    """

    bucket: str
    n: int
    log_loss_delta: float
    brier_delta: float
    underpowered: bool
    verdict: str


@dataclass(frozen=True)
class TcBucketGate:
    """Per-``tc_bucket`` gate, baseline vs the best rating-conditioned challenger (0155)."""

    baseline: str
    model: str
    underpowered_n: int
    buckets: List[TcBucketDelta]  # sorted by log-loss delta ascending (biggest win first)


def tc_bucket_gate(
    rows: Sequence[PositionRow],
    predictors: Dict[str, Predictor],
    *,
    baseline_name: str = "baseline",
    underpowered_n: int = H2H_UNDERPOWERED_N,
) -> Optional[TcBucketGate]:
    """Slice the thesis gate by time-control bucket (task 0155).

    Picks ``baseline_name`` as the rating-blind reference and the non-baseline predictor
    with the lowest *overall* log-loss as its challenger (the same "best rating-conditioned
    model" :func:`head_to_head_deltas` ranks against), groups the rows by ``row.tc_bucket``,
    and per bucket computes the challenger's log-loss and Brier deltas vs the baseline.
    Buckets with fewer than ``underpowered_n`` rows are flagged ``underpowered`` (the task
    0146 floor) so a small bucket's win or loss is not read as the thesis. Returns the
    buckets sorted by log-loss delta (biggest equity win first), or ``None`` when there is
    no baseline or no challenger. Pure computation; no torch needed when the challenger is a
    row predictor (e.g. ``wdl-a``). Pass ``underpowered_n=0`` to disable the floor.
    """
    if baseline_name not in predictors:
        return None
    challengers = [n for n in predictors if n != baseline_name]
    if not challengers:
        return None
    rows = list(rows)
    labels = [r.result for r in rows]
    preds_by_name = {n: [predictors[n](r) for r in rows] for n in predictors}
    base = preds_by_name[baseline_name]
    best = min(challengers, key=lambda n: log_loss(preds_by_name[n], labels))
    model = preds_by_name[best]

    grouped: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        grouped.setdefault(row.tc_bucket, []).append(i)

    deltas: List[TcBucketDelta] = []
    for bucket, idxs in grouped.items():
        n = len(idxs)
        bl = [labels[i] for i in idxs]
        ll_delta = log_loss([model[i] for i in idxs], bl) - log_loss(
            [base[i] for i in idxs], bl
        )
        br_delta = brier_score([model[i] for i in idxs], bl) - brier_score(
            [base[i] for i in idxs], bl
        )
        underpowered = underpowered_n > 0 and n < underpowered_n
        if underpowered:
            verdict = "underpowered"
        elif ll_delta < 0 and br_delta < 0:
            verdict = "beats"
        elif ll_delta > 0 and br_delta > 0:
            verdict = "worse"
        else:
            verdict = "mixed"
        deltas.append(
            TcBucketDelta(bucket, n, ll_delta, br_delta, underpowered, verdict)
        )
    deltas.sort(key=lambda d: d.log_loss_delta)
    return TcBucketGate(baseline_name, best, underpowered_n, deltas)


def format_tc_bucket_gate(gate: TcBucketGate) -> str:
    """Render the per-time-control-bucket gate as a Markdown section (task 0155)."""
    out: List[str] = []
    out.append(
        f"## By time-control bucket: does equity still beat centipawns? "
        f"({gate.baseline} vs {gate.model})"
    )
    out.append("")
    out.append(
        f"Δ = `{gate.model}` − `{gate.baseline}` on each bucket's rows; **Δ < 0 means equity "
        "wins** (lower loss). `beats` = both log-loss and Brier deltas are negative; `worse` "
        "= both positive; `mixed` = the two metrics disagree. A bucket with fewer than "
        f"n={gate.underpowered_n} rows reads `underpowered` and is excluded from any "
        "beats/loses claim — its win or loss is small-n noise, not the thesis. Sorted by "
        "Δ log-loss, biggest equity win first."
    )
    powered = [d for d in gate.buckets if not d.underpowered]
    weak = [d for d in gate.buckets if d.underpowered]
    wins = sum(1 for d in powered if d.verdict == "beats")
    summary = (
        f"Equity beats the baseline on {wins}/{len(powered)} adequately-powered "
        "time-control bucket(s)."
    )
    if weak:
        summary += (
            f" {len(weak)} bucket(s) below n={gate.underpowered_n} excluded as underpowered."
        )
    out.append(summary)
    out.append("")
    out.append("| time control | n | Δ log-loss | Δ Brier | verdict |")
    out.append("|---|--:|--:|--:|:--:|")
    for d in gate.buckets:
        verdict = f"underpowered (n={d.n})" if d.underpowered else d.verdict
        out.append(
            f"| {d.bucket} | {d.n} | {d.log_loss_delta:+.4f} | {d.brier_delta:+.4f} "
            f"| {verdict} |"
        )
    return "\n".join(out)
