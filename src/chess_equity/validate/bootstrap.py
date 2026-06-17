"""Paired-bootstrap confidence intervals on the model-vs-baseline metric deltas (task 0060).

The 0009 gate asks "does a rating-conditioned predictor beat the rating-blind centipawn
baseline?". A bare delta of log-loss / Brier answers *direction* but not *significance* —
on a held-out test set a 0.002 win could be noise. The attended proof (and the gate
verdict) needs error bars.

We use a **paired** bootstrap: both predictors score the *same* rows, so log-loss and
Brier each decompose into per-row terms (:func:`~chess_equity.validate.metrics.log_loss_terms`
/ :func:`~chess_equity.validate.metrics.brier_terms`) and the metric delta is just the mean
of the per-row delta ``model_term[i] - baseline_term[i]``. Resampling rows with replacement
and re-averaging that delta array gives the bootstrap distribution of the delta; the
2.5 / 97.5 percentiles are the 95% CI. Pairing cancels the shared row-difficulty variance,
so the CI is far tighter (and more honest) than bootstrapping each score independently.

Convention: **delta = model - baseline**, and lower loss is better, so a *negative* delta
means the model wins. The CI "clears zero" (``hi < 0``) exactly when the win is significant.

Pure Python + ``random`` — seeded for byte-reproducible CIs, no numpy, no new deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable, List, Optional, Sequence

from chess_equity.validate.metrics import (
    brier_terms,
    expected_calibration_error,
    log_loss_terms,
)

# metric name -> per-row term function (mean of terms == the metric). These are the
# metrics whose delta we bootstrap; ECE is a binned aggregate with no per-row term, so
# it is intentionally excluded.
METRIC_TERMS: dict = {
    "log_loss": log_loss_terms,
    "brier": brier_terms,
}


@dataclass(frozen=True)
class DeltaCI:
    """A model-vs-baseline metric delta with a bootstrap confidence interval.

    ``delta`` is the point estimate (model metric - baseline metric; negative = model
    better). ``lo``/``hi`` are the ``confidence``-level percentile bounds.
    """

    metric: str
    delta: float
    lo: float
    hi: float
    n_resamples: int
    confidence: float

    @property
    def beats_baseline(self) -> bool:
        """True when the model is significantly better: the whole CI sits below zero."""
        return self.hi < 0.0

    @property
    def worse_than_baseline(self) -> bool:
        """True when the model is significantly worse: the whole CI sits above zero."""
        return self.lo > 0.0


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated ``q``-quantile (q in [0, 1]) of an already-sorted sequence."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < n:
        return sorted_vals[lo] + frac * (sorted_vals[lo + 1] - sorted_vals[lo])
    return sorted_vals[lo]


def paired_bootstrap_ci(
    model_terms: Sequence[float],
    baseline_terms: Sequence[float],
    metric: str,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> DeltaCI:
    """Bootstrap CI on ``mean(model_terms) - mean(baseline_terms)`` over paired rows.

    Both term lists must be parallel (same rows, same order). Resamples row indices with
    replacement ``n_resamples`` times — the *same* indices for both predictors, which is
    what makes it paired — and reads the percentile interval off the resampled deltas.
    """
    if len(model_terms) != len(baseline_terms):
        raise ValueError(
            f"paired terms length mismatch: {len(model_terms)} != {len(baseline_terms)}"
        )
    n = len(model_terms)
    if n == 0:
        raise ValueError("need at least one row to bootstrap")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    deltas = [m - b for m, b in zip(model_terms, baseline_terms)]
    point = sum(deltas) / n

    rng = Random(seed)
    boot_means: List[float] = []
    for _ in range(n_resamples):
        # rng.choices is C-level resampling-with-replacement; deterministic under `seed`.
        sample = rng.choices(deltas, k=n)
        boot_means.append(sum(sample) / n)
    boot_means.sort()

    alpha = (1.0 - confidence) / 2.0
    lo = _percentile(boot_means, alpha)
    hi = _percentile(boot_means, 1.0 - alpha)
    return DeltaCI(
        metric=metric,
        delta=point,
        lo=lo,
        hi=hi,
        n_resamples=n_resamples,
        confidence=confidence,
    )


@dataclass(frozen=True)
class EceCI:
    """A predictor's ECE with a bin-resampling bootstrap confidence interval (task 0072).

    ECE is a binned aggregate with no per-row term, so the paired row-bootstrap used for
    log-loss/Brier (:func:`paired_bootstrap_ci`) does not apply: resampling rows changes
    *which* rows land in each reliability bin and how the bin means move. We instead
    resample rows with replacement and **recompute the whole binning + ECE per resample**.

    ``ece`` is the point estimate (lower = better calibrated); ``lo``/``hi`` are its
    percentile bounds. When a baseline was supplied, ``delta`` (= this predictor's ECE
    minus the baseline's, on the *same* resampled rows so the comparison is paired) and
    its ``delta_lo``/``delta_hi`` bounds are also filled in; they stay ``None`` for the
    baseline itself.
    """

    predictor: str
    ece: float
    lo: float
    hi: float
    n_resamples: int
    confidence: float
    bins: int
    delta: Optional[float] = None
    delta_lo: Optional[float] = None
    delta_hi: Optional[float] = None

    @property
    def beats_baseline(self) -> bool:
        """True when this predictor is significantly better calibrated: ECE-delta CI < 0."""
        return self.delta_hi is not None and self.delta_hi < 0.0

    @property
    def worse_than_baseline(self) -> bool:
        """True when this predictor is significantly worse calibrated: ECE-delta CI > 0."""
        return self.delta_lo is not None and self.delta_lo > 0.0


def ece_bootstrap_ci(
    preds: Sequence[float],
    labels: Sequence[float],
    *,
    predictor: str = "model",
    baseline_preds: Optional[Sequence[float]] = None,
    bins: int = 10,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> EceCI:
    """Bin-resampling bootstrap CI on the ECE of ``preds`` vs ``labels``.

    Resamples row indices with replacement ``n_resamples`` times and recomputes the
    reliability binning + ECE on each resample; the percentile interval of those ECEs is
    the CI. If ``baseline_preds`` is given (parallel rows), the *same* resampled indices
    score the baseline too, yielding a paired CI on the ECE delta (this predictor minus
    baseline; negative = better calibrated). Seeded for byte-reproducible CIs.
    """
    n = len(preds)
    if n == 0:
        raise ValueError("need at least one row to bootstrap")
    if len(labels) != n:
        raise ValueError(f"preds/labels length mismatch: {n} != {len(labels)}")
    base: Optional[List[float]] = list(baseline_preds) if baseline_preds is not None else None
    if base is not None and len(base) != n:
        raise ValueError(f"baseline_preds length mismatch: {len(base)} != {n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    point = expected_calibration_error(preds, labels, bins=bins)
    point_delta = (
        point - expected_calibration_error(base, labels, bins=bins)
        if base is not None
        else None
    )

    rng = Random(seed)
    eces: List[float] = []
    deltas: List[float] = []
    for _ in range(n_resamples):
        idx = rng.choices(range(n), k=n)
        rl = [labels[i] for i in idx]
        e = expected_calibration_error([preds[i] for i in idx], rl, bins=bins)
        eces.append(e)
        if base is not None:
            be = expected_calibration_error([base[i] for i in idx], rl, bins=bins)
            deltas.append(e - be)
    eces.sort()

    alpha = (1.0 - confidence) / 2.0
    lo = _percentile(eces, alpha)
    hi = _percentile(eces, 1.0 - alpha)
    delta_lo = delta_hi = None
    if base is not None:
        deltas.sort()
        delta_lo = _percentile(deltas, alpha)
        delta_hi = _percentile(deltas, 1.0 - alpha)
    return EceCI(
        predictor=predictor,
        ece=point,
        lo=lo,
        hi=hi,
        n_resamples=n_resamples,
        confidence=confidence,
        bins=bins,
        delta=point_delta,
        delta_lo=delta_lo,
        delta_hi=delta_hi,
    )


def compare_predictions(
    model_preds: Sequence[float],
    baseline_preds: Sequence[float],
    labels: Sequence[float],
    *,
    metrics: Sequence[str] = ("log_loss", "brier"),
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> List[DeltaCI]:
    """Paired-bootstrap CIs for one model vs the baseline across ``metrics``.

    ``model_preds`` / ``baseline_preds`` are both scored against the same ``labels``.
    Each metric is given its own seed offset so the resamples are independent across
    metrics while staying reproducible.
    """
    out: List[DeltaCI] = []
    for i, metric in enumerate(metrics):
        terms: Callable = METRIC_TERMS[metric]
        out.append(
            paired_bootstrap_ci(
                terms(model_preds, labels),
                terms(baseline_preds, labels),
                metric,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed + i,
            )
        )
    return out
