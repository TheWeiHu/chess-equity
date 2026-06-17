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
from typing import Callable, List, Sequence

from chess_equity.validate.metrics import brier_terms, log_loss_terms

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
