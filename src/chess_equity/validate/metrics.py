"""Scoring rules for "did this prediction match the real outcome?".

All three metrics take parallel sequences of ``preds`` (a predicted White
expected-score in [0, 1], i.e. ``P(win) + 0.5*P(draw)``) and ``labels`` (the actual
White result in {0.0, 0.5, 1.0}). Draws are *soft* 0.5 labels — handled natively by
treating each as a Bernoulli(label) target, so log-loss and Brier both stay valid
without collapsing draws into wins or losses.

Pure functions over plain lists — no numpy — so they live in the light test path and
other tasks can import them without pulling the data extra.
"""

from __future__ import annotations

from math import log
from typing import Dict, List, Sequence, Tuple

# Probabilities are clipped into [EPS, 1-EPS] before log() so a confident-and-wrong
# prediction yields a large-but-finite loss instead of infinity.
EPS = 1e-12


def _clip(p: float) -> float:
    return min(max(p, EPS), 1.0 - EPS)


def brier_terms(preds: Sequence[float], labels: Sequence[float]) -> List[float]:
    """Per-row squared error ``(p - y)**2`` — the un-averaged Brier contributions.

    Exposed so a paired bootstrap (:mod:`chess_equity.validate.bootstrap`) can resample
    these row terms directly: the mean of the list IS :func:`brier_score`.
    """
    _check(preds, labels)
    return [(p - y) ** 2 for p, y in zip(preds, labels)]


def brier_score(preds: Sequence[float], labels: Sequence[float]) -> float:
    """Mean squared error between predicted expected-score and actual result.

    Lower is better; 0 is perfect. Works directly with 0.5 draw labels.
    """
    terms = brier_terms(preds, labels)
    return sum(terms) / len(terms)


def log_loss_terms(preds: Sequence[float], labels: Sequence[float]) -> List[float]:
    """Per-row cross-entropy ``-[y*log(p)+(1-y)*log(1-p)]`` — the un-averaged log-loss.

    The mean of the list IS :func:`log_loss`; exposed for the paired bootstrap so the
    same clipped formula is the single source of truth.
    """
    _check(preds, labels)
    out = []
    for p, y in zip(preds, labels):
        c = _clip(p)
        out.append(-(y * log(c) + (1.0 - y) * log(1.0 - c)))
    return out


def log_loss(preds: Sequence[float], labels: Sequence[float]) -> float:
    """Cross-entropy between Bernoulli(label) and Bernoulli(pred), averaged.

    ``-[y*log(p) + (1-y)*log(1-p)]`` per row. With ``y`` in {0, 0.5, 1} this is the
    proper soft-label generalisation, so a draw rewards a prediction near 0.5. Lower
    is better.
    """
    terms = log_loss_terms(preds, labels)
    return sum(terms) / len(terms)


def reliability_table(
    preds: Sequence[float], labels: Sequence[float], *, bins: int = 10
) -> List[Tuple[float, float, float, int]]:
    """Group rows into ``bins`` equal-width prediction buckets for a calibration view.

    Returns one ``(bin_lo, mean_pred, mean_label, count)`` per non-empty bucket. A
    well-calibrated predictor has ``mean_pred ~= mean_label`` in every row.
    """
    _check(preds, labels)
    buckets: Dict[int, List[Tuple[float, float]]] = {}
    for p, y in zip(preds, labels):
        idx = min(int(_clip(p) * bins), bins - 1)
        buckets.setdefault(idx, []).append((p, y))
    table = []
    for idx in sorted(buckets):
        rows = buckets[idx]
        n = len(rows)
        mean_pred = sum(p for p, _ in rows) / n
        mean_label = sum(y for _, y in rows) / n
        table.append((idx / bins, mean_pred, mean_label, n))
    return table


def expected_calibration_error(
    preds: Sequence[float], labels: Sequence[float], *, bins: int = 10
) -> float:
    """ECE: count-weighted mean ``|mean_pred - mean_label|`` across reliability bins.

    0 is perfectly calibrated. Complements log-loss/Brier — a predictor can be sharp
    but miscalibrated, or calibrated but unsharp.
    """
    n = len(preds)
    if n == 0:
        return 0.0
    table = reliability_table(preds, labels, bins=bins)
    return sum((count / n) * abs(mean_pred - mean_label) for _, mean_pred, mean_label, count in table)


def _check(preds: Sequence[float], labels: Sequence[float]) -> None:
    if len(preds) != len(labels):
        raise ValueError(f"preds/labels length mismatch: {len(preds)} != {len(labels)}")
    if not preds:
        raise ValueError("need at least one prediction to score")
