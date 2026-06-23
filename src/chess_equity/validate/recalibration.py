"""Post-hoc Platt recalibration of a predictor on a held-out calibration split (task 0166).

``maia2`` is the gate's worst-calibrated rating-conditioned model: overall ECE ~0.05 but
the high-rating (2000+) bin ECE blows up to ~0.30. The validation harness only *measures*
ECE today (``validate/metrics.py``, ``validate/bootstrap.py``); nothing repairs it. This
module adds a standard monotonic recalibrator that can be fit on a calibration split and
applied at eval.

The recalibrator is **Platt scaling**: a two-parameter logistic on the *logit* of the base
prediction, ``q = sigmoid(a * logit(p) + b)``, fit by Newton/IRLS to minimise log-loss
against the (soft) labels. Platt is chosen over isotonic regression because it is robust on
the sparse high-rating bin — it has two parameters, not one knot per bin — and is strictly
monotonic, so it never re-orders predictions (a recalibrator must not change *which* side is
favoured, only how confidently). Isotonic is left as a follow-up.

``a = 1, b = 0`` is the identity (``sigmoid(logit(p)) = p``), so a recalibrator fit on data it
already matches stays a no-op. The CLI knob (``--recalibrate-maia2``) is off by default, so
the committed default run is byte-identical; an operator opts in.

Leakage discipline: fit the recalibrator on a **game-disjoint** calibration split, never on
the eval rows (see :func:`chess_equity.validate.split.game_level_split`). The caller passes
the ``--holdout`` *train* portion as the calibration set; the recalibrator is then applied to
the held-out test rows.

Pure Python over plain floats — no numpy, no torch — so it stays in the light test path. The
2x2 Newton step is solved in closed form.
"""

from __future__ import annotations

from math import exp, log
from typing import Sequence

from chess_equity.data.schema import PositionRow
from chess_equity.validate.harness import Predictor

# Probabilities are clipped into [EPS, 1-EPS] before log()/logit so a confident base
# prediction yields a large-but-finite logit instead of +-inf. Matches metrics.EPS in spirit;
# kept local so this module has no metrics dependency.
EPS = 1e-12


def _clip(p: float) -> float:
    return min(max(p, EPS), 1.0 - EPS)


def _logit(p: float) -> float:
    c = _clip(p)
    return log(c / (1.0 - c))


def _sigmoid(z: float) -> float:
    # Numerically stable logistic.
    if z >= 0:
        e = exp(-z)
        return 1.0 / (1.0 + e)
    e = exp(z)
    return e / (1.0 + e)


class PlattRecalibrator:
    """A fitted two-parameter Platt scaler ``q = sigmoid(a * logit(p) + b)``.

    ``a`` is the slope on the base prediction's logit and ``b`` the intercept. ``a = 1,
    b = 0`` is the identity. Strictly monotonic in ``p`` for ``a > 0``, so it never re-orders
    predictions. Call the instance like a function to recalibrate a base probability.
    """

    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b

    def __call__(self, p: float) -> float:
        return _sigmoid(self.a * _logit(p) + self.b)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PlattRecalibrator(a={self.a:.6g}, b={self.b:.6g})"


def fit_platt(
    base_preds: Sequence[float],
    labels: Sequence[float],
    *,
    max_iter: int = 100,
    tol: float = 1e-9,
    ridge: float = 1e-9,
) -> PlattRecalibrator:
    """Fit ``q = sigmoid(a * logit(p) + b)`` by Newton/IRLS minimising log-loss.

    ``base_preds`` are the base predictor's White expected-scores in [0, 1]; ``labels`` are
    the actual White results in {0.0, 0.5, 1.0} (soft 0.5 draws are handled natively — the
    objective is cross-entropy against a Bernoulli(label) target, valid for any label in
    [0, 1]). Starts from the identity ``a = 1, b = 0`` and iterates the closed-form 2x2
    Newton step ``w <- w - H^{-1} g`` until the step is below ``tol`` or ``max_iter``. A tiny
    ``ridge`` is added to the Hessian diagonal so a degenerate (constant-logit) calibration
    set stays solvable.

    Raises ``ValueError`` on a length mismatch or an empty calibration set.
    """
    if len(base_preds) != len(labels):
        raise ValueError(
            f"base_preds/labels length mismatch: {len(base_preds)} != {len(labels)}"
        )
    if not base_preds:
        raise ValueError("need at least one calibration row to fit Platt scaling")

    z = [_logit(p) for p in base_preds]
    y = list(labels)

    a, b = 1.0, 0.0
    for _ in range(max_iter):
        # Gradient g = X^T (q - y); Hessian H = X^T diag(q(1-q)) X with X columns [z, 1].
        g0 = g1 = 0.0
        h00 = h01 = h11 = 0.0
        for zi, yi in zip(z, y):
            qi = _sigmoid(a * zi + b)
            r = qi - yi
            g0 += r * zi
            g1 += r
            wi = qi * (1.0 - qi)
            h00 += wi * zi * zi
            h01 += wi * zi
            h11 += wi
        h00 += ridge
        h11 += ridge
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-30:
            break
        # Newton step: solve H d = g, then w -= d.
        d0 = (h11 * g0 - h01 * g1) / det
        d1 = (-h01 * g0 + h00 * g1) / det
        a -= d0
        b -= d1
        if abs(d0) < tol and abs(d1) < tol:
            break

    return PlattRecalibrator(a, b)


def make_recalibrated_predictor(
    calib_rows: Sequence[PositionRow],
    base_predictor: Predictor,
) -> Predictor:
    """A predictor that Platt-recalibrates ``base_predictor`` using ``calib_rows`` to fit.

    The base predictor is scored on every calibration row to build ``(base_pred, label)``
    pairs, Platt scaling is fit on them, and the returned :data:`Predictor` applies the base
    predictor then the fitted scaler at eval time. ``calib_rows`` MUST be disjoint from the
    rows the returned predictor is later evaluated on (pass the ``--holdout`` train split) so
    the recalibration is genuinely held-out, not fit on the eval set.

    Raises ``ValueError`` if ``calib_rows`` is empty.
    """
    if not calib_rows:
        raise ValueError("make_recalibrated_predictor needs a non-empty calibration set")

    base_preds = [base_predictor(r) for r in calib_rows]
    labels = [r.result for r in calib_rows]
    scaler = fit_platt(base_preds, labels)

    def predict(row: PositionRow) -> float:
        return scaler(base_predictor(row))

    return predict


__all__ = [
    "PlattRecalibrator",
    "fit_platt",
    "make_recalibrated_predictor",
]
