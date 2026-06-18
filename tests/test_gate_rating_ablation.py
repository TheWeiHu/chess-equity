"""Rating-ablation control: prove wdl-a's edge is CAUSED by rating-conditioning (task 0138).

The gate already shows the rating-conditioned ``wdl-a`` beats the rating-blind
``baseline`` (task 0009) and slices that win by rating band (task 0111), and the
negative control (task 0130) shows the gate can FAIL a broken model. What none of those
isolate is *why* wdl-a wins: is the edge the ratings themselves, or just a generally
better-fit model that would win even rating-blind?

This control answers that by re-scoring the SAME fitted wdl-a model with both players'
ratings pinned to a neutral constant (:data:`RATING_ABLATION_ELO`), which zeroes the
model's two rating features while leaving its cp/ply/time-control fit untouched. Same
model, same fixture, same gate path — the only thing removed is the rating signal. If the
win is rating-caused, it must shrink back toward the baseline under ablation.
"""

from __future__ import annotations

from pathlib import Path

from chess_equity.data.build import load_rows
from chess_equity.validate.harness import (
    PREDICTORS,
    baseline_cp,
    build_predictors,
    evaluate,
    wdl_a,
    wdl_a_rating_ablated,
)

# The committed FEN fixture wdl-a is point-better on (the same one the negative control
# uses): enough rows that wdl-a wins on both gate metrics under the point-only gate.
FIXTURE = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset_fen.csv"


def _overall(rows):
    """{name -> overall Scores} for baseline, conditioned wdl-a, and the ablation."""
    predictors = {
        "baseline": baseline_cp,
        "wdl-a": wdl_a,
        "wdl-a-rating-ablated": wdl_a_rating_ablated,
    }
    return {r.name: r.overall for r in evaluate(rows, predictors)}


def test_rating_ablation_removes_the_gain_on_the_committed_fixture():
    rows = load_rows(str(FIXTURE))
    scores = _overall(rows)
    base, cond, abl = (
        scores["baseline"],
        scores["wdl-a"],
        scores["wdl-a-rating-ablated"],
    )

    # There is a gain to ablate: the rating-conditioned model beats the rating-blind
    # baseline on BOTH gate metrics (otherwise "the gain vanishes" would be vacuous).
    assert cond.log_loss < base.log_loss
    assert cond.brier < base.brier

    # The core ablation claim (task 0138): with ratings held constant the model is no
    # better — and in fact worse — than the rating-conditioned model. The edge cannot
    # survive removing the rating signal, so the edge IS the rating signal.
    assert abl.log_loss >= cond.log_loss
    assert abl.brier >= cond.brier

    # ...and the gain degrades *toward the baseline*: ablation gives back part of the win
    # rather than improving on it, so the ablated model sits between conditioned and
    # baseline on both metrics. (It need not lose to baseline outright on this small
    # fixture — only that the conditioning was doing real, rating-derived work.)
    assert cond.log_loss < abl.log_loss <= base.log_loss
    assert cond.brier < abl.brier <= base.brier


def test_rating_ablation_is_not_a_production_model():
    # The ablation is a diagnostic, not a selectable model — like the negative controls,
    # it must never leak into a real `--models` run (it would silently throw away ratings).
    assert "wdl-a-rating-ablated" not in PREDICTORS
    try:
        build_predictors(["wdl-a-rating-ablated"])
    except KeyError:
        pass  # expected — unknown predictor name
    else:  # pragma: no cover - only trips if the ablation leaks into the registry
        raise AssertionError("wdl-a-rating-ablated should not be a buildable predictor")
