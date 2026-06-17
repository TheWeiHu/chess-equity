"""Rating-band calibration for the rating-blind baseline (task 0027).

Task 0009's harness already scores predictors *sliced by rating band*; this module
adds the two things task 0027 asks for on top of the 0002 dataset:

1. :func:`band_reliability` / :func:`format_calibration_report` — the **reliability
   curve** (binned predicted-vs-observed White score) *per rating band*, so the
   rating-blind Lichess Win% can be shown mis-calibrated away from the ~2300 band it
   was fit on (it over-states the stronger side's chances where weaker players fail to
   convert, and vice-versa).
2. :func:`measure_position_classes` — replaces the *hypothesised* practical numbers in
   ``baseline/failure_modes.json`` with the **measured** rating-sliced mean White
   result for each position's class (a centipawn window at the position's rating band).

Pure computation over :class:`~chess_equity.data.schema.PositionRow` lists — no I/O,
no numpy — so it stays in the light test path. The caller loads rows (via
:func:`chess_equity.data.load_rows`) and writes the rendered report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from chess_equity.data.schema import PositionRow
from chess_equity.validate.bootstrap import EceCI, ece_bootstrap_ci
from chess_equity.validate.harness import Predictor, Scores, _score, rating_band
from chess_equity.validate.metrics import reliability_table

# One reliability bucket: (bin_lo, mean_pred, mean_label, count).
ReliabilityRow = Tuple[float, float, float, int]


@dataclass(frozen=True)
class BandCalibration:
    """A predictor's calibration within one slice value (e.g. one rating band).

    ``ece_ci`` is the bin-resampling 95% CI on this band's ECE (task 0076), present only
    when :func:`band_reliability` is called with ``bootstrap > 0``; ``None`` otherwise so
    the bare point-ECE behaviour stays backward-compatible.
    """

    band: str
    scores: Scores
    table: List[ReliabilityRow]
    ece_ci: Optional[EceCI] = None


def band_reliability(
    rows: Sequence[PositionRow],
    predictor: Predictor,
    *,
    slicer: Callable[[PositionRow], str] = rating_band,
    bins: int = 10,
    bootstrap: int = 0,
    confidence: float = 0.95,
    seed: int = 0,
) -> List[BandCalibration]:
    """Per-band scores + reliability table for ``predictor`` over ``rows``.

    Bands come back in sorted order. A band with ``scores.ece`` near 0 and a table
    hugging the diagonal (``mean_pred ~= mean_label``) is well calibrated; the
    baseline is expected to drift in the low-rating bands.

    With ``bootstrap > 0`` each band also carries a bin-resampling ``confidence`` CI on
    its ECE (task 0076) — the error bar a band-level "miscalibrated here" claim needs, so
    a drift can be told from small-sample noise. Each band gets its own seed offset so the
    CIs differ across bands yet stay byte-reproducible. ``bootstrap == 0`` leaves
    ``ece_ci`` ``None`` (unchanged behaviour).
    """
    grouped: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        grouped.setdefault(slicer(row), []).append(i)
    out: List[BandCalibration] = []
    for band_i, band in enumerate(sorted(grouped)):
        idxs = grouped[band]
        preds = [predictor(rows[i]) for i in idxs]
        labels = [rows[i].result for i in idxs]
        ece_ci = (
            ece_bootstrap_ci(
                preds,
                labels,
                predictor=band,
                bins=bins,
                n_resamples=bootstrap,
                confidence=confidence,
                seed=seed + band_i,
            )
            if bootstrap > 0
            else None
        )
        out.append(
            BandCalibration(
                band=band,
                scores=_score(preds, labels),
                table=reliability_table(preds, labels, bins=bins),
                ece_ci=ece_ci,
            )
        )
    return out


def format_calibration_report(
    bands: Sequence[BandCalibration],
    *,
    predictor_name: str = "baseline",
    title: str = "Baseline calibration by rating band",
) -> str:
    """Render per-band scores and reliability tables as Markdown (lower ECE = better)."""
    out: List[str] = [f"# {title}", ""]
    out.append(
        f"Predictor **{predictor_name}** (rating-blind Lichess Win%) vs actual White "
        "result. A calibrated band has `mean_pred ≈ mean_obs` in every bin and ECE ≈ 0; "
        "the rating-blind baseline is fit on ~2300 play, so it should drift in the other "
        "bands (it can't see who is playing)."
    )
    out.append("")
    out.append("## ECE by rating band (lower = better calibrated)")
    out.append("")
    # Show the bin-resampling ECE CI column only when the bands carry one (task 0076);
    # without it the table is the original point-ECE-only view.
    with_ci = any(b.ece_ci is not None for b in bands)
    if with_ci:
        conf_pct = round(next(b.ece_ci.confidence for b in bands if b.ece_ci) * 100)
        out.append(f"Error bars are a bin-resampling bootstrap {conf_pct}% CI on each band's ECE.")
        out.append("")
        out.append(f"| rating band | n | log-loss | Brier | ECE | ECE {conf_pct}% CI |")
        out.append("|---|--:|--:|--:|--:|:--:|")
        for b in bands:
            s = b.scores
            ci = b.ece_ci
            ci_str = f"[{ci.lo:.4f}, {ci.hi:.4f}]" if ci is not None else "—"
            out.append(
                f"| {b.band} | {s.n} | {s.log_loss:.4f} | {s.brier:.4f} | {s.ece:.4f} | {ci_str} |"
            )
    else:
        out.append("| rating band | n | log-loss | Brier | ECE |")
        out.append("|---|--:|--:|--:|--:|")
        for b in bands:
            s = b.scores
            out.append(f"| {b.band} | {s.n} | {s.log_loss:.4f} | {s.brier:.4f} | {s.ece:.4f} |")
    out.append("")
    out.append("## Reliability curves (predicted vs observed White score)")
    for b in bands:
        out.append("")
        out.append(f"### {b.band}  (n={b.scores.n}, ECE={b.scores.ece:.4f})")
        out.append("")
        out.append("| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |")
        out.append("|--:|--:|--:|--:|--:|")
        for bin_lo, mean_pred, mean_obs, count in b.table:
            gap = mean_obs - mean_pred
            out.append(
                f"| {bin_lo:.2f} | {mean_pred:.3f} | {mean_obs:.3f} | {count} | {gap:+.3f} |"
            )
    out.append("")
    return "\n".join(out)


@dataclass(frozen=True)
class ClassMeasurement:
    """Measured rating-sliced White outcome for one failure-mode position class."""

    band: str
    measured_white: Optional[float]  # mean White result in the class, or None if no data
    n: int


def measure_position_classes(
    rows: Sequence[PositionRow],
    engine_cp: float,
    band: str,
    *,
    cp_window: float = 75.0,
    slicer: Callable[[PositionRow], str] = rating_band,
) -> ClassMeasurement:
    """Measured mean White result for the class ``|cp_eval - engine_cp| <= cp_window``
    within rating ``band``.

    This is the empirical answer to "is this engine eval actually borne out at this
    rating?" — e.g. for a drawn (``engine_cp≈0``) endgame at a club band, does White
    really score 50%, or does the stronger side convert? ``measured_white`` is ``None``
    when no dataset row falls in the class (expected on the tiny committed sample; a
    real dump from task 0024 fills it in).
    """
    hits = [
        r.result
        for r in rows
        if slicer(r) == band and abs(r.cp_eval - engine_cp) <= cp_window
    ]
    if not hits:
        return ClassMeasurement(band=band, measured_white=None, n=0)
    return ClassMeasurement(band=band, measured_white=sum(hits) / len(hits), n=len(hits))
