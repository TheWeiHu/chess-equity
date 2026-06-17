"""Reliability-curve plots for the validation gate (task 0036 / 0030 follow-up).

The numeric calibration story already ships: :func:`chess_equity.validate.metrics.
reliability_table` and :func:`chess_equity.validate.calibration.band_reliability` give
the binned predicted-vs-observed White score per rating band, and the Markdown report
tabulates it. This module renders that same data as a **calibration curve** — one line
per rating band against the ``y = x`` diagonal — so the drift of the rating-blind
baseline away from its ~2300 fit is visible at a glance.

matplotlib is an **optional** dependency (the ``plots`` extra). It is imported lazily
inside the render functions, so the rest of the package — and every numeric metric —
works with no plotting dependency installed; only ``--plots`` needs it.
"""

from __future__ import annotations

from typing import Sequence

from chess_equity.validate.calibration import BandCalibration


class MatplotlibNotInstalled(RuntimeError):
    """Raised when a plot is requested but matplotlib is not available."""


def _import_pyplot():
    """Import ``matplotlib.pyplot`` with a non-interactive backend, or fail clearly."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: render to a file, never open a window
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via the wrapper's test
        raise MatplotlibNotInstalled(
            "plotting needs matplotlib; install it with `pip install chess-equity[plots]`"
        ) from exc
    return plt


def save_reliability_plot(
    bands: Sequence[BandCalibration],
    path: str,
    *,
    title: str = "Reliability by rating band",
) -> str:
    """Render per-band reliability curves to ``path`` (PNG) and return the path.

    Each band becomes a predicted-vs-observed line; a dashed ``y = x`` marks perfect
    calibration, so a line bowing away from it shows where the predictor is over- or
    under-confident. Raises :class:`MatplotlibNotInstalled` if matplotlib is absent
    and :class:`ValueError` if there is nothing to plot.
    """
    if not bands:
        raise ValueError("no bands to plot (empty reliability data)")
    plt = _import_pyplot()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    for b in bands:
        if not b.table:
            continue
        xs = [mean_pred for _, mean_pred, _, _ in b.table]
        ys = [mean_obs for _, _, mean_obs, _ in b.table]
        ax.plot(xs, ys, marker="o", label=f"{b.band} (n={b.scores.n}, ECE={b.scores.ece:.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("mean predicted White score")
    ax.set_ylabel("mean observed White score")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
