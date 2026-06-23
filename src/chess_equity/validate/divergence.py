"""How far does the rating/clock-aware equity bar DIVERGE from the Stockfish bar?
(task 0171)

The validation gate (:mod:`chess_equity.validate.harness`) proves the equity bar
*predicts real outcomes* better than the rating-blind centipawn baseline. That is the
accuracy half of the thesis. This module measures the *product-visible* half — the
headline pitch that the rating/clock-aware bar **visibly diverges** from the classic
Stockfish bar. Two bars over the same position:

- **Stockfish bar** — Lichess's rating-blind ``Win%`` of the engine ``cp_eval``
  (:func:`chess_equity.types.lichess_win_percent`). This is what every classic eval bar
  shows: a function of cp alone, identical no matter who is playing.
- **Equity bar** — a rating-conditioned predictor's White-POV expected score (``wdl-a``
  by default; any :data:`~chess_equity.validate.harness.Predictor` works).

Both are White-POV scalars in [0, 1]. Per bin (time control × rating) we report, with
``n`` per bin:

- **signed gap** — mean ``(equity − stockfish)`` in percentage points. The *direction*
  the equity bar pulls the position: positive = it reads more White-favorable than the
  Stockfish bar, negative = less. A rating-aware bar should pull *toward the stronger
  side* and *away from* objectively-decisive-but-unconvertible edges.
- **|gap|** — mean absolute gap (pp): the typical magnitude of disagreement, the number
  a viewer literally sees between the two bars.
- **p90 |gap|** — the 90th-percentile absolute gap: how big the disagreement gets on the
  positions where the two bars part ways most.
- **rank-disagree** — how often the two bars name *different favorites*. Counted over
  "rankable" positions where both bars are at least :data:`FLIP_DEADBAND` from 50% (so a
  49.9-vs-50.1 jitter is not scored as a real flip), as the fraction where one bar is
  >50% and the other <50%.

This is *not* an accuracy claim — no outcomes are read. It quantifies disagreement, the
thing a viewer sees. Pure functions over rows + predictor callables (no numpy, no I/O),
so it lives in the light test path beside :mod:`chess_equity.validate.goodmoves`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from chess_equity.data.schema import PositionRow
from chess_equity.validate.harness import rating_band

# A predictor: a row -> predicted White expected-score in [0, 1] (same contract as the
# gate's harness Predictor; re-declared so this module doesn't depend on it being a
# specific symbol).
Predictor = Callable[[PositionRow], float]

# A position is "rankable" for the flip metric only when a bar is at least this far from
# 50% — so a near-even position whose two bars straddle 50% by a hair is not counted as
# the two bars "disagreeing on the favorite". 0.05 == 5 percentage points.
FLIP_DEADBAND = 0.05


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """The ``q`` quantile (0..1) of an already-sorted list (nearest-rank). Empty -> 0."""
    if not sorted_vals:
        return 0.0
    idx = int(q * (len(sorted_vals) - 1) + 0.5)
    return sorted_vals[max(0, min(len(sorted_vals) - 1, idx))]


@dataclass(frozen=True)
class DivergenceCell:
    """The two-bar disagreement summary over one bin of positions."""

    label: str
    n: int
    mean_signed_gap: float  # mean (equity − stockfish), pp
    mean_abs_gap: float  # mean |equity − stockfish|, pp
    p90_abs_gap: float  # 90th-percentile |gap|, pp
    n_rankable: int  # positions where both bars clear the deadband
    rank_disagree_rate: Optional[float]  # fraction of rankable bins naming diff favorite


def _summarize(label: str, gaps: Sequence[Tuple[float, float]]) -> DivergenceCell:
    """Build a :class:`DivergenceCell` from ``(stockfish, equity)`` pairs in [0, 1]."""
    n = len(gaps)
    signed = [(eq - sf) * 100.0 for sf, eq in gaps]
    abs_gaps = sorted(abs(s) for s in signed)
    rankable = [
        (sf, eq)
        for sf, eq in gaps
        if abs(sf - 0.5) >= FLIP_DEADBAND and abs(eq - 0.5) >= FLIP_DEADBAND
    ]
    disagree = sum(1 for sf, eq in rankable if (sf > 0.5) != (eq > 0.5))
    return DivergenceCell(
        label=label,
        n=n,
        mean_signed_gap=sum(signed) / n if n else 0.0,
        mean_abs_gap=sum(abs_gaps) / n if n else 0.0,
        p90_abs_gap=_percentile(abs_gaps, 0.90),
        n_rankable=len(rankable),
        rank_disagree_rate=(disagree / len(rankable)) if rankable else None,
    )


@dataclass(frozen=True)
class DivergenceReport:
    """The full divergence measurement: overall plus slices by tc, rating, and the cross."""

    equity_name: str
    stockfish_name: str
    overall: DivergenceCell
    by_tc: List[DivergenceCell]
    by_rating: List[DivergenceCell]
    by_tc_rating: List[DivergenceCell]


def measure_divergence(
    rows: Sequence[PositionRow],
    equity: Predictor,
    *,
    equity_name: str = "wdl-a",
    stockfish: Predictor,
    stockfish_name: str = "baseline",
) -> DivergenceReport:
    """Measure how far ``equity`` diverges from the ``stockfish`` bar over ``rows``.

    Both predictors map a row to a White-POV expected score in [0, 1]; the per-row gap is
    ``equity − stockfish`` in percentage points. Returns one :class:`DivergenceCell` for
    the whole set and for every ``tc_bucket``, ``rating_band``, and their cross — so the
    report answers "where does the equity bar visibly disagree with the classic bar?"
    binned by time control & rating, exactly the acceptance criterion of task 0171.
    """
    overall_pairs: List[Tuple[float, float]] = []
    by_tc: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    by_rating: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    by_cross: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)

    for row in rows:
        pair = (stockfish(row), equity(row))
        rb = rating_band(row)
        tc = row.tc_bucket or "unknown"
        overall_pairs.append(pair)
        by_tc[tc].append(pair)
        by_rating[rb].append(pair)
        by_cross[(tc, rb)].append(pair)

    return DivergenceReport(
        equity_name=equity_name,
        stockfish_name=stockfish_name,
        overall=_summarize("overall", overall_pairs),
        by_tc=[_summarize(tc, by_tc[tc]) for tc in sorted(by_tc)],
        by_rating=[_summarize(rb, by_rating[rb]) for rb in sorted(by_rating)],
        by_tc_rating=[
            _summarize(f"{tc} × {rb}", by_cross[(tc, rb)])
            for tc, rb in sorted(by_cross)
        ],
    )


def _fmt_rate(x: Optional[float]) -> str:
    return "—" if x is None else f"{100 * x:4.1f}%"


def _rows_table(cells: Sequence[DivergenceCell], first_col: str) -> List[str]:
    out = [
        f"| {first_col} | n | signed gap (pp) | \\|gap\\| (pp) | p90 \\|gap\\| (pp) | "
        "rank-disagree (n) |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for c in cells:
        out.append(
            f"| {c.label} | {c.n} | {c.mean_signed_gap:+.2f} | {c.mean_abs_gap:.2f} | "
            f"{c.p90_abs_gap:.2f} | {_fmt_rate(c.rank_disagree_rate)} ({c.n_rankable}) |"
        )
    return out


def format_divergence(report: DivergenceReport, *, header: str = "") -> str:
    """Render the divergence measurement as a standalone Markdown report (task 0171).

    ``header`` is a one-line provenance string (dump + n) prepended verbatim so the
    artifact states which real dump it was built from, per the project's data policy.
    """
    eq = report.equity_name
    sf = report.stockfish_name
    out: List[str] = []
    # The header (provenance: dump + n) is the H1 when supplied — mirroring
    # validation_real.md, whose single H1 carries the dump and n. Without one we emit a
    # generic title so the section is never headerless.
    out += [
        header
        or f"# Divergence report — equity bar (`{eq}`) vs Stockfish bar (`{sf}`)",
        "",
        f"How far the rating/clock-aware **equity bar** (`{eq}`) departs from the "
        f"rating-blind **Stockfish bar** (`{sf}`, Lichess Win% of `cp_eval`) on the *same* "
        "real positions. Both are White-POV scalars in [0, 100]%. This measures "
        "product-visible *disagreement*, not predictive accuracy (no outcomes are read; "
        "for accuracy see `reports/validation_real.md`).",
        "",
        "- **signed gap** = mean(equity − stockfish), pp. The direction the equity bar "
        "pulls vs the classic bar (＋ = more White-favorable).",
        "- **|gap|** / **p90 |gap|** = mean and 90th-pct absolute gap, pp — the magnitude "
        "a viewer sees between the two bars.",
        f"- **rank-disagree** = among positions where both bars clear ±"
        f"{100 * FLIP_DEADBAND:.0f}pp of 50% (the rankable n in parens), the fraction "
        "where the two bars name *different favorites*.",
        "",
        "## Overall",
        "",
    ]
    out += _rows_table([report.overall], "")
    out += ["", "## By time control", ""]
    out += _rows_table(report.by_tc, "time control")
    out += ["", "## By rating band", ""]
    out += _rows_table(report.by_rating, "rating")
    out += ["", "## By time control × rating", ""]
    out += _rows_table(report.by_tc_rating, "tc × rating")
    out += [""]
    return "\n".join(out)
