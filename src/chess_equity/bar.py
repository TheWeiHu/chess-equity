"""ASCII rendering of the equity bar.

The bar is always White-POV: a full bar = White is winning, empty = Black. This is
the one place the [0, 100]% scalar becomes something a human reads; the web UI
(task 0010) renders the same ``Equity`` differently.
"""

from __future__ import annotations

from chess_equity.types import Equity


def render_bar(equity: Equity, width: int = 30) -> str:
    """Render a White-POV equity bar like ``[#########---------] 52.3% (W)``.

    ``width`` is the number of cells in the bar. The trailing tag shows which side
    the bar currently favours.
    """
    pct = max(0.0, min(100.0, equity.equity_white))
    filled = round(width * pct / 100.0)
    bar = "#" * filled + "-" * (width - filled)
    favour = "W" if pct >= 50.0 else "B"
    return f"[{bar}] {pct:5.1f}% ({favour})"


def render_eval(equity: Equity, width: int = 30) -> str:
    """A one-line summary: the bar, the WDL triple, and the objective cp if present."""
    w = equity.wdl
    line = render_bar(equity, width=width)
    wdl = f"W/D/L {100 * w.p_win:4.1f}/{100 * w.p_draw:4.1f}/{100 * w.p_loss:4.1f}"
    cp = f"  cp {equity.cp:+.0f}" if equity.cp is not None else ""
    return f"{line}  {wdl}  [{equity.source}]{cp}"
