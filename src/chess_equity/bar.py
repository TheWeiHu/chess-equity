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


def render_svg(
    equity: Equity,
    *,
    white_to_move: bool,
    width: int = 360,
    height: int = 72,
) -> str:
    """Render the equity bar as a self-contained, dependency-free SVG string.

    Mirrors :func:`render_eval` but as a shareable still image: a White-POV
    horizontal bar (light = White's share, dark = Black's), the model's White
    win%, the favoured side, and a side-to-move marker. Orientation matches the
    overlay decision — the bar is always White-POV so it stays stable as turns
    alternate. Pure string generation: no torch, no data, no XML library.
    """
    pct = max(0.0, min(100.0, equity.equity_white))
    favour = "White" if pct >= 50.0 else "Black"
    mover = "White" if white_to_move else "Black"
    cp = f" · cp {equity.cp:+.0f}" if equity.cp is not None else ""

    pad = 12
    bar_w = width - 2 * pad
    bar_h = 18
    bar_y = 30
    fill_w = round(bar_w * pct / 100.0, 2)
    mid_x = round(pad + bar_w / 2.0, 2)
    caption_y = bar_y + bar_h + 13  # below the bar, clear of the bottom edge

    aria = f"Equity bar: White {pct:.1f}% ({favour} ahead), {mover} to move"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{aria}">\n'
        f'  <title>{aria}</title>\n'
        f'  <rect width="{width}" height="{height}" fill="#1b1b1b"/>\n'
        f'  <text x="{pad}" y="22" font-family="sans-serif" font-size="15" '
        f'font-weight="bold" fill="#f0f0f0">White {pct:.1f}%</text>\n'
        f'  <text x="{width - pad}" y="22" font-family="sans-serif" font-size="12" '
        f'fill="#9a9a9a" text-anchor="end">{favour} ahead{cp}</text>\n'
        f'  <rect x="{pad}" y="{bar_y}" width="{bar_w}" height="{bar_h}" rx="3" fill="#2b2b2b"/>\n'
        f'  <rect x="{pad}" y="{bar_y}" width="{fill_w}" height="{bar_h}" rx="3" fill="#f0f0f0"/>\n'
        f'  <line x1="{mid_x}" y1="{bar_y - 3}" x2="{mid_x}" y2="{bar_y + bar_h + 3}" '
        f'stroke="#888" stroke-width="1"/>\n'
        f'  <text x="{pad}" y="{caption_y}" font-family="sans-serif" font-size="10" '
        f'fill="#9a9a9a">{mover} to move</text>\n'
        f'  <text x="{width - pad}" y="{caption_y}" font-family="sans-serif" font-size="9" '
        f'fill="#6a6a6a" text-anchor="end">[{equity.source}]</text>\n'
        f'</svg>\n'
    )
