from chess_equity.bar import render_bar, render_eval
from chess_equity.types import WDL, Equity


def _equity(pct, cp=None):
    # Keep the WDL a valid triple; pct drives only the (possibly out-of-range)
    # White-POV bar value, which render_bar is responsible for clamping.
    clamped = max(0.0, min(100.0, pct)) / 100.0
    wdl = WDL(p_win=clamped, p_draw=0.0, p_loss=1.0 - clamped)
    return Equity(wdl=wdl, equity_white=pct, source="t", cp=cp)


def test_render_bar_full_and_empty():
    assert render_bar(_equity(100.0), width=10).startswith("[##########]")
    assert render_bar(_equity(0.0), width=10).startswith("[----------]")


def test_render_bar_shows_favoured_side():
    assert "(W)" in render_bar(_equity(60.0))
    assert "(B)" in render_bar(_equity(40.0))


def test_render_bar_clamps_out_of_range():
    assert render_bar(_equity(150.0), width=10).startswith("[##########]")


def test_render_eval_includes_wdl_and_cp():
    line = render_eval(_equity(55.0, cp=80))
    assert "W/D/L" in line
    assert "cp +80" in line
    assert "[t]" in line
