from chess_equity.bar import render_bar, render_eval, render_svg
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


# --- SVG snapshot export (task 0217) -----------------------------------------

# Golden string: White ahead, White to move, with an objective cp. The bar fill
# is White's share of the 336px track (336 * 55/100 = 184.8) — White-POV always.
GOLDEN_SVG_WHITE = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="360" height="72" '
    'viewBox="0 0 360 72" role="img" aria-label="Equity bar: White 55.0% (White ahead), White to move">\n'
    '  <title>Equity bar: White 55.0% (White ahead), White to move</title>\n'
    '  <rect width="360" height="72" fill="#1b1b1b"/>\n'
    '  <text x="12" y="22" font-family="sans-serif" font-size="15" font-weight="bold" fill="#f0f0f0">White 55.0%</text>\n'
    '  <text x="348" y="22" font-family="sans-serif" font-size="12" fill="#9a9a9a" text-anchor="end">White ahead · cp +80</text>\n'
    '  <rect x="12" y="30" width="336" height="18" rx="3" fill="#2b2b2b"/>\n'
    '  <rect x="12" y="30" width="184.8" height="18" rx="3" fill="#f0f0f0"/>\n'
    '  <line x1="180.0" y1="27" x2="180.0" y2="51" stroke="#888" stroke-width="1"/>\n'
    '  <text x="12" y="61" font-family="sans-serif" font-size="10" fill="#9a9a9a">White to move</text>\n'
    '  <text x="348" y="61" font-family="sans-serif" font-size="9" fill="#6a6a6a" text-anchor="end">[t]</text>\n'
    '</svg>\n'
)


def test_render_svg_golden_white_to_move():
    assert render_svg(_equity(55.0, cp=80), white_to_move=True) == GOLDEN_SVG_WHITE


def test_render_svg_black_ahead_black_to_move():
    svg = render_svg(_equity(40.0), white_to_move=False)
    # White-POV: bar fill is White's 40% share (336 * 0.40 = 134.4), favour Black.
    assert 'width="134.4" height="18"' in svg
    assert "White 40.0%" in svg
    assert ">Black ahead<" in svg
    assert ">Black to move<" in svg
    assert "· cp" not in svg  # cp omitted when absent


def test_render_svg_clamps_and_is_self_contained():
    svg = render_svg(_equity(150.0), white_to_move=True)
    # Out-of-range bar clamps to the full track width (no overflow past 336).
    assert 'width="336.0" height="18" rx="3" fill="#f0f0f0"' in svg
    # Self-contained: a single inline <svg>, no external refs / scripts / images.
    assert svg.count("<svg") == 1
    for forbidden in ("<image", "<script", "xlink:href", "<?xml-stylesheet"):
        assert forbidden not in svg
