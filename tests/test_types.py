import pytest

from chess_equity.types import WDL, Equity, lichess_win_percent


def test_lichess_win_percent_is_symmetric_around_zero():
    assert lichess_win_percent(0) == pytest.approx(50.0)
    assert lichess_win_percent(300) + lichess_win_percent(-300) == pytest.approx(100.0)


def test_lichess_win_percent_matches_published_anchors():
    """Pin the gate's rating-blind comparator to Lichess's EXACT published curve.

    These are the literal values of ``50 + 50·(2/(1+exp(-0.00368208·cp))−1)`` — the
    constant lila ships in ``rawWinningChances``. The whole thesis is "beats Lichess's
    Win%", so this baseline must be the real curve, not an approximation; if anyone
    nudges :data:`LICHESS_K` or the formula, these anchors fail loudly.
    """
    from chess_equity.types import LICHESS_K

    assert LICHESS_K == 0.00368208
    assert lichess_win_percent(0) == pytest.approx(50.0)
    assert lichess_win_percent(100) == pytest.approx(59.10259, abs=1e-5)
    assert lichess_win_percent(300) == pytest.approx(75.11255, abs=1e-5)
    assert lichess_win_percent(1000) == pytest.approx(97.544744, abs=1e-6)
    assert lichess_win_percent(-1000) == pytest.approx(2.455256, abs=1e-6)
    # Monotone and bounded as |cp| grows large.
    assert lichess_win_percent(4000) < 100.0
    assert lichess_win_percent(-4000) > 0.0


def test_wdl_equity_and_flip():
    wdl = WDL(p_win=0.6, p_draw=0.3, p_loss=0.1)
    assert wdl.equity == pytest.approx(0.75)
    assert wdl.flipped().equity == pytest.approx(0.25)


def test_wdl_from_unnormalized_clamps_and_rescales():
    n = WDL.from_unnormalized(p_win=0.6, p_draw=0.6, p_loss=-0.2)
    assert n.p_loss == 0.0
    assert n.p_win + n.p_draw + n.p_loss == pytest.approx(1.0)


def test_wdl_rejects_negative():
    with pytest.raises(ValueError):
        WDL(p_win=-0.5, p_draw=0.5, p_loss=1.0)


def test_equity_from_side_to_move_is_white_pov():
    wdl = WDL(p_win=0.7, p_draw=0.2, p_loss=0.1)  # side to move winning
    white = Equity.from_side_to_move(wdl, white_to_move=True, source="t")
    black = Equity.from_side_to_move(wdl, white_to_move=False, source="t")
    assert white.equity_white == pytest.approx(80.0)
    assert black.equity_white == pytest.approx(20.0)  # flipped to White POV
