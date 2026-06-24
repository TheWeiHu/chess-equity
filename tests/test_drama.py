"""Tests for drama / clutch detection (task 0020).

Drama is computed from ``MoveEvent``s, so these synthesise events with controlled
equity / Δequity / clocks (decoupled from any model) and assert each storyline fires
on the right move and stays quiet on dull ones — the precision the overlay needs.
"""

import dataclasses

import pytest

from chess_equity.drama import MoveEvent
from chess_equity.drama import (
    CLUTCH_DELTA,
    SLIP_DELTA,
    DramaEvent,
    detect,
    highlights,
    score_event,
)

# A neutral, non-dramatic base event (White just moved; quiet). Override per test.
_BASE = MoveEvent(
    game_id="g1",
    ply=10,
    san="Nf3",
    uci="g1f3",
    fen="rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 0 1",
    white_to_move=False,  # Black to move => White was the mover
    white_clock=120.0,
    black_clock=120.0,
    white_elo=2000,
    black_elo=2000,
    equity=51.0,
    delta_equity=1.0,
    last_move_grade="ok",
    source="Test",
    compute_ms=0.1,
)


def ev(**over):
    return dataclasses.replace(_BASE, **over)


def test_quiet_move_produces_no_drama():
    assert score_event(ev(equity=52.0, delta_equity=1.5)) is None


def test_clutch_on_strong_positive_swing():
    # White (mover) jumps +15 from a balanced position.
    d = score_event(ev(equity=65.0, delta_equity=15.0))
    assert d is not None and d.kind == "clutch"
    assert d.mover_white is True
    assert d.magnitude == pytest.approx(15.0 / 40.0, abs=1e-3)
    assert "clutch" in d.headline.lower()


def test_missed_win_when_a_winning_side_slips():
    # White was at ~85% (after=65, delta=-20 => before=85) and dropped hard.
    d = score_event(ev(equity=65.0, delta_equity=-20.0))
    assert d is not None and d.kind == "missed_win"
    assert "slip" in d.headline.lower()


def test_escape_when_a_losing_side_claws_back():
    # Black is the mover (White to move after). Black after-POV equity:
    # white equity 75 => black after = 25; delta +20 => before = 5 (lost). Recovery.
    d = score_event(ev(white_to_move=True, equity=75.0, delta_equity=20.0))
    assert d is not None and d.kind == "escape"
    assert d.mover_white is False
    assert "claws back" in d.headline.lower()


def test_scramble_needs_low_clock_and_a_swing():
    # A mid-size swing (below SLIP_DELTA) is only drama when the clock is low.
    calm = score_event(ev(equity=58.0, delta_equity=7.0, white_clock=120.0))
    assert calm is None
    pressed = score_event(ev(equity=58.0, delta_equity=7.0, white_clock=8.0))
    assert pressed is not None and pressed.kind == "scramble"
    assert "scramble" in pressed.headline.lower()
    assert "8s" in pressed.headline


def test_none_delta_opening_is_skipped():
    assert score_event(ev(delta_equity=None)) is None


def test_detect_keeps_play_order_and_drops_quiet():
    stream = [
        ev(ply=1, delta_equity=1.0, equity=51.0),            # quiet
        ev(ply=2, delta_equity=15.0, equity=66.0),           # clutch
        ev(ply=3, delta_equity=2.0, equity=64.0),            # quiet
        ev(ply=4, delta_equity=-20.0, equity=44.0),          # missed_win (before 64<70? -> not)
    ]
    out = detect(stream)
    plies = [d.ply for d in out]
    assert plies == [2]  # only the clutch; ply4 before=64 < WIN_LEVEL so no missed_win


def test_highlights_ranked_by_magnitude_and_capped():
    stream = [
        ev(ply=2, delta_equity=CLUTCH_DELTA + 1, equity=60.0),   # small clutch
        ev(ply=4, delta_equity=30.0, equity=80.0),               # huge clutch
        ev(ply=6, delta_equity=SLIP_DELTA + 1, equity=60.0),     # medium clutch
    ]
    reel = highlights(stream, top=2)
    assert [d.ply for d in reel] == [4, 6]  # biggest magnitudes first
    assert all(isinstance(d, DramaEvent) for d in reel)
    assert reel[0].magnitude >= reel[1].magnitude
