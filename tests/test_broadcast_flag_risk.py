"""Per-side flag-risk alert in the broadcast event stream (task 0243).

The clock dimension models each side's P(loses on time) from its remaining clock + the
time control (:func:`chess_equity.clock.flag_risk`). This threads that figure onto every
published :class:`MoveEvent` (``flag_risk_white``/``black``) and surfaces it on the
overlay event as ``flag_risk: {white:{risk,alert}, black:{risk,alert}}`` so the overlay
can light a time-trouble badge. The contract:

- a clocked, time-control-known side carries a ``flag_risk`` in [0, 0.6]; the ``alert``
  flag is ``is_flag_risk_alert(risk)`` — a bullet scramble (~5s) trips it, a comfortable
  clock (~300s) does not;
- a clock-blind side (no ``[%clk]``) carries ``None`` and no ``flag_risk`` block is
  emitted at all, so clock-blind feeds degrade gracefully (no badge);
- correspondence games (no flag pressure) carry risk 0.0 -> never alert.

Engine-free :class:`LichessBaselineModel`, so deterministic without Stockfish.
"""
from chess_equity.broadcast import GameTracker
from chess_equity.clock import FLAG_RISK_ALERT_THRESHOLD, flag_risk
from chess_equity.models import LichessBaselineModel

# A bullet game (60s base) where White is deep in a scramble (~5s) while Black is
# comfortable (~5min) — the classic time-trouble asymmetry the badge exists to surface.
BULLET_SCRAMBLE_PGN = """[Event "Flag-risk test"]
[Site "https://lichess.org/scramble1"]
[White "Scrambler"]
[Black "Comfortable"]
[TimeControl "60+0"]
[WhiteElo "2400"]
[BlackElo "2400"]
[Result "*"]

1. e4 { [%clk 0:00:05] } e5 { [%clk 0:05:00] } 2. Nf3 { [%clk 0:00:04] } Nc6 { [%clk 0:04:58] } *
"""

# Same shape but no [%clk] tags at all — a clock-blind feed.
CLOCK_BLIND_PGN = """[Event "No-clock test"]
[Site "https://lichess.org/blind1"]
[White "A"]
[Black "B"]
[TimeControl "60+0"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *
"""


def _last_overlay_event(pgn):
    tracker = GameTracker("g", LichessBaselineModel(), white_elo=2400, black_elo=2400)
    events = tracker.ingest(pgn)
    assert events, "fixture must yield move events"
    return events[-1].to_overlay_event(), events


def test_high_flag_risk_alerts_low_clock_off_for_comfortable():
    """Bullet scramble: White (~5s) alerts; Black (~5min) does not."""
    evt, _ = _last_overlay_event(BULLET_SCRAMBLE_PGN)
    fr = evt["flag_risk"]
    assert fr["white"]["risk"] >= FLAG_RISK_ALERT_THRESHOLD
    assert fr["white"]["alert"] is True
    assert fr["black"]["risk"] < FLAG_RISK_ALERT_THRESHOLD
    assert fr["black"]["alert"] is False


def test_event_flag_risk_matches_clock_model():
    """The per-side risk equals chess_equity.clock.flag_risk for that side's clock."""
    _, events = _last_overlay_event(BULLET_SCRAMBLE_PGN)
    last = events[-1]
    assert last.flag_risk_white == flag_risk(last.white_clock, "bullet")
    assert last.flag_risk_black == flag_risk(last.black_clock, "bullet")


def test_clock_blind_feed_emits_no_flag_risk_block():
    """No [%clk] -> per-side risk is None and the overlay event omits flag_risk."""
    evt, events = _last_overlay_event(CLOCK_BLIND_PGN)
    assert events[-1].flag_risk_white is None
    assert events[-1].flag_risk_black is None
    assert "flag_risk" not in evt
