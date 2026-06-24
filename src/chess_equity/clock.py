"""The clock / time-pressure dimension (task 0015).

Objective eval — and Maia-2, which takes no clock input at all — treats a winning
position the same with five minutes or five seconds on the clock. In the fast games
streams are made of, the clock is the *dominant* driver of practical results: a
won position you can't convert before the flag falls is not a won position. This
module warps a White-POV equity by how much time pressure the side to move is under.

The model is deliberately tiny and pure (stdlib only, no board, no engine), so it
composes with any :class:`~chess_equity.adapters.EquityModel` output and can be
calibrated against real low-clock outcomes in the 0009 harness:

1. :func:`time_pressure` — clock seconds -> a [0, 1] scramble severity (decays with
   time on the clock; flat ``0`` when the dataset carries no ``[%clk]``).
2. :func:`flag_risk` — severity scaled by the time control (a bullet scramble is
   far deadlier than a classical one) into a probability the side to move throws
   the game on time.
3. :func:`clock_adjusted_white_equity` — apply that risk: with probability
   ``flag_risk`` the mover flags (their result -> 0), so their expected score is
   pulled *down*, holding eval and ratings fixed. The worse the time trouble and
   the more they had to lose, the larger the drop.

:data:`_TC_FLAG_MULTIPLIER` is fit to the real per-time-control time-forfeit rate
(task 0268; see ``reports/clock_calibration_real.md``). The clock-band-shaped knobs —
:data:`SCRAMBLE_SCALE`, :data:`MAX_FLAG_RISK`, :data:`FLAG_RISK_ALERT_THRESHOLD` — are
still heuristics with sensible shape: fitting them needs per-move ``[%clk]`` clocks,
which the cached 2016-05 dump predates, so that calibration waits on a >=2017-04 dump.
"""

from __future__ import annotations

from math import exp
from typing import Optional

# Scramble severity decays as ``exp(-clock / SCRAMBLE_SCALE)``. At 30s the mover is
# ~37% "scrambled"; the danger concentrates in the last few seconds (5s -> ~0.85)
# and is negligible with minutes to spare (120s -> ~0.02), so comfortable positions
# pass through essentially untouched.
SCRAMBLE_SCALE = 30.0

# Even at maximum scramble the mover does not flag with certainty — cap the risk.
MAX_FLAG_RISK = 0.6

# A :func:`flag_risk` at or above this trips the overlay's time-trouble alert (task 0243).
# flag_risk is in [0, MAX_FLAG_RISK=0.6]; 0.2 lights up a real scramble — a bullet/blitz
# side in the last ~30s — while a comfortable clock (minutes to spare -> risk ~0) stays
# dark. A sensible default knob the overlay can override, not a fit.
FLAG_RISK_ALERT_THRESHOLD = 0.2

# A scramble in bullet is far deadlier than the same seconds in classical: moves keep
# coming fast and there is no time to steady. Multiplies flag risk by time control.
#
# Calibrated (task 0268) to the REAL per-time-control time-forfeit rate over 6.2M games
# in the 2016-05 Lichess dump, normalised to bullet=1.0 (see reports/clock_calibration_
# real.md, reproducible via scripts/calibrate_clock_tc.py). Measured forfeit rates:
# bullet 54.1%, blitz 27.4%, rapid 15.6%, classical 14.4% — a far steeper drop than the
# old hand-set guesses (0.8/0.6/0.5). correspondence stays pinned at 0.0 by design: its
# "Time forfeit" is failing to move within days, not a live clock scramble.
_TC_FLAG_MULTIPLIER = {
    "bullet": 1.0,
    "blitz": 0.51,
    "rapid": 0.29,
    "classical": 0.27,
    "correspondence": 0.0,  # days per move — no live flag pressure (not data-derived)
}


# Clock-band edges (seconds remaining) for the coverage diagnostic and the
# ``validate --slice clock`` grouping (task 0249). Each ``(edge, label)`` claims the
# rows whose side-to-move clock is below ``edge``; anything above the last edge is
# ``">3m"`` and a ``None`` clock (no ``[%clk]``) is ``"none"``. Coarse on purpose — the
# point is to see *where* a candidate dump's clock coverage lands, not to fit anything.
_CLOCK_BAND_EDGES = ((10.0, "<10s"), (30.0, "10-30s"), (60.0, "30-60s"), (180.0, "1-3m"))

# Canonical display order for the bands (clock-blind first, then ascending time).
CLOCK_BANDS = ("none", "<10s", "10-30s", "30-60s", "1-3m", ">3m")


def clock_band(clock_remaining: Optional[float]) -> str:
    """Bucket a side-to-move clock (seconds) into a coarse band label.

    ``None`` (the game carried no ``[%clk]`` tags) -> ``"none"``; otherwise the first
    :data:`_CLOCK_BAND_EDGES` band it falls under, or ``">3m"`` above them all. The band
    vocabulary the coverage diagnostic and ``validate --slice clock`` group by (task 0249).
    """
    if clock_remaining is None:
        return "none"
    for edge, label in _CLOCK_BAND_EDGES:
        if clock_remaining < edge:
            return label
    return ">3m"


def time_pressure(clock_remaining: Optional[float]) -> float:
    """Scramble severity in [0, 1] for the side to move from their clock seconds.

    ``1`` at a dead clock, decaying smoothly toward ``0`` with more time. Returns
    ``0`` when ``clock_remaining`` is ``None`` (the game carried no ``[%clk]`` tags),
    so clock-blind data is a no-op rather than an error.
    """
    if clock_remaining is None:
        return 0.0
    return exp(-max(clock_remaining, 0.0) / SCRAMBLE_SCALE)


def flag_risk(clock_remaining: Optional[float], tc_bucket: str) -> float:
    """Probability the side to move loses on time, from clock seconds + time control.

    Time-control aware: the same low clock is far more dangerous in bullet than in
    classical (see :data:`_TC_FLAG_MULTIPLIER`). In [0, :data:`MAX_FLAG_RISK`].
    """
    multiplier = _TC_FLAG_MULTIPLIER.get(tc_bucket, 0.7)
    return MAX_FLAG_RISK * time_pressure(clock_remaining) * multiplier


def is_flag_risk_alert(
    risk: Optional[float], threshold: float = FLAG_RISK_ALERT_THRESHOLD
) -> bool:
    """Should a side's :func:`flag_risk` trip the overlay's time-trouble alert?

    ``True`` when ``risk`` is at or above ``threshold`` (default
    :data:`FLAG_RISK_ALERT_THRESHOLD`). ``risk`` is ``None`` for a clock-blind side (the
    feed carried no ``[%clk]``), which never alerts — so clock-blind data is a no-op.
    """
    return risk is not None and risk >= threshold


def clock_adjusted_white_equity(
    white_equity: float,
    clock_remaining: Optional[float],
    tc_bucket: str,
    white_to_move: bool,
) -> float:
    """Warp a White-POV expected-score by the mover's time pressure.

    ``white_equity`` is in [0, 1] (``P(win) + 0.5·P(draw)`` from White's POV). With
    probability :func:`flag_risk` the side to move flags, collapsing *their* result
    to 0; otherwise the position plays out as scored. So the mover's expected score
    is multiplied by ``1 - flag_risk`` — a winning position with seconds left reads
    as less safe, while a comfortable clock leaves the bar where it was.
    """
    risk = flag_risk(clock_remaining, tc_bucket)
    if risk <= 0.0:
        return white_equity
    mover_equity = white_equity if white_to_move else 1.0 - white_equity
    mover_equity *= 1.0 - risk
    return mover_equity if white_to_move else 1.0 - mover_equity
