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

These are heuristics with sensible shape, not a fit — the constants are the obvious
knobs to calibrate on real data once the validation suite slices by clock.
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

# A scramble in bullet is far deadlier than the same seconds in classical: moves keep
# coming fast and there is no time to steady. Multiplies flag risk by time control.
_TC_FLAG_MULTIPLIER = {
    "bullet": 1.0,
    "blitz": 0.8,
    "rapid": 0.6,
    "classical": 0.5,
    "correspondence": 0.0,  # days per move — no flag pressure
}


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
