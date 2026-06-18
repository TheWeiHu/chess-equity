"""A seeded, torch-free synthetic dataset whose outcomes obey a KNOWN rating-conditioned
WDL law — the *positive* control for the thesis gate (task 0131).

The negative control (task 0130, ``tests/test_gate_negative_control.py``) proves the gate
can say *no*: it FAILs deliberately-broken models. This is the complementary half — a
checked-in, network-free fixture on which the gate must say *yes* with statistical
significance. There is otherwise no committed PASS: ``reports/validation_sample.md`` is
15 illustrative-not-proof rows, and the real n=8000 Lichess run is human-approval-gated
(task 0128).

The law (``true_white_equity``) makes the centipawn→outcome map *sharpen with rating*:
strong players convert an advantage more reliably than weak ones at the same eval. The
rating-blind baseline (Lichess's single fixed logistic) therefore cannot be right for
every band at once, so a predictor that conditions on rating — :func:`oracle`, which
simply returns the true conditional mean — beats it **by construction**. With enough rows
the paired-bootstrap log-loss CI clears zero, so the significance-aware gate PASSes.

Everything here is pure + seeded: regenerating with the same ``(n, seed)`` reproduces the
committed CSV byte-for-byte (a drift-guard test pins this), and nothing imports torch.
"""

from __future__ import annotations

import csv
import random
from math import exp
from pathlib import Path
from typing import List

from chess_equity.data.schema import PositionRow, columns
from chess_equity.types import LICHESS_K

# The committed fixture this module regenerates. Lives beside the other sample datasets;
# named for what it is so no one mistakes synthetic outcomes for real Lichess results.
FIXTURE_PATH = Path(__file__).resolve().parents[3] / "data" / "sample" / "synthetic_ground_truth.csv"

# Default fixture size + seed. n is comfortably above the ~hundreds needed for the
# paired-bootstrap log-loss CI to clear zero on this signal, with margin so the PASS is
# not flaky near the boundary. The seed makes the whole file deterministic.
DEFAULT_N = 2400
DEFAULT_SEED = 0

# Rating sweep: span low club play to titled play so the rating-conditioned effect has
# room to show up across the gate's bands. Both players get near-equal ratings (Lichess
# pairs similar opponents), so the average is a faithful single skill label.
_MIN_RATING = 800
_MAX_RATING = 2600

# How sharply the cp→outcome logistic steepens with rating. At _MIN_RATING the slope is
# half Lichess's; at _MAX_RATING it is 2.5x. The baseline uses one fixed slope, so it is
# mis-calibrated at *both* ends — the gap the rating-aware oracle exploits.
_SLOPE_AT_MIN = 0.5
_SLOPE_AT_MAX = 2.5

# Peak draw rate (in a dead-level position); tapers to zero as the eval becomes decisive.
# Exercises the soft-label (result=0.5) path the metrics are built around.
_DRAW_MAX = 0.40


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def _rating_slope_scale(avg_rating: float) -> float:
    """Multiplier on the logistic slope for a game's average rating (linear in rating).

    Increasing in rating: stronger players' results track the eval more tightly. Clamped
    to the sweep so an out-of-range rating still yields a sane slope.
    """
    r = max(_MIN_RATING, min(_MAX_RATING, avg_rating))
    frac = (r - _MIN_RATING) / (_MAX_RATING - _MIN_RATING)
    return _SLOPE_AT_MIN + frac * (_SLOPE_AT_MAX - _SLOPE_AT_MIN)


def true_white_equity(cp_eval: float, avg_rating: float) -> float:
    """The KNOWN rating-conditioned White expected-score for a position.

    A logistic in ``cp_eval`` whose slope steepens with ``avg_rating`` — the law the
    fixture's outcomes are sampled from, and the Bayes-optimal prediction for it. The
    rating-blind baseline (a fixed-slope logistic) cannot match this for every rating, so
    a predictor that returns this value beats the baseline by construction.
    """
    return _sigmoid(LICHESS_K * _rating_slope_scale(avg_rating) * cp_eval)


def _draw_prob(mu: float) -> float:
    """Draw probability for a position whose true White expected-score is ``mu``.

    Peaks at ``_DRAW_MAX`` for a dead-level position and tapers to zero as the position
    becomes decisive. Capped at ``2*min(mu, 1-mu)`` so the implied win/loss probabilities
    stay non-negative.
    """
    decisiveness = abs(2.0 * mu - 1.0)  # 0 at mu=0.5, 1 at the extremes
    return min(_DRAW_MAX * (1.0 - decisiveness), 2.0 * min(mu, 1.0 - mu))


def oracle(row: PositionRow) -> float:
    """A rating-conditioned predictor that knows the ground-truth law (task 0131).

    The positive-control analog of the negative control's deliberately-broken predictors:
    it returns :func:`true_white_equity` for the row's eval and average rating, so it is
    the best a rating-aware model could do and beats the rating-blind baseline by
    construction. Registered only in the fixture's test, never in the production registry.
    """
    return true_white_equity(row.cp_eval, (row.white_elo + row.black_elo) / 2.0)


def generate_rows(n: int = DEFAULT_N, *, seed: int = DEFAULT_SEED) -> List[PositionRow]:
    """Sample ``n`` PositionRows whose results obey :func:`true_white_equity`, seeded.

    Pure + deterministic: the same ``(n, seed)`` yields identical rows every time. Ratings
    sweep the configured range and evals are drawn around level, so the gate's rating
    bands are all populated. Outcomes are sampled W/D/L from the rating-conditioned law, so
    a rating-aware predictor genuinely beats the rating-blind baseline on these rows.
    """
    rng = random.Random(seed)
    rows: List[PositionRow] = []
    for i in range(n):
        avg_rating = rng.uniform(_MIN_RATING, _MAX_RATING)
        jitter = rng.uniform(-40, 40)
        white_elo = int(round(avg_rating + jitter))
        black_elo = int(round(avg_rating - jitter))
        cp_eval = round(rng.gauss(0.0, 250.0), 1)
        cp_eval = max(-900.0, min(900.0, cp_eval))

        mu = true_white_equity(cp_eval, (white_elo + black_elo) / 2.0)
        p_draw = _draw_prob(mu)
        p_win = mu - 0.5 * p_draw
        # Sample the outcome from the true W/D/L triple (White POV).
        u = rng.random()
        if u < p_win:
            result = 1.0
        elif u < p_win + p_draw:
            result = 0.5
        else:
            result = 0.0

        ply = rng.randint(8, 80)
        phase = "opening" if ply <= 20 else ("endgame" if ply >= 60 else "middlegame")
        rows.append(
            PositionRow(
                cp_eval=cp_eval,
                white_elo=white_elo,
                black_elo=black_elo,
                ply=ply,
                phase=phase,
                time_control="600+0",
                tc_bucket="rapid",
                clock_remaining=None,
                side_to_move="white",
                result=result,
                game_id=f"synthetic-{i:05d}",
            )
        )
    return rows


def write_fixture(path: Path = FIXTURE_PATH, *, n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> int:
    """Regenerate the committed ground-truth CSV from the seeded generator. Returns n.

    Uses the schema's canonical column order so the fixture loads back through
    :func:`chess_equity.data.build.load_rows` unchanged. Deterministic: re-running
    overwrites with byte-identical content (a drift-guard test pins file == generator).
    """
    cols = list(columns())
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(n, seed=seed)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return len(rows)


if __name__ == "__main__":  # pragma: no cover - manual regeneration entry point
    written = write_fixture()
    print(f"wrote {written} rows to {FIXTURE_PATH}")
