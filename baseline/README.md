# Baseline & failure modes (task 0003) — the "before picture"

This is the benchmark the whole project aims to beat: **Lichess's rating-blind
Win%**, plus concrete evidence of the two ways it misleads. It motivates the
rating-conditioned equity bar (0004/0005) and feeds the head-to-head validation
(0009).

> Win% = 50 + 50·(2/(1 + e^(−0.00368208·cp)) − 1) — fit on ~2300-rated games,
> **blind to who is playing.**

## What's here

- **`failure_modes.json`** — a curated set of 7 annotated positions (≥6 required),
  covering both named failure modes, each with a FEN, the engine verdict (`cp`),
  why the baseline mischaracterises it, and a *hypothesised* practical White score.
- **`report.py`** — runs the baseline over the set and prints baseline-vs-practical
  for each position (the "before picture"). Run: `python3 report.py`.
- **`fen_lint.py`** — a stdlib FEN structural validator (so the committed FENs
  can't silently rot before python-chess is a dependency).
- **`test_failure_modes.py`** — schema + FEN sanity, and asserts the baseline
  actually misleads. Run: `python3 test_failure_modes.py` (or `pytest`).

The baseline Win% itself is **already implemented** by the package scaffold (0001)
as `LichessBaselineModel` / `lichess_win_percent`. `report.py` imports that when
`chess_equity` is installed and otherwise falls back to a local copy of the same
constant, so it runs standalone today.

## The two failure modes

1. **`dead-draw-hard` — "dead 0.00 but practically hard."** Positions an engine
   calls ≈0.00 (Philidor R+P vs R, opposite-coloured bishops, K+P vs K, the Réti
   study). The baseline says 50/50, but the result depends on technique a given
   rating may not have — so real outcomes are *asymmetric*, which a rating-blind
   number cannot express.
2. **`absurd-refutation` — "unequal only via an absurd refutation."** Positions an
   engine scores decisively (Saavedra: win only via `c8=R`; the knight-promotion
   fork: win only via `e8=N+`) where the eval banks on a single move almost no
   human of that rating plays. The baseline reports ~95–97%; practically it's far
   closer to equal.

## ⚠️ Honesty note on the numbers

The `hypothesized_practical_white_*` fields are **hypotheses, not measurements** —
the claim this task makes precise, *not* evidence that settles it. For the drawn
studies, `cp = 0` is endgame theory (an engine reports ≈0.00), so the baseline's
50% is exact; the open question is only the practical skew. For the decisive
studies the `cp` is the well-known verdict, hand-entered (no engine is wired yet).

## Deferred (needs the 0002 dataset)

The third acceptance criterion — a **calibration report sliced by rating band**
(reliability curves + Brier/log-loss showing the rating-blind model is
mis-calibrated away from ~2300) — requires the labelled (eval, ratings, outcome)
data from **0002**, which isn't merged yet. Follow-ups:

- Replace the hypothesised practical scores with **measured** rating-sliced
  outcomes for these positions/classes from 0002 data.
- Wire a real **Stockfish** engine (0001's `ObjectiveEngine`) to confirm each `cp`
  and the "only move" claims, rather than hand-entered verdicts.
- Produce the calibration plot + Brier/log-loss by rating band (the quantified
  "before" the rest of the project improves on).
