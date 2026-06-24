# Good moves read as good — real-data report (task 0117)

**Data:** real Lichess dump `lichess_db_standard_rated_2013-01` (cached), sampled to
12000 evaluated positions across 171 games -> 11829 consecutive
ply-pairs (moves). `cp_eval` and results are real `[%eval]` annotations from the dump —
no synthetic data (see CLAUDE.md data policy). Predictors: `baseline` (Lichess Win% of
cp, rating-blind) vs `wdl-a` (the rating-conditioned WDL regression).

**Question (the positive half of the thesis):** does the rating-conditioned equity bar
read *engine-approved* moves as a genuine positive gain — "good moves read as good, not
just less bad" — and how does that compare to the rating-blind centipawn baseline?

**Reproduce:**
```
chess-equity data build --pgn ~/.cache/chess-equity/dumps/lichess_db_standard_rated_2013-01.pgn.zst \
  --sample 12000 --out data/real_0117 --format csv
# the `validate` CLI also emits the good-moves section AND the cutoff-robustness sweep
# (task 0157) inline on any --models baseline,wdl-a run.
```

## Good moves read as good (move-level Δequity, task 0117)

Per consecutive ply-pair, the engine's cp swing (mover POV, clamped ±1000) is the ground-truth move quality. **Good** = mover lost ≤10cp (engine-approved); **blunder** = dropped ≥100cp. `Δgood`/`Δblunder` are the mean mover-POV equity swing (pp) the bar showed on each. The thesis: good moves should read as a *positive* gain, not a saturated ~0.

`sign-acc` (direction on |cp|≥25cp moves) is a sanity floor — any monotone-in-cp bar scores ~1.0. `corr` is **baseline-biased** (the cp swing is the baseline's own input), shown for transparency, not as the win condition.

| predictor | moves | good | blunder | sign-acc | Δgood (pp) | Δblunder (pp) | corr |
|---|--:|--:|--:|:--:|--:|--:|--:|
| baseline | 11829 | 6197 | 1683 | 1.000 | +0.12 | -18.55 | +0.951 |
| wdl-a | 11829 | 6197 | 1683 | 0.993 | +0.08 | -11.59 | +0.829 |

**Direction:** every bar reads engine-approved moves above blunders (Δgood > Δblunder) — good moves read as good, not as bad. ✅

**Rating signal:** every rating-conditioned bar reads blunders as less catastrophic than the rating-blind baseline (Δblunder -18.55pp) — a refutation a rating-peer won't find is discounted. (With cp-delta as ground truth the cp-based baseline is strong by construction; the good-move *upside* needs Maia's rating-relative policy — task 0008/0005.)

## Cutoff-robustness sweep (good × blunder grid, task 0157)

The good/blunder cutoffs above (≤10cp / ≥100cp) are arbitrary defaults, so the headline `Δgood > Δblunder` direction is re-measured across a grid of good cutoffs × blunder cutoffs. `holds` is `Δgood > Δblunder` in that cell. `sign-acc` depends only on the decisive-cp threshold (not the good/blunder cutoffs), so it is constant across the grid and shown once per predictor.

**`baseline`** — sign-acc 1.000 (|cp|≥25cp, grid-invariant)

| good ≤ | blunder ≥ | good | blunder | Δgood (pp) | Δblunder (pp) | holds |
|--:|--:|--:|--:|--:|--:|:--:|
| 5cp | 75cp | 5371 | 2141 | +0.24 | -15.93 | ✅ |
| 5cp | 100cp | 5371 | 1683 | +0.24 | -18.55 | ✅ |
| 5cp | 150cp | 5371 | 1183 | +0.24 | -22.87 | ✅ |
| 10cp | 75cp | 6197 | 2141 | +0.12 | -15.93 | ✅ |
| 10cp | 100cp | 6197 | 1683 | +0.12 | -18.55 | ✅ |
| 10cp | 150cp | 6197 | 1183 | +0.12 | -22.87 | ✅ |
| 20cp | 75cp | 7270 | 2141 | -0.07 | -15.93 | ✅ |
| 20cp | 100cp | 7270 | 1683 | -0.07 | -18.55 | ✅ |
| 20cp | 150cp | 7270 | 1183 | -0.07 | -22.87 | ✅ |

**`wdl-a`** — sign-acc 0.993 (|cp|≥25cp, grid-invariant)

| good ≤ | blunder ≥ | good | blunder | Δgood (pp) | Δblunder (pp) | holds |
|--:|--:|--:|--:|--:|--:|:--:|
| 5cp | 75cp | 5371 | 2141 | +0.17 | -10.13 | ✅ |
| 5cp | 100cp | 5371 | 1683 | +0.17 | -11.59 | ✅ |
| 5cp | 150cp | 5371 | 1183 | +0.17 | -13.81 | ✅ |
| 10cp | 75cp | 6197 | 2141 | +0.08 | -10.13 | ✅ |
| 10cp | 100cp | 6197 | 1683 | +0.08 | -11.59 | ✅ |
| 10cp | 150cp | 6197 | 1183 | +0.08 | -13.81 | ✅ |
| 20cp | 75cp | 7270 | 2141 | -0.08 | -10.13 | ✅ |
| 20cp | 100cp | 7270 | 1683 | -0.08 | -11.59 | ✅ |
| 20cp | 150cp | 7270 | 1183 | -0.08 | -13.81 | ✅ |

**Cutoff-robust:** `Δgood > Δblunder` holds in all 9 cells of the good × blunder grid for `baseline`, `wdl-a` — the direction is not an artifact of the default cutoffs. ✅

## What this proves (and what it doesn't)

- **Direction holds for both bars.** Engine-approved moves read above blunders, and the
  sign of the equity swing matches the engine's verdict on ~99% of decisive moves. The
  equity bar does not read move quality backwards.
- **You cannot prove the rating-conditioned *upside* with this setup.** The declared
  ground truth is the engine cp-delta, which is the rating-blind baseline's *only* input
  — so on any cp-derived move-quality metric the baseline is strong almost by
  construction (note `corr`, biased toward baseline). Both bars read good moves as a
  near-zero gain (Δgood ≈ +0.1pp): in a sampled dump most engine-approved moves are
  ordinary moves in ongoing games, where neither bar should swing. Demonstrating that a
  move *better than the rating-typical mix* earns real upside needs Maia's
  rating-relative move policy — `grade_peer` in task 0008, which needs Maia-2 (0005).
- **The rating signal that DOES show is blunder-leniency.** The rating-conditioned bar
  reads blunders as markedly less catastrophic than the rating-blind baseline
  (Δblunder ≈ -11.6pp vs -18.6pp). That is the thesis's "absurd-refutation" / "a
  refutation a peer won't find is discounted" effect, measured on real outcomes — the
  rating-aware half of "not just less bad".
- **The direction is cutoff-robust (task 0157).** The good (≤10cp) and blunder (≥100cp)
  cutoffs are arbitrary defaults, so the sweep above re-measures across a 5/10/20cp ×
  75/100/150cp grid. `Δgood > Δblunder` holds in all 9 cells for *both* bars — the
  direction is not an artifact of the chosen cutoffs. Note that at the loosest good
  cutoff (≤20cp) Δgood itself dips slightly negative (it now averages in marginally
  inaccurate moves), but it still sits well above the blunder mean, so the direction is
  intact. `sign-acc` (1.000 / 0.993) depends only on the decisive-cp threshold, not the
  good/blunder cutoffs, so it is constant across the grid by construction.

**Bottom line:** good moves read as *good* (above blunders) on both bars, robustly across
the cutoff grid; the rating-conditioned bar's distinct, real edge here is discounting
blunders/refutations, not inflating good-move upside — the upside half awaits the Maia
policy (0008/0005).
