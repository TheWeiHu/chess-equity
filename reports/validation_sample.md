# Validation report — data/sample/dataset.csv

Metric = predicting White expected-score (P(win)+0.5·P(draw)) vs actual result.
**Lower is better** for all three (log-loss, Brier, ECE).

## Gate verdict

Does each rating-conditioned predictor beat the rating-blind `baseline` on held-out outcomes? **PASS** = strictly lower log-loss *and* Brier (deltas are model − baseline; negative is better) **and** the log_loss 95% CI clears zero — a delta whose CI straddles zero is not proof.

- **baseline+clock** beats baseline: logloss +0.0073, brier +0.0011; log_loss 95% CI [-0.0088, +0.0259] (CI straddles zero) -> **FAIL**
- **wdl-a** beats baseline: logloss -0.0385, brier -0.0265; log_loss 95% CI [-0.0888, +0.0203] (CI straddles zero) -> **FAIL**

## Overall

| predictor | n | log-loss | Brier | ECE |
|---|--:|--:|--:|--:|
| baseline | 15 | 0.5794 | 0.1060 | 0.2183 |
| baseline+clock | 15 | 0.5867 | 0.1072 | 0.1570 |
| wdl-a | 15 | 0.5409 | 0.0795 | 0.2361 |

## By rating

| predictor | rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 1200-1599  | 9 | 0.5034 | 0.1766 | 0.3705 |
| baseline | 2000-2399  | 6 | 0.6934 | 0.0001 | 0.0100 |
| baseline+clock | 1200-1599  | 9 | 0.5156 | 0.1785 | 0.2683 |
| baseline+clock | 2000-2399  | 6 | 0.6934 | 0.0001 | 0.0100 |
| wdl-a | 1200-1599  | 9 | 0.4316 | 0.1287 | 0.3432 |
| wdl-a | 2000-2399  | 6 | 0.7048 | 0.0058 | 0.0755 |

## By high_rating

| predictor | high_rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 2000-2199  | 6 | 0.6934 | 0.0001 | 0.0100 |
| baseline | <2000  | 9 | 0.5034 | 0.1766 | 0.3705 |
| baseline+clock | 2000-2199  | 6 | 0.6934 | 0.0001 | 0.0100 |
| baseline+clock | <2000  | 9 | 0.5156 | 0.1785 | 0.2683 |
| wdl-a | 2000-2199  | 6 | 0.7048 | 0.0058 | 0.0755 |
| wdl-a | <2000  | 9 | 0.4316 | 0.1287 | 0.3432 |

## By phase

| predictor | phase | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | opening  | 15 | 0.5794 | 0.1060 | 0.2183 |
| baseline+clock | opening  | 15 | 0.5867 | 0.1072 | 0.1570 |
| wdl-a | opening  | 15 | 0.5409 | 0.0795 | 0.2361 |

## By clock

| predictor | clock | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | comfortable(60s+)  | 13 | 0.6152 | 0.1223 | 0.2530 |
| baseline | low(<60s)  | 1 | 0.0000 | 0.0000 | 0.0000 |
| baseline | no-clock  | 1 | 0.6935 | 0.0002 | 0.0138 |
| baseline+clock | comfortable(60s+)  | 13 | 0.6167 | 0.1230 | 0.1756 |
| baseline+clock | low(<60s)  | 1 | 0.0908 | 0.0075 | 0.0868 |
| baseline+clock | no-clock  | 1 | 0.6935 | 0.0002 | 0.0138 |
| wdl-a | comfortable(60s+)  | 13 | 0.5568 | 0.0895 | 0.2550 |
| wdl-a | low(<60s)  | 1 | 0.1719 | 0.0249 | 0.1579 |
| wdl-a | no-clock  | 1 | 0.7027 | 0.0047 | 0.0689 |

## By rating_gap

| predictor | rating_gap | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | <100  | 15 | 0.5794 | 0.1060 | 0.2183 |
| baseline+clock | <100  | 15 | 0.5867 | 0.1072 | 0.1570 |
| wdl-a | <100  | 15 | 0.5409 | 0.0795 | 0.2361 |

## By failure_mode

| predictor | failure_mode | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | dead-draw-hard  | 13 | 0.6685 | 0.1223 | 0.2519 |
| baseline | none  | 2 | 0.0000 | 0.0000 | 0.0000 |
| baseline+clock | dead-draw-hard  | 13 | 0.6699 | 0.1231 | 0.1744 |
| baseline+clock | none  | 2 | 0.0461 | 0.0038 | 0.0441 |
| wdl-a | dead-draw-hard  | 13 | 0.5991 | 0.0883 | 0.2493 |
| wdl-a | none  | 2 | 0.1628 | 0.0226 | 0.1502 |

## Reliability curve (is the equity bar an honest probability?)

For each predicted-probability bin: mean predicted vs **observed** White expected-score, the bin's row count, and the gap (obs − pred). A calibrated predictor has `gap ≈ 0` in every bin; the count-weighted mean `|gap|` is the ECE.

### baseline  (n=15, ECE=0.2183)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.000 | 0.000 | 1 | -0.000 |
| 0.40 | 0.459 | 0.000 | 2 | -0.459 |
| 0.50 | 0.513 | 0.727 | 11 | +0.214 |
| 0.90 | 1.000 | 1.000 | 1 | +0.000 |

### baseline+clock  (n=15, ECE=0.1570)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.087 | 0.000 | 1 | -0.087 |
| 0.40 | 0.459 | 0.250 | 2 | -0.209 |
| 0.50 | 0.514 | 0.682 | 11 | +0.168 |
| 0.90 | 0.999 | 1.000 | 1 | +0.001 |

### wdl-a  (n=15, ECE=0.2361)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.10 | 0.158 | 0.000 | 1 | -0.158 |
| 0.30 | 0.370 | 0.000 | 2 | -0.370 |
| 0.40 | 0.424 | 0.500 | 6 | +0.076 |
| 0.50 | 0.590 | 1.000 | 5 | +0.410 |
| 0.80 | 0.857 | 1.000 | 1 | +0.143 |


## Head-to-head: where equity wins (baseline vs wdl-a)

Δ log-loss = `baseline` − `wdl-a` on the same rows; **Δ > 0 means equity wins** (lower model log-loss). Sorted by Δ, biggest win first.
Overall Δ: +0.0385
Bands with fewer than n=1000 rows are tagged `(underpowered)` and excluded from the worst-slice beats/loses claim — a single band that small can flip its own win or loss on a handful of games, so its per-band Δ is small-n noise, not the thesis.
**Worst slice:** no adequately-powered band to judge. Equity wins on 0/0 adequately-powered slices. 11 band(s) below n=1000 excluded as underpowered.

| slice | value | n | baseline log-loss | model log-loss | Δ |
|---|---|--:|--:|--:|--:|
| rating | 1200-1599 | 9 | 0.5034 | 0.4316 | +0.0718 (underpowered) |
| high_rating | <2000 | 9 | 0.5034 | 0.4316 | +0.0718 (underpowered) |
| failure_mode | dead-draw-hard | 13 | 0.6685 | 0.5991 | +0.0695 (underpowered) |
| clock | comfortable(60s+) | 13 | 0.6152 | 0.5568 | +0.0583 (underpowered) |
| phase | opening | 15 | 0.5794 | 0.5409 | +0.0385 (underpowered) |
| rating_gap | <100 | 15 | 0.5794 | 0.5409 | +0.0385 (underpowered) |
| clock | no-clock | 1 | 0.6935 | 0.7027 | -0.0092 (underpowered) |
| rating | 2000-2399 | 6 | 0.6934 | 0.7048 | -0.0114 (underpowered) |
| high_rating | 2000-2199 | 6 | 0.6934 | 0.7048 | -0.0114 (underpowered) |
| failure_mode | none | 2 | 0.0000 | 0.1628 | -0.1628 (underpowered) |
| clock | low(<60s) | 1 | 0.0000 | 0.1719 | -0.1719 (underpowered) |

## Significance vs baseline

Paired bootstrap (2000 resamples) on the per-row metric delta vs `baseline`. **Negative delta = the model is better** (lower loss); a verdict of `beats` means the whole 95% CI sits below zero.

| model | metric | delta | 95% CI | verdict |
|---|---|--:|:--:|:--:|
| baseline+clock | log_loss | +0.0073 | [-0.0088, +0.0259] | inconclusive |
| baseline+clock | brier | +0.0011 | [-0.0058, +0.0090] | inconclusive |
| wdl-a | log_loss | -0.0385 | [-0.0888, +0.0203] | inconclusive |
| wdl-a | brier | -0.0265 | [-0.0464, -0.0060] | beats |

## Calibration (ECE) confidence intervals

Bin-resampling bootstrap (2000 resamples) on ECE (**lower = better calibrated**); ECE has no per-row term, so rows are resampled and the binning recomputed each draw. A `beats` verdict means the whole ECE-delta 95% CI vs baseline sits below zero.

| predictor | ECE | 95% CI | Δ vs baseline | Δ 95% CI | verdict |
|---|--:|:--:|--:|:--:|:--:|
| baseline | 0.2183 | [0.0905, 0.3464] | — | — | — |
| baseline+clock | 0.1570 | [0.0363, 0.2934] | -0.0613 | [-0.1939, +0.0147] | inconclusive |
| wdl-a | 0.2361 | [0.1541, 0.3168] | +0.0178 | [-0.0304, +0.0658] | inconclusive |

## Head-to-head significance: per-slice CIs (baseline vs wdl-a)

Paired bootstrap (2000 resamples) on the per-row log-loss delta *within each slice*. Δ = `baseline` − `wdl-a` (**Δ > 0 = equity wins**); `equity` means the whole 95% CI clears zero, so the band-level win is real and not small-n noise. Slices below n=30 read `small-n` (too few rows for a trustworthy CI). A band with fewer than n=1000 rows reads `underpowered` and is excluded from any per-band beats/loses claim — its own win or loss is small-n noise, not the thesis (e.g. a 2000-2399 band at n=415 can flip on a handful of games). Sorted by Δ, biggest win first.

| slice | value | n | Δ log-loss | 95% CI | verdict |
|---|---|--:|--:|:--:|:--:|
| rating | 1200-1599 | 9 | +0.0718 | n<30 | underpowered (n=9) |
| high_rating | <2000 | 9 | +0.0718 | n<30 | underpowered (n=9) |
| failure_mode | dead-draw-hard | 13 | +0.0695 | n<30 | underpowered (n=13) |
| clock | comfortable(60s+) | 13 | +0.0583 | n<30 | underpowered (n=13) |
| phase | opening | 15 | +0.0385 | n<30 | underpowered (n=15) |
| rating_gap | <100 | 15 | +0.0385 | n<30 | underpowered (n=15) |
| clock | no-clock | 1 | -0.0092 | n<30 | underpowered (n=1) |
| rating | 2000-2399 | 6 | -0.0114 | n<30 | underpowered (n=6) |
| high_rating | 2000-2199 | 6 | -0.0114 | n<30 | underpowered (n=6) |
| failure_mode | none | 2 | -0.1628 | n<30 | underpowered (n=2) |
| clock | low(<60s) | 1 | -0.1719 | n<30 | underpowered (n=1) |
## By time-control bucket: does equity still beat centipawns? (baseline vs wdl-a)

Δ = `wdl-a` − `baseline` on each bucket's rows; **Δ < 0 means equity wins** (lower loss). `beats` = both log-loss and Brier deltas are negative; `worse` = both positive; `mixed` = the two metrics disagree. A bucket with fewer than n=1000 rows reads `underpowered` and is excluded from any beats/loses claim — its win or loss is small-n noise, not the thesis. Sorted by Δ log-loss, biggest equity win first.
Equity beats the baseline on 0/0 adequately-powered time-control bucket(s). 3 bucket(s) below n=1000 excluded as underpowered.

| time control | n | Δ log-loss | Δ Brier | verdict |
|---|--:|--:|--:|:--:|
| blitz | 6 | -0.0855 | -0.0515 | underpowered (n=6) |
| bullet | 3 | -0.0442 | -0.0409 | underpowered (n=3) |
| rapid | 6 | +0.0114 | +0.0056 | underpowered (n=6) |
## Good moves read as good (move-level Δequity, task 0117)

Per consecutive ply-pair, the engine's cp swing (mover POV, clamped ±1000) is the ground-truth move quality. **Good** = mover lost ≤10cp (engine-approved); **blunder** = dropped ≥100cp. `Δgood`/`Δblunder` are the mean mover-POV equity swing (pp) the bar showed on each. The thesis: good moves should read as a *positive* gain, not a saturated ~0.

`sign-acc` (direction on |cp|≥25cp moves) is a sanity floor — any monotone-in-cp bar scores ~1.0. `corr` is **baseline-biased** (the cp swing is the baseline's own input), shown for transparency, not as the win condition.

| predictor | moves | good | blunder | sign-acc | Δgood (pp) | Δblunder (pp) | corr |
|---|--:|--:|--:|:--:|--:|--:|--:|
| baseline | 12 | 10 | 2 | 1.000 | +0.64 | -47.24 | +1.000 |
| baseline+clock | 12 | 10 | 2 | 1.000 | +1.50 | -40.96 | +0.975 |
| wdl-a | 12 | 10 | 2 | 1.000 | +0.74 | -24.04 | +0.991 |

**Direction:** every bar reads engine-approved moves above blunders (Δgood > Δblunder) — good moves read as good, not as bad. ✅

**Rating signal:** every rating-conditioned bar reads blunders as less catastrophic than the rating-blind baseline (Δblunder -47.24pp) — a refutation a rating-peer won't find is discounted. (With cp-delta as ground truth the cp-based baseline is strong by construction; the good-move *upside* needs Maia's rating-relative policy — task 0008/0005.)

See [`reports/goodmoves_real.md`](reports/goodmoves_real.md) for the fuller move-level write-up — the reproduce recipe, the rating-signal (blunder-leniency) read, and what this slice proves and does *not* prove (the rating-conditioned good-move upside needs Maia's policy — task 0008/0005).

## Cutoff-robustness sweep (good × blunder grid, task 0157)

The good/blunder cutoffs above (≤10cp / ≥100cp) are arbitrary defaults, so the headline `Δgood > Δblunder` direction is re-measured across a grid of good cutoffs × blunder cutoffs. `holds` is `Δgood > Δblunder` in that cell. `sign-acc` depends only on the decisive-cp threshold (not the good/blunder cutoffs), so it is constant across the grid and shown once per predictor.

**`baseline`** — sign-acc 1.000 (|cp|≥25cp, grid-invariant)

| good ≤ | blunder ≥ | good | blunder | Δgood (pp) | Δblunder (pp) | holds |
|--:|--:|--:|--:|--:|--:|:--:|
| 5cp | 75cp | 9 | 2 | +0.82 | -47.24 | ✅ |
| 5cp | 100cp | 9 | 2 | +0.82 | -47.24 | ✅ |
| 5cp | 150cp | 9 | 2 | +0.82 | -47.24 | ✅ |
| 10cp | 75cp | 10 | 2 | +0.64 | -47.24 | ✅ |
| 10cp | 100cp | 10 | 2 | +0.64 | -47.24 | ✅ |
| 10cp | 150cp | 10 | 2 | +0.64 | -47.24 | ✅ |
| 20cp | 75cp | 10 | 2 | +0.64 | -47.24 | ✅ |
| 20cp | 100cp | 10 | 2 | +0.64 | -47.24 | ✅ |
| 20cp | 150cp | 10 | 2 | +0.64 | -47.24 | ✅ |

**`baseline+clock`** — sign-acc 1.000 (|cp|≥25cp, grid-invariant)

| good ≤ | blunder ≥ | good | blunder | Δgood (pp) | Δblunder (pp) | holds |
|--:|--:|--:|--:|--:|--:|:--:|
| 5cp | 75cp | 9 | 2 | +1.75 | -40.96 | ✅ |
| 5cp | 100cp | 9 | 2 | +1.75 | -40.96 | ✅ |
| 5cp | 150cp | 9 | 2 | +1.75 | -40.96 | ✅ |
| 10cp | 75cp | 10 | 2 | +1.50 | -40.96 | ✅ |
| 10cp | 100cp | 10 | 2 | +1.50 | -40.96 | ✅ |
| 10cp | 150cp | 10 | 2 | +1.50 | -40.96 | ✅ |
| 20cp | 75cp | 10 | 2 | +1.50 | -40.96 | ✅ |
| 20cp | 100cp | 10 | 2 | +1.50 | -40.96 | ✅ |
| 20cp | 150cp | 10 | 2 | +1.50 | -40.96 | ✅ |

**`wdl-a`** — sign-acc 1.000 (|cp|≥25cp, grid-invariant)

| good ≤ | blunder ≥ | good | blunder | Δgood (pp) | Δblunder (pp) | holds |
|--:|--:|--:|--:|--:|--:|:--:|
| 5cp | 75cp | 9 | 2 | +0.91 | -24.04 | ✅ |
| 5cp | 100cp | 9 | 2 | +0.91 | -24.04 | ✅ |
| 5cp | 150cp | 9 | 2 | +0.91 | -24.04 | ✅ |
| 10cp | 75cp | 10 | 2 | +0.74 | -24.04 | ✅ |
| 10cp | 100cp | 10 | 2 | +0.74 | -24.04 | ✅ |
| 10cp | 150cp | 10 | 2 | +0.74 | -24.04 | ✅ |
| 20cp | 75cp | 10 | 2 | +0.74 | -24.04 | ✅ |
| 20cp | 100cp | 10 | 2 | +0.74 | -24.04 | ✅ |
| 20cp | 150cp | 10 | 2 | +0.74 | -24.04 | ✅ |

**Cutoff-robust:** `Δgood > Δblunder` holds in all 9 cells of the good × blunder grid for `baseline`, `baseline+clock`, `wdl-a` — the direction is not an artifact of the default cutoffs. ✅

