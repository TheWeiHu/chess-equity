# Validation report — data/sample/dataset.csv

Metric = predicting White expected-score (P(win)+0.5·P(draw)) vs actual result.
**Lower is better** for all three (log-loss, Brier, ECE).

## Gate verdict

Does each rating-conditioned predictor beat the rating-blind `baseline` on held-out outcomes? **PASS** = strictly lower log-loss *and* Brier (deltas are model − baseline; negative is better).

- **baseline+clock** beats baseline: logloss +0.0073, brier +0.0011 -> **FAIL**
- **wdl-a** beats baseline: logloss -0.0385, brier -0.0265 -> **PASS**

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

## By failure_mode

| predictor | failure_mode | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | dead-draw-hard  | 13 | 0.6685 | 0.1223 | 0.2519 |
| baseline | other  | 2 | 0.0000 | 0.0000 | 0.0000 |
| baseline+clock | dead-draw-hard  | 13 | 0.6699 | 0.1231 | 0.1744 |
| baseline+clock | other  | 2 | 0.0461 | 0.0038 | 0.0441 |
| wdl-a | dead-draw-hard  | 13 | 0.5991 | 0.0883 | 0.2493 |
| wdl-a | other  | 2 | 0.1628 | 0.0226 | 0.1502 |

## Head-to-head: where equity wins (baseline vs wdl-a)

Δ log-loss = `baseline` − `wdl-a` on the same rows; **Δ > 0 means equity wins** (lower model log-loss). Sorted by Δ, biggest win first.
Overall Δ: +0.0385

| slice | value | n | baseline log-loss | model log-loss | Δ |
|---|---|--:|--:|--:|--:|
| rating | 1200-1599 | 9 | 0.5034 | 0.4316 | +0.0718 |
| high_rating | <2000 | 9 | 0.5034 | 0.4316 | +0.0718 |
| failure_mode | dead-draw-hard | 13 | 0.6685 | 0.5991 | +0.0695 |
| clock | comfortable(60s+) | 13 | 0.6152 | 0.5568 | +0.0583 |
| phase | opening | 15 | 0.5794 | 0.5409 | +0.0385 |
| clock | no-clock | 1 | 0.6935 | 0.7027 | -0.0092 |
| rating | 2000-2399 | 6 | 0.6934 | 0.7048 | -0.0114 |
| high_rating | 2000-2199 | 6 | 0.6934 | 0.7048 | -0.0114 |
| failure_mode | other | 2 | 0.0000 | 0.1628 | -0.1628 |
| clock | low(<60s) | 1 | 0.0000 | 0.1719 | -0.1719 |

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

