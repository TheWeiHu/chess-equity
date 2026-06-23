# Validation report — real Lichess dump — lichess_db_standard_rated_2013-01, n=12000, --with-fen (seed 0)

Metric = predicting White expected-score (P(win)+0.5·P(draw)) vs actual result.
**Lower is better** for all three (log-loss, Brier, ECE).

## Gate verdict

Does each rating-conditioned predictor beat the rating-blind `baseline` on held-out outcomes? **PASS** = strictly lower log-loss *and* Brier (deltas are model − baseline; negative is better) **and** the log_loss 95% CI clears zero — a delta whose CI straddles zero is not proof.

- **wdl-a** beats baseline: logloss -0.3403, brier -0.0355; log_loss 95% CI [-0.3881, -0.2978] (CI clears zero) -> **PASS**
- **maia2** beats baseline: logloss -0.2931, brier -0.0229; log_loss 95% CI [-0.3360, -0.2536] (CI clears zero) -> **PASS**

## Overall

| predictor | n | log-loss | Brier | ECE |
|---|--:|--:|--:|--:|
| baseline | 12000 | 0.9046 | 0.2080 | 0.0799 |
| wdl-a | 12000 | 0.5643 | 0.1726 | 0.0486 |
| maia2 | 12000 | 0.6115 | 0.1852 | 0.0523 |

## By rating

| predictor | rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 1200-1599  | 4306 | 0.8713 | 0.1979 | 0.0792 |
| baseline | 1600-1999  | 7279 | 0.9313 | 0.2127 | 0.0898 |
| baseline | 2000-2399  | 415 | 0.7815 | 0.2313 | 0.0854 |
| wdl-a | 1200-1599  | 4306 | 0.5672 | 0.1758 | 0.1090 |
| wdl-a | 1600-1999  | 7279 | 0.5394 | 0.1621 | 0.0404 |
| wdl-a | 2000-2399  | 415 | 0.9725 | 0.3230 | 0.3014 |
| maia2 | 1200-1599  | 4306 | 0.5607 | 0.1699 | 0.0632 |
| maia2 | 1600-1999  | 7279 | 0.6306 | 0.1880 | 0.0741 |
| maia2 | 2000-2399  | 415 | 0.8058 | 0.2922 | 0.3038 |

## By high_rating

| predictor | high_rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 2000-2199  | 415 | 0.7815 | 0.2313 | 0.0854 |
| baseline | <2000  | 11585 | 0.9090 | 0.2072 | 0.0804 |
| wdl-a | 2000-2199  | 415 | 0.9725 | 0.3230 | 0.3014 |
| wdl-a | <2000  | 11585 | 0.5497 | 0.1672 | 0.0510 |
| maia2 | 2000-2199  | 415 | 0.8058 | 0.2922 | 0.3038 |
| maia2 | <2000  | 11585 | 0.6046 | 0.1813 | 0.0529 |

## By phase

| predictor | phase | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | endgame  | 739 | 0.6813 | 0.0721 | 0.1296 |
| baseline | middlegame  | 7853 | 1.0207 | 0.2100 | 0.1222 |
| baseline | opening  | 3408 | 0.6854 | 0.2330 | 0.0365 |
| wdl-a | endgame  | 739 | 0.4666 | 0.0655 | 0.1140 |
| wdl-a | middlegame  | 7853 | 0.5624 | 0.1745 | 0.0635 |
| wdl-a | opening  | 3408 | 0.5901 | 0.1915 | 0.0360 |
| maia2 | endgame  | 739 | 0.4149 | 0.0540 | 0.1124 |
| maia2 | middlegame  | 7853 | 0.6316 | 0.1911 | 0.0857 |
| maia2 | opening  | 3408 | 0.6079 | 0.1998 | 0.0480 |

## By clock

| predictor | clock | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | no-clock  | 12000 | 0.9046 | 0.2080 | 0.0799 |
| wdl-a | no-clock  | 12000 | 0.5643 | 0.1726 | 0.0486 |
| maia2 | no-clock  | 12000 | 0.6115 | 0.1852 | 0.0523 |

## Reliability curve (is the equity bar an honest probability?)

For each predicted-probability bin: mean predicted vs **observed** White expected-score, the bin's row count, and the gap (obs − pred). A calibrated predictor has `gap ≈ 0` in every bin; the count-weighted mean `|gap|` is the ECE.

### baseline  (n=12000, ECE=0.0799)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.025 | 0.220 | 1698 | +0.195 |
| 0.10 | 0.145 | 0.301 | 551 | +0.157 |
| 0.20 | 0.251 | 0.394 | 505 | +0.144 |
| 0.30 | 0.354 | 0.408 | 644 | +0.053 |
| 0.40 | 0.463 | 0.463 | 1778 | +0.000 |
| 0.50 | 0.530 | 0.545 | 3601 | +0.015 |
| 0.60 | 0.646 | 0.513 | 908 | -0.132 |
| 0.70 | 0.746 | 0.611 | 553 | -0.135 |
| 0.80 | 0.848 | 0.654 | 447 | -0.193 |
| 0.90 | 0.979 | 0.903 | 1315 | -0.077 |

### wdl-a  (n=12000, ECE=0.0486)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.066 | 0.019 | 533 | -0.048 |
| 0.10 | 0.160 | 0.304 | 1143 | +0.144 |
| 0.20 | 0.250 | 0.288 | 1397 | +0.038 |
| 0.30 | 0.357 | 0.302 | 1817 | -0.055 |
| 0.40 | 0.448 | 0.431 | 1242 | -0.018 |
| 0.50 | 0.549 | 0.470 | 1107 | -0.079 |
| 0.60 | 0.650 | 0.602 | 1207 | -0.049 |
| 0.70 | 0.750 | 0.767 | 1193 | +0.017 |
| 0.80 | 0.853 | 0.836 | 1180 | -0.017 |
| 0.90 | 0.939 | 0.913 | 1181 | -0.026 |

### maia2  (n=12000, ECE=0.0523)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.048 | 0.154 | 1269 | +0.105 |
| 0.10 | 0.149 | 0.265 | 859 | +0.116 |
| 0.20 | 0.250 | 0.321 | 925 | +0.071 |
| 0.30 | 0.353 | 0.359 | 1143 | +0.007 |
| 0.40 | 0.453 | 0.442 | 1635 | -0.011 |
| 0.50 | 0.549 | 0.553 | 1709 | +0.004 |
| 0.60 | 0.648 | 0.548 | 1226 | -0.100 |
| 0.70 | 0.748 | 0.696 | 1097 | -0.052 |
| 0.80 | 0.848 | 0.766 | 865 | -0.082 |
| 0.90 | 0.958 | 0.923 | 1272 | -0.035 |


## Head-to-head: where equity wins (baseline vs wdl-a)

Δ log-loss = `baseline` − `wdl-a` on the same rows; **Δ > 0 means equity wins** (lower model log-loss). Sorted by Δ, biggest win first.
Overall Δ: +0.3403
**Worst slice:** `high_rating` `2000-2199` (n=415) Δ=-0.1910 — the baseline wins here. Equity wins on 7/9 slices.

| slice | value | n | baseline log-loss | model log-loss | Δ |
|---|---|--:|--:|--:|--:|
| phase | middlegame | 7853 | 1.0207 | 0.5624 | +0.4584 |
| rating | 1600-1999 | 7279 | 0.9313 | 0.5394 | +0.3919 |
| high_rating | <2000 | 11585 | 0.9090 | 0.5497 | +0.3593 |
| clock | no-clock | 12000 | 0.9046 | 0.5643 | +0.3403 |
| rating | 1200-1599 | 4306 | 0.8713 | 0.5672 | +0.3042 |
| phase | endgame | 739 | 0.6813 | 0.4666 | +0.2146 |
| phase | opening | 3408 | 0.6854 | 0.5901 | +0.0954 |
| rating | 2000-2399 | 415 | 0.7815 | 0.9725 | -0.1910 |
| high_rating | 2000-2199 | 415 | 0.7815 | 0.9725 | -0.1910 |

## Significance vs baseline

Paired bootstrap (1000 resamples) on the per-row metric delta vs `baseline`. **Negative delta = the model is better** (lower loss); a verdict of `beats` means the whole 95% CI sits below zero.

| model | metric | delta | 95% CI | verdict |
|---|---|--:|:--:|:--:|
| wdl-a | log_loss | -0.3403 | [-0.3881, -0.2978] | beats |
| wdl-a | brier | -0.0355 | [-0.0383, -0.0327] | beats |
| maia2 | log_loss | -0.2931 | [-0.3360, -0.2536] | beats |
| maia2 | brier | -0.0229 | [-0.0253, -0.0208] | beats |

## Calibration (ECE) confidence intervals

Bin-resampling bootstrap (1000 resamples) on ECE (**lower = better calibrated**); ECE has no per-row term, so rows are resampled and the binning recomputed each draw. A `beats` verdict means the whole ECE-delta 95% CI vs baseline sits below zero.

| predictor | ECE | 95% CI | Δ vs baseline | Δ 95% CI | verdict |
|---|--:|:--:|--:|:--:|:--:|
| baseline | 0.0799 | [0.0746, 0.0889] | — | — | — |
| wdl-a | 0.0486 | [0.0421, 0.0564] | -0.0314 | [-0.0420, -0.0231] | beats |
| maia2 | 0.0523 | [0.0471, 0.0608] | -0.0277 | [-0.0352, -0.0192] | beats |

## Head-to-head significance: per-slice CIs (baseline vs wdl-a)

Paired bootstrap (1000 resamples) on the per-row log-loss delta *within each slice*. Δ = `baseline` − `wdl-a` (**Δ > 0 = equity wins**); `equity` means the whole 95% CI clears zero, so the band-level win is real and not small-n noise. Slices below n=30 read `small-n` (too few rows for a trustworthy CI). Sorted by Δ, biggest win first.

| slice | value | n | Δ log-loss | 95% CI | verdict |
|---|---|--:|--:|:--:|:--:|
| phase | middlegame | 7853 | +0.4584 | [+0.3913, +0.5265] | equity |
| rating | 1600-1999 | 7279 | +0.3919 | [+0.3364, +0.4498] | equity |
| high_rating | <2000 | 11585 | +0.3593 | [+0.3161, +0.4052] | equity |
| clock | no-clock | 12000 | +0.3403 | [+0.2969, +0.3837] | equity |
| rating | 1200-1599 | 4306 | +0.3042 | [+0.2392, +0.3690] | equity |
| phase | endgame | 739 | +0.2146 | [+0.0824, +0.3716] | equity |
| phase | opening | 3408 | +0.0954 | [+0.0790, +0.1185] | equity |
| rating | 2000-2399 | 415 | -0.1910 | [-0.3596, +0.0160] | inconclusive |
| high_rating | 2000-2199 | 415 | -0.1910 | [-0.3672, +0.0327] | inconclusive |

## By time-control bucket: does equity still beat centipawns? (baseline vs wdl-a)

Δ = `wdl-a` − `baseline` on each bucket's rows; **Δ < 0 means equity wins** (lower loss). `beats` = both log-loss and Brier deltas are negative; `worse` = both positive; `mixed` = the two metrics disagree. A bucket with fewer than n=1000 rows reads `underpowered` and is excluded from any beats/loses claim — its win or loss is small-n noise, not the thesis. Sorted by Δ log-loss, biggest equity win first.
Equity beats the baseline on 3/3 adequately-powered time-control bucket(s). 2 bucket(s) below n=1000 excluded as underpowered.

| time control | n | Δ log-loss | Δ Brier | verdict |
|---|--:|--:|--:|:--:|
| blitz | 4266 | -0.4369 | -0.0633 | beats |
| bullet | 4851 | -0.3769 | -0.0282 | beats |
| rapid | 2731 | -0.1475 | -0.0117 | beats |
| classical | 77 | +0.0735 | +0.0112 | underpowered (n=77) |
| correspondence | 75 | +0.0821 | +0.1681 | underpowered (n=75) |

_The time-control slice (task 0155) toward the streaming / time-pressure north star: even
on this clock-blind 2013-01 dump (per-move clocks absent, so the `clock` slice is a single
`no-clock` band), the rating-conditioned model still beats the rating-blind centipawn
baseline within every adequately-powered time-control class — biggest in `blitz`. The two
small buckets (`classical` n=77, `correspondence` n=75) read `wdl-a` worse but are below
the n=1000 floor, so they are flagged, not counted as a thesis loss. Generated torch-free
from the same cached `2013-01` dump (n=12000) via `validate --models baseline,wdl-a`; the
best challenger is `wdl-a` (lowest overall log-loss), the same one the maia2-inclusive run
above ranks against, so these numbers are identical to that run's tc_bucket gate._

## Good moves read as good (move-level Δequity, task 0117)

Per consecutive ply-pair, the engine's cp swing (mover POV, clamped ±1000) is the ground-truth move quality. **Good** = mover lost ≤10cp (engine-approved); **blunder** = dropped ≥100cp. `Δgood`/`Δblunder` are the mean mover-POV equity swing (pp) the bar showed on each. The thesis: good moves should read as a *positive* gain, not a saturated ~0.

`sign-acc` (direction on |cp|≥25cp moves) is a sanity floor — any monotone-in-cp bar scores ~1.0. `corr` is **baseline-biased** (the cp swing is the baseline's own input), shown for transparency, not as the win condition.

| predictor | moves | good | blunder | sign-acc | Δgood (pp) | Δblunder (pp) | corr |
|---|--:|--:|--:|:--:|--:|--:|--:|
| baseline | 11829 | 6197 | 1683 | 1.000 | +0.12 | -18.55 | +0.951 |
| wdl-a | 11829 | 6197 | 1683 | 0.993 | +0.08 | -11.59 | +0.829 |

**Direction:** every bar reads engine-approved moves above blunders (Δgood > Δblunder) — good moves read as good, not as bad. ✅

**Rating signal:** every rating-conditioned bar reads blunders as less catastrophic than the rating-blind baseline (Δblunder -18.55pp) — a refutation a rating-peer won't find is discounted. (With cp-delta as ground truth the cp-based baseline is strong by construction; the good-move *upside* needs Maia's rating-relative policy — task 0008/0005.)

See [`reports/goodmoves_real.md`](reports/goodmoves_real.md) for the fuller move-level write-up — the reproduce recipe, the rating-signal (blunder-leniency) read, and what this slice proves and does *not* prove (the rating-conditioned good-move upside needs Maia's policy — task 0008/0005).

_The positive half of the thesis (task 0117), folded into the headline gate artifact (task
0158): a reader of this single file now sees both that the rating-conditioned bar beats the
centipawn baseline at predicting outcomes (above) AND that it reads engine-approved moves as
good, not backwards. Generated torch-free from the same cached `2013-01` dump (n=12000) via
`validate --models baseline,wdl-a` — `maia2` carries no move-level policy here — so the
move-pair counts and Δ values are identical to the standalone `reports/goodmoves_real.md`
run._
