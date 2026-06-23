# Validation report — cross-dump refit held-out — wdl-a fit on **2013-01**, eval on **2016-05** (n=100000)

**Provenance.** Built torch-free from the two **cached** real Lichess monthly dumps under
`~/.cache/chess-equity/dumps/`. `wdl-a` was **re-fit on the 2013-01 dump** (its training month),
then scored against the **2016-05 dump** (a different, larger, more-recent month) — so 2016-05 is a
genuinely **out-of-distribution held-out** eval for it. Recipe:

```
# 1. build the 2013-01 training set + refit wdl-a on it (stamps meta.fit_month=2013-01)
chess-equity data build  --month 2013-01 --sample 60000  --out train_2013-01
chess-equity train       --data train_2013-01/dataset.csv --out wdl_a_2013-01.json --train-month 2013-01
#    -> n_train=34308, iters=3000, final_log_loss=0.6998, fit_month=2013-01

# 2. build the 2016-05 eval set and score the 2013-01-fit wdl-a on it
chess-equity data build  --month 2016-05 --sample 100000 --out eval_2016-05
chess-equity validate    --data eval_2016-05/dataset.csv --models baseline,wdl-a \
    --wdl-a-artifact wdl_a_2013-01.json --eval-month 2016-05 --bootstrap 1000 --seed 0 --gate
```

**fit_month:** `2013-01` (34308 training rows from the 2013-01 dump). **eval month:** `2016-05`.
**n:** 100000 eval rows (the entire 2016-05 sample is held-out — none of its games were in training,
so no within-dump holdout split is needed; the cross-dump gap *is* the held-out boundary). `maia2` is
**skipped** — no `torch` on this box (the committed 2013-01 report carries the maia2 column).

**Why this report exists (de-coupling the leakage the 0160 follow-up flagged).** The cross-dump gate
[`reports/validation_real_2016-05.md`](validation_real_2016-05.md) PASSes on the larger, well-powered
2016-05 dump, but `wdl-a`'s `fit_month` *was* `2016-05` there, so its column was **in-distribution** —
the leakage guard (task 0112) fired and the wdl-a numbers measured memorization, not held-out skill
(only the model-free `baseline` and the bigger high-rating `n` were independent). This report removes
that confound by **refitting `wdl-a` on a different month (2013-01)** so the well-powered 2016-05 dump
becomes a clean held-out test *for wdl-a itself*. No leakage banner appears below — `fit_month=2013-01`
≠ eval `2016-05` — so this is genuine held-out evidence, now with the high-rating `n` the committed
2013-01 report (`reports/validation_real.md`) lacked (its 2000-2399 bin was only n=415).

**Result.** `wdl-a` (fit 2013-01) **still beats** the rating-blind baseline on the held-out 2016-05
dump: log-loss **−0.2647** (95% CI [−0.2778, −0.2514], clears zero), Brier **−0.0209** (CI clears
zero), ECE **0.0126 vs 0.0500** (CI clears zero) → **GATE: PASS**. The high-rating slices are now
adequately powered and equity wins each: **2000-2199** (n=12167, Δ +0.1746, CI clears zero) and
**2200-2399** (n=5239, Δ +0.3451, CI clears zero). Equity wins on **17/17** adequately-powered slices.
This is the held-out, high-rating-powered proof the 0160 follow-up called for.

> ℹ️ **This is genuine held-out evidence.** Unlike `reports/validation_real_2016-05.md`, `wdl-a`'s
> training month (`2013-01`) differs from the eval month (`2016-05`), so the leakage guard stays
> silent and the PASS below reflects held-out skill, not memorization.

---

# Validation report — /tmp/xdump/eval_2016-05/dataset.csv (data month: 2016-05)

Metric = predicting White expected-score (P(win)+0.5·P(draw)) vs actual result.
**Lower is better** for all three (log-loss, Brier, ECE).

## Gate verdict

Does each rating-conditioned predictor beat the rating-blind `baseline` on held-out outcomes? **PASS** = strictly lower log-loss *and* Brier (deltas are model − baseline; negative is better) **and** the log_loss 95% CI clears zero — a delta whose CI straddles zero is not proof.

- **wdl-a** beats baseline: logloss -0.2647, brier -0.0209; log_loss 95% CI [-0.2778, -0.2514] (CI clears zero) -> **PASS** — cuts log-loss 31.8% (Brier 10.2%) vs baseline

## Overall

| predictor | n | log-loss | Brier | ECE |
|---|--:|--:|--:|--:|
| baseline | 100000 | 0.8319 | 0.2049 | 0.0500 |
| wdl-a | 100000 | 0.5672 | 0.1840 | 0.0126 |

## By rating

| predictor | rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 1200-1599  | 26529 | 0.9567 | 0.2046 | 0.0589 |
| baseline | 1600-1999  | 54069 | 0.7634 | 0.2067 | 0.0498 |
| baseline | 2000-2399  | 17406 | 0.8002 | 0.1936 | 0.0450 |
| baseline | 2400+  | 962 | 0.7478 | 0.2521 | 0.1517 |
| baseline | <1200  | 1034 | 1.8202 | 0.2669 | 0.2703 |
| wdl-a | 1200-1599  | 26529 | 0.5614 | 0.1857 | 0.0339 |
| wdl-a | 1600-1999  | 54069 | 0.5648 | 0.1827 | 0.0208 |
| wdl-a | 2000-2399  | 17406 | 0.5742 | 0.1808 | 0.0470 |
| wdl-a | 2400+  | 962 | 0.7154 | 0.2522 | 0.1688 |
| wdl-a | <1200  | 1034 | 0.5841 | 0.2004 | 0.1243 |

## By high_rating

| predictor | high_rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 2000-2199  | 12167 | 0.7812 | 0.2014 | 0.0802 |
| baseline | 2200-2399  | 5239 | 0.8443 | 0.1756 | 0.1042 |
| baseline | 2400-2599  | 852 | 0.7848 | 0.2654 | 0.2117 |
| baseline | 2600+  | 110 | 0.4609 | 0.1492 | 0.3537 |
| baseline | <2000  | 81632 | 0.8397 | 0.2068 | 0.0526 |
| wdl-a | 2000-2199  | 12167 | 0.6066 | 0.1983 | 0.0747 |
| wdl-a | 2200-2399  | 5239 | 0.4992 | 0.1403 | 0.0482 |
| wdl-a | 2400-2599  | 852 | 0.7757 | 0.2778 | 0.2184 |
| wdl-a | 2600+  | 110 | 0.2482 | 0.0540 | 0.2150 |
| wdl-a | <2000  | 81632 | 0.5639 | 0.1839 | 0.0177 |

## By phase

| predictor | phase | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | endgame  | 2269 | 2.4890 | 0.1026 | 0.1267 |
| baseline | middlegame  | 65259 | 0.8447 | 0.1926 | 0.0637 |
| baseline | opening  | 32472 | 0.6904 | 0.2370 | 0.0204 |
| wdl-a | endgame  | 2269 | 0.5011 | 0.0922 | 0.0574 |
| wdl-a | middlegame  | 65259 | 0.5428 | 0.1739 | 0.0189 |
| wdl-a | opening  | 32472 | 0.6209 | 0.2107 | 0.0096 |

## By clock

| predictor | clock | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | no-clock  | 100000 | 0.8319 | 0.2049 | 0.0500 |
| wdl-a | no-clock  | 100000 | 0.5672 | 0.1840 | 0.0126 |

## By rating_gap

| predictor | rating_gap | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 100-299  | 38539 | 0.7849 | 0.2034 | 0.0547 |
| baseline | 300+  | 12859 | 0.6744 | 0.1828 | 0.0555 |
| baseline | <100  | 48602 | 0.9108 | 0.2120 | 0.0616 |
| wdl-a | 100-299  | 38539 | 0.5604 | 0.1805 | 0.0283 |
| wdl-a | 300+  | 12859 | 0.3578 | 0.1080 | 0.0557 |
| wdl-a | <100  | 48602 | 0.6280 | 0.2069 | 0.0285 |

## By failure_mode

| predictor | failure_mode | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | absurd-refutation  | 1825 | 0.7861 | 0.1918 | 0.1825 |
| baseline | dead-draw-hard  | 41851 | 0.6894 | 0.2372 | 0.0113 |
| baseline | none  | 56324 | 0.9392 | 0.1814 | 0.0745 |
| wdl-a | absurd-refutation  | 1825 | 0.4803 | 0.1503 | 0.0559 |
| wdl-a | dead-draw-hard  | 41851 | 0.6405 | 0.2143 | 0.0169 |
| wdl-a | none  | 56324 | 0.5156 | 0.1626 | 0.0169 |

## Reliability curve (is the equity bar an honest probability?)

For each predicted-probability bin: mean predicted vs **observed** White expected-score, the bin's row count, and the gap (obs − pred). A calibrated predictor has `gap ≈ 0` in every bin; the count-weighted mean `|gap|` is the ECE.

### baseline  (n=100000, ECE=0.0500)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.023 | 0.145 | 10804 | +0.123 |
| 0.10 | 0.151 | 0.265 | 4079 | +0.114 |
| 0.20 | 0.251 | 0.328 | 4006 | +0.076 |
| 0.30 | 0.356 | 0.334 | 6090 | -0.022 |
| 0.40 | 0.462 | 0.431 | 17026 | -0.031 |
| 0.50 | 0.534 | 0.540 | 31797 | +0.007 |
| 0.60 | 0.644 | 0.639 | 6623 | -0.004 |
| 0.70 | 0.749 | 0.697 | 4139 | -0.052 |
| 0.80 | 0.850 | 0.775 | 4148 | -0.075 |
| 0.90 | 0.978 | 0.847 | 11288 | -0.131 |

### wdl-a  (n=100000, ECE=0.0126)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.070 | 0.062 | 3875 | -0.008 |
| 0.10 | 0.155 | 0.161 | 9680 | +0.006 |
| 0.20 | 0.247 | 0.249 | 10850 | +0.002 |
| 0.30 | 0.352 | 0.374 | 11529 | +0.022 |
| 0.40 | 0.452 | 0.472 | 13503 | +0.019 |
| 0.50 | 0.549 | 0.542 | 14976 | -0.006 |
| 0.60 | 0.648 | 0.658 | 12334 | +0.010 |
| 0.70 | 0.751 | 0.732 | 10552 | -0.018 |
| 0.80 | 0.848 | 0.862 | 9326 | +0.014 |
| 0.90 | 0.930 | 0.955 | 3375 | +0.024 |


## Head-to-head: where equity wins (baseline vs wdl-a)

Δ log-loss = `baseline` − `wdl-a` on the same rows; **Δ > 0 means equity wins** (lower model log-loss). Sorted by Δ, biggest win first.
Overall Δ: +0.2647
**Worst slice:** `failure_mode` `dead-draw-hard` (n=41851) Δ=+0.0490 — equity still wins every slice. Equity wins on 17/17 adequately-powered slices. 3 band(s) below n=1000 excluded as underpowered.

| slice | value | n | baseline log-loss | model log-loss | Δ |
|---|---|--:|--:|--:|--:|
| phase | endgame | 2269 | 2.4890 | 0.5011 | +1.9879 |
| rating | <1200 | 1034 | 1.8202 | 0.5841 | +1.2361 |
| failure_mode | none | 56324 | 0.9392 | 0.5156 | +0.4237 |
| rating | 1200-1599 | 26529 | 0.9567 | 0.5614 | +0.3953 |
| high_rating | 2200-2399 | 5239 | 0.8443 | 0.4992 | +0.3451 |
| rating_gap | 300+ | 12859 | 0.6744 | 0.3578 | +0.3166 |
| failure_mode | absurd-refutation | 1825 | 0.7861 | 0.4803 | +0.3059 |
| phase | middlegame | 65259 | 0.8447 | 0.5428 | +0.3019 |
| rating_gap | <100 | 48602 | 0.9108 | 0.6280 | +0.2828 |
| high_rating | <2000 | 81632 | 0.8397 | 0.5639 | +0.2757 |
| clock | no-clock | 100000 | 0.8319 | 0.5672 | +0.2647 |
| rating | 2000-2399 | 17406 | 0.8002 | 0.5742 | +0.2259 |
| rating_gap | 100-299 | 38539 | 0.7849 | 0.5604 | +0.2246 |
| high_rating | 2600+ | 110 | 0.4609 | 0.2482 | +0.2126 (underpowered) |
| rating | 1600-1999 | 54069 | 0.7634 | 0.5648 | +0.1987 |
| high_rating | 2000-2199 | 12167 | 0.7812 | 0.6066 | +0.1746 |
| phase | opening | 32472 | 0.6904 | 0.6209 | +0.0696 |
| failure_mode | dead-draw-hard | 41851 | 0.6894 | 0.6405 | +0.0490 |
| rating | 2400+ | 962 | 0.7478 | 0.7154 | +0.0324 (underpowered) |
| high_rating | 2400-2599 | 852 | 0.7848 | 0.7757 | +0.0091 (underpowered) |

## Significance vs baseline

Paired bootstrap (1000 resamples) on the per-row metric delta vs `baseline`. **Negative delta = the model is better** (lower loss); a verdict of `beats` means the whole 95% CI sits below zero.

| model | metric | delta | 95% CI | verdict |
|---|---|--:|:--:|:--:|
| wdl-a | log_loss | -0.2647 | [-0.2778, -0.2514] | beats |
| wdl-a | brier | -0.0209 | [-0.0217, -0.0202] | beats |

## Calibration (ECE) confidence intervals

Bin-resampling bootstrap (1000 resamples) on ECE (**lower = better calibrated**); ECE has no per-row term, so rows are resampled and the binning recomputed each draw. A `beats` verdict means the whole ECE-delta 95% CI vs baseline sits below zero.

| predictor | ECE | 95% CI | Δ vs baseline | Δ 95% CI | verdict |
|---|--:|:--:|--:|:--:|:--:|
| baseline | 0.0500 | [0.0472, 0.0528] | — | — | — |
| wdl-a | 0.0126 | [0.0103, 0.0154] | -0.0374 | [-0.0409, -0.0334] | beats |

## Head-to-head significance: per-slice CIs (baseline vs wdl-a)

Paired bootstrap (1000 resamples) on the per-row log-loss delta *within each slice*. Δ = `baseline` − `wdl-a` (**Δ > 0 = equity wins**); `equity` means the whole 95% CI clears zero, so the band-level win is real and not small-n noise. Slices below n=30 read `small-n` (too few rows for a trustworthy CI). A band with fewer than n=1000 rows reads `underpowered` and is excluded from any per-band beats/loses claim — its own win or loss is small-n noise, not the thesis (e.g. a 2000-2399 band at n=415 can flip on a handful of games). Sorted by Δ, biggest win first.

| slice | value | n | Δ log-loss | 95% CI | verdict |
|---|---|--:|--:|:--:|:--:|
| phase | endgame | 2269 | +1.9879 | [+1.7631, +2.2384] | equity |
| rating | <1200 | 1034 | +1.2361 | [+0.9390, +1.5436] | equity |
| failure_mode | none | 56324 | +0.4237 | [+0.3998, +0.4479] | equity |
| rating | 1200-1599 | 26529 | +0.3953 | [+0.3634, +0.4280] | equity |
| high_rating | 2200-2399 | 5239 | +0.3451 | [+0.2915, +0.4015] | equity |
| rating_gap | 300+ | 12859 | +0.3166 | [+0.2833, +0.3491] | equity |
| failure_mode | absurd-refutation | 1825 | +0.3059 | [+0.2654, +0.3480] | equity |
| phase | middlegame | 65259 | +0.3019 | [+0.2832, +0.3220] | equity |
| rating_gap | <100 | 48602 | +0.2828 | [+0.2613, +0.3051] | equity |
| high_rating | <2000 | 81632 | +0.2757 | [+0.2608, +0.2906] | equity |
| clock | no-clock | 100000 | +0.2647 | [+0.2517, +0.2785] | equity |
| rating | 2000-2399 | 17406 | +0.2259 | [+0.1992, +0.2563] | equity |
| rating_gap | 100-299 | 38539 | +0.2246 | [+0.2076, +0.2436] | equity |
| high_rating | 2600+ | 110 | +0.2126 | [+0.1885, +0.2343] | underpowered (n=110) |
| rating | 1600-1999 | 54069 | +0.1987 | [+0.1839, +0.2140] | equity |
| high_rating | 2000-2199 | 12167 | +0.1746 | [+0.1414, +0.2105] | equity |
| phase | opening | 32472 | +0.0696 | [+0.0634, +0.0770] | equity |
| failure_mode | dead-draw-hard | 41851 | +0.0490 | [+0.0465, +0.0513] | equity |
| rating | 2400+ | 962 | +0.0324 | [-0.0150, +0.1014] | underpowered (n=962) |
| high_rating | 2400-2599 | 852 | +0.0091 | [-0.0445, +0.0828] | underpowered (n=852) |
## By time-control bucket: does equity still beat centipawns? (baseline vs wdl-a)

Δ = `wdl-a` − `baseline` on each bucket's rows; **Δ < 0 means equity wins** (lower loss). `beats` = both log-loss and Brier deltas are negative; `worse` = both positive; `mixed` = the two metrics disagree. A bucket with fewer than n=1000 rows reads `underpowered` and is excluded from any beats/loses claim — its win or loss is small-n noise, not the thesis. Sorted by Δ log-loss, biggest equity win first.
Equity beats the baseline on 3/4 adequately-powered time-control bucket(s). 1 bucket(s) below n=1000 excluded as underpowered.

| time control | n | Δ log-loss | Δ Brier | verdict |
|---|--:|--:|--:|:--:|
| bullet | 36930 | -0.3857 | -0.0406 | beats |
| blitz | 30171 | -0.2617 | -0.0177 | beats |
| rapid | 29476 | -0.1403 | -0.0052 | beats |
| classical | 3128 | -0.1006 | +0.0166 | mixed |
| correspondence | 295 | +0.4023 | +0.1503 | underpowered (n=295) |
## Good moves read as good (move-level Δequity, task 0117)

Per consecutive ply-pair, the engine's cp swing (mover POV, clamped ±1000) is the ground-truth move quality. **Good** = mover lost ≤10cp (engine-approved); **blunder** = dropped ≥100cp. `Δgood`/`Δblunder` are the mean mover-POV equity swing (pp) the bar showed on each. The thesis: good moves should read as a *positive* gain, not a saturated ~0.

`sign-acc` (direction on |cp|≥25cp moves) is a sanity floor — any monotone-in-cp bar scores ~1.0. `corr` is **baseline-biased** (the cp swing is the baseline's own input), shown for transparency, not as the win condition.

| predictor | moves | good | blunder | sign-acc | Δgood (pp) | Δblunder (pp) | corr |
|---|--:|--:|--:|:--:|--:|--:|--:|
| baseline | 98374 | 49743 | 14524 | 1.000 | -0.01 | -17.52 | +0.939 |
| wdl-a | 98374 | 49743 | 14524 | 0.999 | +0.00 | -12.00 | +0.795 |

**Direction:** every bar reads engine-approved moves above blunders (Δgood > Δblunder) — good moves read as good, not as bad. ✅

**Rating signal:** every rating-conditioned bar reads blunders as less catastrophic than the rating-blind baseline (Δblunder -17.52pp) — a refutation a rating-peer won't find is discounted. (With cp-delta as ground truth the cp-based baseline is strong by construction; the good-move *upside* needs Maia's rating-relative policy — task 0008/0005.)

See [`reports/goodmoves_real.md`](reports/goodmoves_real.md) for the fuller move-level write-up — the reproduce recipe, the rating-signal (blunder-leniency) read, and what this slice proves and does *not* prove (the rating-conditioned good-move upside needs Maia's policy — task 0008/0005).

## Cutoff-robustness sweep (good × blunder grid, task 0157)

The good/blunder cutoffs above (≤10cp / ≥100cp) are arbitrary defaults, so the headline `Δgood > Δblunder` direction is re-measured across a grid of good cutoffs × blunder cutoffs. `holds` is `Δgood > Δblunder` in that cell. `sign-acc` depends only on the decisive-cp threshold (not the good/blunder cutoffs), so it is constant across the grid and shown once per predictor.

**`baseline`** — sign-acc 1.000 (|cp|≥25cp, grid-invariant)

| good ≤ | blunder ≥ | good | blunder | Δgood (pp) | Δblunder (pp) | holds |
|--:|--:|--:|--:|--:|--:|:--:|
| 5cp | 75cp | 41447 | 18155 | +0.11 | -15.28 | ✅ |
| 5cp | 100cp | 41447 | 14524 | +0.11 | -17.52 | ✅ |
| 5cp | 150cp | 41447 | 10344 | +0.11 | -21.19 | ✅ |
| 10cp | 75cp | 49743 | 18155 | -0.01 | -15.28 | ✅ |
| 10cp | 100cp | 49743 | 14524 | -0.01 | -17.52 | ✅ |
| 10cp | 150cp | 49743 | 10344 | -0.01 | -21.19 | ✅ |
| 20cp | 75cp | 59892 | 18155 | -0.21 | -15.28 | ✅ |
| 20cp | 100cp | 59892 | 14524 | -0.21 | -17.52 | ✅ |
| 20cp | 150cp | 59892 | 10344 | -0.21 | -21.19 | ✅ |

**`wdl-a`** — sign-acc 0.999 (|cp|≥25cp, grid-invariant)

| good ≤ | blunder ≥ | good | blunder | Δgood (pp) | Δblunder (pp) | holds |
|--:|--:|--:|--:|--:|--:|:--:|
| 5cp | 75cp | 41447 | 18155 | +0.13 | -10.72 | ✅ |
| 5cp | 100cp | 41447 | 14524 | +0.13 | -12.00 | ✅ |
| 5cp | 150cp | 41447 | 10344 | +0.13 | -14.02 | ✅ |
| 10cp | 75cp | 49743 | 18155 | +0.00 | -10.72 | ✅ |
| 10cp | 100cp | 49743 | 14524 | +0.00 | -12.00 | ✅ |
| 10cp | 150cp | 49743 | 10344 | +0.00 | -14.02 | ✅ |
| 20cp | 75cp | 59892 | 18155 | -0.20 | -10.72 | ✅ |
| 20cp | 100cp | 59892 | 14524 | -0.20 | -12.00 | ✅ |
| 20cp | 150cp | 59892 | 10344 | -0.20 | -14.02 | ✅ |

**Cutoff-robust:** `Δgood > Δblunder` holds in all 9 cells of the good × blunder grid for `baseline`, `wdl-a` — the direction is not an artifact of the default cutoffs. ✅

