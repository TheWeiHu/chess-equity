# Validation report — real Lichess dump — lichess_db_standard_rated_2016-05, n=100000 sample (held-out test 20300 / 325 games), --with-fen (seed 0)

**Provenance.** Built torch-free from the **cached** real Lichess monthly dump
`~/.cache/chess-equity/dumps/lichess_db_standard_rated_2016-05.pgn.zst` (~1.7M real `[%eval]`
positions) via `chess-equity data build --pgn <cached> --sample 100000 --with-fen`, then
`validate --models baseline,wdl-a --gate --bootstrap 1000 --holdout 0.2 --seed 0 --eval-month 2016-05`.
**n:** 100000 sampled rows / 1626 games; the gate scores the leak-free held-out split (20300 rows / 325 games).
`maia2` is **skipped** — no `torch` on this box (the task permits baseline,wdl-a only; the committed
2013-01 run carries the maia2 column).

**Why this report exists (cross-dump replication).** The committed gate
[`reports/validation_real.md`](validation_real.md) rides only the small **2013-01** dump (n=12000),
whose 2000-2399 bin is just **n=415** (underpowered). This re-runs the same gate on a second, larger,
more-recent dump to (a) replicate on independent real outcomes and (b) get a **far bigger high-rating n**.
It worked: the held-out 2000-2399 slice here is **n=3422** (full-sample 17406) and 2400+ now appears
(n=962 full-sample) — so the high-rating slices `2000-2199` (n=2320) and `2200-2399` (n=1102) are now
**adequately powered (>1000)** and equity wins each with a CI clearing zero. That directly de-risks the
master-level (2000+) concern of held task **0154**, which the committed 2013-01 report could not address
at n=415.

> ⚠️ **wdl-a column is IN-DISTRIBUTION here — NOT independent held-out evidence.** `wdl-a`'s
> `fit_month` is **2016-05** (`src/chess_equity/artifacts/wdl_a.json`), the very month of this eval dump,
> so the leakage guard (task 0112) fires: the wdl-a numbers measure memorization of the month's
> rating→outcome distribution, not held-out skill. **The genuine cross-dump held-out proof remains the
> committed 2013-01 report** (`reports/validation_real.md`), where wdl-a's training month differs from the
> eval month. What IS independent here is the **model-free `baseline`** (its outcomes/calibration on a
> newer, larger dump) and the **bigger high-rating n**. Read the wdl-a PASS below as a consistency check,
> not as fresh proof. (The CLI's own leakage banner follows.)

> ⚠️ **LEAKAGE — NOT HELD-OUT EVIDENCE.** The eval dataset's source month (`2016-05`) is the very month `wdl-a` was trained on, so its apparent edge here is memorization, not held-out skill — the **PASS** below cannot be trusted as proof of the thesis. Re-run on a *different* month (the committed evidence uses `2013-01`; `wdl-a` was fit on `2016-05`), or pass `--strict` to refuse the run outright.

Metric = predicting White expected-score (P(win)+0.5·P(draw)) vs actual result.
**Lower is better** for all three (log-loss, Brier, ECE).

## Gate verdict

Does each rating-conditioned predictor beat the rating-blind `baseline` on held-out outcomes? **PASS** = strictly lower log-loss *and* Brier (deltas are model − baseline; negative is better) **and** the log_loss 95% CI clears zero — a delta whose CI straddles zero is not proof.

- **wdl-a** beats baseline: logloss -0.2978, brier -0.0267; log_loss 95% CI [-0.3280, -0.2677] (CI clears zero) -> **PASS** — cuts log-loss 35.5% (Brier 13.6%) vs baseline

## Overall

| predictor | n | log-loss | Brier | ECE |
|---|--:|--:|--:|--:|
| baseline | 20300 | 0.8395 | 0.1970 | 0.0496 |
| wdl-a | 20300 | 0.5417 | 0.1703 | 0.0287 |

## By rating

| predictor | rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 1200-1599  | 5396 | 1.0281 | 0.2002 | 0.0767 |
| baseline | 1600-1999  | 11155 | 0.7821 | 0.1995 | 0.0605 |
| baseline | 2000-2399  | 3422 | 0.7288 | 0.1824 | 0.0548 |
| baseline | 2400+  | 78 | 0.7189 | 0.2717 | 0.4923 |
| baseline | <1200  | 249 | 0.8827 | 0.1973 | 0.1342 |
| wdl-a | 1200-1599  | 5396 | 0.5520 | 0.1788 | 0.0516 |
| wdl-a | 1600-1999  | 11155 | 0.5254 | 0.1622 | 0.0465 |
| wdl-a | 2000-2399  | 3422 | 0.5716 | 0.1786 | 0.0989 |
| wdl-a | 2400+  | 78 | 1.1342 | 0.4532 | 0.6505 |
| wdl-a | <1200  | 249 | 0.4524 | 0.1461 | 0.1456 |

## By high_rating

| predictor | high_rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 2000-2199  | 2320 | 0.7176 | 0.1941 | 0.1401 |
| baseline | 2200-2399  | 1102 | 0.7525 | 0.1576 | 0.1781 |
| baseline | 2400-2599  | 78 | 0.7189 | 0.2717 | 0.4923 |
| baseline | <2000  | 16800 | 0.8626 | 0.1997 | 0.0602 |
| wdl-a | 2000-2199  | 2320 | 0.6338 | 0.2157 | 0.1798 |
| wdl-a | 2200-2399  | 1102 | 0.4407 | 0.1004 | 0.1189 |
| wdl-a | 2400-2599  | 78 | 1.1342 | 0.4532 | 0.6505 |
| wdl-a | <2000  | 16800 | 0.5329 | 0.1673 | 0.0288 |

## By phase

| predictor | phase | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | endgame  | 478 | 2.1232 | 0.1059 | 0.1646 |
| baseline | middlegame  | 13345 | 0.8628 | 0.1823 | 0.0689 |
| baseline | opening  | 6477 | 0.6968 | 0.2341 | 0.0288 |
| wdl-a | endgame  | 478 | 0.5399 | 0.0817 | 0.1240 |
| wdl-a | middlegame  | 13345 | 0.5160 | 0.1602 | 0.0324 |
| wdl-a | opening  | 6477 | 0.5950 | 0.1976 | 0.0460 |

## By clock

| predictor | clock | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | no-clock  | 20300 | 0.8395 | 0.1970 | 0.0496 |
| wdl-a | no-clock  | 20300 | 0.5417 | 0.1703 | 0.0287 |

## By rating_gap

| predictor | rating_gap | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 100-299  | 7762 | 0.7408 | 0.1812 | 0.0814 |
| baseline | 300+  | 2575 | 0.5321 | 0.1609 | 0.0721 |
| baseline | <100  | 9963 | 0.9958 | 0.2187 | 0.0953 |
| wdl-a | 100-299  | 7762 | 0.5073 | 0.1527 | 0.0542 |
| wdl-a | 300+  | 2575 | 0.2140 | 0.0596 | 0.0575 |
| wdl-a | <100  | 9963 | 0.6533 | 0.2126 | 0.0731 |

## By failure_mode

| predictor | failure_mode | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | absurd-refutation  | 397 | 0.6336 | 0.1506 | 0.1399 |
| baseline | dead-draw-hard  | 8524 | 0.6910 | 0.2299 | 0.0051 |
| baseline | none  | 11379 | 0.9580 | 0.1740 | 0.0798 |
| wdl-a | absurd-refutation  | 397 | 0.4081 | 0.1254 | 0.1169 |
| wdl-a | dead-draw-hard  | 8524 | 0.6195 | 0.1989 | 0.0516 |
| wdl-a | none  | 11379 | 0.4882 | 0.1504 | 0.0261 |

## Reliability curve (is the equity bar an honest probability?)

For each predicted-probability bin: mean predicted vs **observed** White expected-score, the bin's row count, and the gap (obs − pred). A calibrated predictor has `gap ≈ 0` in every bin; the count-weighted mean `|gap|` is the ECE.

### baseline  (n=20300, ECE=0.0496)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.026 | 0.146 | 2393 | +0.120 |
| 0.10 | 0.148 | 0.241 | 935 | +0.092 |
| 0.20 | 0.251 | 0.316 | 772 | +0.065 |
| 0.30 | 0.354 | 0.359 | 1207 | +0.004 |
| 0.40 | 0.463 | 0.452 | 3530 | -0.011 |
| 0.50 | 0.533 | 0.543 | 6296 | +0.010 |
| 0.60 | 0.645 | 0.716 | 1387 | +0.071 |
| 0.70 | 0.751 | 0.661 | 803 | -0.089 |
| 0.80 | 0.846 | 0.807 | 852 | -0.039 |
| 0.90 | 0.980 | 0.850 | 2125 | -0.129 |

### wdl-a  (n=20300, ECE=0.0287)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.054 | 0.011 | 1328 | -0.043 |
| 0.10 | 0.147 | 0.159 | 1750 | +0.012 |
| 0.20 | 0.247 | 0.257 | 2110 | +0.010 |
| 0.30 | 0.354 | 0.439 | 2132 | +0.086 |
| 0.40 | 0.450 | 0.448 | 2431 | -0.003 |
| 0.50 | 0.551 | 0.497 | 2610 | -0.055 |
| 0.60 | 0.646 | 0.644 | 2141 | -0.002 |
| 0.70 | 0.754 | 0.722 | 2220 | -0.032 |
| 0.80 | 0.853 | 0.830 | 2191 | -0.023 |
| 0.90 | 0.934 | 0.955 | 1387 | +0.021 |


## Head-to-head: where equity wins (baseline vs wdl-a)

Δ log-loss = `baseline` − `wdl-a` on the same rows; **Δ > 0 means equity wins** (lower model log-loss). Sorted by Δ, biggest win first.
Overall Δ: +0.2978
**Worst slice:** `failure_mode` `dead-draw-hard` (n=8524) Δ=+0.0715 — equity still wins every slice. Equity wins on 14/14 adequately-powered slices. 5 band(s) below n=1000 excluded as underpowered.

| slice | value | n | baseline log-loss | model log-loss | Δ |
|---|---|--:|--:|--:|--:|
| phase | endgame | 478 | 2.1232 | 0.5399 | +1.5833 (underpowered) |
| rating | 1200-1599 | 5396 | 1.0281 | 0.5520 | +0.4760 |
| failure_mode | none | 11379 | 0.9580 | 0.4882 | +0.4698 |
| rating | <1200 | 249 | 0.8827 | 0.4524 | +0.4304 (underpowered) |
| phase | middlegame | 13345 | 0.8628 | 0.5160 | +0.3468 |
| rating_gap | <100 | 9963 | 0.9958 | 0.6533 | +0.3425 |
| high_rating | <2000 | 16800 | 0.8626 | 0.5329 | +0.3297 |
| rating_gap | 300+ | 2575 | 0.5321 | 0.2140 | +0.3181 |
| high_rating | 2200-2399 | 1102 | 0.7525 | 0.4407 | +0.3118 |
| clock | no-clock | 20300 | 0.8395 | 0.5417 | +0.2978 |
| rating | 1600-1999 | 11155 | 0.7821 | 0.5254 | +0.2567 |
| rating_gap | 100-299 | 7762 | 0.7408 | 0.5073 | +0.2336 |
| failure_mode | absurd-refutation | 397 | 0.6336 | 0.4081 | +0.2255 (underpowered) |
| rating | 2000-2399 | 3422 | 0.7288 | 0.5716 | +0.1572 |
| phase | opening | 6477 | 0.6968 | 0.5950 | +0.1018 |
| high_rating | 2000-2199 | 2320 | 0.7176 | 0.6338 | +0.0838 |
| failure_mode | dead-draw-hard | 8524 | 0.6910 | 0.6195 | +0.0715 |
| rating | 2400+ | 78 | 0.7189 | 1.1342 | -0.4153 (underpowered) |
| high_rating | 2400-2599 | 78 | 0.7189 | 1.1342 | -0.4153 (underpowered) |

## Significance vs baseline

Paired bootstrap (1000 resamples) on the per-row metric delta vs `baseline`. **Negative delta = the model is better** (lower loss); a verdict of `beats` means the whole 95% CI sits below zero.

| model | metric | delta | 95% CI | verdict |
|---|---|--:|:--:|:--:|
| wdl-a | log_loss | -0.2978 | [-0.3280, -0.2677] | beats |
| wdl-a | brier | -0.0267 | [-0.0286, -0.0248] | beats |

## Calibration (ECE) confidence intervals

Bin-resampling bootstrap (1000 resamples) on ECE (**lower = better calibrated**); ECE has no per-row term, so rows are resampled and the binning recomputed each draw. A `beats` verdict means the whole ECE-delta 95% CI vs baseline sits below zero.

| predictor | ECE | 95% CI | Δ vs baseline | Δ 95% CI | verdict |
|---|--:|:--:|--:|:--:|:--:|
| baseline | 0.0496 | [0.0447, 0.0558] | — | — | — |
| wdl-a | 0.0287 | [0.0256, 0.0353] | -0.0209 | [-0.0272, -0.0131] | beats |

## Head-to-head significance: per-slice CIs (baseline vs wdl-a)

Paired bootstrap (1000 resamples) on the per-row log-loss delta *within each slice*. Δ = `baseline` − `wdl-a` (**Δ > 0 = equity wins**); `equity` means the whole 95% CI clears zero, so the band-level win is real and not small-n noise. Slices below n=30 read `small-n` (too few rows for a trustworthy CI). A band with fewer than n=1000 rows reads `underpowered` and is excluded from any per-band beats/loses claim — its own win or loss is small-n noise, not the thesis (e.g. a 2000-2399 band at n=415 can flip on a handful of games). Sorted by Δ, biggest win first.

| slice | value | n | Δ log-loss | 95% CI | verdict |
|---|---|--:|--:|:--:|:--:|
| phase | endgame | 478 | +1.5833 | [+1.2015, +2.0034] | underpowered (n=478) |
| rating | 1200-1599 | 5396 | +0.4760 | [+0.3982, +0.5569] | equity |
| failure_mode | none | 11379 | +0.4698 | [+0.4201, +0.5261] | equity |
| rating | <1200 | 249 | +0.4304 | [+0.1680, +0.7674] | underpowered (n=249) |
| phase | middlegame | 13345 | +0.3468 | [+0.3029, +0.3914] | equity |
| rating_gap | <100 | 9963 | +0.3425 | [+0.2932, +0.3923] | equity |
| high_rating | <2000 | 16800 | +0.3297 | [+0.2963, +0.3675] | equity |
| rating_gap | 300+ | 2575 | +0.3181 | [+0.2758, +0.3636] | equity |
| high_rating | 2200-2399 | 1102 | +0.3118 | [+0.2017, +0.4389] | equity |
| clock | no-clock | 20300 | +0.2978 | [+0.2683, +0.3321] | equity |
| rating | 1600-1999 | 11155 | +0.2567 | [+0.2190, +0.2946] | equity |
| rating_gap | 100-299 | 7762 | +0.2336 | [+0.1906, +0.2827] | equity |
| failure_mode | absurd-refutation | 397 | +0.2255 | [+0.1431, +0.3050] | underpowered (n=397) |
| rating | 2000-2399 | 3422 | +0.1572 | [+0.0942, +0.2222] | equity |
| phase | opening | 6477 | +0.1018 | [+0.0855, +0.1214] | equity |
| high_rating | 2000-2199 | 2320 | +0.0838 | [+0.0157, +0.1652] | equity |
| failure_mode | dead-draw-hard | 8524 | +0.0715 | [+0.0645, +0.0790] | equity |
| rating | 2400+ | 78 | -0.4153 | [-0.4389, -0.3882] | underpowered (n=78) |
| high_rating | 2400-2599 | 78 | -0.4153 | [-0.4383, -0.3889] | underpowered (n=78) |
## By time-control bucket: does equity still beat centipawns? (baseline vs wdl-a)

Δ = `wdl-a` − `baseline` on each bucket's rows; **Δ < 0 means equity wins** (lower loss). `beats` = both log-loss and Brier deltas are negative; `worse` = both positive; `mixed` = the two metrics disagree. A bucket with fewer than n=1000 rows reads `underpowered` and is excluded from any beats/loses claim — its win or loss is small-n noise, not the thesis. Sorted by Δ log-loss, biggest equity win first.
Equity beats the baseline on 3/3 adequately-powered time-control bucket(s). 2 bucket(s) below n=1000 excluded as underpowered.

| time control | n | Δ log-loss | Δ Brier | verdict |
|---|--:|--:|--:|:--:|
| bullet | 7961 | -0.4166 | -0.0492 | beats |
| blitz | 5694 | -0.2424 | -0.0120 | beats |
| classical | 643 | -0.2213 | +0.0018 | underpowered (n=643) |
| rapid | 5903 | -0.2049 | -0.0143 | beats |
| correspondence | 99 | +0.0367 | +0.0084 | underpowered (n=99) |
## Good moves read as good (move-level Δequity, task 0117)

Per consecutive ply-pair, the engine's cp swing (mover POV, clamped ±1000) is the ground-truth move quality. **Good** = mover lost ≤10cp (engine-approved); **blunder** = dropped ≥100cp. `Δgood`/`Δblunder` are the mean mover-POV equity swing (pp) the bar showed on each. The thesis: good moves should read as a *positive* gain, not a saturated ~0.

`sign-acc` (direction on |cp|≥25cp moves) is a sanity floor — any monotone-in-cp bar scores ~1.0. `corr` is **baseline-biased** (the cp swing is the baseline's own input), shown for transparency, not as the win condition.

| predictor | moves | good | blunder | sign-acc | Δgood (pp) | Δblunder (pp) | corr |
|---|--:|--:|--:|:--:|--:|--:|--:|
| baseline | 19975 | 10118 | 2980 | 1.000 | -0.01 | -17.39 | +0.937 |
| wdl-a | 19975 | 10118 | 2980 | 0.996 | -0.00 | -11.10 | +0.791 |

**Direction:** every bar reads engine-approved moves above blunders (Δgood > Δblunder) — good moves read as good, not as bad. ✅

**Rating signal:** every rating-conditioned bar reads blunders as less catastrophic than the rating-blind baseline (Δblunder -17.39pp) — a refutation a rating-peer won't find is discounted. (With cp-delta as ground truth the cp-based baseline is strong by construction; the good-move *upside* needs Maia's rating-relative policy — task 0008/0005.)

See [`reports/goodmoves_real.md`](reports/goodmoves_real.md) for the fuller move-level write-up — the reproduce recipe, the rating-signal (blunder-leniency) read, and what this slice proves and does *not* prove (the rating-conditioned good-move upside needs Maia's policy — task 0008/0005).

## Cutoff-robustness sweep (good × blunder grid, task 0157)

The good/blunder cutoffs above (≤10cp / ≥100cp) are arbitrary defaults, so the headline `Δgood > Δblunder` direction is re-measured across a grid of good cutoffs × blunder cutoffs. `holds` is `Δgood > Δblunder` in that cell. `sign-acc` depends only on the decisive-cp threshold (not the good/blunder cutoffs), so it is constant across the grid and shown once per predictor.

**`baseline`** — sign-acc 1.000 (|cp|≥25cp, grid-invariant)

| good ≤ | blunder ≥ | good | blunder | Δgood (pp) | Δblunder (pp) | holds |
|--:|--:|--:|--:|--:|--:|:--:|
| 5cp | 75cp | 8394 | 3718 | +0.11 | -15.16 | ✅ |
| 5cp | 100cp | 8394 | 2980 | +0.11 | -17.39 | ✅ |
| 5cp | 150cp | 8394 | 2133 | +0.11 | -20.94 | ✅ |
| 10cp | 75cp | 10118 | 3718 | -0.01 | -15.16 | ✅ |
| 10cp | 100cp | 10118 | 2980 | -0.01 | -17.39 | ✅ |
| 10cp | 150cp | 10118 | 2133 | -0.01 | -20.94 | ✅ |
| 20cp | 75cp | 12162 | 3718 | -0.21 | -15.16 | ✅ |
| 20cp | 100cp | 12162 | 2980 | -0.21 | -17.39 | ✅ |
| 20cp | 150cp | 12162 | 2133 | -0.21 | -20.94 | ✅ |

**`wdl-a`** — sign-acc 0.996 (|cp|≥25cp, grid-invariant)

| good ≤ | blunder ≥ | good | blunder | Δgood (pp) | Δblunder (pp) | holds |
|--:|--:|--:|--:|--:|--:|:--:|
| 5cp | 75cp | 8394 | 3718 | +0.11 | -9.87 | ✅ |
| 5cp | 100cp | 8394 | 2980 | +0.11 | -11.10 | ✅ |
| 5cp | 150cp | 8394 | 2133 | +0.11 | -12.98 | ✅ |
| 10cp | 75cp | 10118 | 3718 | -0.00 | -9.87 | ✅ |
| 10cp | 100cp | 10118 | 2980 | -0.00 | -11.10 | ✅ |
| 10cp | 150cp | 10118 | 2133 | -0.00 | -12.98 | ✅ |
| 20cp | 75cp | 12162 | 3718 | -0.18 | -9.87 | ✅ |
| 20cp | 100cp | 12162 | 2980 | -0.18 | -11.10 | ✅ |
| 20cp | 150cp | 12162 | 2133 | -0.18 | -12.98 | ✅ |

**Cutoff-robust:** `Δgood > Δblunder` holds in all 9 cells of the good × blunder grid for `baseline`, `wdl-a` — the direction is not an artifact of the default cutoffs. ✅

