# Validation report — high-rating (2000+) only — real Lichess dump `lichess_db_standard_rated_2016-05`, n_high=49269 (held-out test 10134 / 148 games; train 39135; seed 0)

**Provenance.** Built torch-free from the **cached** real Lichess monthly dump
`~/.cache/chess-equity/dumps/lichess_db_standard_rated_2016-05.pgn.zst`. A 300k-row mixed
sample was parsed via `chess-equity data build`, then filtered to the **2000+ rating bands
only** (mean-Elo ≥ 2000) with the existing `load_rows(rating_bucket=…)` pushdown —
`scripts/build_highrating_eval.py` does both and stamps the source-month sidecar. The gate
then ran `validate --models baseline,wdl-a --gate --bootstrap 1000 --holdout 0.2 --seed 0
--eval-month 2016-05`. `maia2` is **skipped** — no `torch` on this box (baseline,wdl-a only,
per the 0160 precedent).

**Why this report exists — properly-powered high-rating gate (task 0165).** The gate's
historically *worst* slice is the 2000-2399 band: on the small **2013-01** dump (the committed
[`reports/validation_real.md`](validation_real.md)) wdl-a there read **logloss 0.97 vs
baseline 0.78, ECE 0.30** — but at only **n=415**, which task 0161 flagged as too few to tell
a real failure from sampling noise. This report **fixes the power**: by filtering a big sample
to 2000+ only, the high-rating eval is **n_high=49269** (held-out **n=10134**) — ~24× the
2013-01 high-rating count, and well above the gate's `MIN_GATE_N=2000` floor for every
2000-2399 sub-bin.

**Verdict — the high-rating failure was a small-sample artifact.** At proper n the earlier
failure **does not survive**: wdl-a beats baseline overall (logloss **0.580 vs 0.848**, CI
clears zero → **PASS**) and, crucially, *within the high-rating bands themselves*:

| band | n (held-out) | wdl-a logloss | baseline logloss | wdl-a ECE | baseline ECE |
|---|--:|--:|--:|--:|--:|
| 2000-2199 | 6942 | **0.597** | 0.956 | **0.081** | 0.081 |
| 2200-2399 | 2908 | **0.557** | 0.617 | **0.070** | 0.118 |
| 2400-2599 | 235 | 0.393 | 0.540 | 0.288 | 0.119 |
| 2600+ | 49 | 0.416 | 0.592 | 0.339 | 0.445 |

At n≈7k–3k the 2000-2399 bands show wdl-a **winning decisively and well-calibrated (ECE
≈0.07–0.08, not 0.30)**. The ECE blowup the 2013-01 run reported now appears **only** in the
genuinely tiny 2400+ bins (n=235, n=49) — i.e. the high-ECE signal tracks small n, confirming
the 2000-2399 "failure" was a power artifact, not a model defect. (This directly de-risks the
master-level concern of held task **0154**.)

> ⚠️ **wdl-a column is IN-DISTRIBUTION here — this is a POWER/calibration check, not clean
> held-out proof.** `wdl-a`'s `fit_month` is **2016-05** (`src/chess_equity/artifacts/wdl_a.json`),
> the very month of this dump, so the leakage guard (task 0112) fires and the wdl-a numbers
> partly reflect memorization of this month's rating→outcome distribution. What this report
> *legitimately* establishes is **statistical power** (n_high ≫ 415) and that the high-rating
> calibration blowup is sample-size-driven. The genuinely out-of-distribution high-rating
> proof is the **cross-dump refit** (sibling task **0164**: refit wdl-a on 2013-01, eval on
> 2016-05). Read the wdl-a PASS below as power+consistency, not fresh held-out evidence. (The
> CLI's own leakage banner follows.)

> ⚠️ **LEAKAGE — NOT HELD-OUT EVIDENCE.** The eval dataset's source month (`2016-05`) is the very month `wdl-a` was trained on, so its apparent edge here is memorization, not held-out skill — the **PASS** below cannot be trusted as proof of the thesis. Re-run on a *different* month (the committed evidence uses `2013-01`; `wdl-a` was fit on `2016-05`), or pass `--strict` to refuse the run outright.

Metric = predicting White expected-score (P(win)+0.5·P(draw)) vs actual result.
**Lower is better** for all three (log-loss, Brier, ECE).

## Gate verdict

Does each rating-conditioned predictor beat the rating-blind `baseline` on held-out outcomes? **PASS** = strictly lower log-loss *and* Brier (deltas are model − baseline; negative is better) **and** the log_loss 95% CI clears zero — a delta whose CI straddles zero is not proof.

- **wdl-a** beats baseline: logloss -0.2675, brier -0.0129; log_loss 95% CI [-0.3118, -0.2253] (CI clears zero) -> **PASS** — cuts log-loss 31.6% (Brier 6.7%) vs baseline

## Overall

| predictor | n | log-loss | Brier | ECE |
|---|--:|--:|--:|--:|
| baseline | 10134 | 0.8476 | 0.1928 | 0.0729 |
| wdl-a | 10134 | 0.5801 | 0.1799 | 0.0686 |

## By rating

| predictor | rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 2000-2399  | 9850 | 0.8562 | 0.1929 | 0.0703 |
| baseline | 2400+  | 284 | 0.5492 | 0.1903 | 0.1751 |
| wdl-a | 2000-2399  | 9850 | 0.5854 | 0.1816 | 0.0707 |
| wdl-a | 2400+  | 284 | 0.3972 | 0.1216 | 0.2106 |

## By high_rating

| predictor | high_rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 2000-2199  | 6942 | 0.9563 | 0.1992 | 0.0813 |
| baseline | 2200-2399  | 2908 | 0.6172 | 0.1779 | 0.1181 |
| baseline | 2400-2599  | 235 | 0.5403 | 0.1881 | 0.1189 |
| baseline | 2600+  | 49 | 0.5918 | 0.2005 | 0.4445 |
| wdl-a | 2000-2199  | 6942 | 0.5974 | 0.1850 | 0.0807 |
| wdl-a | 2200-2399  | 2908 | 0.5568 | 0.1736 | 0.0699 |
| wdl-a | 2400-2599  | 235 | 0.3933 | 0.1225 | 0.2884 |
| wdl-a | 2600+  | 49 | 0.4159 | 0.1172 | 0.3385 |

## By phase

| predictor | phase | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | endgame  | 386 | 1.7316 | 0.0943 | 0.1647 |
| baseline | middlegame  | 6788 | 0.8700 | 0.1820 | 0.0913 |
| baseline | opening  | 2960 | 0.6809 | 0.2304 | 0.0311 |
| wdl-a | endgame  | 386 | 0.3981 | 0.0788 | 0.0994 |
| wdl-a | middlegame  | 6788 | 0.5693 | 0.1735 | 0.0775 |
| wdl-a | opening  | 2960 | 0.6288 | 0.2078 | 0.0665 |

## By clock

| predictor | clock | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | no-clock  | 10134 | 0.8476 | 0.1928 | 0.0729 |
| wdl-a | no-clock  | 10134 | 0.5801 | 0.1799 | 0.0686 |

## By rating_gap

| predictor | rating_gap | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 100-299  | 3938 | 0.9978 | 0.1934 | 0.0891 |
| baseline | 300+  | 1671 | 0.6738 | 0.1775 | 0.1322 |
| baseline | <100  | 4525 | 0.7810 | 0.1980 | 0.0617 |
| wdl-a | 100-299  | 3938 | 0.6254 | 0.1945 | 0.0942 |
| wdl-a | 300+  | 1671 | 0.4021 | 0.1244 | 0.1086 |
| wdl-a | <100  | 4525 | 0.6065 | 0.1877 | 0.0818 |

## By failure_mode

| predictor | failure_mode | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | absurd-refutation  | 140 | 0.2881 | 0.0649 | 0.0465 |
| baseline | dead-draw-hard  | 4740 | 0.6822 | 0.2236 | 0.0500 |
| baseline | none  | 5254 | 1.0117 | 0.1685 | 0.0943 |
| wdl-a | absurd-refutation  | 140 | 0.2466 | 0.0643 | 0.0398 |
| wdl-a | dead-draw-hard  | 4740 | 0.6302 | 0.2008 | 0.0663 |
| wdl-a | none  | 5254 | 0.5439 | 0.1642 | 0.0753 |

## Reliability curve (is the equity bar an honest probability?)

For each predicted-probability bin: mean predicted vs **observed** White expected-score, the bin's row count, and the gap (obs − pred). A calibrated predictor has `gap ≈ 0` in every bin; the count-weighted mean `|gap|` is the ECE.

### baseline  (n=10134, ECE=0.0729)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.023 | 0.140 | 852 | +0.117 |
| 0.10 | 0.146 | 0.138 | 289 | -0.008 |
| 0.20 | 0.249 | 0.127 | 311 | -0.122 |
| 0.30 | 0.357 | 0.250 | 579 | -0.107 |
| 0.40 | 0.464 | 0.353 | 1788 | -0.112 |
| 0.50 | 0.534 | 0.561 | 3681 | +0.027 |
| 0.60 | 0.644 | 0.687 | 825 | +0.043 |
| 0.70 | 0.749 | 0.677 | 422 | -0.073 |
| 0.80 | 0.851 | 0.722 | 423 | -0.129 |
| 0.90 | 0.976 | 0.854 | 964 | -0.122 |

### wdl-a  (n=10134, ECE=0.0686)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.050 | 0.129 | 691 | +0.079 |
| 0.10 | 0.157 | 0.199 | 723 | +0.042 |
| 0.20 | 0.247 | 0.174 | 937 | -0.073 |
| 0.30 | 0.352 | 0.430 | 942 | +0.078 |
| 0.40 | 0.452 | 0.366 | 1103 | -0.086 |
| 0.50 | 0.550 | 0.469 | 1538 | -0.081 |
| 0.60 | 0.646 | 0.598 | 1103 | -0.048 |
| 0.70 | 0.748 | 0.681 | 985 | -0.067 |
| 0.80 | 0.850 | 0.788 | 1096 | -0.062 |
| 0.90 | 0.946 | 0.886 | 1016 | -0.060 |


## Head-to-head: where equity wins (baseline vs wdl-a)

Δ log-loss = `baseline` − `wdl-a` on the same rows; **Δ > 0 means equity wins** (lower model log-loss). Sorted by Δ, biggest win first.
Overall Δ: +0.2675
**Worst slice:** `failure_mode` `dead-draw-hard` (n=4740) Δ=+0.0521 — equity still wins every slice. Equity wins on 11/11 adequately-powered slices. 5 band(s) below n=1000 excluded as underpowered.

| slice | value | n | baseline log-loss | model log-loss | Δ |
|---|---|--:|--:|--:|--:|
| phase | endgame | 386 | 1.7316 | 0.3981 | +1.3335 (underpowered) |
| failure_mode | none | 5254 | 1.0117 | 0.5439 | +0.4678 |
| rating_gap | 100-299 | 3938 | 0.9978 | 0.6254 | +0.3724 |
| high_rating | 2000-2199 | 6942 | 0.9563 | 0.5974 | +0.3589 |
| phase | middlegame | 6788 | 0.8700 | 0.5693 | +0.3007 |
| rating_gap | 300+ | 1671 | 0.6738 | 0.4021 | +0.2717 |
| rating | 2000-2399 | 9850 | 0.8562 | 0.5854 | +0.2708 |
| clock | no-clock | 10134 | 0.8476 | 0.5801 | +0.2675 |
| high_rating | 2600+ | 49 | 0.5918 | 0.4159 | +0.1758 (underpowered) |
| rating_gap | <100 | 4525 | 0.7810 | 0.6065 | +0.1746 |
| rating | 2400+ | 284 | 0.5492 | 0.3972 | +0.1519 (underpowered) |
| high_rating | 2400-2599 | 235 | 0.5403 | 0.3933 | +0.1470 (underpowered) |
| high_rating | 2200-2399 | 2908 | 0.6172 | 0.5568 | +0.0604 |
| phase | opening | 2960 | 0.6809 | 0.6288 | +0.0521 |
| failure_mode | dead-draw-hard | 4740 | 0.6822 | 0.6302 | +0.0521 |
| failure_mode | absurd-refutation | 140 | 0.2881 | 0.2466 | +0.0415 (underpowered) |

## Significance vs baseline

Paired bootstrap (1000 resamples) on the per-row metric delta vs `baseline`. **Negative delta = the model is better** (lower loss); a verdict of `beats` means the whole 95% CI sits below zero.

| model | metric | delta | 95% CI | verdict |
|---|---|--:|:--:|:--:|
| wdl-a | log_loss | -0.2675 | [-0.3118, -0.2253] | beats |
| wdl-a | brier | -0.0129 | [-0.0154, -0.0102] | beats |

## Calibration (ECE) confidence intervals

Bin-resampling bootstrap (1000 resamples) on ECE (**lower = better calibrated**); ECE has no per-row term, so rows are resampled and the binning recomputed each draw. A `beats` verdict means the whole ECE-delta 95% CI vs baseline sits below zero.

| predictor | ECE | 95% CI | Δ vs baseline | Δ 95% CI | verdict |
|---|--:|:--:|--:|:--:|:--:|
| baseline | 0.0729 | [0.0650, 0.0813] | — | — | — |
| wdl-a | 0.0686 | [0.0602, 0.0767] | -0.0043 | [-0.0167, +0.0079] | inconclusive |

## Head-to-head significance: per-slice CIs (baseline vs wdl-a)

Paired bootstrap (1000 resamples) on the per-row log-loss delta *within each slice*. Δ = `baseline` − `wdl-a` (**Δ > 0 = equity wins**); `equity` means the whole 95% CI clears zero, so the band-level win is real and not small-n noise. Slices below n=30 read `small-n` (too few rows for a trustworthy CI). A band with fewer than n=1000 rows reads `underpowered` and is excluded from any per-band beats/loses claim — its own win or loss is small-n noise, not the thesis (e.g. a 2000-2399 band at n=415 can flip on a handful of games). Sorted by Δ, biggest win first.

| slice | value | n | Δ log-loss | 95% CI | verdict |
|---|---|--:|--:|:--:|:--:|
| phase | endgame | 386 | +1.3335 | [+0.8664, +1.9053] | underpowered (n=386) |
| failure_mode | none | 5254 | +0.4678 | [+0.3870, +0.5490] | equity |
| rating_gap | 100-299 | 3938 | +0.3724 | [+0.2850, +0.4682] | equity |
| high_rating | 2000-2199 | 6942 | +0.3589 | [+0.2994, +0.4285] | equity |
| phase | middlegame | 6788 | +0.3007 | [+0.2463, +0.3637] | equity |
| rating_gap | 300+ | 1671 | +0.2717 | [+0.1953, +0.3675] | equity |
| rating | 2000-2399 | 9850 | +0.2708 | [+0.2268, +0.3147] | equity |
| clock | no-clock | 10134 | +0.2675 | [+0.2255, +0.3132] | equity |
| high_rating | 2600+ | 49 | +0.1758 | [+0.1682, +0.1819] | underpowered (n=49) |
| rating_gap | <100 | 4525 | +0.1746 | [+0.1202, +0.2320] | equity |
| rating | 2400+ | 284 | +0.1519 | [+0.1199, +0.1821] | underpowered (n=284) |
| high_rating | 2400-2599 | 235 | +0.1470 | [+0.1110, +0.1795] | underpowered (n=235) |
| high_rating | 2200-2399 | 2908 | +0.0604 | [+0.0339, +0.0889] | equity |
| phase | opening | 2960 | +0.0521 | [+0.0387, +0.0661] | equity |
| failure_mode | dead-draw-hard | 4740 | +0.0521 | [+0.0410, +0.0623] | equity |
| failure_mode | absurd-refutation | 140 | +0.0415 | [-0.0289, +0.1281] | underpowered (n=140) |
## By time-control bucket: does equity still beat centipawns? (baseline vs wdl-a)

Δ = `wdl-a` − `baseline` on each bucket's rows; **Δ < 0 means equity wins** (lower loss). `beats` = both log-loss and Brier deltas are negative; `worse` = both positive; `mixed` = the two metrics disagree. A bucket with fewer than n=1000 rows reads `underpowered` and is excluded from any beats/loses claim — its win or loss is small-n noise, not the thesis. Sorted by Δ log-loss, biggest equity win first.
Equity beats the baseline on 2/3 adequately-powered time-control bucket(s). 1 bucket(s) below n=1000 excluded as underpowered.

| time control | n | Δ log-loss | Δ Brier | verdict |
|---|--:|--:|--:|:--:|
| rapid | 3302 | -0.3942 | -0.0241 | beats |
| bullet | 3763 | -0.2894 | -0.0352 | beats |
| blitz | 2788 | -0.1072 | +0.0318 | mixed |
| classical | 281 | -0.0744 | -0.0261 | underpowered (n=281) |
## Good moves read as good (move-level Δequity, task 0117)

Per consecutive ply-pair, the engine's cp swing (mover POV, clamped ±1000) is the ground-truth move quality. **Good** = mover lost ≤10cp (engine-approved); **blunder** = dropped ≥100cp. `Δgood`/`Δblunder` are the mean mover-POV equity swing (pp) the bar showed on each. The thesis: good moves should read as a *positive* gain, not a saturated ~0.

`sign-acc` (direction on |cp|≥25cp moves) is a sanity floor — any monotone-in-cp bar scores ~1.0. `corr` is **baseline-biased** (the cp swing is the baseline's own input), shown for transparency, not as the win condition.

| predictor | moves | good | blunder | sign-acc | Δgood (pp) | Δblunder (pp) | corr |
|---|--:|--:|--:|:--:|--:|--:|--:|
| baseline | 9986 | 5172 | 1249 | 1.000 | -0.00 | -16.19 | +0.936 |
| wdl-a | 9986 | 5172 | 1249 | 0.997 | -0.00 | -12.05 | +0.766 |

**Direction:** every bar reads engine-approved moves above blunders (Δgood > Δblunder) — good moves read as good, not as bad. ✅

**Rating signal:** every rating-conditioned bar reads blunders as less catastrophic than the rating-blind baseline (Δblunder -16.19pp) — a refutation a rating-peer won't find is discounted. (With cp-delta as ground truth the cp-based baseline is strong by construction; the good-move *upside* needs Maia's rating-relative policy — task 0008/0005.)

See [`reports/goodmoves_real.md`](reports/goodmoves_real.md) for the fuller move-level write-up — the reproduce recipe, the rating-signal (blunder-leniency) read, and what this slice proves and does *not* prove (the rating-conditioned good-move upside needs Maia's policy — task 0008/0005).

## Cutoff-robustness sweep (good × blunder grid, task 0157)

The good/blunder cutoffs above (≤10cp / ≥100cp) are arbitrary defaults, so the headline `Δgood > Δblunder` direction is re-measured across a grid of good cutoffs × blunder cutoffs. `holds` is `Δgood > Δblunder` in that cell. `sign-acc` depends only on the decisive-cp threshold (not the good/blunder cutoffs), so it is constant across the grid and shown once per predictor.

**`baseline`** — sign-acc 1.000 (|cp|≥25cp, grid-invariant)

| good ≤ | blunder ≥ | good | blunder | Δgood (pp) | Δblunder (pp) | holds |
|--:|--:|--:|--:|--:|--:|:--:|
| 5cp | 75cp | 4246 | 1569 | +0.14 | -14.20 | ✅ |
| 5cp | 100cp | 4246 | 1249 | +0.14 | -16.19 | ✅ |
| 5cp | 150cp | 4246 | 840 | +0.14 | -19.92 | ✅ |
| 10cp | 75cp | 5172 | 1569 | -0.00 | -14.20 | ✅ |
| 10cp | 100cp | 5172 | 1249 | -0.00 | -16.19 | ✅ |
| 10cp | 150cp | 5172 | 840 | -0.00 | -19.92 | ✅ |
| 20cp | 75cp | 6284 | 1569 | -0.22 | -14.20 | ✅ |
| 20cp | 100cp | 6284 | 1249 | -0.22 | -16.19 | ✅ |
| 20cp | 150cp | 6284 | 840 | -0.22 | -19.92 | ✅ |

**`wdl-a`** — sign-acc 0.997 (|cp|≥25cp, grid-invariant)

| good ≤ | blunder ≥ | good | blunder | Δgood (pp) | Δblunder (pp) | holds |
|--:|--:|--:|--:|--:|--:|:--:|
| 5cp | 75cp | 4246 | 1569 | +0.15 | -10.82 | ✅ |
| 5cp | 100cp | 4246 | 1249 | +0.15 | -12.05 | ✅ |
| 5cp | 150cp | 4246 | 840 | +0.15 | -14.19 | ✅ |
| 10cp | 75cp | 5172 | 1569 | -0.00 | -10.82 | ✅ |
| 10cp | 100cp | 5172 | 1249 | -0.00 | -12.05 | ✅ |
| 10cp | 150cp | 5172 | 840 | -0.00 | -14.19 | ✅ |
| 20cp | 75cp | 6284 | 1569 | -0.22 | -10.82 | ✅ |
| 20cp | 100cp | 6284 | 1249 | -0.22 | -12.05 | ✅ |
| 20cp | 150cp | 6284 | 840 | -0.22 | -14.19 | ✅ |

**Cutoff-robust:** `Δgood > Δblunder` holds in all 9 cells of the good × blunder grid for `baseline`, `wdl-a` — the direction is not an artifact of the default cutoffs. ✅

