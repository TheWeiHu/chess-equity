# Approach D — end-to-end rating-conditioned WDL net (task 0013) — real Lichess data

**Train:** `lichess_db_standard_rated_2016-05`, n=80,000 evaluated positions, `--with-fen`.
**Eval (this report):** `lichess_db_standard_rated_2013-01`, n=12,000, `--with-fen` — a
**different month** from training, so this is genuine held-out evidence, not memorization
(the same eval set `reports/validation_real.md` uses, so `baseline`/`wdl-a` numbers match it).

Reproduce:
```
chess-equity data build --pgn .../2016-05.pgn.zst --sample 80000 --with-fen --format parquet --out build_train
chess-equity data build --pgn .../2013-01.pgn.zst --sample 12000 --with-fen --format parquet --out build_eval
chess-equity train-net --data build_train/dataset.parquet --epochs 6 --train-month 2016-05   # dropout 0.4, wd 1e-3
chess-equity validate --data build_eval/dataset.parquet --models baseline,wdl-a,wdl-net --eval-month 2013-01
```

## Verdict: end-to-end-from-the-board is **not worth the complexity at this scale**

`wdl-net` predicts WDL straight from the board + ratings (no Stockfish at inference) — the
clean long-term goal. On real held-out data it **fails the 0009 gate**: log-loss 1.19 vs the
rating-blind `baseline` 0.90 and Approach A (`wdl-a`, regression on Stockfish `cp_eval`) 0.56.
It is *worse than the centipawn baseline it was meant to replace*, and badly miscalibrated
(ECE 0.25 vs baseline 0.08), even with dropout 0.4 + weight-decay 1e-3 reining in the
overfit (train log-loss 0.23 → test 1.19).

Why, honestly:
- **The eval is the signal.** `cp_eval` is a deep-search summary of the position; a small MLP
  re-deriving that from raw piece planes on 80k positions cannot compete with regressing on it.
- **Data scale.** Maia-2 makes end-to-end work with a 23.3M-param residual tower on *millions*
  of games; 80k positions + a 2-layer MLP overfit instead (concept-equity-bar: Maia-2's value
  head already *is* the working end-to-end estimator).
- **It does learn *some* rating conditioning** — in the 300+ Elo-gap band it nearly matches the
  baseline (0.67 vs 0.51) — but absolute calibration never gets there.

**Takeaway:** keep `wdl-a` / Maia-2's value head on the critical path; revisit end-to-end only
with Maia-2-scale data + architecture (deferred — see PR "what's deferred"). The scaffold
(`wdl_net.py`, `chess-equity train-net`, the `wdl-net` predictor) is in place to retry at scale.

---

Metric = predicting White expected-score (P(win)+0.5·P(draw)) vs actual result.
**Lower is better** for all three (log-loss, Brier, ECE).

## Gate verdict

Does each rating-conditioned predictor beat the rating-blind `baseline` on held-out outcomes? **PASS** = strictly lower log-loss *and* Brier (deltas are model − baseline; negative is better) **and** the log_loss 95% CI clears zero — a delta whose CI straddles zero is not proof.

- **wdl-a** beats baseline: logloss -0.3403, brier -0.0355; log_loss 95% CI [-0.3853, -0.2967] (CI clears zero) -> **PASS** — cuts log-loss 37.6% (Brier 17.0%) vs baseline
- **wdl-net** beats baseline: logloss +0.2835, brier +0.0807; log_loss 95% CI [+0.2322, +0.3363] (CI straddles zero) -> **FAIL**

## Overall

| predictor | n | log-loss | Brier | ECE |
|---|--:|--:|--:|--:|
| baseline | 12000 | 0.9046 | 0.2080 | 0.0799 |
| wdl-a | 12000 | 0.5643 | 0.1726 | 0.0486 |
| wdl-net | 12000 | 1.1881 | 0.2887 | 0.2485 |

## By rating

| predictor | rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 1200-1599  | 4306 | 0.8713 | 0.1979 | 0.0792 |
| baseline | 1600-1999  | 7279 | 0.9313 | 0.2127 | 0.0898 |
| baseline | 2000-2399  | 415 | 0.7815 | 0.2313 | 0.0854 |
| wdl-a | 1200-1599  | 4306 | 0.5672 | 0.1758 | 0.1090 |
| wdl-a | 1600-1999  | 7279 | 0.5394 | 0.1621 | 0.0404 |
| wdl-a | 2000-2399  | 415 | 0.9725 | 0.3230 | 0.3014 |
| wdl-net | 1200-1599  | 4306 | 1.2503 | 0.2974 | 0.2564 |
| wdl-net | 1600-1999  | 7279 | 1.0921 | 0.2743 | 0.2349 |
| wdl-net | 2000-2399  | 415 | 2.2251 | 0.4510 | 0.4323 |

## By high_rating

| predictor | high_rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 2000-2199  | 415 | 0.7815 | 0.2313 | 0.0854 |
| baseline | <2000  | 11585 | 0.9090 | 0.2072 | 0.0804 |
| wdl-a | 2000-2199  | 415 | 0.9725 | 0.3230 | 0.3014 |
| wdl-a | <2000  | 11585 | 0.5497 | 0.1672 | 0.0510 |
| wdl-net | 2000-2199  | 415 | 2.2251 | 0.4510 | 0.4323 |
| wdl-net | <2000  | 11585 | 1.1509 | 0.2829 | 0.2420 |

## By phase

| predictor | phase | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | endgame  | 739 | 0.6813 | 0.0721 | 0.1296 |
| baseline | middlegame  | 7853 | 1.0207 | 0.2100 | 0.1222 |
| baseline | opening  | 3408 | 0.6854 | 0.2330 | 0.0365 |
| wdl-a | endgame  | 739 | 0.4666 | 0.0655 | 0.1140 |
| wdl-a | middlegame  | 7853 | 0.5624 | 0.1745 | 0.0635 |
| wdl-a | opening  | 3408 | 0.5901 | 0.1915 | 0.0360 |
| wdl-net | endgame  | 739 | 1.0017 | 0.2037 | 0.2508 |
| wdl-net | middlegame  | 7853 | 1.3817 | 0.3176 | 0.2968 |
| wdl-net | opening  | 3408 | 0.7822 | 0.2406 | 0.1481 |

## By clock

| predictor | clock | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | no-clock  | 12000 | 0.9046 | 0.2080 | 0.0799 |
| wdl-a | no-clock  | 12000 | 0.5643 | 0.1726 | 0.0486 |
| wdl-net | no-clock  | 12000 | 1.1881 | 0.2887 | 0.2485 |

## By rating_gap

| predictor | rating_gap | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 100-299  | 6031 | 0.8269 | 0.1978 | 0.0774 |
| baseline | 300+  | 1613 | 0.5120 | 0.1717 | 0.0656 |
| baseline | <100  | 4356 | 1.1575 | 0.2356 | 0.1263 |
| wdl-a | 100-299  | 6031 | 0.5711 | 0.1698 | 0.0726 |
| wdl-a | 300+  | 1613 | 0.3141 | 0.0842 | 0.0663 |
| wdl-a | <100  | 4356 | 0.6476 | 0.2092 | 0.0611 |
| wdl-net | 100-299  | 6031 | 1.2070 | 0.2952 | 0.2678 |
| wdl-net | 300+  | 1613 | 0.6740 | 0.1267 | 0.0787 |
| wdl-net | <100  | 4356 | 1.3522 | 0.3397 | 0.2911 |

## By failure_mode

| predictor | failure_mode | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | absurd-refutation  | 262 | 0.9963 | 0.2463 | 0.2419 |
| baseline | dead-draw-hard  | 4697 | 0.6883 | 0.2236 | 0.0113 |
| baseline | none  | 7041 | 1.0455 | 0.1962 | 0.1220 |
| wdl-a | absurd-refutation  | 262 | 0.4976 | 0.1567 | 0.1658 |
| wdl-a | dead-draw-hard  | 4697 | 0.5897 | 0.1773 | 0.0626 |
| wdl-a | none  | 7041 | 0.5499 | 0.1700 | 0.0863 |
| wdl-net | absurd-refutation  | 262 | 1.4385 | 0.3296 | 0.3077 |
| wdl-net | dead-draw-hard  | 4697 | 1.0516 | 0.2707 | 0.2047 |
| wdl-net | none  | 7041 | 1.2698 | 0.2992 | 0.2757 |

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

### wdl-net  (n=12000, ECE=0.2485)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.029 | 0.365 | 2613 | +0.336 |
| 0.10 | 0.146 | 0.402 | 823 | +0.256 |
| 0.20 | 0.250 | 0.384 | 602 | +0.134 |
| 0.30 | 0.352 | 0.428 | 611 | +0.076 |
| 0.40 | 0.453 | 0.404 | 659 | -0.049 |
| 0.50 | 0.547 | 0.458 | 706 | -0.089 |
| 0.60 | 0.653 | 0.524 | 573 | -0.129 |
| 0.70 | 0.753 | 0.557 | 716 | -0.197 |
| 0.80 | 0.852 | 0.578 | 962 | -0.274 |
| 0.90 | 0.975 | 0.655 | 3735 | -0.320 |


## Head-to-head: where equity wins (baseline vs wdl-a)

Δ log-loss = `baseline` − `wdl-a` on the same rows; **Δ > 0 means equity wins** (lower model log-loss). Sorted by Δ, biggest win first.
Overall Δ: +0.3403
**Worst slice:** `phase` `opening` (n=3408) Δ=+0.0954 — equity still wins every slice. Equity wins on 11/11 adequately-powered slices. 4 band(s) below n=1000 excluded as underpowered.

| slice | value | n | baseline log-loss | model log-loss | Δ |
|---|---|--:|--:|--:|--:|
| rating_gap | <100 | 4356 | 1.1575 | 0.6476 | +0.5099 |
| failure_mode | absurd-refutation | 262 | 0.9963 | 0.4976 | +0.4987 (underpowered) |
| failure_mode | none | 7041 | 1.0455 | 0.5499 | +0.4956 |
| phase | middlegame | 7853 | 1.0207 | 0.5624 | +0.4584 |
| rating | 1600-1999 | 7279 | 0.9313 | 0.5394 | +0.3919 |
| high_rating | <2000 | 11585 | 0.9090 | 0.5497 | +0.3593 |
| clock | no-clock | 12000 | 0.9046 | 0.5643 | +0.3403 |
| rating | 1200-1599 | 4306 | 0.8713 | 0.5672 | +0.3042 |
| rating_gap | 100-299 | 6031 | 0.8269 | 0.5711 | +0.2558 |
| phase | endgame | 739 | 0.6813 | 0.4666 | +0.2146 (underpowered) |
| rating_gap | 300+ | 1613 | 0.5120 | 0.3141 | +0.1979 |
| failure_mode | dead-draw-hard | 4697 | 0.6883 | 0.5897 | +0.0986 |
| phase | opening | 3408 | 0.6854 | 0.5901 | +0.0954 |
| rating | 2000-2399 | 415 | 0.7815 | 0.9725 | -0.1910 (underpowered) |
| high_rating | 2000-2199 | 415 | 0.7815 | 0.9725 | -0.1910 (underpowered) |

## Significance vs baseline

Paired bootstrap (2000 resamples) on the per-row metric delta vs `baseline`. **Negative delta = the model is better** (lower loss); a verdict of `beats` means the whole 95% CI sits below zero.

| model | metric | delta | 95% CI | verdict |
|---|---|--:|:--:|:--:|
| wdl-a | log_loss | -0.3403 | [-0.3853, -0.2967] | beats |
| wdl-a | brier | -0.0355 | [-0.0382, -0.0326] | beats |
| wdl-net | log_loss | +0.2835 | [+0.2322, +0.3363] | worse |
| wdl-net | brier | +0.0807 | [+0.0743, +0.0876] | worse |

## Calibration (ECE) confidence intervals

Bin-resampling bootstrap (2000 resamples) on ECE (**lower = better calibrated**); ECE has no per-row term, so rows are resampled and the binning recomputed each draw. A `beats` verdict means the whole ECE-delta 95% CI vs baseline sits below zero.

| predictor | ECE | 95% CI | Δ vs baseline | Δ 95% CI | verdict |
|---|--:|:--:|--:|:--:|:--:|
| baseline | 0.0799 | [0.0741, 0.0888] | — | — | — |
| wdl-a | 0.0486 | [0.0419, 0.0563] | -0.0314 | [-0.0421, -0.0231] | beats |
| wdl-net | 0.2485 | [0.2403, 0.2569] | +0.1686 | [+0.1558, +0.1780] | worse |

## Head-to-head significance: per-slice CIs (baseline vs wdl-a)

Paired bootstrap (2000 resamples) on the per-row log-loss delta *within each slice*. Δ = `baseline` − `wdl-a` (**Δ > 0 = equity wins**); `equity` means the whole 95% CI clears zero, so the band-level win is real and not small-n noise. Slices below n=30 read `small-n` (too few rows for a trustworthy CI). A band with fewer than n=1000 rows reads `underpowered` and is excluded from any per-band beats/loses claim — its own win or loss is small-n noise, not the thesis (e.g. a 2000-2399 band at n=415 can flip on a handful of games). Sorted by Δ, biggest win first.

| slice | value | n | Δ log-loss | 95% CI | verdict |
|---|---|--:|--:|:--:|:--:|
| rating_gap | <100 | 4356 | +0.5099 | [+0.4181, +0.6107] | equity |
| failure_mode | absurd-refutation | 262 | +0.4987 | [+0.3542, +0.6448] | underpowered (n=262) |
| failure_mode | none | 7041 | +0.4956 | [+0.4224, +0.5691] | equity |
| phase | middlegame | 7853 | +0.4584 | [+0.3941, +0.5269] | equity |
| rating | 1600-1999 | 7279 | +0.3919 | [+0.3360, +0.4498] | equity |
| high_rating | <2000 | 11585 | +0.3593 | [+0.3164, +0.4050] | equity |
| clock | no-clock | 12000 | +0.3403 | [+0.2977, +0.3833] | equity |
| rating | 1200-1599 | 4306 | +0.3042 | [+0.2364, +0.3704] | equity |
| rating_gap | 100-299 | 6031 | +0.2558 | [+0.2026, +0.3094] | equity |
| phase | endgame | 739 | +0.2146 | [+0.0820, +0.3663] | underpowered (n=739) |
| rating_gap | 300+ | 1613 | +0.1979 | [+0.1616, +0.2336] | equity |
| failure_mode | dead-draw-hard | 4697 | +0.0986 | [+0.0878, +0.1091] | equity |
| phase | opening | 3408 | +0.0954 | [+0.0783, +0.1175] | equity |
| rating | 2000-2399 | 415 | -0.1910 | [-0.3599, +0.0143] | underpowered (n=415) |
| high_rating | 2000-2199 | 415 | -0.1910 | [-0.3654, +0.0235] | underpowered (n=415) |
