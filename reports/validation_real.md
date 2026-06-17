# Validation report — real Lichess data (the headline thesis gate)

> **This is the real-data answer to the project's north-star gate** (task 0009 / 0087):
> does a rating-conditioned equity predictor beat the rating-blind centipawn baseline at
> predicting *actual* human outcomes? Unlike `reports/validation_sample.md` (a 15-row
> smoke fixture, explicitly *not* evidence), this report is computed on **n = 8,000 real
> Lichess positions** with the real Maia-2 value head and a real Stockfish-backed baseline.
>
> **Provenance / reproduce:**
> - **Data:** Lichess `2013-01` standard rated dump, `--with-fen`, `--sample 8000`
>   (`chess-equity data build --month 2013-01 --with-fen --sample 8000`).
> - **Out-of-sample:** `wdl-a` was fit on a *different* month (`2016-05`, 50k rows), so the
>   whole `2013-01` set is held out for it; `maia2` is externally pretrained and `baseline`
>   is fit-free — none of the three saw these rows.
> - **Command:** `chess-equity validate --data <dataset> --models baseline,wdl-a,maia2
>   --bootstrap 2000 --seed 0 --out reports/validation_real.md
>   --calibration reports/validation_real_calibration.md`
> - **Per-rating-band calibration** (with paired ECE deltas vs baseline) is in the
>   companion **[reports/validation_real_calibration.md](validation_real_calibration.md)**.
>
> **Bottom line:** both rating-conditioned predictors **PASS** — strictly lower log-loss
> *and* Brier than the baseline, with every paired-bootstrap 95% CI clearing zero. The
> biggest wins are in the **middlegame** and **1200-1599** band — exactly where the
> rating-blind centipawn bar is most wrong about how real games actually resolve.

Metric = predicting White expected-score (P(win)+0.5·P(draw)) vs actual result.
**Lower is better** for all three (log-loss, Brier, ECE).

## Gate verdict

Does each rating-conditioned predictor beat the rating-blind `baseline` on held-out outcomes? **PASS** = strictly lower log-loss *and* Brier (deltas are model − baseline; negative is better).

- **wdl-a** beats baseline: logloss -0.3255, brier -0.0383 -> **PASS**
- **maia2** beats baseline: logloss -0.2737, brier -0.0269 -> **PASS**

## Overall

| predictor | n | log-loss | Brier | ECE |
|---|--:|--:|--:|--:|
| baseline | 8000 | 0.9064 | 0.2195 | 0.1010 |
| wdl-a | 8000 | 0.5809 | 0.1813 | 0.0709 |
| maia2 | 8000 | 0.6327 | 0.1926 | 0.0682 |

## By rating

| predictor | rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 1200-1599  | 2866 | 1.0369 | 0.2198 | 0.1124 |
| baseline | 1600-1999  | 4863 | 0.8456 | 0.2196 | 0.1100 |
| baseline | 2000-2399  | 271 | 0.6170 | 0.2146 | 0.2523 |
| wdl-a | 1200-1599  | 2866 | 0.6072 | 0.1932 | 0.1422 |
| wdl-a | 1600-1999  | 4863 | 0.5628 | 0.1713 | 0.0601 |
| wdl-a | 2000-2399  | 271 | 0.6271 | 0.2335 | 0.3165 |
| maia2 | 1200-1599  | 2866 | 0.6025 | 0.1871 | 0.0822 |
| maia2 | 1600-1999  | 4863 | 0.6527 | 0.1951 | 0.0960 |
| maia2 | 2000-2399  | 271 | 0.5908 | 0.2073 | 0.2719 |

## By high_rating

| predictor | high_rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 2000-2199  | 271 | 0.6170 | 0.2146 | 0.2523 |
| baseline | <2000  | 7729 | 0.9165 | 0.2197 | 0.0971 |
| wdl-a | 2000-2199  | 271 | 0.6271 | 0.2335 | 0.3165 |
| wdl-a | <2000  | 7729 | 0.5793 | 0.1794 | 0.0795 |
| maia2 | 2000-2199  | 271 | 0.5908 | 0.2073 | 0.2719 |
| maia2 | <2000  | 7729 | 0.6341 | 0.1921 | 0.0781 |

## By phase

| predictor | phase | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | endgame  | 543 | 0.4721 | 0.0765 | 0.1357 |
| baseline | middlegame  | 5245 | 1.0363 | 0.2262 | 0.1466 |
| baseline | opening  | 2212 | 0.7049 | 0.2388 | 0.0405 |
| wdl-a | endgame  | 543 | 0.4244 | 0.0607 | 0.1338 |
| wdl-a | middlegame  | 5245 | 0.5844 | 0.1854 | 0.0846 |
| wdl-a | opening  | 2212 | 0.6109 | 0.2010 | 0.0737 |
| maia2 | endgame  | 543 | 0.3809 | 0.0548 | 0.1279 |
| maia2 | middlegame  | 5245 | 0.6635 | 0.2019 | 0.0923 |
| maia2 | opening  | 2212 | 0.6214 | 0.2044 | 0.0540 |

## By clock

| predictor | clock | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | no-clock  | 8000 | 0.9064 | 0.2195 | 0.1010 |
| wdl-a | no-clock  | 8000 | 0.5809 | 0.1813 | 0.0709 |
| maia2 | no-clock  | 8000 | 0.6327 | 0.1926 | 0.0682 |

## Head-to-head: where equity wins (baseline vs wdl-a)

Δ log-loss = `baseline` − `wdl-a` on the same rows; **Δ > 0 means equity wins** (lower model log-loss). Sorted by Δ, biggest win first.
Overall Δ: +0.3255

| slice | value | n | baseline log-loss | model log-loss | Δ |
|---|---|--:|--:|--:|--:|
| phase | middlegame | 5245 | 1.0363 | 0.5844 | +0.4519 |
| rating | 1200-1599 | 2866 | 1.0369 | 0.6072 | +0.4296 |
| high_rating | <2000 | 7729 | 0.9165 | 0.5793 | +0.3373 |
| clock | no-clock | 8000 | 0.9064 | 0.5809 | +0.3255 |
| rating | 1600-1999 | 4863 | 0.8456 | 0.5628 | +0.2829 |
| phase | opening | 2212 | 0.7049 | 0.6109 | +0.0940 |
| phase | endgame | 543 | 0.4721 | 0.4244 | +0.0477 |
| rating | 2000-2399 | 271 | 0.6170 | 0.6271 | -0.0101 |
| high_rating | 2000-2199 | 271 | 0.6170 | 0.6271 | -0.0101 |

## Significance vs baseline

Paired bootstrap (2000 resamples) on the per-row metric delta vs `baseline`. **Negative delta = the model is better** (lower loss); a verdict of `beats` means the whole 95% CI sits below zero.

| model | metric | delta | 95% CI | verdict |
|---|---|--:|:--:|:--:|
| wdl-a | log_loss | -0.3255 | [-0.3774, -0.2770] | beats |
| wdl-a | brier | -0.0383 | [-0.0418, -0.0345] | beats |
| maia2 | log_loss | -0.2737 | [-0.3207, -0.2295] | beats |
| maia2 | brier | -0.0269 | [-0.0298, -0.0240] | beats |

## Calibration (ECE) confidence intervals

Bin-resampling bootstrap (2000 resamples) on ECE (**lower = better calibrated**); ECE has no per-row term, so rows are resampled and the binning recomputed each draw. A `beats` verdict means the whole ECE-delta 95% CI vs baseline sits below zero.

| predictor | ECE | 95% CI | Δ vs baseline | Δ 95% CI | verdict |
|---|--:|:--:|--:|:--:|:--:|
| baseline | 0.1010 | [0.0921, 0.1114] | — | — | — |
| wdl-a | 0.0709 | [0.0631, 0.0810] | -0.0301 | [-0.0436, -0.0167] | beats |
| maia2 | 0.0682 | [0.0611, 0.0787] | -0.0327 | [-0.0439, -0.0196] | beats |

