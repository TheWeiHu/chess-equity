# Calibration by rating band — real Lichess 2013-01 (n=8000)

> Companion to **[reports/validation_real.md](validation_real.md)**: the rating-blind
> `baseline`'s per-rating-band reliability on the same 8,000 real `2013-01` positions,
> with a bin-resampling bootstrap 95% CI on each band's ECE.


Predictor **baseline** (rating-blind Lichess Win%) vs actual White result. A calibrated band has `mean_pred ≈ mean_obs` in every bin and ECE ≈ 0; the rating-blind baseline is fit on ~2300 play, so it should drift in the other bands (it can't see who is playing).

## ECE by rating band (lower = better calibrated)

Error bars are a bin-resampling bootstrap 95% CI on each band's ECE.

| rating band | n | log-loss | Brier | ECE | ECE 95% CI |
|---|--:|--:|--:|--:|:--:|
| 1200-1599 | 2866 | 1.0369 | 0.2198 | 0.1124 | [0.0980, 0.1296] |
| 1600-1999 | 4863 | 0.8456 | 0.2196 | 0.1100 | [0.0980, 0.1227] |
| 2000-2399 | 271 | 0.6170 | 0.2146 | 0.2523 | [0.2139, 0.3013] |

## Reliability curves (predicted vs observed White score)

### 1200-1599  (n=2866, ECE=0.1124)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.022 | 0.288 | 469 | +0.265 |
| 0.10 | 0.142 | 0.383 | 175 | +0.241 |
| 0.20 | 0.252 | 0.459 | 148 | +0.207 |
| 0.30 | 0.350 | 0.489 | 183 | +0.139 |
| 0.40 | 0.461 | 0.489 | 348 | +0.027 |
| 0.50 | 0.531 | 0.497 | 805 | -0.035 |
| 0.60 | 0.648 | 0.515 | 199 | -0.133 |
| 0.70 | 0.750 | 0.747 | 99 | -0.002 |
| 0.80 | 0.856 | 0.763 | 95 | -0.093 |
| 0.90 | 0.981 | 0.904 | 345 | -0.077 |

### 1600-1999  (n=4863, ECE=0.1100)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.032 | 0.284 | 566 | +0.252 |
| 0.10 | 0.151 | 0.346 | 205 | +0.196 |
| 0.20 | 0.245 | 0.300 | 175 | +0.055 |
| 0.30 | 0.357 | 0.239 | 234 | -0.118 |
| 0.40 | 0.463 | 0.487 | 706 | +0.025 |
| 0.50 | 0.529 | 0.555 | 1407 | +0.025 |
| 0.60 | 0.645 | 0.506 | 454 | -0.139 |
| 0.70 | 0.748 | 0.496 | 271 | -0.251 |
| 0.80 | 0.847 | 0.564 | 249 | -0.283 |
| 0.90 | 0.978 | 0.877 | 596 | -0.101 |

### 2000-2399  (n=271, ECE=0.2523)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.020 | 0.048 | 21 | +0.028 |
| 0.10 | 0.136 | 0.000 | 9 | -0.136 |
| 0.20 | 0.271 | 0.364 | 22 | +0.093 |
| 0.30 | 0.344 | 0.417 | 12 | +0.073 |
| 0.40 | 0.469 | 0.710 | 69 | +0.242 |
| 0.50 | 0.523 | 0.872 | 125 | +0.349 |
| 0.60 | 0.623 | 1.000 | 6 | +0.377 |
| 0.70 | 0.723 | 1.000 | 3 | +0.277 |
| 0.80 | 0.865 | 1.000 | 2 | +0.135 |
| 0.90 | 0.996 | 1.000 | 2 | +0.004 |

