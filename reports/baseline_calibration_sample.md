# Calibration by rating band — data/sample/dataset.csv

> ⚠️ **SMOKE TEST, NOT EVIDENCE.** Generated from the 15-row `data/sample/` fixture to
> show the task-0027 report format end-to-end. The per-band numbers are meaningless at
> this size (one game per band). Real evidence needs a real dataset (task 0024).
> Regenerate with `chess-equity validate --data <dataset> --models baseline --calibration reports/<name>.md`.

Predictor **baseline** (rating-blind Lichess Win%) vs actual White result. A calibrated band has `mean_pred ≈ mean_obs` in every bin and ECE ≈ 0; the rating-blind baseline is fit on ~2300 play, so it should drift in the other bands (it can't see who is playing).

## ECE by rating band (lower = better calibrated)

| rating band | n | log-loss | Brier | ECE |
|---|--:|--:|--:|--:|
| 1200-1599 | 9 | 0.5034 | 0.1766 | 0.3705 |
| 2000-2399 | 6 | 0.6934 | 0.0001 | 0.0100 |

## Reliability curves (predicted vs observed White score)

### 1200-1599  (n=9, ECE=0.3705)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.00 | 0.000 | 0.000 | 1 | -0.000 |
| 0.40 | 0.459 | 0.000 | 2 | -0.459 |
| 0.50 | 0.517 | 1.000 | 5 | +0.483 |
| 0.90 | 1.000 | 1.000 | 1 | +0.000 |

### 2000-2399  (n=6, ECE=0.0100)

| pred ≥ | mean pred | mean obs | n | gap (obs−pred) |
|--:|--:|--:|--:|--:|
| 0.50 | 0.510 | 0.500 | 6 | -0.010 |

