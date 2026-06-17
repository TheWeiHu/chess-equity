# Validation report — data/sample/dataset.csv

> ⚠️ **SMOKE TEST, NOT EVIDENCE.** Generated from the 15-row `data/sample/` fixture
> to show the harness output format end-to-end. The numbers are meaningless at this
> size and only the rating-blind `baseline` is wired up so far. Real headline
> evidence needs a real dataset (task 0024) and the rating-conditioned models
> (0004/0005) registered as predictors. Regenerate with
> `chess-equity validate --data <dataset> --out reports/<name>.md`.

Metric = predicting White expected-score (P(win)+0.5·P(draw)) vs actual result.
**Lower is better** for all three (log-loss, Brier, ECE).

## Overall

| predictor | n | log-loss | Brier | ECE |
|---|--:|--:|--:|--:|
| baseline | 15 | 0.5794 | 0.1060 | 0.2183 |

## By rating

| predictor | rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 1200-1599  | 9 | 0.5034 | 0.1766 | 0.3705 |
| baseline | 2000-2399  | 6 | 0.6934 | 0.0001 | 0.0100 |

## By phase

| predictor | phase | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | opening  | 15 | 0.5794 | 0.1060 | 0.2183 |

