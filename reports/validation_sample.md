# Validation report — data/sample/dataset.csv

> ⚠️ **SMOKE TEST, NOT EVIDENCE.** Generated from the 15-row `data/sample/dataset.csv`
> fixture to show the gate's output format end-to-end — the numbers are meaningless at
> this size, and `wdl-a`'s committed artifact was *trained on these very 15 rows*, so its
> apparent edge here is memorization, doubly meaningless. This file exists to make the
> thesis gate — rating-conditioned `wdl-a` vs the rating-blind `baseline` — a *visible,
> reviewable artifact*: the **PASS** verdict and the head-to-head "where equity wins"
> ranking. It does **not** claim the thesis is proven. Real headline evidence needs a real
> dataset (task 0024), a held-out split (task 0030), and the Maia-2 value head (0005)
> compared against `wdl-a`.
>
> Regenerate with:
> `chess-equity validate --data data/sample/dataset.csv --models baseline,baseline+clock,wdl-a --out reports/validation_sample.md`

Metric = predicting White expected-score (P(win)+0.5·P(draw)) vs actual result.
**Lower is better** for all three (log-loss, Brier, ECE).

## Gate verdict

Does each rating-conditioned predictor beat the rating-blind `baseline` on held-out outcomes? **PASS** = strictly lower log-loss *and* Brier (deltas are model − baseline; negative is better).

- **baseline+clock** beats baseline: logloss +0.0073, brier +0.0011 -> **FAIL**
- **wdl-a** beats baseline: logloss -0.0830, brier -0.0486 -> **PASS**

## Overall

| predictor | n | log-loss | Brier | ECE |
|---|--:|--:|--:|--:|
| baseline | 15 | 0.5794 | 0.1060 | 0.2183 |
| baseline+clock | 15 | 0.5867 | 0.1072 | 0.1570 |
| wdl-a | 15 | 0.4964 | 0.0574 | 0.1961 |

## By rating

| predictor | rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 1200-1599  | 9 | 0.5034 | 0.1766 | 0.3705 |
| baseline | 2000-2399  | 6 | 0.6934 | 0.0001 | 0.0100 |
| baseline+clock | 1200-1599  | 9 | 0.5156 | 0.1785 | 0.2683 |
| baseline+clock | 2000-2399  | 6 | 0.6934 | 0.0001 | 0.0100 |
| wdl-a | 1200-1599  | 9 | 0.3635 | 0.0948 | 0.3022 |
| wdl-a | 2000-2399  | 6 | 0.6959 | 0.0014 | 0.0368 |

## By high_rating

| predictor | high_rating | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | 2000-2199  | 6 | 0.6934 | 0.0001 | 0.0100 |
| baseline | <2000  | 9 | 0.5034 | 0.1766 | 0.3705 |
| baseline+clock | 2000-2199  | 6 | 0.6934 | 0.0001 | 0.0100 |
| baseline+clock | <2000  | 9 | 0.5156 | 0.1785 | 0.2683 |
| wdl-a | 2000-2199  | 6 | 0.6959 | 0.0014 | 0.0368 |
| wdl-a | <2000  | 9 | 0.3635 | 0.0948 | 0.3022 |

## By phase

| predictor | phase | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | opening  | 15 | 0.5794 | 0.1060 | 0.2183 |
| baseline+clock | opening  | 15 | 0.5867 | 0.1072 | 0.1570 |
| wdl-a | opening  | 15 | 0.4964 | 0.0574 | 0.1961 |

## By clock

| predictor | clock | n | log-loss | Brier | ECE |
|---|---|--:|--:|--:|--:|
| baseline | comfortable(60s+)  | 13 | 0.6152 | 0.1223 | 0.2530 |
| baseline | low(<60s)  | 1 | 0.0000 | 0.0000 | 0.0000 |
| baseline | no-clock  | 1 | 0.6935 | 0.0002 | 0.0138 |
| baseline+clock | comfortable(60s+)  | 13 | 0.6167 | 0.1230 | 0.1756 |
| baseline+clock | low(<60s)  | 1 | 0.0908 | 0.0075 | 0.0868 |
| baseline+clock | no-clock  | 1 | 0.6935 | 0.0002 | 0.0138 |
| wdl-a | comfortable(60s+)  | 13 | 0.4978 | 0.0616 | 0.2047 |
| wdl-a | low(<60s)  | 1 | 0.2793 | 0.0594 | 0.2437 |
| wdl-a | no-clock  | 1 | 0.6957 | 0.0013 | 0.0359 |

## Head-to-head: where equity wins (baseline vs wdl-a)

Δ log-loss = `baseline` − `wdl-a` on the same rows; **Δ > 0 means equity wins** (lower model log-loss). Sorted by Δ, biggest win first.
Overall Δ: +0.0830

| slice | value | n | baseline log-loss | model log-loss | Δ |
|---|---|--:|--:|--:|--:|
| rating | 1200-1599 | 9 | 0.5034 | 0.3635 | +0.1399 |
| high_rating | <2000 | 9 | 0.5034 | 0.3635 | +0.1399 |
| clock | comfortable(60s+) | 13 | 0.6152 | 0.4978 | +0.1174 |
| phase | opening | 15 | 0.5794 | 0.4964 | +0.0830 |
| clock | no-clock | 1 | 0.6935 | 0.6957 | -0.0022 |
| rating | 2000-2399 | 6 | 0.6934 | 0.6959 | -0.0025 |
| high_rating | 2000-2199 | 6 | 0.6934 | 0.6959 | -0.0025 |
| clock | low(<60s) | 1 | 0.0000 | 0.2793 | -0.2793 |

