# Validation report — data/sample/dataset.csv

> ⚠️ **SMOKE TEST, NOT EVIDENCE.** Generated from the 15-row `data/sample/` fixture
> to show the harness output format end-to-end. The numbers are meaningless at this
> size. `baseline` (rating-blind) and `baseline+clock` are wired; `wdl-a` (Approach A,
> task 0004) is now registered too — but its committed artifact was *trained on these
> very 15 rows*, so its apparent edge here is memorization, doubly meaningless. Real
> headline evidence needs a real dataset (task 0024), a held-out split (task 0030), and
> the Maia-2 value head (0005) compared against `wdl-a`. Regenerate with
> `chess-equity validate --data <dataset> --models baseline,wdl-a --out reports/<name>.md`.

Metric = predicting White expected-score (P(win)+0.5·P(draw)) vs actual result.
**Lower is better** for all three (log-loss, Brier, ECE).

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

