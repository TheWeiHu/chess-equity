# Divergence — real Lichess dump `2013-01`, n=12000 (equity=`wdl-a` vs Stockfish=`baseline`)

How far the rating/clock-aware **equity bar** (`wdl-a`) departs from the rating-blind **Stockfish bar** (`baseline`, Lichess Win% of `cp_eval`) on the *same* real positions. Both are White-POV scalars in [0, 100]%. This measures product-visible *disagreement*, not predictive accuracy (no outcomes are read; for accuracy see `reports/validation_real.md`).

- **signed gap** = mean(equity − stockfish), pp. The direction the equity bar pulls vs the classic bar (＋ = more White-favorable).
- **|gap|** / **p90 |gap|** = mean and 90th-pct absolute gap, pp — the magnitude a viewer sees between the two bars.
- **rank-disagree** = among positions where both bars clear ±5pp of 50% (the rankable n in parens), the fraction where the two bars name *different favorites*.

## Overall

|  | n | signed gap (pp) | \|gap\| (pp) | p90 \|gap\| (pp) | rank-disagree (n) |
|---|--:|--:|--:|--:|--:|
| overall | 12000 | +2.51 | 15.40 | 31.18 |  8.7% (7329) |

## By time control

| time control | n | signed gap (pp) | \|gap\| (pp) | p90 \|gap\| (pp) | rank-disagree (n) |
|---|--:|--:|--:|--:|--:|
| blitz | 4266 | +6.73 | 14.57 | 28.87 |  5.6% (2574) |
| bullet | 4851 | +2.01 | 16.32 | 33.94 |  9.2% (3060) |
| classical | 77 | +7.24 | 8.93 | 15.11 |  1.7% (59) |
| correspondence | 75 | -16.02 | 16.59 | 32.83 | 12.3% (57) |
| rapid | 2731 | -2.79 | 15.22 | 32.37 | 12.8% (1579) |

## By rating band

| rating | n | signed gap (pp) | \|gap\| (pp) | p90 \|gap\| (pp) | rank-disagree (n) |
|---|--:|--:|--:|--:|--:|
| 1200-1599 | 4306 | +0.80 | 16.36 | 34.36 |  9.8% (2780) |
| 1600-1999 | 7279 | +2.98 | 14.28 | 28.62 |  7.0% (4396) |
| 2000-2399 | 415 | +12.16 | 25.12 | 50.42 | 37.9% (153) |

## By time control × rating

| tc × rating | n | signed gap (pp) | \|gap\| (pp) | p90 \|gap\| (pp) | rank-disagree (n) |
|---|--:|--:|--:|--:|--:|
| blitz × 1200-1599 | 1600 | +5.56 | 15.41 | 31.95 |  5.5% (995) |
| blitz × 1600-1999 | 2666 | +7.43 | 14.06 | 28.15 |  5.7% (1579) |
| bullet × 1200-1599 | 1337 | +2.94 | 15.02 | 37.65 |  3.6% (960) |
| bullet × 1600-1999 | 3099 | +0.24 | 15.70 | 30.50 |  9.8% (1947) |
| bullet × 2000-2399 | 415 | +12.16 | 25.12 | 50.42 | 37.9% (153) |
| classical × 1200-1599 | 43 | +8.62 | 11.49 | 14.98 |  2.4% (41) |
| classical × 1600-1999 | 34 | +5.51 | 5.70 | 19.82 |  0.0% (18) |
| correspondence × 1200-1599 | 75 | -16.02 | 16.59 | 32.83 | 12.3% (57) |
| rapid × 1200-1599 | 1251 | -6.83 | 19.15 | 35.72 | 23.9% (727) |
| rapid × 1600-1999 | 1480 | +0.62 | 11.89 | 22.26 |  3.3% (852) |

