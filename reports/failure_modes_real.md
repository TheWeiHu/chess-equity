# Failure modes on REAL binned Lichess outcomes — lichess_db_standard_rated_2013-01, n=12000 (seed 0)

Each cell is **measured** from real game results: `measured` is the mean White score (1=win, 0.5=draw, 0=loss) of the games in that cp×rating cell, with its `n`. `baseline` is Lichess's rating-blind cp→Win%; `wdl-a` is the rating-conditioned predictor. The question is which prediction sits closer to what actually happened.

## All cells (cp bin × rating band)

| cp bin | rating | n | measured | baseline | wdl-a |
|---|---|--:|--:|--:|--:|
| <= -1000 | 1200-1599 | 432 | 0.154 | 0.004 | 0.240 |
| <= -1000 | 1600-1999 | 588 | 0.168 | 0.004 | 0.199 |
| <= -1000 | 2000-2399 | 35 | 0.029 | 0.002 | 0.512 |
| (-1000, -500] | 1200-1599 | 387 | 0.344 | 0.082 | 0.215 |
| (-1000, -500] | 1600-1999 | 489 | 0.307 | 0.072 | 0.241 |
| (-500, -200] | 1200-1599 | 376 | 0.372 | 0.232 | 0.279 |
| (-500, -200] | 1600-1999 | 530 | 0.359 | 0.230 | 0.282 |
| (-500, -200] | 2000-2399 | 33 | 0.273 | 0.253 | 0.529 |
| (-200, -75] | 1200-1599 | 364 | 0.514 | 0.385 | 0.396 |
| (-200, -75] | 1600-1999 | 429 | 0.339 | 0.382 | 0.377 |
| (-75, 75] | 1200-1599 | 1543 | 0.493 | 0.507 | 0.468 |
| (-75, 75] | 1600-1999 | 2873 | 0.524 | 0.507 | 0.516 |
| (-75, 75] | 2000-2399 | 270 | 0.548 | 0.504 | 0.553 |
| (75, 200] | 1200-1599 | 375 | 0.595 | 0.616 | 0.600 |
| (75, 200] | 1600-1999 | 721 | 0.522 | 0.618 | 0.652 |
| (75, 200] | 2000-2399 | 35 | 0.400 | 0.622 | 0.679 |
| (200, 500] | 1200-1599 | 280 | 0.773 | 0.755 | 0.751 |
| (200, 500] | 1600-1999 | 740 | 0.549 | 0.763 | 0.772 |
| (500, 1000] | 1200-1599 | 164 | 0.841 | 0.920 | 0.818 |
| (500, 1000] | 1600-1999 | 374 | 0.730 | 0.925 | 0.892 |
| > 1000 | 1200-1599 | 385 | 0.951 | 0.996 | 0.848 |
| > 1000 | 1600-1999 | 535 | 0.948 | 0.997 | 0.903 |

## Failure mode — "hard 0.00 isn't 50/50" (|cp| ≤ 75)

The rating-blind baseline reads a ≈0.00 eval as ≈0.50 by construction. These are the cells where that's tested against the real result.

| cp bin | rating | n | measured | baseline | wdl-a |
|---|---|--:|--:|--:|--:|
| (-75, 75] | 1200-1599 | 1543 | 0.493 | 0.507 | 0.468 |
| (-75, 75] | 1600-1999 | 2873 | 0.524 | 0.507 | 0.516 |
| (-75, 75] | 2000-2399 | 270 | 0.548 | 0.504 | 0.553 |

n-weighted mean |prediction − measured|: baseline 0.0177, wdl-a 0.0137 → **wdl-a tracks the measured rate closer** in the dead-draw band.

## Failure mode — "good moves read as good" (|cp| ≥ 1000)

Decisive positions the engine reads as near-won. Whether the *result* is near-won — and how much that hinges on who is playing — is the measured question.

| cp bin | rating | n | measured | baseline | wdl-a |
|---|---|--:|--:|--:|--:|
| <= -1000 | 1200-1599 | 432 | 0.154 | 0.004 | 0.240 |
| <= -1000 | 1600-1999 | 588 | 0.168 | 0.004 | 0.199 |
| <= -1000 | 2000-2399 | 35 | 0.029 | 0.002 | 0.512 |
| > 1000 | 1200-1599 | 385 | 0.951 | 0.996 | 0.848 |
| > 1000 | 1600-1999 | 535 | 0.948 | 0.997 | 0.903 |

n-weighted mean |prediction − measured|: baseline 0.1044, wdl-a 0.0688 → **wdl-a tracks the measured rate closer** in the decisive band.
