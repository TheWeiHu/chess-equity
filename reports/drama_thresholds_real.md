# Drama trigger thresholds — calibrated on the real Δequity distribution (task 0170)

**Provenance.** Built from the cached real Lichess monthly dump **`lichess_db_standard_rated_2016-05`**
via `chess-equity data build --month <YYYY-MM>` (cached under
`~/.cache/chess-equity/dumps/`). Per-move practical-equity swings computed by
`scripts/calibrate_drama_thresholds.py`. **n = 295,140 per-move transitions** over
**4,860 games**. Real positions, real Stockfish `cp_eval`, no synthetic data.

**Δequity definition.** Each position's White-POV equity is the rating-blind Lichess
logistic over the real `cp_eval` (`chess_equity.types.lichess_win_percent`, in [0,100] —
exactly `LichessBaselineModel`'s cp→equity map). The per-move Δequity is the **mover**'s
POV change across one half-move (after − before), matching `broadcast.GameTracker`.

## The key asymmetry (why clutch is calibrated on magnitude, not the positive tail)

Signed mover-POV Δequity (pts):

| quantile | Δ |
|---|---|
| p1 | -42.66 |
| p5 | -17.91 |
| p10 | -9.86 |
| p25 | -3.52 |
| p50 | -0.79 |
| p75 | +0.00 |
| p90 | +0.27 |
| p95 | +0.62 |
| p99 | +1.53 |

The distribution is heavily **negative-skewed**: under a rating-*blind* best-play eval the
mover can almost never raise their *own* eval — the eval already assumed their best reply —
so the positive side is bounded by eval depth-noise (p99 ≈ +1.5pt) while blunders form a
long negative tail. A clutch's *positive-swing* bar therefore **cannot** be set from the
positive-only quantiles (degenerate on the baseline). It is set on the **|Δequity| magnitude**
scale below — the same scale that becomes real on a rating-conditioned model (Maia-2), where
a below-rating-typical baseline lets a strong move genuinely lift practical equity. So on the
baseline these constants make the reel *quiet* (the documented "muted on baseline" behaviour);
on Maia-2 they fire at the calibrated rarity.

## |Δequity| magnitude distribution (the calibration basis)

| quantile | \|Δ\| (pts) |
|---|---|
| p50 | 0.92 |
| p75 | 3.56 |
| p80 | 4.74 |
| p85 | 6.59 |
| p90 | 9.87 |
| p95 | 17.92 |
| p97.5 | 28.17 |
| p99 | 42.67 |

## Derived thresholds (rounded to 0.5)

| constant | target tail | percentile of \|Δ\| | derived | realised rate on this dump |
|---|---|---|---|---|
| `SLIP_DELTA` | top 5% | p95 = 17.92 | **18.0** | missed_win 1.70%, escape 0.00% |
| `CLUTCH_DELTA` | top 10% | p90 = 9.87 | **10.0** | clutch 0.01% (positive-swing, ≈eval-noise on baseline) |
| `SCRAMBLE_DELTA` | top 15% | p85 = 6.59 | **6.5** | clock-gated; not measurable here (sparse clocks) |

`WIN_LEVEL` (70) / `LOSS_LEVEL` (30) are equity *levels*, not swing magnitudes, so the
Δequity distribution does not inform them — kept as principled round numbers ("practically
winning/losing"). `SCRAMBLE_SECS` (20) is a clock gate, unchanged.

Ordering holds: `SCRAMBLE_DELTA` (6.5) < `CLUTCH_DELTA` (10.0) < `SLIP_DELTA` (18.0) —
the scramble bar is the lowest (clock is the story), slip the highest (a "real" let-it-slip).

_Regenerate: `uv run --extra data python scripts/calibrate_drama_thresholds.py --data <parquet> --dump lichess_db_standard_rated_2016-05 --out reports/drama_thresholds_real.md`._
