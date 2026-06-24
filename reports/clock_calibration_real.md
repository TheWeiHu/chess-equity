# Clock flag-risk: per-time-control multiplier calibration (real data)

**Dump:** `lichess_db_standard_rated_2016-05`  ·  **games (n):** 6,225,957  ·  task 0268

Real time-forfeit rates by time control, used to set
`chess_equity.clock._TC_FLAG_MULTIPLIER`. The multiplier is the per-bucket
time-forfeit rate normalised so **bullet = 1.0** (the model's reference TC).

## Measured time-forfeit rate by time control

| time control | games | time forfeits | forfeit rate | current mult | measured mult |
| --- | ---: | ---: | ---: | ---: | ---: |
| bullet | 1,841,397 | 995,960 | 54.1% | 1.00 | 1.00 |
| blitz | 2,692,369 | 738,513 | 27.4% | 0.80 | 0.51 |
| rapid | 1,550,534 | 241,149 | 15.6% | 0.60 | 0.29 |
| classical | 116,441 | 16,743 | 14.4% | 0.50 | 0.27 |
| correspondence | 25,216 | 6,073 | 24.1% | 0.00 | 0.00 |

**Reading it:** the forfeit rate falls steeply from bullet to classical — the
same low clock is far more often fatal in bullet. Normalised to bullet=1.0 the
measured multipliers replace the hand-set guesses. `correspondence` stays pinned
at 0.0 by design (days per move -> no flag pressure), never derived from data.

## What is NOT calibrated here (blocked on a [%clk] dump)

`SCRAMBLE_SCALE` (the decay of `time_pressure` vs seconds remaining) and the
band split of `MAX_FLAG_RISK` / `FLAG_RISK_ALERT_THRESHOLD` need per-move
`[%clk]` clocks to fit by clock band. **The cached 2016-05 dump predates
`[%clk]`** (confirmed: zero `%clk` tags; see `clock_coverage.py`), so the
clock-band slice is degenerate on it. That half of the calibration needs a
>=2017-04 dump (a download) and is left for a follow-up — it cannot be done on
cached data and is not faked here (CLAUDE.md: real data only).

## Reproduce

```
uv run python scripts/calibrate_clock_tc.py \
    --dump ~/.cache/chess-equity/dumps/lichess_db_standard_rated_2016-05.pgn.zst \
    --out reports/clock_calibration_real.md
```
