# Clock-coverage audit — cached real Lichess dumps — 2013-01 (n=34,308 full) + 2016-05 (n=200,000 sample) (task 0267)

**Purpose.** A prerequisite for the clock-proof thread (tasks 0153 / 0250 / 0269): before
paying to download a newer Lichess dump to *prove* the clock dimension (`clock.py`,
flag-risk model), establish which cached dump — if any — actually carries `[%clk]` tags.
A dump with no clocks cannot validate the flag-risk model no matter how it is sliced.

**Method.** The model-free 0249 coverage diagnostic (`chess_equity/clock_coverage.py`),
invoked as `chess-equity validate --slice clock --data <built-dataset>`. It tallies the
fraction of parsed rows carrying a side-to-move clock (`clock_remaining is not None`) and
buckets them over `chess_equity.clock.clock_band` (edges 10/30/60/180s). Datasets were
built with `chess-equity data build --pgn <cached dump> --format csv` (real positions,
real `cp_eval` from the dump's `[%eval]` tags — no synthetic data). Stdlib-only, no
Stockfish, no model.

**Dumps audited** (cached under `~/.cache/chess-equity/dumps/`):

| dump | size (zst) | rows built | sampling |
|---|--:|--:|---|
| `lichess_db_standard_rated_2013-01` | 17 MB | 34,308 | full dump |
| `lichess_db_standard_rated_2016-05` | 1.1 GB | 200,000 | `--sample 200000` (head) |

The 2016-05 dump is a 1.1 GB stream; a full parse is ~100 min, so it was sampled at the
first 200,000 rows. Clock presence is a structural property of the dump format (a dump
either emits `[%clk]` move tags or it does not), so a 200k-row sample settles coverage
just as decisively as a full pass — see the verdict.

## Per-`CLOCK_BAND` coverage

### `lichess_db_standard_rated_2013-01` — n = 34,308 (full)

```
clock coverage: 0/34308 rows carry [%clk] (0.0%)
     <10s:        0  (  0.0%)
   10-30s:        0  (  0.0%)
   30-60s:        0  (  0.0%)
     1-3m:        0  (  0.0%)
      >3m:        0  (  0.0%)
  (no clock-bearing rows — this dump predates [%clk] or has none)
```

| CLOCK_BAND | n | share of clock-bearing |
|---|--:|--:|
| none (no `[%clk]`) | 34,308 | — |
| <10s | 0 | 0.0% |
| 10-30s | 0 | 0.0% |
| 30-60s | 0 | 0.0% |
| 1-3m | 0 | 0.0% |
| >3m | 0 | 0.0% |

### `lichess_db_standard_rated_2016-05` — n = 200,000 (sample)

```
clock coverage: 0/200000 rows carry [%clk] (0.0%)
     <10s:        0  (  0.0%)
   10-30s:        0  (  0.0%)
   30-60s:        0  (  0.0%)
     1-3m:        0  (  0.0%)
      >3m:        0  (  0.0%)
  (no clock-bearing rows — this dump predates [%clk] or has none)
```

| CLOCK_BAND | n | share of clock-bearing |
|---|--:|--:|
| none (no `[%clk]`) | 200,000 | — |
| <10s | 0 | 0.0% |
| 10-30s | 0 | 0.0% |
| 30-60s | 0 | 0.0% |
| 1-3m | 0 | 0.0% |
| >3m | 0 | 0.0% |

## Verdict

**Neither cached dump is usable for calibrating or proving the clock dimension.** Both
carry `[%clk]` on **0%** of rows — every row lands in the `none` band, leaving the four
time-pressure bands (`<10s`, `10-30s`, `30-60s`, `1-3m`) and `>3m` completely empty. This
confirms what `drama_thresholds_real.md` notes about 2013-01 and extends it to 2016-05:
both dumps predate the era when Lichess monthly dumps began emitting per-move `[%clk]`
clock tags (the standard dumps gain clocks in **2017**, not before).

**Use dump X → there is no usable cached dump; a 2017-04-or-later monthly dump must be
downloaded** before tasks 0153 (prove the clock dimension on a real clock-bearing dump),
0250 (clock-dimension PROOF on real low-clock data), and 0269 (clock-aware gate) can run.
That download is a multi-GB fetch and needs human approval (it is the gate those held
tasks are parked behind).

---

*Reproduce:*

```sh
chess-equity data build --pgn ~/.cache/chess-equity/dumps/lichess_db_standard_rated_2013-01.pgn.zst \
  --out /tmp/clk/2013-01 --format csv
chess-equity validate --slice clock --data /tmp/clk/2013-01/dataset.csv

chess-equity data build --pgn ~/.cache/chess-equity/dumps/lichess_db_standard_rated_2016-05.pgn.zst \
  --out /tmp/clk/2016-05 --sample 200000 --format csv
chess-equity validate --slice clock --data /tmp/clk/2016-05/dataset.csv
```
