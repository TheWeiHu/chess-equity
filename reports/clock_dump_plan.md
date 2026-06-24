# Clock-dump locator — the smallest fetchable `[%clk]`-bearing Lichess dump (task 0271)

**One human decision, not a nightly hold.** Task 0267's coverage audit
(`clock_coverage_real.md`) proved both cached dumps (2013-01, 2016-05) carry `[%clk]` on
**0%** of rows, so every clock-proof task gets parked "needs a clock dump" each night
(0153 / 0173 / 0174 / 0250 / 0269 — see *Unblocks* below). This report converts that
recurring churn into a single approve-once artifact: **which** dump to fetch, **how big**
it is, and the **exact one-liner** a human runs to fetch the smallest viable clock-bearing
slice. Doc/research only — no download was performed, no synthetic data.

## Verdict — fetch `2017-04`

| | |
|---|---|
| **Earliest `[%clk]`-bearing month** | **2017-04** (April 2017) |
| **Compressed size** | **3,481,200,992 bytes ≈ 3.24 GiB** (`.pgn.zst`) |
| **URL** | `https://database.lichess.org/standard/lichess_db_standard_rated_2017-04.pgn.zst` |

Lichess standard monthly dumps grow monotonically, so the **earliest** clock-bearing
month is also the **smallest** clock-bearing month — there is no smaller fetchable slice
that still carries clocks. 2017-04 is therefore the single best target for the clock
thread.

### The one-liner a human approves

```sh
# Full month — ~3.24 GiB download, cached under ~/.cache/chess-equity/dumps/
chess-equity data build --month 2017-04 --with-fen --out data/2017-04

# OR cap the PARSE to keep the build fast (the download is still the full month;
#    --sample only caps rows parsed). 500k rows is ample for the low-clock bands:
chess-equity data build --month 2017-04 --sample 500000 --with-fen --out data/2017-04
```

Then confirm clocks actually landed before spending model time on it:

```sh
chess-equity validate --slice clock --data data/2017-04/dataset.csv
# expect a non-zero share in the <10s / 10-30s / 30-60s / 1-3m bands
```

`--with-fen` is included because the downstream clock-validation tasks pair clock state
with board models (Maia); drop it if a given task is pure cp/clock and doesn't need FENs
(~3× smaller dataset).

## Evidence

Sizes are exact `Content-Length` values from HTTP **HEAD** requests against
`database.lichess.org` (metadata only — no dump bytes were downloaded and nothing was
parsed):

| month | compressed size | Δ vs prior |
|---|--:|--:|
| 2016-12 | 1.58 GiB | — |
| 2017-01 | 1.77 GiB | +0.19 |
| 2017-02 | 1.68 GiB | −0.09 |
| 2017-03 | 2.02 GiB | +0.34 |
| **2017-04** | **3.24 GiB** | **+1.22 (+60%)** |
| 2017-05 | 3.34 GiB | +0.10 |
| 2017-06 | 3.28 GiB | −0.06 |

The **+60% step at 2017-04** — far larger than the gentle organic month-over-month growth
on either side of it — is the structural fingerprint of per-move `[%clk]` annotations
entering the export: a clock tag on every half-move inflates the PGN by roughly that
proportion, and the inflation then persists. This corroborates, from an independent
signal, three things that already agree:

1. **Repo audit** (`clock_coverage_real.md`, task 0267): standard dumps "gain clocks in
   **2017**, not before"; both pre-2017 cached dumps are 0% `[%clk]`.
2. **Memory** `cached-dumps-have-no-clocks`: clock-proof tasks blocked until a 2017+ dump
   is downloaded.
3. **Lichess's documented history**: clock times were added to the database PGN exports
   starting with the **April 2017** dump.

**Caveat (no download → no parse).** This report names 2017-04 from the size-jump signal +
documented history; it does **not** itself parse the dump (the task forbids the download).
The `validate --slice clock` step in the one-liner above is the confirmation gate — run it
right after the build and expect non-zero low-clock bands. If 2017-04 somehow disappoints,
2017-05/2017-06 are the immediate fallbacks (also ~3.3 GiB, definitely post-clock).

## Note on the CLI's size warning

`chess-equity data build --month` prints `note: the <month> dump is ~30 GB compressed`
(generic constant `APPROX_DUMP_SIZE_GB = 30` in `chess_equity/data/download.py:36`). That
figure tracks recent (2020s) dumps and is **~10× too large** for 2017-04, which is 3.24
GiB. Don't let the warning deter the approval — the real fetch is a few GiB, not 30.

## Unblocks (cross-linked held tasks)

Approving the 2017-04 fetch once clears the "needs a clock dump" gate for all of:

- **0153** — Validation: prove the clock dimension on a real clock-bearing dump
- **0173** — Clock: calibrate time-pressure / flag_risk constants on real low-clock outcomes
- **0174** — Validation: clock-remaining slice of the thesis gate
- **0250** — Clock-dimension PROOF on real low-clock data
- **0269** — Clock-aware gate: does the clock adjustment beat clock-blind?

---

*Reproduce the size measurements (metadata-only HEAD, no download):*

```sh
for m in 2017-03 2017-04 2017-05; do
  curl -sI "https://database.lichess.org/standard/lichess_db_standard_rated_${m}.pgn.zst" \
    | awk -F': ' 'tolower($1)=="content-length"{print "'"$m"'", $2, "bytes"}'
done
```
