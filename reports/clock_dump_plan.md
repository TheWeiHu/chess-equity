# Clock-dump plan — the smallest fetchable `[%clk]`-bearing Lichess dump (task 0271)

**One human decision, made once.** Task 0267's coverage audit
(`reports/clock_coverage_real.md`) proved both cached dumps (2013-01, 2016-05) carry
`[%clk]` on **0%** of rows, so every clock-proof task gets parked "needs a clock dump"
each night with no decision artifact to act on. This report converts that recurring churn
into a single approve-able fetch: it names the **earliest** (and therefore **smallest**)
Lichess monthly dump that actually carries per-move `[%clk]` clock tags, its real
compressed size, and the exact `chess-equity data build --month` one-liner to fetch the
smallest viable clock-bearing slice.

## Verdict — fetch `2017-04`

| | value |
|---|---|
| **Earliest `[%clk]`-bearing month** | **`2017-04`** (April 2017) |
| **Compressed size (`.pgn.zst`)** | **3,481,200,992 bytes ≈ 3.48 GB** (3.24 GiB) |
| **URL** | `https://database.lichess.org/standard/lichess_db_standard_rated_2017-04.pgn.zst` |
| **Clock coverage** | present on essentially all rated rows (see evidence) |

`2017-04` is both the **earliest** clock-bearing month and the **smallest** one: Lichess
monthly volume grows roughly monotonically, so the first month to carry clocks is also the
least bytes you can download to get them. The immediately-prior month (`2017-03`) has **no**
clock tags, so there is no smaller clock-bearing slice to fetch.

## Evidence (real data, no full download, no synthetic data)

Two structural probes against `database.lichess.org`, both well short of downloading a dump:

**1. Compressed sizes** — `HEAD` `Content-Length` for the candidate months:

| month | compressed size | note |
|---|--:|---|
| `2017-01` | 1.90 GB | no clocks |
| `2017-02` | 1.80 GB | no clocks |
| `2017-03` | 2.17 GB | no clocks |
| **`2017-04`** | **3.48 GB** | **first clock-bearing month** |
| `2017-05` | 3.59 GB | clocks |

The **+60% jump from 2017-03 (2.17 GB) to 2017-04 (3.48 GB)** is the `[%clk]` tag
inflation: a `{ [%clk H:MM:SS] }` comment is appended after every ply, roughly doubling
movetext bytes.

**2. Tag probe** — fetched only the first ~4 MB of each `.zst` via an HTTP `Range`
request, decompressed the prefix with `zstd -dc`, and grepped the movetext:

| month | rows decompressed | `%clk` hits | `%eval` hits |
|---|--:|--:|--:|
| `2017-03` | 118,813 lines | **0** | 697 |
| `2017-04` | 50,783 lines | **2,799** | 234 |

A `2017-04` movetext line, verbatim from the probe:

```
1. e4 { [%clk 0:03:00] } e5 { [%clk 0:03:00] } 2. f4 { [%clk 0:02:58] } exf4 { [%clk 0:02:58] } 3. Nf3 { [%clk 0:02:57] } Nf6 { [%clk 0:02:57] } ...
```

Every ply carries a `[%clk]` tag in `2017-04`; `2017-03` carries none. This is the
structural confirmation the 0267 audit lacked — it concluded "clocks arrive in 2017" but
did not pin the month. It is now pinned to **April 2017**.

> Both probes read only headers and a 4 MB prefix — no dump was downloaded and no dataset
> was built, consistent with this task's "doc/research only" scope and the project's
> real-data-only policy.

## The one-liner a human approves

The fetch streams the full `.zst` into `~/.cache/chess-equity/dumps/` first
(`data/download.py`), then builds. `--sample N` caps **build** rows/time but **not** the
download, so the human is approving a **~3.48 GB stream** either way. Build the smallest
useful slice (sampling matches the 0267 audit's 200k cap, enough to populate every
`CLOCK_BAND`):

```sh
chess-equity data build --month 2017-04 --sample 200000 --format csv --out data/clock_2017-04
```

To audit clock coverage on the built slice exactly as 0267 did:

```sh
chess-equity validate --slice clock --data data/clock_2017-04/dataset.csv
```

The built dataset is auto-stamped with source-month `2017-04`
(`data/source_month.py`), so the validation leakage guard (task 0126) keeps it distinct
from any month the wdl-a model was fit on.

## What this unblocks (cross-link — approve once, unblock all)

The held clock-thread tasks all wait on exactly this download. One approval of the
one-liner above clears the gate for:

- **0153** — Validation: prove the clock dimension on a real clock-bearing dump
- **0173** — Clock: calibrate time-pressure / flag_risk constants on real low-clock outcomes
- **0174** — Validation: clock-remaining slice of the thesis gate
- **0250** — Clock-dimension PROOF on real low-clock data
- **0269** — Clock-aware gate: does the clock adjustment beat clock-blind?

Each is parked behind a "needs a clock dump" hold today. `2017-04` (3.48 GB) is the
smallest dump that satisfies all of them.

---

*Reproduce the size + tag probes (header + 4 MB prefix only — no dump download):*

```sh
for m in 2017-03 2017-04; do
  url="https://database.lichess.org/standard/lichess_db_standard_rated_${m}.pgn.zst"
  curl -sI "$url" | awk 'tolower($1)=="content-length:"{print "'"$m"'", $2}'
  curl -s -r 0-4194304 "$url" | zstd -dc 2>/dev/null | grep -c '%clk'   # 2017-03 -> 0, 2017-04 -> many
done
```
