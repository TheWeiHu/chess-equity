# reports/REPRODUCE.md — how each committed real-data gate report was built

`reports/SUMMARY.md` is the gate **index** (one row per report: dump, n, verdict). This
file is the gate **provenance**: one documented command per committed `reports/*_real.md`,
so an auditor can map every number back to the exact `chess-equity …` invocation (dump
month, n, model, seed/flags) that produced it. Every command here is harvested from that
report's own header `Reproduce:` / `Provenance.` block — nothing here reads data or computes
new numbers (see the real-data-only policy in `CLAUDE.md`).

All dumps are cached real Lichess monthly dumps under `~/.cache/chess-equity/dumps/`
(`lichess_db_standard_rated_<YYYY-MM>.pgn.zst`); `chess-equity data build --month <YYYY-MM>`
resolves the cache (or `--pgn <path>` points at it explicitly). Prefix CLI calls with
`uv run` in the repo. See `DEPENDENCIES.md` for installs and the
`theweihu__chess-equity/validation-proof-runbook` brain page for the attended one-shot.

## ⚠ Attended-only (HUMAN-APPROVED) vs unattended

Three steps need **`maia2` + torch** (a ~23M-param checkpoint downloaded on first
`evaluate()`) and therefore are the **attended, human-approved headline** path — they do
**not** run on the unattended nightshift sandbox:

- **`validation_real.md`** — its `maia2` column (the `baseline,wdl-a` columns are torch-free).
- **`recalibration_maia2_real.md`** — Platt recalibration of `maia2`.
- **`wdl_net_real.md`** — `train-net` (Approach D end-to-end board→WDL net).

Everything else below is **torch-free** (`baseline,wdl-a` only) and reproducible unattended
from the cached dumps. Rows are flagged **[torch/attended]** or **[torch-free]** accordingly.

## One command per report

### Headline gate — `validation_real.md`  [torch/attended for the maia2 column]
Dump `lichess_db_standard_rated_2013-01`, **n=12000**, `--with-fen`, seed 0. Re-fit `wdl-a`
on the 2013-01 month first (the packaged artifact is fit on the tiny fixture), then score
all three predictors on the leak-free held-out split:
```
chess-equity data build  --month 2013-01 --sample 12000 --with-fen --format parquet --out data/real_2013-01
chess-equity train       --data data/real_2013-01/dataset.parquet --train-month 2013-01   # refits wdl_a.json
chess-equity validate    --data data/real_2013-01/dataset.parquet --models baseline,wdl-a,maia2 \
    --gate --bootstrap 2000 --holdout 0.2 --seed 0 --eval-month 2013-01 --out reports/validation_real.md
```
`maia2` needs the `--with-fen` dataset + the `maia2` extra; drop it from `--models` for a
torch-free `baseline,wdl-a` run (the committed report carries the maia2 column from an
attended run). This single `validate` run also emits the folded-in **good-moves** and
**time-control** sections inline.

### Cross-dump refit held-out — `validation_real_xdump_refit.md`  [torch-free]
Fit on **2013-01**, eval out-of-distribution on **2016-05**, n=100000:
```
chess-equity data build  --month 2013-01 --sample 60000  --out train_2013-01
chess-equity train       --data train_2013-01/dataset.csv --out wdl_a_2013-01.json --train-month 2013-01
chess-equity data build  --month 2016-05 --sample 100000 --out eval_2016-05
chess-equity validate    --data eval_2016-05/dataset.csv --models baseline,wdl-a \
    --wdl-a-artifact wdl_a_2013-01.json --eval-month 2016-05 --bootstrap 1000 --seed 0 --gate
```

### Cross-dump replication — `validation_real_2016-05.md`  [torch-free]
Dump `lichess_db_standard_rated_2016-05`, n=100000 sample (held-out 20300 / 325 games),
`--with-fen`, seed 0:
```
chess-equity data build  --pgn ~/.cache/chess-equity/dumps/lichess_db_standard_rated_2016-05.pgn.zst \
    --sample 100000 --with-fen --out build_2016-05
chess-equity validate    --data build_2016-05/dataset.parquet --models baseline,wdl-a \
    --gate --bootstrap 1000 --holdout 0.2 --seed 0 --eval-month 2016-05 --out reports/validation_real_2016-05.md
```
`maia2` skipped (no torch on this box). Caveat: `wdl-a` is in-distribution here
(`fit_month`=2016-05; the leakage guard fires), so read it as a consistency check.

### Properly-powered high-rating gate — `validation_real_2016-05_high.md`  [torch-free]
Dump `lichess_db_standard_rated_2016-05`, filtered to **2000+ only**, n_high=49269 (held-out
10134 / 148 games), seed 0. A helper builds the 300k mixed sample and filters to the
high-rating bands, stamping the source-month sidecar:
```
chess-equity data build  --pgn ~/.cache/chess-equity/dumps/lichess_db_standard_rated_2016-05.pgn.zst \
    --month 2016-05 --sample 300000 --out data/highrating_2016-05.csv          # via scripts/build_highrating_eval.py
python scripts/build_highrating_eval.py --pgn ~/.cache/chess-equity/dumps/lichess_db_standard_rated_2016-05.pgn.zst \
    --month 2016-05 --sample 300000 --out data/highrating_2016-05.csv
chess-equity validate    --data data/highrating_2016-05.csv --models baseline,wdl-a \
    --gate --bootstrap 1000 --holdout 0.2 --seed 0 --eval-month 2016-05 \
    --out reports/validation_real_2016-05_high.md
```

### Positive half (good moves) — `goodmoves_real.md`  [torch-free]
Dump `lichess_db_standard_rated_2013-01`, n=12000 (→ 11829 ply-pairs), `baseline,wdl-a`:
```
chess-equity data build  --pgn ~/.cache/chess-equity/dumps/lichess_db_standard_rated_2013-01.pgn.zst \
    --sample 12000 --out data/real_0117 --format csv
chess-equity validate    --data data/real_0117/dataset.csv --models baseline,wdl-a   # emits the good-moves + cutoff-robustness sections
```

### Maia2 Platt recalibration — `recalibration_maia2_real.md`  [torch/attended]
Dump `lichess_db_standard_rated_2013-01`, 8000 positions `--with-fen`; held-out n=1398 / 22
games (seed 0); Platt scaler fit on the disjoint 6602-row train split. maia2 from the cached
value head (`~/.cache/chess-equity/maia2.pkl`):
```
chess-equity data build  --month 2013-01 --sample 8000 --with-fen --out data/recal_2013-01
chess-equity validate    --data data/recal_2013-01/dataset.parquet --models baseline,wdl-a,maia2 \
    --recalibrate-maia2 --holdout 0.2 --seed 0 --out reports/recalibration_maia2_real.md
```

### Approach D end-to-end net — `wdl_net_real.md`  [torch/attended]
Train on **2016-05** (n=80000, `--with-fen`), eval on **2013-01** (n=12000, the same eval set
`validation_real.md` uses):
```
chess-equity data build  --pgn .../lichess_db_standard_rated_2016-05.pgn.zst --sample 80000 --with-fen --format parquet --out build_train
chess-equity data build  --pgn .../lichess_db_standard_rated_2013-01.pgn.zst --sample 12000 --with-fen --format parquet --out build_eval
chess-equity train-net   --data build_train/dataset.parquet --epochs 6 --train-month 2016-05   # dropout 0.4, wd 1e-3
chess-equity validate    --data build_eval/dataset.parquet --models baseline,wdl-a,wdl-net \
    --eval-month 2013-01 --out reports/wdl_net_real.md
```

### Calibration by rating band — `calibration_real.md`  [torch-free] *(info)*
The `--calibration` side-report of a `baseline` validate run on the same 2013-01 n=12000
`--with-fen` dataset (seed 0):
```
chess-equity validate    --data data/real_2013-01/dataset.parquet --models baseline \
    --holdout 0.2 --seed 0 --calibration reports/calibration_real.md
```

### Failure modes on binned outcomes — `failure_modes_real.md`  [torch-free] *(info)*
Feeds the same 2013-01 n=12000 dataset to the binned-outcomes report builder:
```
python scripts/failure_modes_real.py \
    --data data/real_2013-01/dataset.parquet \
    --dump lichess_db_standard_rated_2013-01 \
    --out reports/failure_modes_real.md
```

### Equity vs Stockfish divergence — `divergence_real.md`  [torch-free] *(info)*
The dedicated `divergence` subcommand on the same 2013-01 n=12000 dataset (`wdl-a` vs
`baseline`; reads no outcomes):
```
chess-equity divergence  --data data/real_2013-01/dataset.parquet \
    --equity wdl-a --stockfish baseline --out reports/divergence_real.md
```

### Drama trigger thresholds — `drama_thresholds_real.md`  [torch-free] *(info)*
Dump `lichess_db_standard_rated_2016-05`, n=295140 per-move transitions / 4860 games:
```
chess-equity data build  --month 2016-05 --out /tmp/drama_calib_2016-05
python scripts/calibrate_drama_thresholds.py \
    --data /tmp/drama_calib_2016-05/dataset.parquet \
    --dump lichess_db_standard_rated_2016-05 \
    --out reports/drama_thresholds_real.md
```

## Notes

- **Seeds / flags travel in the report header.** Where a report's header states `seed 0`,
  `--bootstrap N`, `--holdout 0.2`, or `--eval-month`, that is the verbatim flag — reproduce
  with the same values. The gate PASS rule (stated in every validation report): a
  rating-conditioned model PASSes when it has strictly lower log-loss **and** Brier than
  `baseline` **and** its log-loss 95% bootstrap CI clears zero.
- **n may differ from `--sample`.** `--sample` is the rows parsed; the report `n` is what
  survives parsing/filtering (and the held-out split is smaller again). The headers state
  both — trust the header's `n`.
- **The `*_sample.md` reports are excluded** (`validation_sample`, `calibration_sample`,
  `baseline_calibration_sample`) — they are illustrative offline smoke artifacts, not
  evidence, built from the 15-row `data/sample/` fixture, not a dump.
- When a new real-data report lands, add its one command here and a row to `SUMMARY.md`.
