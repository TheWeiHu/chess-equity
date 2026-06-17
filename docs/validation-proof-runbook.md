# Runbook — proving practical equity beats the centipawn bar

This is the **one-shot recipe for the project's scientific gate** ([task 0009]): does a
rating-conditioned equity predictor predict *actual* Lichess results better than the
rating-blind centipawn baseline? The answer needs real data + heavy models, so it can't
run on the unattended nightshift sandbox — this file captures the exact sequence so a
human/GPU session executes it in one pass instead of rediscovering the steps.

> **Why it's parked.** Every step below needs at least one *attended-only* dependency
> (a Stockfish binary, a multi-GB Lichess dump, the `maia2`/torch extra, Maia-2 weights,
> outbound network). See **[../DEPENDENCIES.md](../DEPENDENCIES.md)** — the single source
> of truth — for each row's install command and which tasks need it. The core
> `uv sync --extra dev` path proves nothing here; it only runs the 15-row `data/sample/`
> smoke fixture.

## 0 · Provision (once)

```bash
uv sync --extra data --extra maia2 --extra plots   # pandas/pyarrow/zstandard + torch + matplotlib
brew install stockfish                             # or: apt-get install stockfish
export STOCKFISH_PATH=/path/to/stockfish           # only if not on $PATH
```

`maia2` pulls torch and downloads the ~23M-param checkpoint on first `evaluate(...)`.
Stockfish is the objective engine behind the `baseline` centipawn bar (resolution order:
explicit `path=` → `$STOCKFISH_PATH` → `stockfish` on `PATH`). Confirm both resolve:

```bash
uv run chess-equity eval "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"   # cp from a real engine, not material
uv run python -c "import maia2, torch; print('maia2+torch OK')"
```

## 1 · Build a real dataset **with FENs** ([task 0024])

The board models (Maia-2) need each position's FEN, so pass `--with-fen` (≈3× size). The
`.zst` is streamed, never fully unpacked:

```bash
uv run chess-equity data build \
  --month 2026-05 --sample 50000 \
  --with-fen --format parquet \
  --out data/
# -> data/dataset.parquet  (one row per [%eval] position: cp, ratings, ply, phase, clocks, result, fen)
```

`data build --month` prints the canonical `database.lichess.org` URL it streams. Use a
larger `--sample` (or drop it) for tighter confidence intervals; 50k is the documented
default for a first pass.

## 2 · Re-fit the `wdl-a` regression on the real data ([task 0034])

The packaged `wdl_a.json` was fit on the tiny sample fixture and is a placeholder. Re-fit
it on the real dataset so the dependency-free baseline-beater is honest:

```bash
uv run chess-equity train --data data/dataset.parquet
# default --out overwrites the packaged src/chess_equity/artifacts/wdl_a.json
```

## 3 · Run the validation gate with every predictor ([task 0009] / [task 0031])

Score the rating-blind `baseline` (Stockfish cp → Lichess Win%) against the
rating-conditioned predictors on a **leak-free, game-level** held-out split, and emit the
per-rating-band reliability artifacts ([task 0027] / [task 0036]):

```bash
uv run chess-equity validate \
  --data data/dataset.parquet \
  --models baseline,wdl-a,maia2 \
  --holdout 0.2 --seed 0 \
  --out docs/validation-report.md \
  --calibration docs/validation-calibration.md \
  --plots docs/validation-reliability.png
```

`maia2` as a predictor requires the `--with-fen` dataset (step 1) and the `maia2` extra
(step 0) for real numbers; without them it has nothing to score. `baseline` uses the real
Stockfish cp from step 0.

## 4 · Read the result — the gate

The report scores each model by **log-loss, Brier, and ECE**, overall and sliced by rating
band and phase. The thesis holds iff a **rating-conditioned** model (`maia2`, and ideally
`wdl-a`) beats `baseline` on log-loss/Brier — especially in the **low-** and **high-rating
bands** and under **time pressure**, where the rating-blind centipawn bar is most wrong.
The calibration report + reliability PNG show *where* the centipawn bar is miscalibrated
(it over-states the stronger side's chances where weaker players fail to refute).

### Expected artifacts
- `data/dataset.parquet` — the real labelled dataset (with FENs).
- `src/chess_equity/artifacts/wdl_a.json` — re-fit regression coefficients.
- `docs/validation-report.md` — the headline metrics table (the gate verdict).
- `docs/validation-calibration.md` — per-rating-band reliability numbers.
- `docs/validation-reliability.png` — reliability curves per band.

## Smoke-test the pipeline shape first (no download)

To verify the commands wire up before committing to a multi-GB run, use the bundled
15-row fixture (its numbers are a smoke test, **not** evidence):

```bash
uv run chess-equity validate --data data/sample/dataset_fen.csv --models baseline,wdl-a --holdout 0.5
```

## Related tasks
- **[task 0009]** — the validation harness + this gate (the deliverable this runbook unblocks).
- **[task 0034]** — re-fit `wdl-a` on the real dataset (step 2).
- **[task 0024]** — build the real tens-of-thousands-row dataset (step 1).
- **[task 0031]** — register `Maia2Equity` as a 0009 predictor (step 3).
- **[task 0013] / [task 0014]** — held neural stretch work (end-to-end WDL net, personal equity) that rides on the same provisioned stack.

[task 0009]: ../README.md#data--validation
[task 0013]: ../DEPENDENCIES.md
[task 0014]: ../DEPENDENCIES.md
[task 0024]: ../DEPENDENCIES.md
[task 0027]: ../README.md#data--validation
[task 0031]: ../README.md#data--validation
[task 0034]: ../DEPENDENCIES.md
[task 0036]: ../README.md#data--validation
