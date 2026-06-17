# chess-equity

> ⚠️ **Experimental prototype — nukeable at any time.** This is a research spike, not
> a product. We optimize for *speed of learning*, not stability: expect aggressive
> prototyping, throwaway code, hard pivots, force-pushes, and whole modules deleted
> without ceremony. Nothing here is a stable API or a compatibility promise. If a
> sharper approach appears, we rewrite or scrap. Don't build anything important on
> top of it — and don't be precious about the code that's here.

A chess evaluation bar that shows your **equity** — the probability *you* win given
your rating and your opponent's — instead of objective centipawns.

## Why

The classic Stockfish/Lc0 centipawn bar assumes both sides play perfectly. A
consequence: **a move can only ever be bad, never good.** The best a move can do is
leave the eval unchanged; any human imperfection only moves the bar against you.
There is no upside for *finding* a strong move.

We reframe the bar as equity, in the poker sense — expected value against the
*realistic* distribution of play at your rating:

```
equity = P(win) + 0.5 * P(draw)        # conditioned on (white_elo, black_elo)
```

rendered from White's POV on a 0–100% win scale. Consequences we want:

1. **Moves can be good.** Finding a move better than a player of your rating
   typically finds *increases* your equity.
2. **Absurd refutations read ~equal.** A position that is "−1.5" only because the
   opponent could find a move no human of their rating would play reads near-even.
3. **"Dead 0.00" reflects real difficulty.** Endgames an engine calls 0.00 but that
   are genuinely hard for a human to hold show real conversion/blunder odds.

The engine that knows *how a human of a given rating actually plays* is
[Maia-2](https://github.com/CSSLab/maia2); its value head already emits a
rating-conditioned win probability. This project turns that into a usable eval bar,
grades moves by Δequity, and aims to **prove** it predicts real outcomes better than
the rating-blind centipawn bar.

## Status

The package defines the contract and ships a **placeholder** model — the rating-blind
Lichess Win% over a trivial material eval — so the CLI runs end-to-end with zero heavy
deps (`--model baseline`, the default). The first **real** rating-conditioned model,
Maia-2's `win_prob` (`--model maia2`, task 0005), plugs in behind the same `EquityModel`
interface; the regression baseline (0004) lands the same way.

## Install

```bash
uv sync --extra dev      # creates .venv with python-chess + pytest
```

## Use

```bash
uv run chess-equity eval                                    # startpos, 1500 vs 1500
uv run chess-equity eval "<fen>" --white-elo 1800 --black-elo 1600
uv run chess-equity eval --pgn game.pgn                     # annotate every move
```

Example:

```
[###############---------------]  50.0% (W)  W/D/L 33.4/33.2/33.4  [lichess-baseline]  cp +0
```

## Maia-2 equity (task 0005) — the real rating-conditioned bar

`--model maia2` swaps the placeholder for [Maia-2](https://github.com/CSSLab/maia2)
(NeurIPS 2024), a single rating-conditioned model whose value head was trained on real
Lichess outcomes. Its `win_prob` is the side-to-move's expected score in `[0, 1]` — i.e.
**exactly our equity** `P(win) + 0.5·P(draw)` — so the bar comes straight from Maia-2,
conditioned on *both* players' ratings. This is the first principled `EquityModel`; the
regression/search baselines (0004/0006/0007) become comparisons against it.

```bash
chess-equity eval "<fen>" --white-elo 1800 --black-elo 1600 --model maia2
chess-equity grade --pgn game.pgn --model maia2
chess-equity broadcast --pgn game.pgn --model maia2
```

Setup (the real model is heavyweight, so it's optional and lazy-loaded):

```bash
pip install maia2          # pulls torch; the ~23M-param checkpoint downloads on first use
```

- **Deps / weights:** `maia2` + `torch`; the checkpoint is fetched on the first
  `evaluate(...)` call and then cached by the library. Without `maia2` installed, the CLI
  prints a clear install hint and exits non-zero (it never tries the network just to scaffold).
- **Caching:** evaluations are memoised by `(fen, white_elo, black_elo)` to a pickle at
  `~/.cache/chess-equity/maia2.pkl`, so repeats and restarts don't re-run the model
  (search/rollout baselines hammer the same positions; precompute/batching is 0012).
- **WDL split:** Maia-2 gives the scalar equity; the win/loss-vs-draw partition is modelled
  (`wdl_from_equity`, draw mass peaking near 50%) so `P(win)+0.5·P(draw)` stays faithful to
  `win_prob`.
- **Latency:** one CPU forward pass per uncached position; budget for the interactive bar
  is set in 0012. **Calibration caveat (feeds 0009):** the value head is a *secondary*
  objective in the paper with few reported numbers — verify it in the validation harness
  before trusting it as the shipped bar, especially at extreme ratings and in endgames.

Tests inject a fake backend (the `Backend` seam: `(fen, elo_self, elo_oppo) ->
(move_probs, win_prob)`), so the suite needs neither torch nor weights.

## Approach A — rating-conditioned WDL regression (task 0004)

`--model wdl-a` is a transparent, dependency-free alternative to Maia-2: a pure-Python
multinomial-logistic model fitting `P(White W/D/L | cp_eval, white_elo, black_elo, ply,
time_control)`. Re-scoped now that Maia-2 is the principled core — this is the cheap
**baseline to compare against Maia-2's value head** in the 0009 validation (and a
Stockfish-only fallback when no learned head is available). The feature that makes it
genuinely rating-conditioned is the `cp × skill` interaction: the *same* engine eval is
more/less decisive depending on who's playing, which a rating-blind logistic can't express.

```bash
chess-equity train --data data/dataset.csv          # fits + writes the wdl_a.json artifact
chess-equity eval "<fen>" --white-elo 1800 --black-elo 1600 --model wdl-a
chess-equity validate --data data/dataset.csv --models baseline,wdl-a   # the A/B
```

The artifact (`src/chess_equity/artifacts/wdl_a.json`) is a small, diff-friendly JSON of
weights. **The committed one is a placeholder fit on the 15-row sample** (`n_train=15` in
its `meta`) — it proves the train→persist→serve→validate wiring, but its numbers are not
meaningful (15 rows from 3 games carry a spurious rating↔outcome signal). A real fit, and
the "beats baseline on held-out log-loss/calibration, especially off-2300" evidence, wait
on a real tens-of-thousands-row dataset (task 0024). The model and CLI need no extra deps;
training is plain batch gradient descent, fine for the sample and documented to scale.

## Data (task 0002)

The training + validation substrate. `chess-equity data build` turns a Lichess
monthly PGN dump into a tabular `(cp_eval, white_elo, black_elo, ply, phase,
time_control, tc_bucket, clock_remaining, side_to_move, result)` dataset — one row
per `[%eval]`-annotated position, streamed so a multi-GB `.zst` is never unpacked to
disk.

```bash
# Build from a downloaded dump (plain .pgn or .zst). Needs the data extra for .zst:
uv sync --extra data
uv run chess-equity data build --pgn lichess_db_standard_rated_2026-05.pgn.zst \
    --sample 50000 --out data/ --format parquet

# Or let --month stream + cache the dump for you (resumable), then build from it.
# The .zst dump needs the data extra; it lands in ~/.cache/chess-equity/dumps:
uv run chess-equity data build --month 2026-05 --sample 50000 --out data/ --format parquet

# CSV is the default and needs no extra:
uv run chess-equity data build --pgn data/sample/sample_games.pgn --out data/sample

# Add --with-fen to record each position so board models (Maia, 0005) can be
# scored in validation — off by default because the FEN ~triples row size:
uv run chess-equity data build --pgn data/sample/sample_games.pgn --out data/sample --with-fen
```

A small committed fixture lives in `data/sample/` so tests and downstream tasks
(0003/0004/0009) have substrate without a download. Load a built dataset with
`chess_equity.data.load_rows(path)` (typed rows, dependency-free) or
`load_dataframe(path)` (pandas, needs the data extra).

`cp_eval` and `result` are both White-POV; mate scores are clamped to ±10000 cp.
With `--with-fen`, an optional `fen` column is appended; datasets built without it
load unchanged (`fen=None`). `validate.harness.model_predictor(model)` adapts any
board-based `EquityModel` into a 0009 predictor by reading that column.
`--month YYYY-MM` streams the canonical Lichess dump to a cache dir (`--dump-dir` to
override), resuming a partial download via an HTTP `Range` request and never holding
the multi-GB file in memory, then builds from it. Checksum verification is opt-in
(`download_month(expected_sha256=...)`) since Lichess publishes no stable per-dump
hash. See `data/schema.py` for the column contract.

## Validation (task 0009)

The scientific gate for the whole thesis: does a rating-conditioned predictor beat
the rating-blind centipawn baseline at predicting **actual** game results?

```bash
uv run chess-equity validate --data data/dataset.csv --models baseline \
    --out reports/validation.md
```

A **predictor** maps a dataset row to a predicted White expected-score
(`P(win)+0.5·P(draw)`); the harness scores each with **log-loss, Brier, and ECE**
(calibration) — overall and sliced by **rating band** and **game phase** — so a model
that only wins in the off-2300 bands still shows up. Shipped today: `baseline`
(Lichess's rating-blind Win% over the row's centipawns — the thing to beat), `baseline+clock`,
and `wdl-a` (Approach A, the rating-conditioned regression — task 0004). A demonstration run on the
sample fixture lives in [`reports/validation_sample.md`](reports/validation_sample.md)
(smoke test — meaningless at 15 rows, real evidence needs a real dataset). Models that
need the full board (Maia, 0005) wait on positions being added to the dataset schema.

## Move grading by Δequity (`grade`)

The classic centipawn-loss grade can only ever be ≤ 0 — perfect play is the ceiling,
so every human move is "less bad." Equity grading flips that: a move is scored by the
change in the **mover's** equity, benchmarked against what a player of their rating
was expected to do, so a move stronger than the rating-typical mix reads **positive**.

```bash
uv run chess-equity grade --pgn game.pgn --white-elo 1200 --black-elo 1200
```

```
  7. Qxf7#   brilliant   Δpeer +48.2  Δbest  +0.0
```

- **`Δpeer` = equity_after − expected_equity** (the headline) where
  `expected_equity = Σ_move P(move | rating) · equity(after move)`. Positive ⇒ you
  beat your peers. `P(move | rating)` comes from a `HumanPolicy` — Maia-2 (task 0005);
  a uniform placeholder until then, so quiet moves read ~0 under the material model.
- **`Δbest`** = equity_after − best-legal-move equity (≤ 0) — the classic "left on the
  table," on the equity scale.
- **`cp_loss`** is shown alongside so the flagship case is visible: a move can *lose
  centipawns yet gain equity* (a sound trap a rating-peer opponent likely walks into).
  `grading.py`'s tests stage that case directly. Thresholds are rating-aware (wider at
  lower ratings, where the peer mix is noisier); calibration lands with 0005/0009.

## Live broadcast (streaming wedge)

Stream per-move equity from a live game — the plumbing the OBS overlay (task 0019)
consumes. It emits one JSON event per new move on stdout (tail it, or pipe to a file):

```bash
# Live Lichess broadcast round (poll its public PGN every 2s):
uv run chess-equity broadcast --round <roundId>

# Replay a finished PGN move-by-move as if it were live (no network — great for demos/CI):
uv run chess-equity broadcast --pgn game.pgn --interval 0

# Any public PGN URL (chess.com export, a static file, …), with manual ratings for OTB:
uv run chess-equity broadcast --url <pgnUrl> --white-elo 2700 --black-elo 2650
```

Each event:

```json
{"game_id": "...", "ply": 12, "san": "Bb5", "fen": "...",
 "white_clock": 55.0, "black_clock": 54.0, "white_elo": 2900, "black_elo": 2850,
 "equity": 50.0, "delta_equity": 0.0, "last_move_grade": "ok", "compute_ms": 0.1}
```

`equity` is the White-POV bar (0–100%); `delta_equity` is the change from the
*mover's* POV (positive = the move improved their practical chances — the whole
point of the reframe). `[%clk]` tags are parsed and carried on every event so the
clock-aware model (task 0015) can use them; this module does not yet feed the clock
into the equity computation. The ingestor tracks every game in a round at once,
de-dupes repeated polls, resyncs on broadcast corrections, and retries through
transient feed errors (reconnect). Latency: with the placeholder model, equity
compute is ~0.1 ms/move — well under the sub-second target (Maia-2 in 0005 sets the
real number; cache/batch live in 0012).

## Performance: caching + precompute (task 0012)

A live overlay needs interactive per-move latency, so equity is cacheable.
`CachingEquityModel` wraps **any** `EquityModel` and memoises by `(model, fen,
white_elo, black_elo)` — the inputs the result depends on — returning a result
identical to the uncached model (a warm lookup recomputes nothing). With a `--cache`
path the cache is a small JSON that survives restarts.

`precompute` evaluates a whole game in one cache-backed pass and emits a UI-ready JSON,
so the web demo (0010) can scrub a finished game with no live backend:

```bash
chess-equity precompute --pgn game.pgn --model wdl-a --cache .cache.json --out game.json
# stderr: "# 8 plies, 2.6 ms total (0.33 ms/ply), cache 0 hit / 8 miss"  (cold)
# re-run:  "# 8 plies, 0.6 ms total (0.07 ms/ply), cache 8 hit / 0 miss"  (warm)
```

Each ply record carries `equity_white` (the [0,100]% bar), the side-to-move WDL split,
and the objective `cp`. **Deferred:** leaf-eval **batching** (GPU batch for Maia-2,
multi-PV Stockfish) and search-parameter (`depth`/`k`) cache keys land with the search
baselines (0006/0007); a measured p50/p95 interactive target waits on a real model
(Maia-2) under load — the baseline is already sub-millisecond.

## Architecture

| Type | Role |
|------|------|
| `Equity` / `WDL` (`types.py`) | model-agnostic eval values — full WDL + the White-POV bar |
| `EquityModel` (`adapters.py`) | `(fen, white_elo, black_elo) -> Equity` — the core contract |
| `ObjectiveEngine` | `fen -> cp/mate` — Stockfish/Lc0 plug in here (placeholder: material only) |
| `HumanPolicy` | `fen, elo -> P(move)` — Maia plugs in here (task 0005) |
| `bar.py` | ASCII rendering of the bar |
| `data/` | Lichess PGN dump -> `(eval, ratings, outcome)` dataset (task 0002) |
| `validate/` | score predictors vs real outcomes — log-loss/Brier/ECE (task 0009) |
| `grading.py` | Δequity move grading — peer-relative + classic (task 0008) |
| `broadcast.py` | live feeds + per-move equity event stream (task 0018) |
| `cli.py` | `chess-equity` entry point; depends only on `EquityModel` |

Swap the model in `cli.build_model()` and everything else is unchanged.

## Test

```bash
uv run pytest
```
