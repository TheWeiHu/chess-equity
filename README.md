# chess-equity

[![CI](https://github.com/TheWeiHu/chess-equity/actions/workflows/ci.yml/badge.svg)](https://github.com/TheWeiHu/chess-equity/actions/workflows/ci.yml)

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

## Web demo (task 0010) — the idea, made tangible

**Live:** https://theweihu.github.io/chess-equity/

A static, dependency-free page that puts the **rating-conditioned equity bar** next to
the **classic centipawn bar** and lets you drag both players' rating sliders. The pitch
in one screen: *move a slider and the equity bar moves while the centipawn bar can't.*
The bundled game is **Légal's Mate** — White's queen "sacrifice" tanks the material
count (centipawn bar collapses) while it is in fact a forced mate (equity bar stays
winning): a move that is **green on equity but red on centipawns**.

```bash
python3 -m http.server -d web 8000     # then open http://localhost:8000
```

The page reads `web/demo-game.json` (a precomputed fixed game — no backend). Regenerate it:

```bash
python web/build_demo.py                # default: illustrative rating skew (no heavy deps)
python web/build_demo.py --model maia2  # real rating-conditioned numbers (needs Maia-2)
```

The committed equity is **illustrative** (the centipawns are real material counts);
swap in real numbers with `--model maia2`. Schema + the two headline acceptance checks
(slider moves the bar; the green/red flagship move exists) are gated by
`python3 web/test_demo.py`.

**Deployment:** the live site is published from `web/` by
[`.github/workflows/pages.yml`](.github/workflows/pages.yml) on every push to `main`
(GitHub Pages, no build step, no secrets). One-time setup: in **Settings → Pages**, set
**Source** to **GitHub Actions**. Import a real game with
`python web/import_game.py <lichess-url>` — or grab a player's latest game with
`python web/import_game.py --user <name>` (task 0039) — then open
`…/chess-equity/?game=imported-game.json`.

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

## Maia rollout oracle (task 0007) — the slow ground truth

`--model maia-rollout` estimates equity the honest, expensive way: **play the
position out.** It samples White moves from Maia-2 at `--white-elo` and Black moves at
`--black-elo` (both sides err like their rating), to checkmate / a draw rule / a ply
cutoff; positions surviving the cutoff are scored by Maia-2's value head. Averaging
`--n` rollouts gives the equity plus a 95% confidence interval on it.

```bash
chess-equity eval "<fen>" --white-elo 1800 --black-elo 1600 --model maia-rollout --n 500 --seed 0
# [#####---] 41.2% (B)  W/D/L ...  [maia-rollout]  95% CI [38.1, 44.3]  n=500 (123 terminal, 47 avg plies)
```

- **Reference, not the bar.** This is the oracle for validation (0009) and for
  sanity-checking the expectimax search (0006) — explicitly **non-interactive**.
- **Cost:** ~`n · mean_plies` Maia inferences per position (n=500 to an 80-ply cutoff
  ≈ up to ~40k forward passes → minutes). The same `(fen, elo, elo)` cache as `maia2`
  amortises repeats; batching/precompute is 0012.
- **Decoupled + testable:** the model takes a `HumanPolicy` (move sampler) + a leaf
  `EquityModel` (cutoff scorer) by injection, so the suite drives it with the uniform
  policy + material baseline — no torch/weights. `--seed` makes a run reproducible.

## Maia-weighted expectimax (task 0006) — the deterministic middle

`--model maia-search` computes equity by an explicit **expectimax** tree instead of
sampling: every node averages its children weighted by how likely Maia thinks each
move is at the *mover's* rating,
`equity(node) = Σ_move P_maia(move | side_to_move_elo) · equity(child)`, to a fixed
ply `--depth`, then scores leaves with Maia-2's value head. Both sides are expectation
nodes (not min/max), so an "absurd refutation" Maia gives ~0% probability barely moves
the bar, while a position that bleeds to likely human errors loses equity.

```bash
chess-equity eval "<fen>" --white-elo 1800 --black-elo 1600 --model maia-search --depth 2 --k 4
# [#####---] 41.2% (B)  depth=2 k=4  (16 leaves, 0 terminal, trunc=0.31)
```

- **Comparison, not the bar (re-scoped).** Maia-2's value head already bakes in the
  rating-conditioned "only bad if they find it" intuition in one forward pass, so this
  exists to *test* whether explicit look-ahead beats the implicit version on the 0009
  metrics — not as a faster estimator. Non-interactive at real depth.
- **Truncation, not silent capping:** each node keeps the top-`k` Maia moves and
  **renormalizes** their mass; the dropped mass is reported (`trunc=`) rather than
  hidden. Cost is `O(k**depth)` leaf scorings — `--depth`/`--k` are the knobs feeding
  the 0012 perf work.
- **Decoupled + testable:** like the rollout oracle it takes a `HumanPolicy` + a leaf
  `EquityModel` by injection, so the suite drives it with a scripted policy + a stub
  leaf — no torch/weights. Deterministic given `depth`/`k`.

## Data (task 0002)

The training + validation substrate. `chess-equity data build` turns a Lichess
monthly PGN dump into a tabular `(cp_eval, white_elo, black_elo, ply, phase,
time_control, tc_bucket, clock_remaining, white_clock, black_clock, side_to_move,
result)` dataset — one row per `[%eval]`-annotated position, streamed so a multi-GB
`.zst` is never unpacked to disk. `clock_remaining` is the **side-to-move's** clock;
`white_clock`/`black_clock` carry both players' clocks (task 0026) so the
time-pressure work (0015) can see the opponent's clock too.

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

For the **realistic ~50k-row sample** that tasks 0003/0004/0009 actually train and
validate on, use the reproducible wrapper (task 0024) instead of committing the data:

```bash
uv sync --extra data                          # zstandard (.zst) + pyarrow (parquet)
scripts/build_real_sample.sh                  # → data/dataset.parquet from Lichess 2026-05
ROWS=100000 MONTH=2026-04 scripts/build_real_sample.sh   # override size / month
WITH_FEN=1 scripts/build_real_sample.sh       # add FENs for board-model validation
scripts/build_real_sample.sh --dry-run        # print the command, build nothing
```

The output lands under `data/` (gitignored except `data/sample/`), so the repo stays
small — the reproducible command is the deliverable, not a checked-in multi-GB file.

A small committed fixture lives in `data/sample/` so tests and downstream tasks
(0003/0004/0009) have substrate without a download: `dataset.csv` is the cp-only
sample, and `dataset_fen.csv` is its FEN-bearing companion (same 15 rows + a `fen`
column) so board models (Maia, 0005/0031) can be scored end-to-end on checked-in
data with no PGN rebuild. Load a built dataset with
`chess_equity.data.load_rows(path)` (typed rows, dependency-free) or
`load_dataframe(path)` (pandas, needs the data extra).

`cp_eval` and `result` are both White-POV; mate scores are clamped to ±10000 cp.
Each row also carries a `game_id` (the Lichess game slug) so validation can split
train/test at the game level without leakage (task 0030); datasets built before it
existed load with `game_id=None`. With `--with-fen`, an optional `fen` column is
appended; datasets built without it load unchanged (`fen=None`). `validate.harness.model_predictor(model)` adapts any
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
# held-out, leak-free evaluation: score only a test split of whole games (task 0030)
uv run chess-equity validate --data data/dataset.csv --models baseline \
    --holdout 0.2 --seed 0

# Score Maia-2's rating-conditioned win_prob beside the baseline (task 0031).
# Needs a --with-fen dataset and `pip install maia2` (torch) for real numbers:
uv run chess-equity validate --data data/dataset_fen.csv --models baseline,maia2
```

A **predictor** maps a dataset row to a predicted White expected-score
(`P(win)+0.5·P(draw)`); the harness scores each with **log-loss, Brier, and ECE**
(calibration) — overall and sliced by **rating band** and **game phase** — so a model
that only wins in the off-2300 bands still shows up.

`--holdout FRACTION` scores only a held-out **game-level** split: positions from one
game are correlated, so a random row split would leak near-identical positions across
train/test and flatter every predictor. The split partitions *whole games* (keyed on
the new `game_id` column) so no game spans the boundary — the honest measure of
generalisation (task 0030). It's deterministic per `--seed`, and errors loudly on a
dataset built before `game_id` existed. *Deferred:* reliability-curve plots committed
to `reports/` (the numeric reliability table already ships in `validate.metrics`).

Shipped predictors: `baseline` (Lichess's rating-blind Win% over the row's centipawns
— the thing to beat), `baseline+clock`, `wdl-a` (Approach A, the rating-conditioned
regression — task 0004, a row predictor with no harness change), and the board models
**`maia2`** (Maia-2's value head, scored via `harness.model_predictor` on the row's
FEN — the thesis comparison; needs a `--with-fen` dataset) and **`maia-search`** (the
Maia-weighted expectimax, task 0006 — registered as a board predictor so 0009 can ask
whether explicit look-ahead beats the implicit value head; the comparison run needs
Maia weights). A demonstration run on the sample fixture lives in
[`reports/validation_sample.md`](reports/validation_sample.md) (smoke test —
meaningless at 15 rows, real evidence needs a real dataset).

### Calibration by rating band (task 0027)

`--calibration <path>` additionally writes a **reliability report**: the predicted-vs-
observed White score binned *within each rating band*, so the rating-blind baseline can
be shown drifting away from the ~2300 band it was fit on (it can't see who is playing).

```bash
uv run chess-equity validate --data data/dataset.csv --models baseline \
    --calibration reports/baseline_calibration.md \
    --plots reports/calibration.png        # optional: needs chess-equity[plots]
```

`--plots PATH` renders the same per-band reliability data as a **calibration curve**
(one predicted-vs-observed line per rating band against the `y = x` diagonal — task
0036), so the baseline's drift is visible at a glance. matplotlib is an optional extra
(`pip install chess-equity[plots]`); every numeric metric and the Markdown reports work
without it, and `--plots` errors with a clear install hint when it's absent.

A sample run is in [`reports/baseline_calibration_sample.md`](reports/baseline_calibration_sample.md),
with the matching curve at [`reports/calibration_sample.png`](reports/calibration_sample.png).
Separately, `python baseline/measure_practical.py --data <dataset> --write` replaces the
*hypothesised* practical numbers in `baseline/failure_modes.json` with the **measured**
rating-sliced mean White result for each position's class (its engine-eval band) — `null`
where the dataset has no row in the class (the committed fixture is tiny; a real dump from
0024 makes both decisive).

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
