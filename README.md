# chess-equity

[![CI](https://github.com/TheWeiHu/chess-equity/actions/workflows/ci.yml/badge.svg)](https://github.com/TheWeiHu/chess-equity/actions/workflows/ci.yml)

> ⚠️ **Experimental prototype — nukeable at any time.** A research spike, not a
> product: expect hard pivots, force-pushes, and throwaway code. Don't build on it.

A chess evaluation bar that shows your **equity** — the probability *you* win given
your rating and your opponent's — instead of rating-blind centipawns.

## Why

The classic Stockfish/Lc0 centipawn bar assumes both sides play perfectly, so **a move
can only ever be bad, never good**: the best it can do is hold the eval, and any human
imperfection moves the bar against you. There's no upside for *finding* a strong move.

We reframe the bar as equity, in the poker sense — expected value against the
*realistic* distribution of play at your rating:

```
equity = P(win) + 0.5 * P(draw)        # conditioned on (white_elo, black_elo)
```

rendered from White's POV on a 0–100% scale. What that buys us:

1. **Moves can be good.** A move stronger than your rating-typical play *increases* your equity.
2. **Absurd refutations read ~even.** A line that wins only because the opponent could
   find a move no human of their rating plays barely moves the bar.
3. **"Dead 0.00" reflects real difficulty.** Endgames an engine calls drawn but that
   humans routinely botch show real conversion odds.

The model that knows *how a human of a given rating actually plays* is
[Maia-2](https://github.com/CSSLab/maia2); its value head emits a rating-conditioned win
probability. This project turns that into an eval bar, grades moves by Δequity, and aims
to **prove** it predicts real outcomes better than the centipawn bar.

## Results

**The claim, precisely:** a *rating-conditioned* predictor beats the *rating-blind*
centipawn baseline at predicting **real human outcomes** — not at out-calculating a deep
engine on forced lines. `chess-equity validate` makes that machine-checkable: it scores
each model on log-loss, Brier, and ECE against actual results, and only calls a model a
winner when its 95% paired-bootstrap CI on the delta vs the baseline clears zero.

On a **real Lichess** run (n=12,000 positions from the `2013-01` standard dump) the
rating-conditioned `wdl-a` and Maia-2 both clear that bar — the largest gains land in the
lower rating band and the middlegame, exactly where the rating-blind bar is most wrong.
The committed evidence (gate verdict + the "where equity wins" slice table) is
**[reports/validation_real.md](reports/validation_real.md)**, with the per-rating-band
calibration curves in **[reports/calibration_real.md](reports/calibration_real.md)**:

| predictor | log-loss | Brier | gate |
|---|--:|--:|:--:|
| baseline (rating-blind Win%) | 0.905 | 0.208 | — |
| `wdl-a` (rating-conditioned) | **0.564** | **0.173** | **PASS** (CI clears zero) |
| Maia-2 value head | 0.612 | 0.185 | **PASS** (CI clears zero) |

(A no-download, illustrative 15-row stand-in is kept at
[reports/validation_sample.md](reports/validation_sample.md) for the offline smoke path.)

Reproduce the real proof in one pinned command, then read the report it writes:

```bash
uv run chess-equity headline --data <full --with-fen dump>   # -> reports/validation_headline.md
```

Both `validate --out` and `headline` write a machine-readable `*.verdict.json` beside the
report (schema `chess-equity-gate/v1`, top-level `pass`). CI and badges assert the gate from
that field instead of re-parsing the markdown — `chess-equity gate-check <verdict.json>`
exits 0 only when `pass` is true (and is run on every push in [CI](.github/workflows/ci.yml)).

Full method, the gate's exact meaning, and a no-download sample run are in
[Data & validation](#data--validation) below.

## Install

```bash
uv sync --extra dev      # python-chess + pytest in .venv
```

That covers the core CLI, tests, and CI.

The classic centipawn bar we compare against is a real engine eval, not a material
count. The objective engine is **Stockfish** (open source, a small local UCI binary
that `python-chess` drives); install it to get real centipawns:

```bash
brew install stockfish          # macOS  (apt-get install stockfish on Debian/Ubuntu)
export STOCKFISH_PATH=/path/...  # or point at any UCI binary
```

`StockfishEngine` resolves `path=` → `$STOCKFISH_PATH` → `stockfish` on `PATH`, and
raises a clear hint rather than silently falling back to material. Tests stay
engine-free via an injectable backend, so `uv run pytest` needs no binary. (The bar
still defaults to the material placeholder until Stockfish is wired in as the default.)

Anything beyond `uv sync --extra dev` — the Stockfish binary, the `data`/`maia2`/`plots`
extras, a Lichess dump, Maia-2 weights, network access — is opt-in per task and
catalogued in **[DEPENDENCIES.md](DEPENDENCIES.md)**: one row per external requirement
with its install command, which tasks need it, and whether CI needs it (it never should).

## Use

```bash
uv run chess-equity eval                                    # startpos, 1500 vs 1500
uv run chess-equity eval "<fen>" --white-elo 1800 --black-elo 1600
uv run chess-equity eval --pgn game.pgn                     # annotate every move
uv run chess-equity eval "<fen>" --json                     # structured payload (one object; --pgn -> one per ply)
uv run chess-equity grade --pgn game.pgn --white-elo 1200 --black-elo 1200
uv run chess-equity broadcast --pgn game.pgn --interval 0   # stream per-move equity
uv run chess-equity eval --white-profile magnuscarlsen      # personalize to a Lichess player
uv run chess-equity eval --white-profile "Alice@games.pgn"  # ...or profile offline from a PGN
```

```
[###############---------------]  50.0% (W)  W/D/L 33.4/33.2/33.4  [lichess-baseline]  cp +0
```

`grade` scores a move by the change in the **mover's** equity against what their
rating-peers were expected to play, so a strong move can read **positive** (the classic
centipawn-loss grade is capped at 0). `broadcast` emits one JSON event per move for live
feeds and the OBS overlay (`overlay/`) — point it at a live Lichess broadcast round and
into OBS with the [streamer quickstart](overlay/README.md#streamer-quickstart--a-live-lichess-broadcast-round--obs).

## Web demo

Run it locally (see [Deployment](#deployment) — this project ships **no hosted
site**):

```bash
python3 -m http.server -d web 8000     # then open http://localhost:8000
```

A static, dependency-free page that puts the equity bar next to a centipawn bar and
lets you drag both players' rating sliders — *move a slider and the equity bar moves
while the centipawn bar can't.* The bundled game is **Légal's Mate**: White's queen
"sacrifice" tanks the material count while it is in fact a forced mate, so the move is
**green on equity but red on centipawns**.

The committed demo's equity bar is **real Maia-2** output (rating-conditioned win-equity
from its value head), regenerated with `python web/build_demo.py --all --model maia2`. Its
centipawn bar, by contrast, is a deliberately **shallow material count**, not the deep
engine — a real Stockfish *solves* this mate, so a shallow bar is what makes the
contradiction visible (`--cp-engine stockfish` is an opt-in source for positional games).
The project's actual *"equity beats centipawns"* claim is the validation gate below,
which compares equity against the **real, rating-blind Stockfish** eval — not this demo.

The page reads a precomputed `web/demo-game.json` (no backend). Import a real game with
`python web/import_game.py <lichess-url>` (or `--user <name>`), then open
`http://localhost:8000/?game=imported-game.json`.

## Deployment

There is **none** — this is a local-only runnable, by design. Don't add GitHub Pages
or any other hosted-site deployment: there's no audience for a hosted page on a nukeable
research spike, and a static host can't run the live board's Stockfish + Maia-2 backend
anyway.

- **Static demo:** `python3 -m http.server -d web 8000` → http://localhost:8000
- **Live interactive board** (needs Stockfish + Maia-2, see [DEPENDENCIES.md](DEPENDENCIES.md)):
  `uv run python web/server.py` → http://localhost:8000/live.html

## Models

Every model plugs in behind one `EquityModel` interface; pick with `--model`:

| `--model` | What it is |
|-----------|------------|
| `baseline` *(default)* | Rating-blind Lichess Win% over the objective engine eval (material placeholder until Stockfish is the default). Zero heavy deps — the thing to beat. |
| `maia2` | The real rating-conditioned bar: [Maia-2](https://github.com/CSSLab/maia2)'s value head, trained on real Lichess outcomes. `pip install maia2` (pulls torch; checkpoint downloads on first use). |
| `wdl-a` | Transparent dependency-free regression: `P(W/D/L | cp, ratings, ply, tc)` with a `cp × skill` interaction. The shipped artifact is fit on **50k real Lichess positions** (`n_train=50000`); re-fit it with `chess-equity train`. |
| `maia-rollout` | Slow ground-truth oracle: play the position out, both sides erring like their rating, average `--n` rollouts (with a 95% CI). |
| `maia-search` | Maia-weighted expectimax to a fixed `--depth`/`--k`. |

Maia-2 is the first principled model; the others exist to compare against it.

## Data & validation

`chess-equity data build` turns a Lichess monthly PGN dump into a tabular dataset
(`cp_eval, ratings, ply, phase, clocks, result` — one row per `[%eval]` position),
streamed so a multi-GB `.zst` is never unpacked to disk:

```bash
uv sync --extra data
uv run chess-equity data build --month 2026-05 --sample 50000 --out data/ --format parquet
```

`chess-equity validate` is the scientific gate: does a rating-conditioned predictor beat
the rating-blind centipawn baseline at predicting **actual** results? It scores each
model with **log-loss, Brier, and ECE**, overall and sliced by rating band and phase, on
a held-out game-level split:

```bash
uv run chess-equity validate --data data/dataset.csv --models baseline,wdl-a --holdout 0.2
```

A small fixture under `data/sample/` lets tests and demos run with no download — its
numbers are a smoke test, not evidence (real evidence needs a real dump).

### Does equity beat centipawns?

"Beats centipawns" has a precise meaning here: a rating-conditioned predictor beats a
**rating-blind OBJECTIVE eval** (real Stockfish) at predicting **actual human outcomes**
— i.e. in *practical* terms. It is **not** a claim about out-tactic-ing a deep engine on
forced lines: a deep engine is right about the board, but blind to *this* player against
*that* one. (The web demo's material bar is a separate, shallow teaching foil.)

The gate's own answer is checked in at **[reports/validation_sample.md](reports/validation_sample.md)**:
a **Gate verdict** line (does each rating-conditioned model beat the rating-blind baseline on
log-loss *and* Brier, with a *significant* — CI-clears-zero — log-loss win?) followed by a
**head-to-head "where equity wins"** table that ranks slices by the baseline-minus-model
log-loss gap. On the sample, rating-conditioned equity (`wdl-a`) wins most in the lower rating
band — exactly where the rating-blind bar is most wrong — but the gate honestly reads **FAIL**
there: on 15 rows the log-loss CI straddles zero, so the point win isn't yet proof. **These
numbers are illustrative only — a 15-row sample, not statistically powered, and never the
headline proof.** The headline evidence comes from `chess-equity headline` on a real dump (see
[Results](#results) and `reports/validation_headline.md`); the route to a committed real
artifact is task 0128. Regenerate this *sample* with:

```bash
uv run chess-equity validate --data data/sample/dataset.csv --models baseline,baseline+clock,wdl-a --out reports/validation_sample.md
```

Running the full gate (real dump + Maia-2 + Stockfish, all attended-only) is an
attended/GPU one-shot — `uv sync --extra data --extra maia2 --extra plots`, build a
`--with-fen` dataset, re-fit `wdl-a`, then `chess-equity validate` against
`baseline,wdl-a,maia2`. See [DEPENDENCIES.md](DEPENDENCIES.md) for each install.

## Architecture

| Type | Role |
|------|------|
| `Equity` / `WDL` (`types.py`) | model-agnostic eval values — full WDL + the White-POV bar |
| `EquityModel` (`adapters.py`) | `(fen, white_elo, black_elo) -> Equity` — the core contract |
| `ObjectiveEngine` | `fen -> cp/mate` — `StockfishEngine` (real UCI) or the `MaterialEngine` placeholder |
| `HumanPolicy` | `fen, elo -> P(move)` — Maia plugs in here |
| `data/` | Lichess dump → `(eval, ratings, outcome)` dataset |
| `validate/` | score predictors vs real outcomes — log-loss/Brier/ECE |
| `grading.py` | Δequity move grading |
| `broadcast.py` | live feeds + per-move equity event stream |
| `cli.py` | `chess-equity` entry point; depends only on `EquityModel` |

Swap the model in `cli.build_model()` and everything else is unchanged.

## Test

```bash
uv run pytest
```
