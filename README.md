# chess-equity

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

Scaffold (task 0001). The package defines the contract and ships a **placeholder**
model — the rating-blind Lichess Win% over a trivial material eval — so the CLI runs
end-to-end. The real rating-conditioned models (regression in 0004, Maia-2's
`win_prob` in 0005) plug in behind the same `EquityModel` interface.

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

# CSV is the default and needs no extra:
uv run chess-equity data build --pgn data/sample/sample_games.pgn --out data/sample
```

A small committed fixture lives in `data/sample/` so tests and downstream tasks
(0003/0004/0009) have substrate without a download. Load a built dataset with
`chess_equity.data.load_rows(path)` (typed rows, dependency-free) or
`load_dataframe(path)` (pandas, needs the data extra).

`cp_eval` and `result` are both White-POV; mate scores are clamped to ±10000 cp.
`--month YYYY-MM` prints the canonical Lichess dump URL to fetch (auto-download is a
follow-up). See `data/schema.py` for the column contract.

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
(Lichess's rating-blind Win% over the row's centipawns — the thing to beat). Approach
A (0004) registers as a predictor with no harness change. A demonstration run on the
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
