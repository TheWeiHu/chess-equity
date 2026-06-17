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

## Architecture

| Type | Role |
|------|------|
| `Equity` / `WDL` (`types.py`) | model-agnostic eval values — full WDL + the White-POV bar |
| `EquityModel` (`adapters.py`) | `(fen, white_elo, black_elo) -> Equity` — the core contract |
| `ObjectiveEngine` | `fen -> cp/mate` — Stockfish/Lc0 plug in here (placeholder: material only) |
| `HumanPolicy` | `fen, elo -> P(move)` — Maia plugs in here (task 0005) |
| `bar.py` | ASCII rendering of the bar |
| `data/` | Lichess PGN dump -> `(eval, ratings, outcome)` dataset (task 0002) |
| `cli.py` | `chess-equity` entry point; depends only on `EquityModel` |

Swap the model in `cli.build_model()` and everything else is unchanged.

## Test

```bash
uv run pytest
```
