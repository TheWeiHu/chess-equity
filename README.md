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

## Architecture

| Type | Role |
|------|------|
| `Equity` / `WDL` (`types.py`) | model-agnostic eval values — full WDL + the White-POV bar |
| `EquityModel` (`adapters.py`) | `(fen, white_elo, black_elo) -> Equity` — the core contract |
| `ObjectiveEngine` | `fen -> cp/mate` — Stockfish/Lc0 plug in here (placeholder: material only) |
| `HumanPolicy` | `fen, elo -> P(move)` — Maia plugs in here (task 0005) |
| `bar.py` | ASCII rendering of the bar |
| `grading.py` | Δequity move grading — peer-relative + classic (task 0008) |
| `cli.py` | `chess-equity` entry point; depends only on `EquityModel` |

Swap the model in `cli.build_model()` and everything else is unchanged.

## Test

```bash
uv run pytest
```
