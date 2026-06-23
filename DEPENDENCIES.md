# Dependencies

The single source of truth for **every external/runtime requirement** of this project —
not just pip. `pyproject.toml` extras cover pip deps; the things that actually *block*
work (a Stockfish binary, a multi-GB Lichess dump, Maia-2 weights, torch, network access)
live outside pip, and this file is where they're declared.

Read it top-to-bottom on a fresh machine and every currently-parked task becomes runnable.

> **The core path needs none of this.** `uv sync --extra dev` + `python-chess` runs the
> baseline CLI, the test suite, and CI. Everything below is opt-in for a specific task.
> Tests/CI must stay in the **"CI?: no"** column — heavy/external deps sit behind
> injectable fakes so `uv run pytest` never needs a binary, a dump, weights, or a network.

## Manifest

| Dependency | Category | Install / fetch | Needed by | CI? |
|---|---|---|---|---|
| **python-chess** (`chess>=1.10`) | pip (core) | `uv sync` (always installed) | everything | yes |
| **pytest** (`dev` extra) | pip-extra | `uv sync --extra dev` | the test suite | yes |
| **data** extra (`zstandard`, `pyarrow`, `pandas`) | pip-extra | `uv sync --extra data` | 0002/0024/0034/0040 — `.zst` streaming + Parquet/DataFrame | no (CSV path is fake-free) |
| **plots** extra (`matplotlib>=3.5`) | pip-extra | `uv sync --extra plots` | 0036 — `validate --plots` reliability PNGs | no |
| **maia2** extra (`maia2`) | pip-extra (heavy, pulls **torch**) | `uv sync --extra maia2` (or `pip install maia2`) | 0005/0014 — `--model maia2` | no (fake inference backend) |
| **wdl-net** extra (`torch`) | pip-extra (heavy: torch only) | `uv sync --extra wdl-net` | 0013 — `train-net` / `--model wdl-net` (end-to-end board→WDL net; lighter than `maia2`) | no (torch-gated tests `importorskip`; encoder is pure-chess) |
| **torch** | pip (heavy, via `maia2` or `wdl-net` extras) | comes with the `maia2` or `wdl-net` extra | 0005/0013/0014 — neural models | no |
| **Stockfish binary** (UCI) | system-binary | `brew install stockfish` / `apt-get install stockfish`; or `export STOCKFISH_PATH=/path/to/stockfish` | 0028/0035/0042/0043 — the real centipawn bar | no (injectable `Backend`; never silently falls back to material) |
| **Maia-2 weights** (~23M-param checkpoint) | model-weights | auto-downloaded by `maia2` on first `evaluate(...)`, cached by the library | 0005/0014 | no |
| **Lichess monthly dump** (`.pgn.zst`, multi-GB) | dataset | `chess-equity data build --month YYYY-MM` prints the canonical URL to `curl`; the build **streams** the `.zst` (never fully unpacked) | 0024/0034/0013 — real ~50k-row training/validation data | no (15-row `data/sample/` fixture covers tests) |
| **Lichess API** (public, optional `$LICHESS_TOKEN` for rate limit) | network/API | none to install; needs outbound network | 0011/0014/0039 — game import / history mining | no (importer caches; tests use a fake opener) |

**Resolution order for the engine:** explicit `path=` → `$STOCKFISH_PATH` → `stockfish`
on `PATH` (see `src/chess_equity/stockfish.py`).

**Verify a provisioned host:** `chess-equity doctor` resolves Stockfish and runs a real
eval, then imports Maia-2 and runs a real inference, printing PASS/FAIL per engine (exit
non-zero if any is missing/broken). Use it after `brew install stockfish` +
`uv sync --extra maia2` to confirm the two non-core bars actually work.

## Why a task gets "parked" on the unattended host

The nightshift loop runs in a sandbox that **cannot** provision: a system binary
(Stockfish), a multi-GB dataset download, heavy pip installs (torch), model weights, or
arbitrary network. A task needing any **non-`CI?: yes`** row above is *attended-only* —
do it on a real machine after provisioning that row. Tasks needing only core/`dev` are
sandbox-doable.

## Upkeep rule

**Any task that introduces a new external dependency adds its row here in the same PR**
— and a parked/attended task's body should link to its row instead of re-deriving install
steps inline. Keep this file tooling-free (a plain markdown table). A future CI grep could
fail the build if a new `pyproject` extra or a `brew install` / binary reference appears
with no matching row here (deferred — see task follow-ups).
