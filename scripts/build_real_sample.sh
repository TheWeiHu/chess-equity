#!/usr/bin/env bash
# Build a realistic, tens-of-thousands-row dataset for tasks 0003/0004/0009 (task 0024).
#
# The committed data/sample/ fixture is intentionally tiny (15 rows, 3 games) — enough
# to prove the wiring, too small to mean anything statistically. This script produces
# the *real* substrate by streaming one recent Lichess month through the already-built
# `chess-equity data build --month` path. The output is NOT committed: it lands under
# data/ (gitignored except data/sample/), so the repo stays small — the reproducible
# command IS the deliverable, not a checked-in multi-GB file.
#
# Usage:
#   scripts/build_real_sample.sh                 # defaults below
#   ROWS=100000 MONTH=2026-04 scripts/build_real_sample.sh
#   WITH_FEN=1 scripts/build_real_sample.sh       # add FENs for board-model validation
#   scripts/build_real_sample.sh --dry-run        # print the command, build nothing
#
# Needs the data extra (zstandard for .zst dumps, pyarrow for parquet):
#   uv sync --extra data
set -euo pipefail

# A recent *complete* month — Lichess publishes a month's dump after it ends. Override
# with MONTH=YYYY-MM. The dump is multi-GB; --sample caps how many evaluated rows we keep.
MONTH="${MONTH:-2026-05}"
ROWS="${ROWS:-50000}"
FORMAT="${FORMAT:-parquet}"   # parquet keeps a 50k-row sample compact; csv for diffs
OUT="${OUT:-data}"
DUMP_DIR="${DUMP_DIR:-}"      # empty -> CLI default (~/.cache/chess-equity/dumps)

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] || [ "${1:-}" = "-n" ] && DRY_RUN=1

# Prefer `uv run` (resolves the project + the data extra); fall back to whatever
# `chess-equity` is already on PATH.
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run chess-equity)
else
  RUN=(chess-equity)
fi

cmd=("${RUN[@]}" data build --month "$MONTH" --sample "$ROWS" --out "$OUT" --format "$FORMAT")
[ -n "$DUMP_DIR" ] && cmd+=(--dump-dir "$DUMP_DIR")
[ "${WITH_FEN:-0}" = "1" ] && cmd+=(--with-fen)

echo "Building ~${ROWS}-row ${FORMAT} sample from Lichess ${MONTH} into ${OUT}/" >&2
printf '  ' >&2; printf '%q ' "${cmd[@]}" >&2; echo >&2

if [ "$DRY_RUN" = "1" ]; then
  echo "(dry run — nothing built)" >&2
  exit 0
fi

exec "${cmd[@]}"
