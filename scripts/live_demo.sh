#!/usr/bin/env bash
# scripts/live_demo.sh — clone-to-live-bar streaming demo in one command (task 0248).
#
# The North-star deliverable is the streamer-facing OBS overlay, but its pieces live
# across `broadcast` (the per-move event producer) and `overlay/` (the browser source
# that paints the bar). This script chains them: it replays a bundled sample PGN through
# `chess-equity broadcast --serve-sse`, which serves BOTH the transparent `overlay/`
# browser source AND a live Server-Sent-Events feed on one port — so a freshly-cloned
# checkout goes straight to a live, move-by-move equity bar in your browser / OBS.
#
# Offline + unattended-safe: it replays data/sample/sample_games.pgn (the sanctioned
# tiny fixture — illustrative, not evidence; see reports/validation_sample.md). No
# network, no torch, no Lichess round needed.
#
# Usage:
#   scripts/live_demo.sh                 # serve the live overlay, print the URL, block
#   PORT=9001 scripts/live_demo.sh       # pick the port
#   PGN=mygame.pgn scripts/live_demo.sh  # replay a different PGN
#   INTERVAL=0 scripts/live_demo.sh      # blast all moves instantly (no pacing)
#   scripts/live_demo.sh --check         # offline smoke: finite replay, assert non-empty
#
# Open the printed URL in any browser (transparent background), or add it as an OBS
# "Browser" source. Ctrl-C to stop the server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PGN="${PGN:-$ROOT/data/sample/sample_games.pgn}"
PORT="${PORT:-8779}"
# Pace the replay so the bar visibly *moves* move-by-move (seconds between plies).
# --check overrides this to 0 for an instant, terminating smoke run.
INTERVAL="${INTERVAL:-1.5}"

# Resolve a runner for the `chess-equity` CLI without assuming a global install:
# an on-PATH entry point, the repo venv, then `uv run`, then the module.
resolve_cli() {
  if command -v chess-equity >/dev/null 2>&1; then echo "chess-equity"; return; fi
  if [ -x "$ROOT/.venv/bin/chess-equity" ]; then echo "$ROOT/.venv/bin/chess-equity"; return; fi
  if command -v uv >/dev/null 2>&1; then echo "uv run --project $ROOT chess-equity"; return; fi
  echo "python3 -m chess_equity.cli"
}
CLI="$(resolve_cli)"

if [ ! -f "$PGN" ]; then
  echo "error: sample PGN not found: $PGN" >&2
  exit 1
fi

# --check: the offline smoke path the test drives. Run the *finite* JSONL replay (no
# --serve-sse, --interval 0 so it terminates instantly) and assert it produced a
# non-empty overlay event stream — at least one `position` event in the overlay schema.
#
# This path is HERMETIC by design: it pins the CLI to THIS script's own source tree
# (PYTHONPATH=$ROOT/src) instead of resolve_cli's PATH lookup. The nightshift fleet runs
# many git worktrees behind one shared PATH, so a global `chess-equity` entry point is an
# editable install pointing at *whichever* worktree last ran `pip install -e` — replaying
# a sibling branch's code that may emit zero overlay events and turn the gate red
# (task 0252). The user-facing serve path below still uses resolve_cli for a friendly
# clone-to-live experience; only the CI smoke must be reproducible regardless of PATH.
if [ "${1:-}" = "--check" ]; then
  out="$(PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m chess_equity.cli broadcast --pgn "$PGN" --interval 0 2>/dev/null)"
  positions="$(printf '%s\n' "$out" | grep -c '"type": *"position"' || true)"
  if [ "${positions:-0}" -lt 1 ]; then
    echo "FAIL: live_demo produced no overlay position events" >&2
    printf '%s\n' "$out" | head -5 >&2
    exit 1
  fi
  echo "ok: live_demo emitted $positions overlay position event(s) from $(basename "$PGN")"
  exit 0
fi

URL="http://localhost:${PORT}/?src=/sse"
cat >&2 <<BANNER
─────────────────────────────────────────────────────────────────
 chess-equity live demo — clone to live equity bar
   feed   : $(basename "$PGN") (bundled sample, illustrative-not-evidence)
   overlay: $URL
─────────────────────────────────────────────────────────────────
 Open that URL in a browser, or add it as an OBS "Browser" source.
 The equity bar moves as each move replays. Ctrl-C to stop.
─────────────────────────────────────────────────────────────────
BANNER

# --serve-sse serves the overlay/ static files AND the /sse feed on one port; the
# finite --pgn replay paces a move every $INTERVAL seconds, then the stream ends.
exec $CLI broadcast --pgn "$PGN" --serve-sse "$PORT" --interval "$INTERVAL"
