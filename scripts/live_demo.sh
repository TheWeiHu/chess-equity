#!/usr/bin/env bash
# scripts/live_demo.sh — clone-to-live-bar in ONE command (task 0248).
#
# The north-star deliverable is the streamer-facing OBS overlay, but its pieces
# (`broadcast`, `overlay/`, `grade`) were scattered with no single entry point.
# This script chains them: it replays a bundled `data/sample/` PGN through
# `chess-equity broadcast --serve-sse`, which serves BOTH the transparent
# overlay browser-source AND a live `/sse` event stream on one port. A reviewer
# or streamer goes from `git clone` to a live, moving equity bar with one line —
# no pip extras, no torch, no network, no Lichess token.
#
# Usage:
#   scripts/live_demo.sh                 # start the demo; prints the overlay URL
#   PORT=9000 scripts/live_demo.sh       # pick the port (default 8788)
#   PGN=data/sample/round_games.pgn scripts/live_demo.sh   # swap the bundled PGN
#   scripts/live_demo.sh --open          # also open the overlay in your browser
#   scripts/live_demo.sh --check         # OFFLINE SMOKE: assert a non-empty
#                                        #   overlay event stream, then exit
#
# DATA POLICY: the bundled data/sample/ PGNs are illustrative fixtures, NOT
# validation evidence (see CLAUDE.md / reports/validation_sample.md). Point
# `broadcast` at `--round <id>` or `--url <pgn-url>` for a real live feed.
set -euo pipefail

# Run from the repo root regardless of the caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PGN="${PGN:-data/sample/sample_games.pgn}"
PORT="${PORT:-8788}"
RUN="${CHESS_EQUITY:-uv run chess-equity}"   # override to a bare `chess-equity` if installed

if [ ! -f "$PGN" ]; then
  echo "live_demo: bundled PGN not found: $PGN" >&2
  exit 1
fi

overlay_url() { echo "http://localhost:$1/?src=/sse"; }

# --check: offline self-test of the whole chain. Start the SAME server the demo
# uses on an OS-picked port, connect to /sse as the overlay's EventSource would,
# and assert it emits at least one overlay event. Exits 0 on success.
if [ "${1:-}" = "--check" ]; then
  errlog="$(mktemp)"
  trap 'rm -f "$errlog"; [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null || true' EXIT
  # port 0 → the OS picks a free port; broadcast logs the bound port to stderr.
  $RUN broadcast --pgn "$PGN" --serve-sse 0 >/dev/null 2>"$errlog" &
  pid=$!
  bound=""
  for _ in $(seq 1 60); do
    bound="$(sed -n 's#.*localhost:\([0-9][0-9]*\)/sse.*#\1#p' "$errlog" | head -1)"
    [ -n "$bound" ] && break
    kill -0 "$pid" 2>/dev/null || { echo "live_demo --check: server exited early" >&2; cat "$errlog" >&2; exit 1; }
    sleep 0.25
  done
  if [ -z "$bound" ]; then
    echo "live_demo --check: SSE server never reported a bound port" >&2
    cat "$errlog" >&2
    exit 1
  fi
  # Read /sse to EOF (a finite --pgn replay terminates) and count overlay events.
  count="$(python3 - "http://localhost:$bound/sse" <<'PY'
import sys, urllib.request
url = sys.argv[1]
n = 0
with urllib.request.urlopen(url, timeout=15) as resp:
    ctype = resp.headers.get("Content-Type", "")
    if "text/event-stream" not in ctype:
        sys.stderr.write(f"unexpected Content-Type: {ctype!r}\n")
        sys.exit(2)
    for raw in resp:
        if raw.decode("utf-8", "replace").startswith("data:"):
            n += 1
print(n)
PY
)"
  if [ "${count:-0}" -ge 1 ] 2>/dev/null; then
    echo "OK: $count overlay events on $(overlay_url "$bound")"
    exit 0
  fi
  echo "live_demo --check: overlay event stream was empty (count=$count)" >&2
  exit 1
fi

# Default: launch the live demo and run until Ctrl-C.
URL="$(overlay_url "$PORT")"
cat >&2 <<BANNER
┌─ chess-equity live demo ───────────────────────────────────────────
│ Replaying $PGN move-by-move as a live equity feed.
│
│ Overlay (open in a browser, or add as an OBS Browser source):
│   $URL
│
│ Add it in OBS as a transparent Browser source — see overlay/README.md.
│ data/sample is an illustrative fixture, not validation evidence.
│ Ctrl-C to stop.
└────────────────────────────────────────────────────────────────────
BANNER

if [ "${1:-}" = "--open" ]; then
  ( sleep 1
    if command -v open >/dev/null 2>&1; then open "$URL"
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
    fi ) >/dev/null 2>&1 &
fi

exec $RUN broadcast --pgn "$PGN" --serve-sse "$PORT"
