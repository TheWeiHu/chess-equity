#!/usr/bin/env bash
# The single source of truth for "is the working tree green?"
#
# Run this LOCALLY before you push — it is the SAME check CI runs. Both the CI
# workflow (.github/workflows/ci.yml) and the nightshift fleet's --test-cmd call this
# script, so the local green-gate and CI can never drift apart again. (Before this
# existed, the fleet gated on `pytest -q` = 857 tests/ only, while CI ran the wider
# `pytest tests baseline overlay web` + the Node suites — so branches merged green and
# then failed CI on the suites the gate never ran. That gap was ~half of all CI failures.)
#
# Usage:  scripts/gate.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== pytest (tests + baseline + overlay + web) =="
# Prefer python3.11 (the version CI pins via setup-python) so local and CI run the SAME
# interpreter; fall back to plain python3 if 3.11 isn't installed. Not the bare `python`
# alias — that only exists in interactive shells, so scripts must use python3.x.
PY="$(command -v python3.11 || command -v python3)"
echo "   interpreter: $("$PY" -V 2>&1)"
"$PY" -m pytest tests baseline web -q

# Dependency-free Node test suites. GLOBBED on purpose: a new overlay/*.test.js (or
# web/test_*.js) added by a task is gated automatically, so CI and this gate never drift
# even as the suite grows. (CI used to list each `node …` step by hand and already missed
# overlay/test_toast.test.js — exactly the drift this prevents.)
echo "== node test suites =="
shopt -s nullglob
for t in web/test_live.js overlay/*.test.js; do
  echo "-- node $t"
  node "$t"
done

echo "== GATE PASS =="
