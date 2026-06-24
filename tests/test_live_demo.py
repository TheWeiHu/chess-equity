"""The one-command live demo (task 0248) actually produces a live overlay stream.

``scripts/live_demo.sh`` is the clone-to-live-bar entry point: it chains a bundled
``data/sample/`` PGN through ``chess-equity broadcast --serve-sse`` into the
``overlay/`` browser-source. The acceptance bar is that a reviewer who just cloned
the repo can run ONE command, offline, and get a non-empty overlay event stream.

The script's own ``--check`` mode is that end-to-end self-test: it starts the same
SSE server the demo uses on an OS-picked port, connects to ``/sse`` as the overlay's
``EventSource`` would, and asserts at least one overlay event arrives. This test runs
that path and asserts it succeeds with a positive event count — no network, no torch
(the default ``LichessBaselineModel``), no Lichess token.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "live_demo.sh"


def _env() -> dict:
    """Prefer the installed ``chess-equity`` console script over ``uv run`` so the
    self-test doesn't pay uv's resolve/startup cost (pytest already runs inside the
    project venv); fall back to the script's ``uv run`` default if it's absent."""
    env = dict(os.environ)
    installed = shutil.which("chess-equity")
    if installed:
        env["CHESS_EQUITY"] = installed
    return env


def test_live_demo_check_emits_overlay_events():
    if not SCRIPT.exists():  # pragma: no cover - script is committed
        pytest.skip("live_demo.sh not present")
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--check"],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"--check failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    # The script prints "OK: N overlay events on http://localhost:PORT/?src=/sse".
    m = re.search(r"OK:\s*(\d+)\s+overlay events", proc.stdout)
    assert m, f"no 'OK: N overlay events' line in:\n{proc.stdout}"
    assert int(m.group(1)) >= 1, "overlay event stream was empty"
    # The reviewer-facing line must point at the one-command overlay URL.
    assert "/?src=/sse" in proc.stdout


def test_live_demo_default_banner_prints_overlay_url():
    """The default (server) mode prints the overlay URL before it blocks on serve.

    We can't let the server run forever in a test, so drive the script with a tiny
    timeout and assert the pre-serve banner already advertised the overlay URL — the
    "prints the overlay URL/file" half of the acceptance criteria.
    """
    if not SCRIPT.exists():  # pragma: no cover - script is committed
        pytest.skip("live_demo.sh not present")
    env = _env()
    env["PORT"] = "8799"
    try:
        proc = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=8,
        )
        combined = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        # Expected: the server blocks. The banner is already on stderr by now.
        combined = (exc.stdout or b"")
        combined = combined.decode() if isinstance(combined, bytes) else (combined or "")
        err = (exc.stderr or b"")
        combined += err.decode() if isinstance(err, bytes) else (err or "")
    assert "http://localhost:8799/?src=/sse" in combined, combined
