"""The realistic-sample build script (task 0024) assembles the right CLI command.

We can't download a multi-GB Lichess dump in the test gate, so we exercise the
script's ``--dry-run`` path: it must resolve env overrides into the exact
``chess-equity data build`` invocation, without building anything.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_real_sample.sh"


def _run(env_overrides: dict) -> str:
    if not SCRIPT.exists():  # pragma: no cover - script is committed
        pytest.skip("build_real_sample.sh not present")
    env = dict(os.environ)
    # Strip uv from PATH so the script takes the bare `chess-equity` fallback —
    # makes the asserted command deterministic regardless of the dev's tooling.
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    env.update(env_overrides)
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stderr


def test_dry_run_uses_defaults():
    msg = _run({"MONTH": "2026-05", "ROWS": "50000", "FORMAT": "parquet"})
    assert "chess-equity data build" in msg
    assert "--month 2026-05" in msg
    assert "--sample 50000" in msg
    assert "--format parquet" in msg
    assert "dry run" in msg.lower()


def test_dry_run_honors_overrides():
    msg = _run({"MONTH": "2026-04", "ROWS": "100000", "FORMAT": "csv", "WITH_FEN": "1"})
    assert "--month 2026-04" in msg
    assert "--sample 100000" in msg
    assert "--format csv" in msg
    assert "--with-fen" in msg
