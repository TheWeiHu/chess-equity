"""Smoke test for the one-command streaming demo (task 0248).

``scripts/live_demo.sh`` is the clone-to-live-bar entry point: it replays the bundled
sample PGN through ``broadcast --serve-sse`` so a reviewer/streamer goes from a fresh
checkout to a live equity bar in one command. The default mode blocks on a server, so
the script's ``--check`` flag runs the *finite* JSONL replay path that the same demo
chains under the hood and asserts it produced a non-empty overlay event stream.

This test drives that ``--check`` path so CI catches a regression where the demo stops
emitting overlay events (a broken CLI runner, a moved sample fixture, a schema break).
Offline + unattended-safe — replays ``data/sample/sample_games.pgn`` (illustrative, not
evidence; see ``reports/validation_sample.md``). No server, no network.
"""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "scripts", "live_demo.sh")


def test_live_demo_check_emits_overlay_events():
    assert os.access(SCRIPT, os.X_OK), "live_demo.sh must be executable"
    proc = subprocess.run(
        [SCRIPT, "--check"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"--check failed: {proc.stderr}\n{proc.stdout}"
    # The bundled first sample game is a 7-ply scholar's mate, so the demo must emit a
    # handful of overlay position events — assert it reported at least one.
    assert "overlay position event" in proc.stdout, proc.stdout
    count = int(proc.stdout.split("emitted", 1)[1].strip().split()[0])
    assert count >= 1, f"expected a non-empty overlay event stream, got {count}"
