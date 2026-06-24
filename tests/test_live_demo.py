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

A second test (task 0270) goes one step further and guards the *go-live* path the
streamer actually uses: it launches the demo's **serve** mode, connects to the live
``/sse`` endpoint exactly as an OBS browser source would, asserts the frames are
well-formed per-move equity events, AND asserts the OBS URL the demo prints is the same
one ``docs/STREAMING.md`` / ``overlay/README.md`` tell streamers to paste. That catches
the overlay / demo / docs falling out of sync as broadcast features churn — all still
offline (loopback only, bundled PGN, no network).
"""
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
SCRIPT = os.path.join(ROOT, "scripts", "live_demo.sh")
SRC = os.path.join(ROOT, "src")
DOC_PATHS = (
    os.path.join(ROOT, "docs", "STREAMING.md"),
    os.path.join(ROOT, "overlay", "README.md"),
)


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


# --- go-live serve path (task 0270) -----------------------------------------------

# The OBS browser-source URL the demo prints and the docs document. The port differs by
# context (the demo defaults to 8779, the docs use 8777), so the invariant that must not
# drift is the path+query — `/?src=/sse` is what makes the overlay read the live push.
_OBS_URL_RE = re.compile(r"http://localhost:(\d+)(/\?src=/sse)\b")


def _free_loopback_port():
    """Reserve an OS-assigned loopback port, then release it for the demo to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _hermetic_cli_dir(dirpath):
    """Write a ``chess-equity`` shim into ``dirpath`` that runs THIS worktree's CLI.

    ``live_demo.sh``'s serve path resolves the CLI via ``resolve_cli`` (PATH first), so
    on a box with a global ``chess-equity`` — e.g. an editable install pointing at a
    *sibling* nightshift worktree — the demo would replay foreign code and flake the gate
    red (the shared-PATH contamination of task 0252). Prepending this shim pins the demo
    to the tree under test, hermetically, without touching the script. The shim's
    ``python3`` finds ``chess_equity`` via the ``PYTHONPATH=src`` we pass to the demo.
    """
    shim = os.path.join(dirpath, "chess-equity")
    with open(shim, "w") as fh:
        fh.write('#!/usr/bin/env bash\nexec python3 -m chess_equity.cli "$@"\n')
    os.chmod(shim, 0o755)
    return dirpath


def _wait_for_port(port, deadline):
    """Block until ``port`` accepts a loopback connection (the demo's server bound)."""
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


def _read_first_position(port, *, timeout):
    """Connect to ``/sse`` as the overlay would; return the first ``position`` event.

    Validates every data frame against the documented overlay schema en route, so a
    malformed feed fails loudly rather than silently painting a dead bar.
    """
    from chess_equity.doctor import validate_overlay_event

    url = f"http://127.0.0.1:{port}/sse"
    frames = 0
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type", "")
        assert "text/event-stream" in ctype, f"/sse Content-Type was {ctype!r}"
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue  # SSE comments (": keepalive") and blank separators
            frames += 1
            event = json.loads(line[len("data:"):].strip())
            validate_overlay_event(event, where="live_demo /sse frame")
            if event.get("type") == "position":
                return event, frames
    return None, frames


def test_live_demo_serves_sse_and_obs_url_matches_docs(tmp_path):
    """End-to-end go-live guard: the demo serves a well-formed live ``/sse`` equity
    feed AND prints the same OBS browser-source URL the docs tell streamers to paste."""
    assert os.access(SCRIPT, os.X_OK), "live_demo.sh must be executable"

    port = _free_loopback_port()
    env = dict(os.environ)
    # Pin the CLI to this tree (see _hermetic_cli_dir) and blast every ply instantly so
    # the finite sample replay is over in well under a second.
    env["PATH"] = _hermetic_cli_dir(str(tmp_path)) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = SRC + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PORT"] = str(port)
    env["INTERVAL"] = "0"

    # Drain the demo's banner (stderr) on a thread so its pipe never blocks the server.
    proc = subprocess.Popen(
        [SCRIPT],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,  # own process group → clean teardown of the exec'd CLI
    )
    banner = []

    def _drain():
        assert proc.stderr is not None  # stderr=PIPE above
        for line in proc.stderr:
            banner.append(line)

    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()

    try:
        assert _wait_for_port(port, time.monotonic() + 20), (
            f"demo server never bound on :{port}\n{''.join(banner)}"
        )
        # make_events is a per-connection factory, so this connection replays the sample
        # from scratch — no race against the first replay finishing.
        position, frames = _read_first_position(port, timeout=15)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        drainer.join(timeout=2)

    # 1. The live feed served a well-formed per-move equity event.
    assert position is not None, (
        f"/sse reachable but emitted no position event ({frames} frame(s))\n{''.join(banner)}"
    )
    equity = position.get("equity")
    assert isinstance(equity, (int, float)) and 0.0 <= equity <= 1.0, (
        f"position event carried a bad White-POV equity: {position!r}"
    )

    # 2. The OBS URL the demo printed is the one the docs document (path+query, any port).
    text = "".join(banner)
    match = _OBS_URL_RE.search(text)
    assert match, f"demo did not print an OBS browser-source URL:\n{text}"
    printed_path = match.group(2)  # "/?src=/sse"
    assert int(match.group(1)) == port, "printed URL port should match the demo's PORT"
    for doc in DOC_PATHS:
        with open(doc, encoding="utf-8") as fh:
            doc_text = fh.read()
        assert printed_path in doc_text, (
            f"demo prints OBS path {printed_path!r} but {os.path.relpath(doc, ROOT)} "
            f"does not document it — overlay/demo/docs have drifted apart"
        )
