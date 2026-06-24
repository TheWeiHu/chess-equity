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
import json
import os
import re
import subprocess
import sys
import threading
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRIPT = os.path.join(HERE, "..", "scripts", "live_demo.sh")
SAMPLE_PGN = os.path.join(ROOT, "data", "sample", "sample_games.pgn")
STREAMING_DOC = os.path.join(ROOT, "docs", "STREAMING.md")
OVERLAY_README = os.path.join(ROOT, "overlay", "README.md")

# The one-command OBS browser-source URL the demo surfaces. The PORT is configurable
# (--serve-sse / PORT=), so the durable cross-surface contract is the *path + query*:
# the overlay static page (`/`) consuming the live SSE push (`?src=/sse`). If the
# overlay's feed param or SSE path ever changes, the script banner, the CLI log line,
# and both streamer docs must move together — or this constant's guards fail.
OBS_PATH_QUERY = "/?src=/sse"
# Matches `http://localhost:<port>/?src=/sse` in docs/banners (port-agnostic).
OBS_URL_RE = re.compile(r"http://localhost:\d+/\?src=/sse")


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


# --------------------------------------------------------------------------- #
# Go-live smoke test: the real --serve-sse path + OBS-URL/docs drift (task 0270)
# --------------------------------------------------------------------------- #
#
# The --check path above proves the demo's finite JSONL replay still emits events, but
# the actual streamer go-live is `broadcast --serve-sse PORT`: one process that serves
# the transparent `overlay/` browser source AND a `/sse` push on one port, which a
# streamer drops into OBS as `http://localhost:PORT/?src=/sse`. Nothing guarded that
# end-to-end — `test_broadcast_replay_e2e.py` drives the in-process server WITHOUT the
# overlay static dir, so it can't catch the overlay assets failing to serve (which the
# 0247 cli/ package refactor silently did, by leaving `_overlay_static_dir()` one
# directory level too shallow). These tests close that gap and pin the OBS URL to docs.


def _serve_sse(port="0", interval="0", banner_timeout=20.0):
    """Start `broadcast --serve-sse` hermetically and return (proc, port, stderr_lines).

    HERMETIC like live_demo.sh's --check path (task 0252): pinned to THIS tree via
    `PYTHONPATH=src` + `python3 -m chess_equity.cli`, never a PATH/`uv` lookup that the
    nightshift shared-PATH fleet can point at a sibling worktree. `--serve-sse 0` lets the
    OS pick a free port; the bound port is parsed back from the banner the CLI logs.

    Caller MUST terminate the returned proc (serve_sse runs forever).
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"  # so the startup banner flushes through the pipe at once
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "chess_equity.cli",
            "broadcast", "--pgn", SAMPLE_PGN, "--serve-sse", port, "--interval", interval,
        ],
        cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )

    # Read the startup banner (a few lines, ending in "Ctrl-C to stop.") off a thread so
    # a crash-before-banner can't hang the test forever.
    lines, done = [], threading.Event()

    def _read_banner():
        for line in proc.stderr:  # type: ignore[union-attr]
            lines.append(line)
            if "Ctrl-C to stop." in line:
                break
        done.set()

    threading.Thread(target=_read_banner, daemon=True).start()
    done.wait(timeout=banner_timeout)
    banner = "".join(lines)
    m = re.search(r"http://localhost:(\d+)/sse", banner)
    assert m, f"serve-sse never printed its bound-port banner:\n{banner}\nrc={proc.poll()}"
    return proc, int(m.group(1)), banner


def _read_sse_events(port, timeout=15.0):
    """Connect to /sse and parse the `data: <json>` frames, exactly like overlay/feed.js."""
    body = urllib.request.urlopen(f"http://127.0.0.1:{port}/sse", timeout=timeout).read().decode("utf-8")
    events = []
    for frame in body.split("\n\n"):
        frame = frame.strip()
        if frame.startswith("data: "):
            events.append(json.loads(frame[len("data: ") :]))
    return events


def test_demo_serve_sse_serves_overlay_and_wellformed_equity_events():
    """The real go-live path: `--serve-sse` serves the overlay AND a per-move equity feed.

    This is the smoke test that would have caught the 0247 refactor breaking the
    one-command overlay: it asserts BOTH halves of the one port — the `/sse` push emits
    well-formed per-move equity events, and `/` serves the overlay browser source (so
    `http://localhost:PORT/?src=/sse` actually paints a bar, not a 404).
    """
    proc, port, banner = _serve_sse()
    try:
        # (1) /sse emits the documented overlay schema: a leading `game` metadata event,
        #     then one `position` per ply carrying a usable White-POV equity in [0, 1].
        events = _read_sse_events(port)
        assert events, "the /sse feed produced no events at all"
        assert events[0]["type"] == "game", f"first event must be `game` metadata, got {events[0]}"
        positions = [e for e in events if e.get("type") == "position"]
        assert positions, "the /sse feed emitted no per-move `position` events"
        for evt in positions:
            eq = evt["equity"]
            assert isinstance(eq, float) and 0.0 <= eq <= 1.0, f"bad equity on ply {evt.get('ply')}: {eq!r}"

        # (2) the overlay static page is served on the SAME port, so the one-command OBS
        #     URL works. `--serve-sse` must composite the browser source, not just /sse.
        index = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).read().decode("utf-8")
        assert "streaming overlay" in index.lower(), "`/` did not serve the overlay index.html"
        assert "overlay.js" in index, "overlay index served but missing its overlay.js bundle"

        # (3) the CLI's printed one-command OBS URL is the documented path+query.
        assert OBS_PATH_QUERY in banner, f"serve-sse banner is missing the OBS URL {OBS_PATH_QUERY!r}:\n{banner}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_obs_browser_source_url_matches_docs():
    """Doc-drift guard: the OBS URL the demo surfaces matches both streamer docs.

    The script banner, the CLI serve log, and the two docs a streamer follows
    (docs/STREAMING.md and overlay/README.md) must all point OBS at the same
    `/?src=/sse` browser source. If any drifts (e.g. the overlay's feed param is
    renamed), the go-live instructions stop matching reality — this fails first.
    """
    with open(SCRIPT, encoding="utf-8") as fh:
        script_src = fh.read()
    assert OBS_PATH_QUERY in script_src, f"live_demo.sh no longer prints the {OBS_PATH_QUERY!r} OBS URL"

    cli_init = os.path.join(ROOT, "src", "chess_equity", "cli", "__init__.py")
    serve_src = os.path.join(ROOT, "src", "chess_equity", "broadcast.py")
    # The CLI logs the one-command overlay URL from broadcast.serve_sse.
    with open(serve_src, encoding="utf-8") as fh:
        assert OBS_PATH_QUERY in fh.read(), f"broadcast.serve_sse no longer logs the {OBS_PATH_QUERY!r} OBS URL"
    assert os.path.isfile(cli_init), "cli package missing"

    for doc in (STREAMING_DOC, OVERLAY_README):
        with open(doc, encoding="utf-8") as fh:
            text = fh.read()
        assert OBS_URL_RE.search(text), (
            f"{os.path.relpath(doc, ROOT)} no longer documents the OBS browser-source "
            f"URL matching {OBS_URL_RE.pattern!r} — streamer go-live docs have drifted"
        )
