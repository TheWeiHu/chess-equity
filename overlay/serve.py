#!/usr/bin/env python3
"""Dependency-light dev server for the chess-equity OBS overlay.

Serves the static overlay files AND an SSE endpoint at /sse that replays
mock-game.json as a live event stream — so you can point the overlay at a real
push feed (the path the live ingestion task, 0018, will use), not just the
.json replay.

Usage:
    python3 serve.py [--port 8777] [--game mock-game.json] [--speed 1.0]

Then in OBS add a Browser source:
    Static replay:  http://localhost:8777/
    Live SSE feed:  http://localhost:8777/?src=/sse

Stdlib only — no pip install.
"""
import argparse
import http.server
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def load_events(game_path):
    with open(game_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["events"] if isinstance(data, dict) else data


class Handler(http.server.SimpleHTTPRequestHandler):
    # Set by main() before serving.
    events = []
    speed = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def do_GET(self):  # noqa: N802 (stdlib API)
        if self.path.split("?")[0] == "/sse":
            return self._stream_sse()
        return super().do_GET()

    def _stream_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            for evt in self.events:
                payload = json.dumps(evt)
                self.wfile.write(("data: " + payload + "\n\n").encode("utf-8"))
                self.wfile.flush()
                delay = evt.get("delayMs", 0) / 1000.0 / max(self.speed, 1e-6)
                if delay:
                    time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError):
            pass  # OBS / browser closed the source — normal.

    def log_message(self, format, *args):  # quieter console
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--game", default=os.path.join(HERE, "mock-game.json"))
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    Handler.events = load_events(args.game)
    Handler.speed = args.speed

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("chess-equity overlay server: http://localhost:%d/" % args.port)
    print("  static replay : http://localhost:%d/" % args.port)
    print("  live SSE feed : http://localhost:%d/?src=/sse" % args.port)
    print("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
