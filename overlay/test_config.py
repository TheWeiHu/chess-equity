#!/usr/bin/env python3
"""Content tests for the overlay setup page (task 0021).

The overlay is configured by URL query params; ``config.html`` is the streamer-
facing form that builds that URL (no hand-editing) and exposes a rating override.
There's no JS runtime in the test harness, so — like ``test_overlay.py`` — these
assert the contract by content: the page wires every param the overlay reads, and
``overlay.js`` actually honours the new ``welo``/``belo`` overrides.

Runs under pytest *or* plain ``python3 test_config.py`` (stdlib only).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config.html")
OVERLAY_JS = os.path.join(HERE, "overlay.js")


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def test_config_page_exists():
    assert os.path.exists(CONFIG), "overlay/config.html setup page must exist"


def test_config_builds_overlay_url_with_every_param():
    html = _read(CONFIG)
    # The form must let a streamer set each param the overlay understands.
    for param in ("src", "layout", "theme", "cp", "cpbar", "caster", "welo", "belo"):
        assert param in html, "config page must wire the '%s' param" % param
    # It targets the overlay page and offers the live SSE endpoint.
    assert "index.html" in html
    assert "/sse" in html


def test_config_shows_live_ingestor_command():
    """For a live game the page must point the user at the 0018 ingestor, not a mock."""
    html = _read(CONFIG)
    assert "broadcast" in html, "config page should show the broadcast ingestor command"
    assert "lichess.org/broadcast" in html, "config page should accept a Lichess round URL"


def test_overlay_reads_rating_overrides():
    """overlay.js must parse and apply ?welo=/?belo= (the override the page sets)."""
    js = _read(OVERLAY_JS)
    assert 'p.get("welo")' in js and 'p.get("belo")' in js, "params() must read welo/belo"
    assert "overrideRating" in js, "overlay.js must have a rating-override path"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except AssertionError as exc:
            failures += 1
            print("FAIL", t.__name__, "-", exc)
    raise SystemExit(1 if failures else 0)
