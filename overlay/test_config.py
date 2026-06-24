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
INDEX = os.path.join(HERE, "index.html")


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


def test_config_persists_setup_in_localstorage():
    """An OBS browser-source reload must keep the streamer's setup (task 0056).

    No JS runtime in the harness, so assert the wiring by content: the page both
    writes to and reads from localStorage, and re-applies the saved state on load.
    """
    html = _read(CONFIG)
    assert "localStorage.setItem" in html, "config must SAVE the setup to localStorage"
    assert "localStorage.getItem" in html, "config must READ the setup back on load"
    # restore() must run before the first render so saved fields are rehydrated.
    assert "restore()" in html and html.index("restore()") < html.index("build();")


def test_legend_toggle_is_wired_and_off_by_default():
    """The legend key (task 0201) must be a config-driven, default-off toggle.

    No JS runtime, so assert the contract by content across the three files:
    - index.html ships the legend element, ``hidden`` by default (off);
    - overlay.js reads ``?legend=1`` and reveals it only then;
    - config.html offers the toggle and emits the param into the built URL;
    - the toggle is a persisted form field (rides the localStorage save like 0056).
    """
    index, js, html = _read(INDEX), _read(OVERLAY_JS), _read(CONFIG)
    # Element present and hidden (off) by default.
    assert "data-legend" in index, "index.html must ship the legend element"
    legend_tag = index[index.index("data-legend"):]
    legend_tag = legend_tag[: legend_tag.index(">")]
    assert "hidden" in legend_tag, "legend must be hidden (off) by default"
    # overlay.js gates it on ?legend=1.
    assert 'p.get("legend")' in js, "overlay.js must read the ?legend param"
    assert "[data-legend]" in js, "overlay.js must reveal the legend element"
    # config.html toggles + emits the param, and persists it (in FIELDS/CHECKS).
    assert 'id="legend"' in html, "config page must offer a legend checkbox"
    assert 'params.set("legend"' in html, "config must emit ?legend into the URL"
    assert '"legend"' in html, "legend must be a persisted form field (FIELDS)"


def test_pov_toggle_is_wired_and_white_by_default():
    """The bar POV toggle (task 0206) must be config-driven and default to White-POV.

    No JS runtime, so assert the contract by content across the three files:
    - overlay.js reads ``?pov`` (defaulting to "white") and exposes the pure mapping
      helpers (``orient``/``whiteToMove``) plus the side-to-move readout element;
    - index.html ships the ``data-stm-pct`` readout element, ``hidden`` by default (off);
    - config.html offers the toggle, emits the param only when non-default, and persists it.
    """
    index, js, html = _read(INDEX), _read(OVERLAY_JS), _read(CONFIG)
    # overlay.js parses the param (default white) and exposes the mapping helpers.
    assert 'p.get("pov")' in js, "overlay.js must read the ?pov param"
    assert "orient" in js and "whiteToMove" in js, "overlay.js must expose orient/whiteToMove"
    assert "[data-stm-pct]" in js, "overlay.js must drive the side-to-move readout"
    # index.html ships the readout element, hidden (off) by default.
    assert "data-stm-pct" in index, "index.html must ship the side-to-move readout element"
    stm_tag = index[index.index("data-stm-pct"):]
    stm_tag = stm_tag[: stm_tag.index(">")]
    assert "hidden" in stm_tag, "side-to-move readout must be hidden (off) by default"
    # config.html toggles + emits the param (only when not the white default), and persists it.
    assert 'id="pov"' in html, "config page must offer a POV select"
    assert 'params.set("pov"' in html, "config must emit ?pov into the URL"
    assert '!== "white"' in html, "config must emit ?pov only when it differs from the default"
    assert '"pov"' in html, "pov must be a persisted form field (FIELDS)"


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
