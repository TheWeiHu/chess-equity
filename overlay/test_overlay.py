#!/usr/bin/env python3
"""Schema + content tests for the overlay's event feed.

Runs with pytest *or* plain `python3 test_overlay.py` (stdlib only, no deps) so
it satisfies the green-gate before the pytest harness from 0001 lands.

It validates the event contract that the overlay consumes (and that the live
ingestion task, 0018, must emit) and asserts the headline acceptance criterion:
the practical equity bar demonstrably diverges from the classic centipawn eval.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GAME = os.path.join(HERE, "mock-game.json")


def load_events():
    with open(GAME, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["events"] if isinstance(data, dict) else data


def cp_to_white_pos(cp):
    """Mirror overlay.js cpToWhitePos — classic Lichess logistic."""
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


def test_first_event_is_game_metadata():
    events = load_events()
    assert events, "feed must not be empty"
    first = events[0]
    assert first["type"] == "game"
    for side in ("white", "black"):
        player = first["players"][side]
        assert player["name"]
        assert isinstance(player["rating"], int)


def test_position_events_conform_to_schema():
    events = load_events()
    positions = [e for e in events if e.get("type") == "position"]
    assert len(positions) >= 5, "need a few moves to show a swing"
    for e in positions:
        assert 0.0 <= e["equity"] <= 1.0, "equity is White-POV probability"
        assert isinstance(e["cp"], (int, float))
        clk = e["clock"]
        assert clk["white"] >= 0 and clk["black"] >= 0
        if "grade" in e:
            assert e["grade"]["label"]
            assert -1.0 <= e["grade"]["delta"] <= 1.0


def test_equity_diverges_from_centipawns():
    """Acceptance criterion: the equity bar must NOT just track the cp bar.

    There must be a position where the clock-aware practical equity and the
    classic centipawn eval point to materially different win chances.
    """
    events = load_events()
    max_gap = 0.0
    for e in events:
        if e.get("type") != "position":
            continue
        cp_pos = cp_to_white_pos(e["cp"])
        max_gap = max(max_gap, abs(e["equity"] - cp_pos))
    assert max_gap >= 0.20, (
        "equity should diverge from the centipawn bar by >=20pts somewhere; "
        "max gap was %.2f" % max_gap
    )


def test_time_pressure_is_present():
    """The wedge is clock-aware: a side must actually hit time trouble."""
    events = load_events()
    low = [
        e
        for e in events
        if e.get("type") == "position" and min(e["clock"]["white"], e["clock"]["black"]) < 5.0
    ]
    assert low, "fixture should include a real time-scramble"


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
