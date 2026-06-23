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


def test_caster_mode_has_an_engine_blind_swing():
    """Acceptance for the caster-mode drama indicator (task 0022).

    Mirrors overlay.js ``dramaSwing``: the bundled replay must contain a move where
    the PRACTICAL equity swings hard (>=10 pts) while the centipawn bar barely moves
    (practical swing >= 2x the engine's) — the "swing the engine bar misses" the
    flare is built to catch. Without one, caster mode would have nothing to fire on.
    """
    events = load_events()
    positions = [e for e in events if e.get("type") == "position"]
    found = False
    prev = None
    for e in positions:
        if prev is not None:
            swing = abs(e["equity"] - prev["equity"])
            cp_swing = abs(cp_to_white_pos(e["cp"]) - cp_to_white_pos(prev["cp"]))
            if swing >= 0.10 and swing >= 2.0 * cp_swing:
                found = True
                break
        prev = e
    assert found, "fixture needs a big practical swing the centipawn bar misses"


def human_edge(equity_white, cp, threshold=0.15):
    """Mirror overlay.js humanEdge: a 'human edge' fires when the practical equity
    bar and the centipawn bar disagree on the position by >= threshold points."""
    gap = equity_white - cp_to_white_pos(cp)
    if abs(gap) < threshold:
        return None
    return {"side": "white" if gap > 0 else "black", "gap": gap}


def test_human_edge_badge_fires_and_clears(threshold=0.15):
    """Acceptance for the human-edge divergence badge (task 0048).

    The bundled replay must contain at least one position where the practical
    equity and the classic centipawn eval disagree past the threshold (the badge
    SHOWS) and at least one where they agree (the badge HIDES) — otherwise the
    indicator would be either always-on or never-on and prove nothing.
    """
    positions = [e for e in load_events() if e.get("type") == "position"]
    edges = [(e, human_edge(e["equity"], e["cp"], threshold)) for e in positions]
    fires = [(e, edge) for e, edge in edges if edge is not None]
    clears = [e for e, edge in edges if edge is None]
    assert fires, "fixture needs a position where practical equity diverges from the engine bar"
    assert clears, "fixture needs a position where the two agree (badge hidden)"
    # `side` must point to whoever the practical bar favors relative to the engine.
    for e, edge in fires:
        favored_white = e["equity"] > cp_to_white_pos(e["cp"])
        assert edge["side"] == ("white" if favored_white else "black")


def test_optional_drama_field_schema():
    """If an event carries a server-side `drama` payload (chess_equity.drama, once
    0018/0020 emit it), it must match the shape overlay.js reads."""
    for e in load_events():
        drama = e.get("drama")
        if drama is None:
            continue
        assert drama.get("headline"), "drama needs a caster-facing headline"
        if "magnitude" in drama:
            assert 0.0 <= drama["magnitude"] <= 1.0


def test_time_pressure_is_present():
    """The wedge is clock-aware: a side must actually hit time trouble."""
    events = load_events()
    low = [
        e
        for e in events
        if e.get("type") == "position" and min(e["clock"]["white"], e["clock"]["black"]) < 5.0
    ]
    assert low, "fixture should include a real time-scramble"


def time_pressure(secs, threshold):
    """Mirror overlay.js EquityOverlay.timePressure — the cue's boolean predicate."""
    return isinstance(secs, (int, float)) and secs >= 0 and secs <= threshold


def test_time_pressure_predicate_boundaries():
    """The cue fires at/under the threshold, and never on null/negative clocks."""
    assert time_pressure(30, 30) is True          # at the threshold -> pressure
    assert time_pressure(5.0, 30) is True
    assert time_pressure(31, 30) is False         # above -> no cue
    assert time_pressure(None, 30) is False       # missing clock -> no cue
    assert time_pressure(-1, 30) is False         # malformed -> no cue


def test_default_threshold_lights_the_cue_on_the_fixture():
    """At the default 30s threshold a side's nameplate would light up on the fixture,
    so the visual time-pressure cue is exercised by the committed mock game."""
    events = load_events()
    lit = [
        e
        for e in events
        if e.get("type") == "position"
        and (time_pressure(e["clock"]["white"], 30) or time_pressure(e["clock"]["black"], 30))
    ]
    assert lit, "fixture should drive the time-pressure cue at the default threshold"


class StaleTracker:
    """Mirror of feed.js ``makeStaleTracker`` — the pure stale-state machine the
    overlay uses to show a STALE/reconnecting state when the live feed drops.

    No real timers: the caller passes ``now`` (ms). Each method returns a
    transition string only on the edge ("stale"/"recovered"), else None, so the
    overlay fires its UI side-effect exactly once per transition.
    """

    def __init__(self, stale_ms=10000):
        self.stale_ms = stale_ms or 10000
        self.last_event_at = None
        self.stale = False

    def event(self, now):
        self.last_event_at = now
        if self.stale:
            self.stale = False
            return "recovered"
        return None

    def fail(self):
        if not self.stale:
            self.stale = True
            return "stale"
        return None

    def poll(self, now):
        if self.stale or self.last_event_at is None:
            return None
        if now - self.last_event_at >= self.stale_ms:
            self.stale = True
            return "stale"
        return None

    def is_stale(self):
        return self.stale


def test_stale_tracker_enters_stale_on_silence():
    """No event for >= staleMs -> the bar goes STALE (silence-driven)."""
    t = StaleTracker(stale_ms=10000)
    assert t.event(0) is None and t.is_stale() is False
    # Polling before the threshold keeps it live...
    assert t.poll(5000) is None
    assert t.is_stale() is False
    # ...and crossing the threshold flips it exactly once.
    assert t.poll(10000) == "stale"
    assert t.is_stale() is True
    assert t.poll(20000) is None, "stale transition should fire only on the edge"


def test_stale_tracker_enters_stale_on_transport_error():
    """An EventSource/WebSocket error forces STALE immediately, before any timeout."""
    t = StaleTracker(stale_ms=10000)
    t.event(0)
    assert t.fail() == "stale"
    assert t.is_stale() is True
    assert t.fail() is None, "already-stale fail must not re-fire"


def test_stale_tracker_recovers_on_next_event():
    """The next event after going stale clears the state exactly once (recover)."""
    t = StaleTracker(stale_ms=10000)
    t.event(0)
    assert t.poll(10000) == "stale"
    assert t.is_stale() is True
    # Next event recovers...
    assert t.event(12000) == "recovered"
    assert t.is_stale() is False
    # ...and a normal event while live reports no transition.
    assert t.event(13000) is None


def test_stale_tracker_no_op_before_first_event():
    """Polling before any event ever arrives must not declare a frozen-from-birth feed."""
    t = StaleTracker(stale_ms=10000)
    assert t.poll(999999) is None
    assert t.is_stale() is False


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
