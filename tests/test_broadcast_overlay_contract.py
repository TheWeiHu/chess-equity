"""Producer <-> consumer JSON contract: broadcast.py emits what overlay.js reads.

The bridge (``src/chess_equity/broadcast.py``) and the overlay (``overlay/*.js``)
evolved on separate fleet branches and keep colliding on schema: a dropped or
renamed field silently breaks the overlay with no test catching it. This test is
that catch.

It drives a fake PGN feed through the bridge to overlay-shaped JSON events and
asserts the *consumer-visible* contract: the ``position`` event carries the fields
``overlay/overlay.js`` actually reads (equity + white/black clocks + cp + grade),
with the right shapes and White-POV scaling. The expected field list is *derived
from overlay.js* (not hand-copied), so if the overlay starts reading a new field
the producer doesn't emit, this test fails.

Scope: the ``position`` event (everything :class:`MoveEvent` can supply today). The
``game`` event's ``players.{white,black}.{name,rating}`` is plumbed by task 0047 and
pinned there; this test guards the per-move half. No network/torch — a bundled PGN
is replayed through :class:`GameTracker`.
"""
import os
import re

from chess_equity.broadcast import GameTracker
from chess_equity.models import LichessBaselineModel

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY_JS = os.path.join(HERE, "..", "overlay", "overlay.js")

# A short game with [%clk] tags + ratings + named players — the shape a live
# Lichess broadcast emits.
GAME_PGN = """[Event "Contract Test"]
[Site "https://lichess.org/abcd1234"]
[White "Carlsen"]
[Black "Nakamura"]
[Round "1"]
[WhiteElo "2850"]
[BlackElo "2780"]
[Result "*"]

1. e4 { [%clk 0:03:00] } e5 { [%clk 0:02:58] } 2. Nf3 { [%clk 0:02:55] } Nc6 { [%clk 0:02:50] } *
"""

# Fields the overlay tolerates being absent from a position event (degrade
# gracefully, per overlay/README.md): late-arriving player metadata and the
# optional caster-mode drama payload. Everything else the overlay reads on a
# position event must be emitted by the producer.
OPTIONAL_POSITION_READS = {"players", "drama"}


def drive_events():
    """Replay GAME_PGN through the bridge and return the overlay-shaped events."""
    tracker = GameTracker(
        "contract-test", LichessBaselineModel(), white_elo=2850, black_elo=2780
    )
    move_events = tracker.ingest(GAME_PGN)
    assert move_events, "fixture must yield at least one move event"
    return move_events, [e.to_overlay_event() for e in move_events]


def overlay_position_reads():
    """Fields overlay.js reads off a position event, scraped from applyPosition.

    Derived from source so consumer drift (a new ``evt.<field>`` read) surfaces as a
    test failure instead of a silent break.
    """
    with open(OVERLAY_JS, "r", encoding="utf-8") as fh:
        src = fh.read()
    # Isolate applyPosition's body so we don't pick up reads from applyGame etc.
    body = src[src.index("function applyPosition") :]
    body = body[: body.index("\n  function ", 1)]
    return set(re.findall(r"evt\.([A-Za-z_]+)", body))


# --------------------------------------------------------------------------- #
# Structural contract
# --------------------------------------------------------------------------- #


def test_every_overlay_position_read_is_emitted():
    """Drift guard: the producer emits every field overlay.js reads (or it's optional)."""
    _, events = drive_events()
    emitted = set().union(*(e.keys() for e in events))
    for field in overlay_position_reads():
        assert (
            field in emitted or field in OPTIONAL_POSITION_READS
        ), f"overlay.js reads evt.{field} on a position event but the bridge never emits it"


def test_position_event_required_shape():
    """type + equity are required; equity is White-POV in [0, 1] (not [0, 100])."""
    _, events = drive_events()
    for evt in events:
        assert evt["type"] == "position"
        assert "equity" in evt, "equity is REQUIRED on a position event"
        assert isinstance(evt["equity"], float)
        assert 0.0 <= evt["equity"] <= 1.0, "overlay clamp01s equity to [0, 1]"


def test_position_event_clock_is_nested_with_both_sides():
    """Overlay reads evt.clock.white / evt.clock.black — both keys must exist."""
    _, events = drive_events()
    for evt in events:
        clock = evt["clock"]
        assert isinstance(clock, dict)
        assert "white" in clock and "black" in clock


def test_position_event_carries_cp_key():
    """Overlay reads evt.cp for the ghost tick; the key must be present (None is ok)."""
    _, events = drive_events()
    for evt in events:
        assert "cp" in evt


def test_graded_event_has_label_and_delta():
    """Overlay reads evt.grade.label / evt.grade.delta — both must be present when graded."""
    _, events = drive_events()
    graded = [e for e in events if "grade" in e]
    assert graded, "fixture must contain at least one graded move"
    for evt in graded:
        grade = evt["grade"]
        assert isinstance(grade["label"], str) and grade["label"]
        assert grade["delta"] is None or isinstance(grade["delta"], float)


# --------------------------------------------------------------------------- #
# Value mapping (flat MoveEvent -> nested overlay event)
# --------------------------------------------------------------------------- #


def test_values_map_from_moveevent():
    """The serializer rescales/reshapes correctly: equity/clocks/grade all line up."""
    move_events, events = drive_events()
    for src, evt in zip(move_events, events):
        assert evt["equity"] == src.equity / 100.0
        assert evt["clock"]["white"] == src.white_clock
        assert evt["clock"]["black"] == src.black_clock
        if "grade" in evt:
            assert evt["grade"]["label"] == src.last_move_grade
            expected = None if src.delta_equity is None else src.delta_equity / 100.0
            assert evt["grade"]["delta"] == expected
