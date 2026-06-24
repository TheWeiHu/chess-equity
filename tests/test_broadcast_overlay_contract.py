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
import io
import os
import re

import chess.pgn

from chess_equity.adapters import EquityModel, white_to_move
from chess_equity.broadcast import GameTracker, game_event, model_label
from chess_equity.models import LichessBaselineModel
from chess_equity.types import WDL, Equity

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
OPTIONAL_POSITION_READS = {"players", "drama", "flag_risk"}


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


def test_position_event_carries_authoritative_white_to_move():
    """Overlay reads evt.white_to_move for the ?pov=stm readout (task 0212).

    The flag is the post-move FEN's side-to-move, threaded straight from the
    internal MoveEvent — the overlay no longer has to guess from ply parity. It must
    be present, boolean, agree with the MoveEvent, and match ply parity (White moves
    on odd plies, so the position after an even ply is White-to-move).
    """
    move_events, events = drive_events()
    for src, evt in zip(move_events, events):
        assert "white_to_move" in evt, "white_to_move is REQUIRED on a position event"
        assert isinstance(evt["white_to_move"], bool)
        assert evt["white_to_move"] == src.white_to_move
        assert evt["white_to_move"] == (evt["ply"] % 2 == 0)


def test_cp_flows_through_and_is_white_pov():
    """A resolvable engine yields a numeric, White-POV cp on the overlay event (task 0052).

    Without this the overlay's centipawn ghost tick and the human-edge divergence
    badge are dead on a live feed (cp was hard-coded to None).
    """
    move_events, events = drive_events()
    assert any(e.cp is not None for e in move_events), "baseline engine must supply a cp"
    for src, evt in zip(move_events, events):
        assert evt["cp"] == src.cp, "cp must thread through to the overlay event"
        if src.cp is not None:
            # Independent White-POV re-derivation from the event's own (post-move) fen.
            eq = LichessBaselineModel().evaluate(src.fen, 1500, 1500)
            assert eq.cp is not None
            white_pov = eq.cp if white_to_move(src.fen) else -eq.cp
            assert src.cp == white_pov


def test_cp_none_degrades_when_no_engine_cp():
    """A model that exposes no objective cp leaves cp=None end to end (CI-safe)."""

    class _NoCpModel(EquityModel):
        def evaluate(self, fen, white_elo, black_elo):
            return Equity.from_side_to_move(
                WDL(p_win=0.4, p_draw=0.2, p_loss=0.4),
                white_to_move=white_to_move(fen),
                source="nocp",
                cp=None,
            )

    tracker = GameTracker("nocp", _NoCpModel(), white_elo=1500, black_elo=1500)
    move_events = tracker.ingest(GAME_PGN)
    assert move_events
    for e in move_events:
        assert e.cp is None
        assert e.to_overlay_event()["cp"] is None


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


# --------------------------------------------------------------------------- #
# Model badge contract (task 0222): the `game` event names the bar's model so a
# viewer can tell the fill is a human win-probability model, not Stockfish.
# --------------------------------------------------------------------------- #


def _headers():
    return chess.pgn.read_headers(io.StringIO(GAME_PGN))


def overlay_game_reads():
    """Fields overlay.js reads off a `game` event, scraped from applyGame.

    Same drift guard as :func:`overlay_position_reads` but for the one-time game
    event: a new ``evt.<field>`` read in applyGame the producer doesn't emit fails.
    """
    with open(OVERLAY_JS, "r", encoding="utf-8") as fh:
        src = fh.read()
    body = src[src.index("function applyGame") :]
    body = body[: body.index("\n  function ", 1)]
    return set(re.findall(r"evt\.([A-Za-z_]+)", body))


def test_model_label_maps_known_sources():
    """model_label maps a model's SOURCE to its human-readable badge label."""
    assert model_label(LichessBaselineModel()) == "baseline"


def test_model_label_none_for_no_model():
    """A None model yields no label, so the overlay shows no badge."""
    assert model_label(None) is None


def test_game_event_emits_model_badge():
    """game_event(model=...) surfaces a human-readable model on the overlay event."""
    ev = game_event(_headers(), "g", model=LichessBaselineModel())
    assert ev.model == "baseline"
    assert ev.to_overlay()["model"] == "baseline"


def test_game_event_without_model_omits_badge():
    """Default/None model: the overlay event carries no model key (fallback: no badge)."""
    ev = game_event(_headers(), "g")
    assert ev.model is None
    assert "model" not in ev.to_overlay()


def test_overlay_reads_model_off_game_event():
    """Drift guard: if overlay.js reads evt.model on a game event, the bridge emits it."""
    emitted = set(game_event(_headers(), "g", model=LichessBaselineModel()).to_overlay())
    if "model" in overlay_game_reads():
        assert "model" in emitted, "overlay.js reads evt.model but the bridge never emits it"
