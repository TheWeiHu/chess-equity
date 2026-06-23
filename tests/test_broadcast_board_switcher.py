"""Producer side of the live board switcher (task 0185).

A multi-game broadcast round must let a caster flip boards live. The consumer half
(``overlay/overlay.js``'s ``makeBoardRouter`` + the ``<select>``) is tested in
``overlay/test_routing.js``; this is the producer half: the feed must *announce* the
available boards in its event stream so the overlay can build the selector.

Contract asserted here:
  - a single-board round emits NO ``boards`` event (the overlay then shows no
    selector — the unchanged default);
  - a two-board round emits a ``boards`` event listing both boards (index + players),
    and every ``position`` event carries a ``game_id`` so the overlay can route it.

No network/torch: a two-game round PGN is replayed through the real
``BroadcastIngestor`` -> ``overlay_events`` bridge with a tiny in-memory feed.
"""
from chess_equity.broadcast import (
    BroadcastFeed,
    BroadcastIngestor,
    HEARTBEAT,
    overlay_events,
)
from chess_equity.models import LichessBaselineModel

ONE_BOARD = """[Event "Round"]
[Site "https://lichess.org/g0"]
[White "Carlsen"]
[Black "Nakamura"]
[WhiteElo "2850"]
[BlackElo "2780"]
[Result "*"]

1. e4 { [%clk 0:03:00] } e5 { [%clk 0:02:58] } 2. Nf3 { [%clk 0:02:55] } *
"""

TWO_BOARDS = ONE_BOARD + """
[Event "Round"]
[Site "https://lichess.org/g1"]
[White "Caruana"]
[Black "Firouzja"]
[WhiteElo "2790"]
[BlackElo "2770"]
[Result "*"]

1. d4 { [%clk 0:03:00] } Nf6 { [%clk 0:02:57] } 2. c4 { [%clk 0:02:54] } *
"""


class _OneShotFeed(BroadcastFeed):
    """Reveal a fixed (possibly multi-game) round PGN once, then end the stream."""

    def __init__(self, pgn_text):
        self._pgn = pgn_text
        self._sent = False

    def poll(self):
        if self._sent:
            return None
        self._sent = True
        return self._pgn


def _drive(pgn_text):
    events = overlay_events(
        BroadcastIngestor(
            _OneShotFeed(pgn_text),
            LichessBaselineModel(),
            white_elo=None,
            black_elo=None,
        ),
        interval=0,
        max_polls=None,
        max_idle_polls=1,
    )
    return [e for e in events if e is not HEARTBEAT]


def test_single_board_round_emits_no_boards_event():
    events = _drive(ONE_BOARD)
    assert not [e for e in events if e.get("type") == "boards"], (
        "a single-board round must not announce a switcher roster"
    )


def test_two_board_round_announces_both_boards():
    events = _drive(TWO_BOARDS)
    rosters = [e for e in events if e.get("type") == "boards"]
    assert rosters, "a multi-board round must announce a boards roster"
    boards = rosters[-1]["boards"]  # last roster is the complete one
    assert len(boards) == 2, "both boards must be listed"
    # Index + players are what the overlay's selector renders.
    assert [b["index"] for b in boards] == [0, 1]
    names = {(b["white"], b["black"]) for b in boards}
    assert ("Carlsen", "Nakamura") in names
    assert ("Caruana", "Firouzja") in names
    # Each board's game_id must be distinct so the router can route on it.
    gids = {b["game_id"] for b in boards}
    assert len(gids) == 2


def test_position_events_carry_game_id_for_routing():
    events = _drive(TWO_BOARDS)
    positions = [e for e in events if e.get("type") == "position"]
    assert positions, "round must produce moves"
    assert all(e.get("game_id") for e in positions), (
        "every position event needs a game_id so the overlay can route it to a board"
    )
    # The two boards' moves are distinguishable by game_id.
    assert len({e["game_id"] for e in positions}) == 2
