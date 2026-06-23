"""GameTracker hardening against real Lichess broadcast-PGN quirks (task 0172).

Live broadcast PGN is messier than a finished game export: the mainline carries
sideline *variations*, inline *comments* with `[%eval]`/`[%clk]` tags and NAG glyphs
(`!?`, `$1`), the `Result` header flips when a game ends, and a poll can arrive
*truncated* mid-token or *corrected* (an operator fixes a wrong move). The ingestor
must emit the genuinely new mainline moves, ignore the annotations, and resync cleanly
on a correction — or the live overlay desyncs.

All offline: every PGN here is a committed fixture (illustrative parser input, not
validation evidence — see CLAUDE.md data policy). No network. Basic clock/dedup/resync
behaviour lives in ``test_broadcast.py``; this file covers the annotation/correction
shapes a real feed produces.
"""

from chess_equity.broadcast import GameTracker
from chess_equity.models import LichessBaselineModel


def _model():
    return LichessBaselineModel()


def _tracker():
    return GameTracker("g", _model(), white_elo=None, black_elo=None)


_HDR = """[Event "Real Broadcast"]
[Site "https://lichess.org/q1q2q3q4"]
[White "A"]
[Black "B"]
[WhiteElo "2700"]
[BlackElo "2700"]
[TimeControl "300+0"]
[Result "*"]

"""


# --------------------------------------------------------------------------- #
# Variations, NAGs, and [%eval] in the same comment as [%clk]
# --------------------------------------------------------------------------- #

# The shape a Lichess broadcast actually emits: every move annotated with [%eval] and
# [%clk] in one comment, move NAGs (!?, $1), and an analysis sideline in parentheses.
QUIRKS_PGN = _HDR + (
    "1. e4 { [%eval 0.17] [%clk 0:05:00] } e5!? { [%eval 0.20] [%clk 0:04:58] } "
    "2. Nf3 (2. Bc4 Nc6 3. Qh5) Nc6 $1 { [%eval 0.15] [%clk 0:04:50] } *\n"
)


def test_only_mainline_moves_emitted_variations_ignored():
    # The sideline (2. Bc4 Nc6 3. Qh5) must not leak into the event stream: only the
    # four mainline half-moves are published, in order.
    events = _tracker().ingest(QUIRKS_PGN)
    assert [e.ply for e in events] == [1, 2, 3, 4]
    assert [e.san for e in events] == ["e4", "e5", "Nf3", "Nc6"]


def test_nags_and_eval_comments_do_not_break_parsing():
    # NAG glyphs (!?, $1) and the broadcast's own [%eval] are annotations, not moves —
    # they're ignored, and the SAN is the bare move (no "!?" suffix).
    events = _tracker().ingest(QUIRKS_PGN)
    assert all("!" not in e.san and "$" not in e.san for e in events)
    # The objective bar comes from our model on the FEN, never from the PGN's [%eval].
    assert all(0.0 <= e.equity <= 100.0 for e in events)


def test_clk_parsed_even_when_eval_precedes_it_in_the_comment():
    # [%clk] sits *after* [%eval] in each comment; node.clock() must still find it.
    events = _tracker().ingest(QUIRKS_PGN)
    assert events[0].white_clock == 300.0  # e4 -> 0:05:00
    assert events[1].black_clock == 298.0  # e5 -> 0:04:58


# --------------------------------------------------------------------------- #
# Mid-stream truncation (a poll cut off before the PGN is complete)
# --------------------------------------------------------------------------- #


def test_truncated_mid_comment_emits_complete_moves_only():
    # A poll snipped inside a trailing { [%clk ... comment: the three complete moves
    # parse; the dangling comment is dropped, not crashed on.
    truncated = _HDR + "1. e4 { [%clk 0:05:00] } e5 { [%clk 0:04:58] } 2. Nf3 { [%clk 0:04"
    events = _tracker().ingest(truncated)
    assert [e.san for e in events] == ["e4", "e5", "Nf3"]


def test_truncated_mid_move_token_drops_the_partial_move():
    # A poll cut inside a move token ("Nf"): the two complete moves emit; the partial
    # token is ignored rather than parsed as an illegal move.
    events = _tracker().ingest(_HDR + "1. e4 e5 2. Nf")
    assert [e.san for e in events] == ["e4", "e5"]


def test_resync_when_a_later_poll_is_shorter():
    # First a full game, then a walk-back poll with fewer moves (a correction that
    # rewinds): the tracker resets and re-emits from ply 1, flagged resync.
    t = _tracker()
    assert len(t.ingest(_HDR + "1. e4 e5 2. Nf3 Nc6 *")) == 4
    rewound = t.ingest(_HDR + "1. e4 e5 *")
    assert [e.ply for e in rewound] == [1, 2]
    assert all(e.resync for e in rewound)


# --------------------------------------------------------------------------- #
# Result corrections
# --------------------------------------------------------------------------- #


def test_result_header_correction_emits_no_new_moves():
    # When a game ends the Result flips * -> 1-0 and a result token is appended, but the
    # MOVES are unchanged — so a re-poll yields nothing new (no spurious re-emit).
    t = _tracker()
    assert len(t.ingest(_HDR + "1. e4 e5 2. Nf3 Nc6 *")) == 4
    finished = _HDR.replace('[Result "*"]', '[Result "1-0"]') + "1. e4 e5 2. Nf3 Nc6 1-0"
    assert t.ingest(finished) == []


def test_same_length_move_correction_resyncs_and_reemits():
    # The hardening case (task 0172): an operator typo is corrected in place — the PGN
    # keeps the same length but ply 3 changes (Nf3 -> Nc3). The old shrink-only check
    # would silently keep the stale move; now the prefix divergence triggers a resync so
    # the corrected line (and its equity) is re-emitted from the start.
    t = _tracker()
    assert [e.san for e in t.ingest(_HDR + "1. e4 e5 2. Nf3 Nc6 *")] == [
        "e4", "e5", "Nf3", "Nc6",
    ]
    corrected = t.ingest(_HDR + "1. e4 e5 2. Nc3 Nc6 *")
    assert [e.san for e in corrected] == ["e4", "e5", "Nc3", "Nc6"]
    assert all(e.resync for e in corrected)


def test_correction_then_growth_continues_cleanly():
    # After a correction the tracker must keep diffing normally: a subsequent grown poll
    # (same corrected prefix + a new move) emits only the new move, no resync.
    t = _tracker()
    t.ingest(_HDR + "1. e4 e5 2. Nf3 Nc6 *")
    t.ingest(_HDR + "1. e4 e5 2. Nc3 Nc6 *")  # correction (resync)
    grown = t.ingest(_HDR + "1. e4 e5 2. Nc3 Nc6 3. d4 *")
    assert [e.san for e in grown] == ["d4"]
    assert not any(e.resync for e in grown)
