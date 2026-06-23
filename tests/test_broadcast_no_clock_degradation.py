"""OTB classical: graceful no-clock degradation end-to-end (task 0186).

The north-star prestige target is OTB classical broadcasts, and those relay PGNs
often carry NO per-move ``[%clk]`` clock (especially at round start) — yet the live
pipeline is now clock-aware throughout: the published bar and Δequity grade are warped
by remaining clock (tasks 0097/0103/0106), the ``scramble`` drama kind is clock-gated
(``drama.py``), and the overlay shows a ``⏱`` time-pressure cue driven by the clock
field (task 0105). This test locks the contract that on a clockless feed all of that
degrades *gracefully*: the whole ``broadcast -> grade -> drama -> overlay-event`` chain
runs to completion, the clock cues are simply OMITTED (clocks stay ``null`` — never
fabricated), and the clock-aware grade falls back to exactly its clock-blind value.

The fixture (``data/sample/otb_classical_no_clock.pgn``) is a real OTB classical game
shaped like an OTB relay snapshot: a **classical** ``TimeControl`` header but no
``[%clk]`` tags. Classical (not correspondence) matters — it means the clock warp is
genuinely *armed* (``clock_aware=True``, a non-trivial tc bucket), so this proves the
no-op comes from ``stm_clock is None`` (no clock data), not merely from a zero bucket
multiplier. ``data/sample`` is the sanctioned offline-smoke fixture (illustrative, not
evidence — see project CLAUDE.md). No browser, no torch, no network — unattended-safe.
"""
import io
import os

import chess.pgn

from chess_equity.broadcast import BroadcastIngestor, LocalPgnFeed, overlay_events
from chess_equity.models import LichessBaselineModel

HERE = os.path.dirname(os.path.abspath(__file__))
NO_CLOCK_PGN = os.path.join(HERE, "..", "data", "sample", "otb_classical_no_clock.pgn")


def _fixture():
    """PGN text plus the ply count and final SAN, derived from the fixture itself so
    the assertions track the file rather than hard-coded constants."""
    with open(NO_CLOCK_PGN, "r", encoding="utf-8") as fh:
        text = fh.read()
    game = chess.pgn.read_game(io.StringIO(text))
    assert game is not None, "fixture must contain a game"
    nodes = list(game.mainline())
    assert nodes, "fixture game must have moves"
    assert not any(n.clock() is not None for n in nodes), (
        "fixture must be clockless — no [%clk] tags"
    )
    last_san = nodes[-1].parent.board().san(nodes[-1].move)
    return text, len(nodes), last_san


def _overlay_events(pgn_text, *, clock_aware):
    """Run the full producer->overlay chain over a finite replay and return the
    overlay-shaped dicts (drops the HEARTBEAT sentinel, which a finite replay omits)."""
    ingestor = BroadcastIngestor(
        LocalPgnFeed(pgn_text),
        LichessBaselineModel(),
        white_elo=None,  # OTB relay: ratings often absent ([WhiteElo "?"])
        black_elo=None,
        clock_aware=clock_aware,
    )
    return [
        e
        for e in overlay_events(ingestor, interval=0, max_polls=None, max_idle_polls=1)
        if isinstance(e, dict)
    ]


# --------------------------------------------------------------------------- #
# The clock-aware pipeline degrades gracefully on a clockless feed
# --------------------------------------------------------------------------- #


def test_clockless_feed_runs_end_to_end_with_clocks_omitted():
    """broadcast -> grade -> drama -> overlay-event completes; clocks stay null."""
    pgn_text, n_plies, final_san = _fixture()
    events = _overlay_events(pgn_text, clock_aware=True)

    assert events, "clockless feed produced no events"
    positions = [e for e in events if e["type"] == "position"]

    # Full coverage: one position per half-move, in order, reaching the final move —
    # the pipeline ran to completion rather than crashing on the missing clocks.
    assert [e["ply"] for e in positions] == list(range(1, n_plies + 1))
    assert positions[-1]["move"]["san"] == final_san

    for evt in positions:
        # (1) Clocks are OMITTED, not fabricated: both sides null on every move.
        assert evt["clock"] == {"white": None, "black": None}, (
            f"ply {evt['ply']} fabricated a clock: {evt['clock']!r}"
        )
        # (2) The equity bar is still a usable White-POV value in [0, 1].
        eq = evt["equity"]
        assert isinstance(eq, float) and 0.0 <= eq <= 1.0, f"bad equity: {eq!r}"
        # (3) The clock-driven drama kind never fires without a clock (drama.py gate).
        drama = evt.get("drama")
        assert drama is None or drama["kind"] != "scramble", (
            f"ply {evt['ply']} fired a clock-only 'scramble' drama with no clock"
        )


def test_clock_aware_grade_falls_back_to_clock_blind_value():
    """With no clocks, the clock-aware path emits exactly the clock-blind events.

    The whole degradation guarantee in one assertion: warping the bar by a clock that
    isn't there must be a no-op, so the entire overlay event stream produced with
    ``clock_aware=True`` is byte-for-byte what ``clock_aware=False`` produces — same
    equity, same grade label, same grade delta, same drama.
    """
    pgn_text, _, _ = _fixture()
    aware = _overlay_events(pgn_text, clock_aware=True)
    blind = _overlay_events(pgn_text, clock_aware=False)

    assert aware == blind, "clock-aware stream diverged from clock-blind on a clockless feed"

    # And the grade really is present (we're comparing real grades, not two empties).
    graded = [e for e in aware if e["type"] == "position" and "grade" in e]
    assert graded, "expected graded position events on the fixture"
    assert all(g["grade"]["label"] for g in graded), "every grade must carry a label"
