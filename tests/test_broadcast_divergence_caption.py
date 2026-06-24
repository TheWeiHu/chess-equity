"""Human-vs-engine DIVERGENCE caption callout (task 0273).

The overlay already paints a per-move human-edge divergence BADGE (visual only); this
pins its *spoken* counterpart: when the rating-conditioned practical bar (``equity``)
disagrees with the rating-blind engine bar (``lichess_win_percent(cp)``) by at least the
threshold, the caster caption appends a DIVERGENCE callout naming the magnitude and which
side the human read leans toward. Reuses the exact ``|equity - cp_win|`` definition the
offline reel category uses (``chess_equity.reel.detect_divergence``, task 0272) so the
live callout and the post-game recap never disagree.

Pure-fixture unit test (no dump, no torch): hand-built :class:`MoveEvent`s of a graded
move, exercising the threshold gate, direction, cp-less silence, and the flag threading
through ``live_caption`` / ``build_captions_vtt`` / ``build_captions_srt``.
"""
from chess_equity.broadcast import (
    DIVERGENCE_CAPTION_THRESHOLD,
    MoveEvent,
    build_captions_srt,
    build_captions_vtt,
    divergence_callout,
    live_caption,
)
from chess_equity.types import lichess_win_percent


def _event(*, equity: float, cp, ply: int = 4, san: str = "Nf3", grade="ok") -> MoveEvent:
    """A graded White move with a given practical ``equity`` and engine ``cp``.

    ``white_to_move=False`` (Black to move in the post-move FEN) means White just moved.
    ``delta_equity`` is kept tiny so the drama classifier never fires — this isolates the
    divergence callout from the drama headline.
    """
    return MoveEvent(
        game_id="g1",
        ply=ply,
        san=san,
        uci="g1f3",
        fen="rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 0 1",
        white_to_move=False,
        white_clock=None,
        black_clock=None,
        white_elo=1800,
        black_elo=1800,
        equity=equity,
        delta_equity=0.5,
        last_move_grade=grade,
        source="test",
        compute_ms=0.0,
        cp=cp,
    )


def test_big_gap_toward_black_fires_callout():
    # Engine reads White clearly better (cp +300 -> high Win%), but the human practical
    # bar gives White only ~50%: the classic "humans don't convert the engine's edge".
    cp = 300.0
    cp_win = lichess_win_percent(cp)
    ev = _event(equity=50.0, cp=cp)
    assert cp_win - 50.0 >= DIVERGENCE_CAPTION_THRESHOLD  # fixture really is a divergence

    callout = divergence_callout(ev)
    assert callout is not None
    # Magnitude + direction: both bars, the signed gap, and the leaning side are spoken.
    assert "Divergence" in callout
    assert "50%" in callout and f"{cp_win:.0f}%" in callout
    assert "toward Black" in callout
    assert "don't convert" in callout


def test_big_gap_toward_white_fires_callout():
    # Engine reads roughly level (cp 0 -> 50%), human bar likes White at 70%.
    ev = _event(equity=70.0, cp=0.0)
    callout = divergence_callout(ev)
    assert callout is not None
    assert "toward White" in callout
    assert "likes White more" in callout


def test_near_agreement_is_silent():
    # Human 50% vs engine ~50% (cp 0): gap below threshold -> no callout.
    assert divergence_callout(_event(equity=50.0, cp=0.0)) is None


def test_cp_none_is_silent():
    # No engine bar to diverge from (mate / cp-less feed) -> never fires, even at a huge
    # practical-bar value.
    assert divergence_callout(_event(equity=95.0, cp=None)) is None


def test_threshold_flag_is_tunable():
    ev = _event(equity=50.0, cp=120.0)  # a modest ~13-pt gap
    gap = lichess_win_percent(120.0) - 50.0
    # Default threshold (~15) does not fire on this modest gap...
    assert divergence_callout(ev) is None
    # ...but a lower threshold does.
    assert 0 < gap < DIVERGENCE_CAPTION_THRESHOLD
    assert divergence_callout(ev, threshold=5.0) is not None


def test_live_caption_appends_callout_after_the_move_line():
    ev = _event(equity=50.0, cp=300.0, san="Re1")
    line = live_caption(ev)
    assert line is not None
    # The move's own caster line comes first, then the divergence callout after a separator.
    assert line.startswith("Re1 — ")
    assert "  ·  📊 Divergence" in line
    # Raising the threshold above the gap drops the callout but keeps the move line.
    bare = live_caption(ev, divergence_threshold=99.0)
    assert bare == "Re1 — ok, +0% for a 1800 here"


def test_callout_rides_the_vtt_and_srt_tracks():
    events = [_event(equity=50.0, cp=300.0, san="Re1")]
    vtt = build_captions_vtt(events)
    srt = build_captions_srt(events)
    assert "📊 Divergence" in vtt
    assert "📊 Divergence" in srt
    # Threading the threshold through both builders suppresses it consistently.
    assert "📊 Divergence" not in build_captions_vtt(events, divergence_threshold=99.0)
    assert "📊 Divergence" not in build_captions_srt(events, divergence_threshold=99.0)
