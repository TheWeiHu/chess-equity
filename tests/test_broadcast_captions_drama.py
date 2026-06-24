"""Drama-kind prefix on broadcast caption cues (task 0246).

``build_captions_vtt`` / ``build_captions_srt`` (tasks 0211/0229) render one caster cue
per graded move. The reel already stamps a kind callout (emoji + label, e.g. ``🎯 Clutch``)
on its lower-thirds; this pins that the *broadcast* caption cues do the same: a move that
fired a drama event gets its cue text **prefixed** with that kind's callout, keyed by ply
so it never drifts from the move it labels, while undramatic cues are byte-identical to
the bare caster caption.

Fixture: ``data/sample/sample_games.pgn`` — the sanctioned offline-smoke fixture
(illustrative, not evidence; see project CLAUDE.md). Its first game is a 7-ply scholar's
mate ending in ``Qxf7#``, a >50% swing that fires the drama classifier's ``clutch`` on
the engine-free :class:`LichessBaselineModel` (deterministic, Stockfish-independent).
"""
from chess_equity.broadcast import (
    _caption_cues,
    _drama_callout,
    build_captions_srt,
    build_captions_vtt,
    live_caption,
)
from chess_equity.drama import score_event
from chess_equity.reel import _KIND_LABEL

from test_broadcast_captions_vtt import _first_game_sans, _replay_events


def test_dramatic_cue_is_prefixed_with_kind_callout_others_unchanged():
    events = _replay_events()
    cues = _caption_cues(events)
    graded = [e for e in events if live_caption(e) is not None]
    assert len(cues) == len(graded)

    saw_drama = False
    for event, (_start, _end, text) in zip(graded, cues):
        cap = live_caption(event)
        drama = score_event(event)
        if drama is not None:
            saw_drama = True
            callout = _drama_callout(drama.kind)
            # The cue is the kind callout (emoji + label) prefixed onto the caster line.
            assert text == f"{callout} — {cap}", text
            emoji, label = _KIND_LABEL[drama.kind]
            assert emoji in text and label in text, text
        else:
            # Undramatic cues carry the bare caster caption, unchanged.
            assert text == cap, text

    # The scholar's-mate Qxf7# fires a clutch, so at least one cue is prefixed.
    assert saw_drama, "expected the closing Qxf7# clutch to fire a drama callout"
    last_san = _first_game_sans()[-1]
    last_text = cues[-1][2]
    assert "Clutch" in last_text and "🎯" in last_text and last_san in last_text, last_text


def test_vtt_and_srt_carry_the_drama_prefix_at_the_clutch_ply():
    """The prefix rides the shared ``_caption_cues`` source, so both export dialects
    surface it at the same (clutch) cue."""
    events = _replay_events()
    vtt = build_captions_vtt(events)
    srt = build_captions_srt(events)
    # The closing clutch callout appears verbatim in both tracks' cue text.
    assert "🎯 Clutch — " in vtt
    assert "🎯 Clutch — " in srt
