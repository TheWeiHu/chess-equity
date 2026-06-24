"""FocusDirector cut cue threaded into the SRT/VTT caption track (task 0263).

The live SSE path (`overlay_events`) threads `FocusDirector.last_reason` (task 0260)
onto the `focus` event so a `--board auto` cut can be voiced. The offline export
(`build_captions_srt` / `build_captions_vtt`, tasks 0211/0229) never saw that cue. This
pins the bridge: with `auto_follow=True`, a `--board auto` snapshot replays the director
and each focus cut becomes its own caption cue (e.g. "cut to Bd2: +0.9 swing vs +0.4")
placed at the cut ply, on the cut-to board's own per-game timeline. Without `auto_follow`
(the default), a single-board export stays byte-identical.

Fixture: a two-board round where board 0 is a quiet opening and board 1 is a scholar's
mate (`Qxf7#`), whose >50% final swing the engine-free baseline model scores as drama —
the same shape the live `_two_board_round_with_drama` routing test uses. Illustrative,
not thesis evidence (see project CLAUDE.md).
"""
import os

from chess_equity.broadcast import (
    BroadcastIngestor,
    LocalPgnFeed,
    _caption_cues,
    build_captions_srt,
    build_captions_vtt,
)
from chess_equity.models import LichessBaselineModel

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PGN = os.path.join(HERE, "..", "data", "sample", "sample_games.pgn")

# Board 0: quiet opening (no drama). Board 1: scholar's mate, a huge final-move swing the
# material baseline scores as drama, so the director cuts to it. Two distinct game ids.
_QUIET = """[Event "Round"]
[Site "https://lichess.org/quietgame"]
[White "Anna"]
[Black "Boris"]
[Round "1"]
[WhiteElo "1500"]
[BlackElo "1500"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *
"""

_DRAMA = """[Event "Round"]
[Site "https://lichess.org/dramagame"]
[White "Hero"]
[Black "Victim"]
[Round "1"]
[WhiteElo "1500"]
[BlackElo "1500"]
[Result "1-0"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""


def _round_events():
    """Ingest the two-board round in one snapshot through the engine-free baseline model,
    exactly as the `--captions-srt/--captions-vtt` CLI path does (`ingest_snapshot` on the
    full multi-game PGN — `LocalPgnFeed` only replays a single game)."""
    ingestor = BroadcastIngestor(
        LocalPgnFeed(_QUIET),  # feed is unused for snapshot ingest; satisfies the ctor
        LichessBaselineModel(),
        white_elo=1500,
        black_elo=1500,
    )
    return ingestor.ingest_snapshot(_QUIET + "\n" + _DRAMA)


def _single_game_events():
    with open(SAMPLE_PGN, "r", encoding="utf-8") as fh:
        text = fh.read()
    feed = LocalPgnFeed(text)
    ingestor = BroadcastIngestor(
        feed, LichessBaselineModel(), white_elo=1800, black_elo=1800
    )
    events = []
    while True:
        snap = feed.poll()
        if snap is None:
            break
        events.extend(ingestor.ingest_snapshot(snap))
    return events


def test_auto_follow_adds_a_focus_cut_cue_at_the_cut_ply():
    """The scholar's-mate Qxf7# steals focus -> a dedicated "cut to Bd2" cue lands right
    before that move's caption, naming the board being cut TO."""
    events = _round_events()
    base = _caption_cues(events, auto_follow=False)
    auto = _caption_cues(events, auto_follow=True)

    # auto_follow only ADDS focus cues — every base cue is preserved, in order.
    base_texts = [text for _start, _end, text in base]
    auto_texts = [text for _start, _end, text in auto]
    cut_texts = [t for t in auto_texts if t.startswith("cut to ")]
    assert cut_texts, "a dramatic board must add at least one focus-cut cue"
    # The cut names board 1 (0-based) as "Bd2" and reads as the director cue.
    assert any("cut to Bd2" in t for t in cut_texts), cut_texts
    assert [t for t in auto_texts if not t.startswith("cut to ")] == base_texts

    # The cut cue sits immediately before the move's own caption (the Qxf7# clutch line),
    # i.e. the director announces the cut, then the dramatic move plays.
    cut_idx = next(i for i, t in enumerate(auto_texts) if t.startswith("cut to "))
    assert "Qxf7" in auto_texts[cut_idx + 1], auto_texts[cut_idx : cut_idx + 2]
    # ...and it lands at the cut ply's start time (cue is well-formed, end > start).
    assert auto[cut_idx][1] > auto[cut_idx][0]


def test_focus_cut_cue_rides_both_vtt_and_srt_dialects():
    """The cue rides the shared `_caption_cues` source, so it surfaces verbatim in both
    export containers, and only when auto_follow is on."""
    events = _round_events()
    vtt_on = build_captions_vtt(events, auto_follow=True)
    srt_on = build_captions_srt(events, auto_follow=True)
    assert "cut to Bd2" in vtt_on
    assert "cut to Bd2" in srt_on

    vtt_off = build_captions_vtt(events, auto_follow=False)
    srt_off = build_captions_srt(events, auto_follow=False)
    assert "cut to" not in vtt_off
    assert "cut to" not in srt_off


def test_single_board_export_is_byte_identical_with_auto_follow():
    """A single-game snapshot has no rival board to cut to, so auto_follow changes
    nothing — the default export is preserved byte-for-byte."""
    events = _single_game_events()
    assert build_captions_vtt(events, auto_follow=True) == build_captions_vtt(events)
    assert build_captions_srt(events, auto_follow=True) == build_captions_srt(events)
