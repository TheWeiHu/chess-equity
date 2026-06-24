"""SRT (SubRip) caption export for ``broadcast --captions-srt`` (task 0229).

``broadcast --captions-vtt`` (task 0211) writes the per-move caster captions as a
timestamped WebVTT track; this is the SRT parity sibling for non-web editors
(Premiere/Resolve/CapCut) that can't ingest WebVTT — exactly as the reel path already
ships both VTT and SRT (task 0216). Both exporters render the *same* ``_caption_cues``
timeline, so the SRT track is cue-for-cue identical to the VTT track and differs only in
the container dialect: no ``WEBVTT`` header, numbered cues, ``HH:MM:SS,mmm`` comma-decimal
timestamps, and raw (un-escaped) cue text. These tests pin the acceptance facts:

* ``build_captions_srt`` emits a valid SubRip document — one numbered cue per graded move,
  comma-millisecond timestamps, each cue's text the caster caption; and
* the SRT shares the VTT's cue count, payloads, and timings (the shared cue source); and
* the ``--captions-srt OUT`` CLI flag writes that file from a local ``--pgn`` and refuses
  to run without one.

Fixture: ``data/sample/sample_games.pgn`` — the sanctioned offline-smoke fixture
(illustrative, not evidence; see project CLAUDE.md). Its first game is a 7-ply scholar's
mate carrying [%clk] tags, so the cue timings are deterministic. The unit path drives an
engine-free :class:`LichessBaselineModel` so captions are independent of Stockfish.
"""
import re

from chess_equity.broadcast import (
    _drama_callout,
    build_captions_srt,
    build_captions_vtt,
    live_caption,
)
from chess_equity.cli import main
from chess_equity.drama import score_event

from test_broadcast_captions_vtt import (  # reuse the sibling test's fixtures
    SAMPLE_PGN,
    _first_game_sans,
    _replay_all_games,
    _replay_events,
)

# SRT timestamps use the SubRip comma decimal (``HH:MM:SS,mmm``), not WebVTT's dot.
_TS = r"\d{2}:\d{2}:\d{2},\d{3}"
_CUE_RE = re.compile(rf"^({_TS}) --> ({_TS})$")


def _ts_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def test_build_captions_srt_one_numbered_cue_per_graded_move():
    events = _replay_events()
    sans = _first_game_sans()
    srt = build_captions_srt(events)

    # No WEBVTT header — SubRip starts straight at cue 1.
    assert not srt.startswith("WEBVTT")
    assert srt.startswith("1\n")

    blocks = [b for b in srt.split("\n\n") if b.strip()]
    captions = [live_caption(e) for e in events]
    assert len(blocks) == len(sans) == len(captions)

    starts = []
    for idx, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        # Block: <index>, <timestamp line>, <caption text>.
        assert lines[0] == str(idx), lines[0]
        m = _CUE_RE.match(lines[1])
        assert m, lines[1]
        start, end = _ts_to_seconds(m.group(1)), _ts_to_seconds(m.group(2))
        assert end > start, (start, end)
        starts.append(start)
        payload = lines[2]
        san, cap = sans[idx - 1], captions[idx - 1]
        # A dramatic move (the closing Qxf7# clutch) is prefixed with its emoji+label
        # drama callout; undramatic cues are the bare caster caption.
        d = score_event(events[idx - 1])
        expected = f"{_drama_callout(d.kind)} — {cap}" if d is not None else cap
        assert payload == expected, (payload, expected)
        assert f"{san} — " in payload, payload

    # Starts strictly increase, keyed by [%clk] deltas (same as the VTT track): 3s and 6s
    # opening fallbacks, then White's 180→178 lands the third cue at 8s, not a flat 9s.
    assert all(starts[i] < starts[i + 1] for i in range(len(starts) - 1)), starts
    assert starts[0] == 3.0 and starts[1] == 6.0 and starts[2] == 8.0, starts


def test_build_captions_srt_resets_timeline_per_game():
    """Multi-game --pgn: clocks reset each game, so the SRT timeline must restart at each
    board boundary — board 2's first cue lands near t=0, not after board 1's total
    (task 0232, the SRT half of the shared ``_caption_cues`` reset added in 0230). The VTT
    test pins the same fact directly; the cue-for-cue parity test only spans one game, so
    this asserts the SRT track inherits the reset on its own. Numbering stays globally
    sequential and single-game output is unchanged (covered by the per-cue-timing test)."""
    events = _replay_all_games()
    # The sample is a multi-game round, so events span >1 distinct game.
    game_ids = list(dict.fromkeys(e.game_id for e in events))
    assert len(game_ids) >= 2, game_ids

    srt = build_captions_srt(events)
    cue_lines = [ln for ln in srt.splitlines() if _CUE_RE.match(ln)]
    starts = [_ts_to_seconds(_CUE_RE.match(c).group(1)) for c in cue_lines]

    # Map each graded cue back to its game so we can find each board's first cue.
    captioned = [e for e in events if live_caption(e) is not None]
    assert len(captioned) == len(starts)
    first_start_by_game = {}
    for e, start in zip(captioned, starts):
        first_start_by_game.setdefault(e.game_id, start)

    # Board 1's first cue is at the 3s opening fallback; every later board's first cue is
    # likewise near t=0 — NOT pushed past the prior board's accumulated elapsed time.
    board1_last = max(s for e, s in zip(captioned, starts) if e.game_id == game_ids[0])
    for gid in game_ids[1:]:
        assert first_start_by_game[gid] <= board1_last, (gid, first_start_by_game[gid], board1_last)
        assert first_start_by_game[gid] < 5.0, (gid, first_start_by_game[gid])

    # Numbering is still globally sequential 1..N across all boards (one track).
    nums = [int(ln) for ln in srt.splitlines() if ln.isdigit()]
    assert nums == list(range(1, len(cue_lines) + 1)), nums


def test_srt_matches_vtt_cue_for_cue():
    """The two exporters share one cue source, so the tracks are identical bar the
    container: same cue count, same payloads, same start/end seconds."""
    events = _replay_events()
    srt = build_captions_srt(events)
    vtt = build_captions_vtt(events)

    srt_cues = [_CUE_RE.match(ln) for ln in srt.splitlines()]
    srt_cues = [m for m in srt_cues if m]
    vtt_ts = r"\d{2}:\d{2}:\d{2}\.\d{3}"
    vtt_re = re.compile(rf"^({vtt_ts}) --> ({vtt_ts})$")
    vtt_cues = [vtt_re.match(ln) for ln in vtt.splitlines()]
    vtt_cues = [m for m in vtt_cues if m]

    assert len(srt_cues) == len(vtt_cues)
    for s, v in zip(srt_cues, vtt_cues):
        # Comma vs dot is the only difference; compare the numeric seconds.
        assert _ts_to_seconds(s.group(1)) == _vtt_ts_to_seconds(v.group(1))
        assert _ts_to_seconds(s.group(2)) == _vtt_ts_to_seconds(v.group(2))


def _vtt_ts_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def test_cli_captions_srt_writes_valid_track(tmp_path):
    out = tmp_path / "captions.srt"
    rc = main(
        [
            "broadcast",
            "--pgn",
            SAMPLE_PGN,
            "--captions-srt",
            str(out),
            "--white-elo",
            "1800",
            "--black-elo",
            "1800",
            "--interval",
            "0",
        ]
    )
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert not text.startswith("WEBVTT")
    assert text.startswith("1\n")
    cues = [ln for ln in text.splitlines() if _CUE_RE.match(ln)]
    assert len(cues) >= len(_first_game_sans())  # at least the first game's graded moves


def test_cli_captions_srt_requires_pgn(capsys):
    """A live feed (--round) has no fixed timeline to subtitle, so the export refuses it
    and returns 2 — the guard fires before the feed is built, so no network."""
    rc = main(["broadcast", "--round", "deadbeef", "--captions-srt", "/tmp/should-not-write.srt"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "requires --pgn" in err
