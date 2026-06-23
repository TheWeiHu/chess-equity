"""Timestamped WebVTT caption export for ``broadcast --captions-vtt`` (task 0211).

``broadcast --captions`` (task 0190) prints one caster sentence per graded move to
stdout; this turns that same stream into a *timestamped* WebVTT subtitle track so the
caster line becomes a real caption/TTS track for a recorded stream. One cue per graded
move, keyed by the game's own [%clk]: each cue starts at the elapsed game time the move
was played, so the subtitles line up with a screen recording paced by the players'
clocks. These tests pin the acceptance facts:

* ``build_captions_vtt`` emits a valid ``WEBVTT`` document — one cue per graded move,
  each cue's text the caster caption, with strictly increasing, clock-keyed timings; and
* the ``--captions-vtt OUT`` CLI flag writes that file from a local ``--pgn`` and refuses
  to run without one (no live feed to time against).

Fixture: ``data/sample/sample_games.pgn`` — the sanctioned offline-smoke fixture
(illustrative, not evidence; see project CLAUDE.md). Its first game is a 7-ply scholar's
mate carrying [%clk] tags, so the cue timings are deterministic. Like the sibling
``--captions`` test, the unit path drives an engine-free :class:`LichessBaselineModel`
so the captions are independent of whether Stockfish is installed.
"""
import io
import os
import re

import chess
import chess.pgn

from chess_equity.broadcast import (
    BroadcastIngestor,
    LocalPgnFeed,
    build_captions_vtt,
    live_caption,
)
from chess_equity.cli import main
from chess_equity.models import LichessBaselineModel

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PGN = os.path.join(HERE, "..", "data", "sample", "sample_games.pgn")

_TS = r"\d{2}:\d{2}:\d{2}\.\d{3}"
_CUE_RE = re.compile(rf"^({_TS}) --> ({_TS})$")


def _first_game_sans():
    with open(SAMPLE_PGN, "r", encoding="utf-8") as fh:
        game = chess.pgn.read_game(io.StringIO(fh.read()))
    assert game is not None
    board = game.board()
    sans = []
    for move in game.mainline_moves():
        sans.append(board.san(move))
        board.push(move)
    return sans


def _replay_events():
    """Replay the sample's first game through the engine-free baseline model."""
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


def _ts_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def test_build_captions_vtt_one_clock_keyed_cue_per_graded_move():
    events = _replay_events()
    sans = _first_game_sans()
    vtt = build_captions_vtt(events)

    lines = vtt.splitlines()
    assert lines[0] == "WEBVTT"

    cue_lines = [ln for ln in lines if _CUE_RE.match(ln)]
    captions = [live_caption(e) for e in events]
    # One cue per graded move; each move past the opening grades on the baseline.
    assert len(cue_lines) == len(sans) == len(captions)

    # Cue payloads are the caster captions, in order, each naming its move.
    payloads = [lines[lines.index(c) + 1] for c in cue_lines]
    for san, cap, payload in zip(sans, captions, payloads):
        assert payload == cap, (payload, cap)
        assert payload.startswith(f"{san} — "), payload

    # Each cue is well-formed (end > start) and starts strictly increase — the moves
    # play out over the recording's timeline, never overlapping or going backwards.
    starts, ends = [], []
    for cue in cue_lines:
        m = _CUE_RE.match(cue)
        starts.append(_ts_to_seconds(m.group(1)))
        ends.append(_ts_to_seconds(m.group(2)))
    for i in range(len(starts)):
        assert ends[i] > starts[i], (starts[i], ends[i])
    assert all(starts[i] < starts[i + 1] for i in range(len(starts) - 1)), starts

    # Keyed by the [%clk] deltas, not a flat per-move dwell: the first two cues fall on
    # the 3s opening fallback (no prior clock reading for each side), then White's
    # 180→178 over Bc4 lands the third cue at 8s (3 + 3 + 2), not a uniform 9s.
    assert starts[0] == 3.0 and starts[1] == 6.0 and starts[2] == 8.0, starts


def test_cli_captions_vtt_writes_valid_track(tmp_path):
    out = tmp_path / "captions.vtt"
    rc = main(
        [
            "broadcast",
            "--pgn",
            SAMPLE_PGN,
            "--captions-vtt",
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
    assert text.startswith("WEBVTT")
    cues = [ln for ln in text.splitlines() if _CUE_RE.match(ln)]
    assert len(cues) >= len(_first_game_sans())  # at least the first game's graded moves


def test_cli_captions_vtt_requires_pgn(capsys):
    """A live feed (--round) has no fixed timeline to subtitle, so the export refuses
    it and returns 2 — the guard fires before the feed is built, so no network."""
    rc = main(["broadcast", "--round", "deadbeef", "--captions-vtt", "/tmp/should-not-write.vtt"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "requires --pgn" in err
