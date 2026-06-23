"""Tests for the cross-game round recap (task 0198).

A tournament round PGN holds many games; the recap pools the drama across ALL boards
and ranks one cross-game reel. Pooling itself is free (the broadcast ingestor tags every
``MoveEvent`` with its ``game_id`` and ``drama.score_event`` is stateless), so these
tests use synthetic ``MoveEvent``s from >1 game — the baseline model's swings are muted,
so a real replay won't reliably surface drama (mirrors ``test_reel``'s fixture convention).
What's new and tested here: moments from different games are interleaved and ranked by
magnitude, ``--top`` caps the pooled list, and each moment names its source board + pairing.
"""

import dataclasses
import json

from chess_equity.broadcast import MoveEvent
from chess_equity.reel import (
    GameSource,
    build_reel,
    game_sources,
    render_html,
    render_json,
    render_markdown,
)

# Two-game round fixture: each MoveEvent carries the game_id its source PGN would mint.
_BASE = MoveEvent(
    game_id="g?",
    ply=10,
    san="Nf3",
    uci="g1f3",
    fen="rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 0 1",
    white_to_move=False,
    white_clock=120.0,
    black_clock=120.0,
    white_elo=2000,
    black_elo=2000,
    equity=51.0,
    delta_equity=1.0,
    last_move_grade="ok",
    source="Test",
    compute_ms=0.1,
)


def ev(game_id, **over):
    return dataclasses.replace(_BASE, game_id=game_id, **over)


# Board 1 (alice-bob) and Board 2 (carol-dave), each contributing drama of a different
# magnitude so the pooled rank order is unambiguous.
_G1 = "alice-bob-R?#0"
_G2 = "carol-dave-R?#1"
_ROUND_EVENTS = [
    ev(_G1, ply=4, equity=65.0, delta_equity=-20.0),   # board 1 missed_win, mag 0.50
    ev(_G2, ply=2, equity=65.0, delta_equity=15.0),    # board 2 clutch,     mag 0.375
    ev(_G1, ply=8, equity=58.0, delta_equity=7.0, white_clock=8.0),  # board 1 scramble, mag ~0.18
]

# A 2-game PGN whose split mints exactly _G1 / _G2 (no Site/GameId → pairing-based ids).
_ROUND_PGN = """[Event "Round 1"]
[White "alice"]
[Black "bob"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "Round 1"]
[White "carol"]
[Black "dave"]
[Result "0-1"]

1. d4 d5 2. c4 e6 0-1
"""


def test_pooled_reel_interleaves_moments_from_multiple_games():
    reel = build_reel(_ROUND_EVENTS)
    game_ids = {d.game_id for d in reel}
    assert len(game_ids) > 1  # the recap pooled across boards
    assert {_G1, _G2} <= game_ids


def test_pooled_reel_sorted_by_magnitude_desc():
    reel = build_reel(_ROUND_EVENTS)
    mags = [d.magnitude for d in reel]
    assert mags == sorted(mags, reverse=True)
    # Across games, the biggest swing (board 1 missed_win) leads the cross-game reel.
    assert reel[0].game_id == _G1
    assert reel[0].kind == "missed_win"
    # The #2 moment comes from the OTHER board — moments interleave, not group by game.
    assert reel[1].game_id == _G2


def test_top_caps_the_pooled_list():
    assert len(build_reel(_ROUND_EVENTS, top=2)) == 2


def test_game_sources_maps_ids_to_board_and_pairing():
    sources = game_sources(_ROUND_PGN)
    assert set(sources) == {_G1, _G2}
    assert sources[_G1] == GameSource(game_id=_G1, board=1, white="alice", black="bob")
    assert sources[_G2] == GameSource(game_id=_G2, board=2, white="carol", black="dave")
    assert sources[_G2].label == "Board 2 · carol vs dave"


def test_round_markdown_names_each_moment_source():
    reel = build_reel(_ROUND_EVENTS)
    sources = game_sources(_ROUND_PGN)
    md = render_markdown(reel, title="Round recap", sources=sources)
    # Each contributing board is named on its moment(s).
    assert "Board 1 · alice vs bob" in md
    assert "Board 2 · carol vs dave" in md
    # The summary states the pool spanned multiple boards.
    assert "across 2 board(s)" in md


def test_round_json_carries_source_and_board_per_moment():
    reel = build_reel(_ROUND_EVENTS)
    sources = game_sources(_ROUND_PGN)
    payload = json.loads(render_json(reel, sources=sources))
    assert payload["games"] == 2
    for m in payload["moments"]:
        assert m["source"].startswith("Board ")
        assert m["board"] in (1, 2)
    # Board 1's leading missed_win names its source.
    assert payload["moments"][0]["source"] == "Board 1 · alice vs bob"


def test_round_html_names_source_board():
    reel = build_reel(_ROUND_EVENTS)
    sources = game_sources(_ROUND_PGN)
    doc = render_html(reel, title="Round recap", sources=sources)
    assert doc.startswith("<!doctype html>")
    assert "Board 1 · alice vs bob" in doc
    assert "Board 2 · carol vs dave" in doc
    # Still self-contained — nothing fetched from the network.
    assert "http://" not in doc and "https://" not in doc


def test_single_game_output_unchanged_without_sources():
    # Regression: without sources the moment location is the bare game id, as before.
    reel = build_reel([ev(_G1, ply=4, equity=65.0, delta_equity=-20.0)])
    md = render_markdown(reel)
    assert f"game {_G1}" in md
    assert "Board" not in md
    payload = json.loads(render_json(reel))
    assert "games" not in payload
    assert "source" not in payload["moments"][0]


def test_cli_round_smoke_writes_artifacts(tmp_path):
    from chess_equity.cli import main

    out = tmp_path / "round"
    rc = main(
        [
            "reel",
            "--round",
            "--pgn",
            "data/sample/sample_games.pgn",
            "--out-dir",
            str(out),
            "--model",
            "baseline",
        ]
    )
    assert rc == 0
    payload = json.loads((out / "reel.json").read_text())
    # Round framing landed even if the baseline surfaces no drama on this fixture.
    assert payload["title"] == "Round recap"
    assert "games" in payload
    mags = [m["magnitude"] for m in payload["moments"]]
    assert mags == sorted(mags, reverse=True)
    assert (out / "reel.md").read_text().startswith("# Round recap")
