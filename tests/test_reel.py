"""Tests for the auto-highlight reel export (task 0168).

The reel composes over ``DramaEvent``s, so — like ``test_drama`` — the per-kind
coverage test synthesises events with controlled equity/Δequity/clocks (the baseline
model's swings are muted, so a real PGN replay won't reliably surface every kind).
A separate smoke test drives the real CLI over the committed sample fixture and only
asserts both artifacts land and are well-formed.
"""

import dataclasses
import json

from chess_equity.broadcast import MoveEvent
from chess_equity.drama import score_event
from chess_equity.reel import (
    build_reel,
    by_kind,
    caption,
    caption_payload,
    rank,
    reel_payload,
    render_captions,
    render_json,
    render_markdown,
)

# Neutral base event (White just moved; quiet). Mirror test_drama's fixture.
_BASE = MoveEvent(
    game_id="g1",
    ply=10,
    san="Nf3",
    uci="g1f3",
    fen="rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 0 1",
    white_to_move=False,  # Black to move => White was the mover
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


def ev(**over):
    return dataclasses.replace(_BASE, **over)


# One MoveEvent that fires each of the four drama kinds (validated below).
_ONE_OF_EACH = [
    ev(ply=2, equity=65.0, delta_equity=15.0),                       # clutch (+15)
    ev(ply=4, equity=65.0, delta_equity=-20.0),                      # missed_win
    ev(ply=6, white_to_move=True, equity=75.0, delta_equity=20.0),   # escape (Black mover)
    ev(ply=8, equity=58.0, delta_equity=7.0, white_clock=8.0),       # scramble (low clock)
]


def test_fixture_surfaces_one_of_each_kind():
    kinds = {score_event(e).kind for e in _ONE_OF_EACH}
    assert kinds == {"clutch", "missed_win", "escape", "scramble"}


def test_reel_ranked_by_magnitude_desc():
    reel = build_reel(_ONE_OF_EACH)
    mags = [d.magnitude for d in reel]
    assert mags == sorted(mags, reverse=True)
    # The biggest swing (missed_win, |−20|) leads; the smallest (scramble, 7) trails.
    assert reel[0].kind == "missed_win"
    assert reel[-1].kind == "scramble"


def test_each_trigger_type_surfaces_in_reel():
    reel = build_reel(_ONE_OF_EACH)
    assert set(by_kind(reel)) == {"clutch", "missed_win", "escape", "scramble"}


def test_rank_breaks_ties_by_kind_priority_then_ply():
    # Two equal-magnitude events of different kinds: missed_win outranks clutch.
    same = [
        ev(ply=4, equity=70.0, delta_equity=20.0),    # clutch, mag 20/40
        ev(ply=2, equity=64.0, delta_equity=-20.0),   # missed_win, mag 20/40
    ]
    ranked = rank([score_event(e) for e in same])
    assert ranked[0].magnitude == ranked[1].magnitude
    assert ranked[0].kind == "missed_win"  # tie broken by drama-type priority


def test_top_caps_the_reel():
    assert len(build_reel(_ONE_OF_EACH, top=2)) == 2


def test_render_json_payload_shape():
    reel = build_reel(_ONE_OF_EACH)
    payload = json.loads(render_json(reel, title="My reel"))
    assert payload["title"] == "My reel"
    assert payload["count"] == len(reel)
    assert payload["by_kind"] == by_kind(reel)
    assert len(payload["moments"]) == len(reel)
    # Moments keep the ranked order in the serialised payload.
    assert [m["kind"] for m in payload["moments"]] == [d.kind for d in reel]
    assert reel_payload(reel)["count"] == len(reel)


def test_render_markdown_lists_every_kind_and_top_section():
    md = render_markdown(build_reel(_ONE_OF_EACH), title="My reel")
    assert md.startswith("# My reel")
    assert "## Top moments" in md
    assert "## By drama type" in md
    for kind in ("clutch", "missed_win", "escape", "scramble"):
        assert kind in md


def test_render_markdown_empty_reel_is_graceful():
    md = render_markdown([])
    assert "No highlight-worthy moments" in md
    assert "## Top moments" not in md


def test_caption_payload_shape():
    reel = build_reel(_ONE_OF_EACH)
    payload = json.loads(render_captions(reel, title="My reel"))
    assert payload["title"] == "My reel"
    assert payload["count"] == len(reel)
    caps = payload["captions"]
    assert len(caps) == len(reel)
    # Captions keep the ranked order and carry exactly the OBS lower-third schema.
    assert [c["kind"] for c in caps] == [d.kind for d in reel]
    assert [c["ply"] for c in caps] == [d.ply for d in reel]
    for c in caps:
        assert set(c) == {"text", "kind", "ply", "duration_s"}
        assert isinstance(c["text"], str) and c["text"]
        assert 3.0 <= c["duration_s"] <= 6.0
    assert caption_payload(reel)["count"] == len(reel)


def test_caption_text_reuses_kind_label_and_signed_delta():
    # missed_win on White, Δ −20: text uses the shared _KIND_LABEL string + signed pts.
    d = score_event(ev(ply=4, equity=65.0, delta_equity=-20.0))
    c = caption(d)
    assert "Missed win" in c["text"]
    assert "White" in c["text"]
    assert "-20 pts" in c["text"]


def test_caption_duration_scales_with_magnitude():
    # A bigger swing lingers longer on screen than a smaller one.
    big = caption(score_event(ev(equity=65.0, delta_equity=-20.0)))     # mag 0.5
    small = caption(score_event(ev(ply=8, equity=58.0, delta_equity=7.0, white_clock=8.0)))
    assert big["duration_s"] > small["duration_s"]


def test_render_captions_empty_reel_is_graceful():
    payload = json.loads(render_captions([]))
    assert payload["count"] == 0
    assert payload["captions"] == []


def test_cli_reel_writes_both_artifacts(tmp_path):
    from chess_equity.cli import main

    out = tmp_path / "reel"
    rc = main(
        ["reel", "--pgn", "data/sample/sample_games.pgn", "--out-dir", str(out)]
    )
    assert rc == 0
    json_path = out / "reel.json"
    md_path = out / "reel.md"
    assert json_path.exists() and md_path.exists()

    payload = json.loads(json_path.read_text())
    assert "moments" in payload and "by_kind" in payload
    # Whatever drama the baseline surfaces, the JSON reel is magnitude-ranked.
    mags = [m["magnitude"] for m in payload["moments"]]
    assert mags == sorted(mags, reverse=True)
    assert md_path.read_text().startswith("# Highlight reel")
