"""Tests for the auto-highlight reel export (task 0168).

The reel composes over ``DramaEvent``s, so — like ``test_drama`` — the per-kind
coverage test synthesises events with controlled equity/Δequity/clocks (the baseline
model's swings are muted, so a real PGN replay won't reliably surface every kind).
A separate smoke test drives the real CLI over the committed sample fixture and only
asserts both artifacts land and are well-formed.
"""

import base64
import dataclasses
import json
import re

import pytest

from chess_equity.broadcast import MoveEvent
from chess_equity.drama import score_event
from chess_equity.reel import (
    _KIND_LABEL,
    build_chapters,
    build_poster_svg,
    build_reel,
    build_srt,
    build_webvtt,
    by_kind,
    caption,
    caption_payload,
    clip_durations,
    detect_divergence,
    divergence_payload,
    drop_below_magnitude,
    rank,
    reel_payload,
    render_captions,
    render_divergence_markdown,
    render_html,
    render_json,
    render_markdown,
    social_caption,
    write_posters,
)
from chess_equity.types import lichess_win_percent

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


# --- Human-vs-engine divergence category (task 0272) -------------------------
#
# Divergence ranks moves by |equity − cp_win| (cp_win = the rating-blind Lichess Win%
# of the White-POV cp). cp=0 ⇒ cp_win=50, so a move with equity=80% diverges 30 pts.
# These fixtures set cp explicitly; the base _BASE has no cp (skipped by design).

# A QUIET move (Δequity≈0 ⇒ no drama) that nonetheless diverges hugely from the engine:
# the human bar reads 82% White while the engine (cp=0) reads 50% — a 32-pt gap.
_QUIET_BUT_DIVERGENT = ev(ply=12, equity=82.0, delta_equity=1.0, cp=0.0)


def test_detect_divergence_ranks_by_abs_gap_desc():
    events = [
        ev(ply=2, equity=55.0, cp=0.0),    # gap |55−50| = 5
        ev(ply=4, equity=90.0, cp=0.0),    # gap |90−50| = 40  ← biggest
        ev(ply=6, equity=20.0, cp=300.0),  # cp_win≈75 ⇒ gap ≈55  ← actually biggest
    ]
    moments = detect_divergence(events)
    gaps = [m.divergence for m in moments]
    assert gaps == sorted(gaps, reverse=True)
    # The engine-disagreement leader: human says Black-favored (20%) while the engine
    # (cp=+300) says White is winning (~75%).
    assert moments[0].ply == 6
    assert moments[0].divergence == pytest.approx(abs(20.0 - lichess_win_percent(300.0)))


def test_detect_divergence_skips_moves_without_cp():
    # _BASE carries no cp (a cp-less / mate feed) → no engine bar to diverge from.
    assert detect_divergence([ev(ply=2, equity=99.0)]) == []


def test_detect_divergence_surfaces_quiet_high_divergence_move():
    # The move never registers as drama (Δequity≈0) but tops the divergence list — the
    # whole point of a SEPARATE category.
    assert build_reel([_QUIET_BUT_DIVERGENT]) == []  # no drama
    moments = detect_divergence([_QUIET_BUT_DIVERGENT])
    assert len(moments) == 1
    assert moments[0].divergence == pytest.approx(32.0)
    assert moments[0].signed_gap == pytest.approx(32.0)  # human bar above engine


def test_detect_divergence_top_caps_the_list():
    events = [ev(ply=p, equity=50.0 + p, cp=0.0) for p in range(1, 6)]
    assert len(detect_divergence(events, top=2)) == 2


def test_divergence_payload_shape_and_caption():
    payload = divergence_payload(detect_divergence([_QUIET_BUT_DIVERGENT]))
    assert payload["count"] == 1
    moment = payload["moments"][0]
    assert moment["cp_win"] == pytest.approx(50.0)
    assert "human bar 82% vs engine 50%" in moment["caption"]


def test_reel_payload_embeds_divergence_block_when_given():
    reel = build_reel(_ONE_OF_EACH)
    divergence = detect_divergence([_QUIET_BUT_DIVERGENT])
    payload = reel_payload(reel, divergence=divergence)
    assert payload["divergence"]["count"] == 1
    # Omitted entirely when not requested (drama-only contract unchanged).
    assert "divergence" not in reel_payload(reel)


def test_render_markdown_appends_divergence_section():
    reel = build_reel(_ONE_OF_EACH)
    md = render_markdown(reel, divergence=detect_divergence([_QUIET_BUT_DIVERGENT]))
    assert "## Human-vs-engine divergence" in md
    # Not present when divergence is not requested.
    assert "Human-vs-engine divergence" not in render_markdown(reel)


def test_render_divergence_markdown_graceful_on_empty():
    md = render_divergence_markdown([])
    assert md.startswith("# Human-vs-engine divergence")
    assert "No human-vs-engine divergence" in md


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


def test_drop_below_magnitude_keeps_only_at_or_above_floor():
    reel = build_reel(_ONE_OF_EACH)
    floor = 0.3
    kept, dropped = drop_below_magnitude(reel, floor)
    # Strictly shorter-or-equal, only moments at/above the floor survive.
    assert len(kept) <= len(reel)
    assert all(d.magnitude >= floor for d in kept)
    assert dropped == len(reel) - len(kept)
    # Order is preserved (still magnitude-ranked since the input was).
    assert [d.magnitude for d in kept] == sorted(
        (d.magnitude for d in kept), reverse=True
    )


def test_drop_below_magnitude_zero_floor_keeps_everything():
    reel = build_reel(_ONE_OF_EACH)
    kept, dropped = drop_below_magnitude(reel, 0.0)
    assert kept == reel and dropped == 0


def test_cli_reel_min_magnitude_shortens_reel(tmp_path):
    from chess_equity.cli import main

    base = tmp_path / "all"
    floored = tmp_path / "floored"
    assert main(["reel", "--pgn", "data/sample/sample_games.pgn", "--out-dir", str(base)]) == 0
    assert (
        main(
            [
                "reel",
                "--pgn",
                "data/sample/sample_games.pgn",
                "--min-magnitude",
                "0.1",
                "--out-dir",
                str(floored),
            ]
        )
        == 0
    )
    all_moments = json.loads((base / "reel.json").read_text())["moments"]
    kept = json.loads((floored / "reel.json").read_text())["moments"]
    # Strictly shorter-or-equal, and every surviving moment clears the floor.
    assert len(kept) <= len(all_moments)
    assert all(m["magnitude"] >= 0.1 for m in kept)


def test_cli_reel_rejects_out_of_range_floor(tmp_path):
    from chess_equity.cli import main

    rc = main(
        ["reel", "--pgn", "data/sample/sample_games.pgn", "--min-magnitude", "1.5"]
    )
    assert rc == 1


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


def test_each_moment_carries_a_social_caption():
    # Acceptance: every reel.json moment carries a 'caption' string.
    payload = json.loads(render_json(build_reel(_ONE_OF_EACH)))
    for m in payload["moments"]:
        assert isinstance(m["caption"], str) and m["caption"]


def test_social_caption_single_game_names_side_move_grade_and_swing():
    # A clutch: White finds the move, grade label + signed practical-equity swing.
    d = score_event(ev(ply=2, san="Qxf7#", equity=65.0, delta_equity=15.0))
    cap = social_caption(d)
    assert cap == "White finds Qxf7#, clutch (+15 vs peers)"
    # No board prefix and no player name without a round recap's sources map.
    assert not cap.startswith("Board ")


def test_social_caption_missed_win_uses_signed_negative_swing():
    d = score_event(ev(ply=4, san="Rd1", equity=65.0, delta_equity=-20.0))
    assert social_caption(d) == "White lets a win slip on Rd1, missed win (-20 vs peers)"


def test_html_card_shows_the_social_caption():
    reel = build_reel(_ONE_OF_EACH)
    doc = render_html(reel)
    assert 'class="share"' in doc
    assert social_caption(reel[0]) in doc


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


# --- HTML clip player (task 0184) --------------------------------------------

def test_render_html_is_self_contained_and_lists_moments():
    reel = build_reel(_ONE_OF_EACH)
    doc = render_html(reel, title="My reel")
    # A well-formed, standalone document.
    assert doc.startswith("<!doctype html>")
    assert "<title>My reel</title>" in doc
    assert doc.rstrip().endswith("</html>")
    # Self-contained: no external deps / CDN / scripts. The only src= permitted is
    # the inline WebVTT data: URI (the captions track) — never an external fetch.
    assert "http://" not in doc and "https://" not in doc
    assert "<script" not in doc and "<link" not in doc
    for src in re.findall(r'src="([^"]*)"', doc):
        assert src.startswith("data:"), f"non-inline src in self-contained doc: {src}"
    # Every drama kind's caster label/emoji surfaces.
    for kind in ("clutch", "missed_win", "escape", "scramble"):
        d = next(x for x in reel if x.kind == kind)
        emoji, label = _KIND_LABEL[kind]
        assert label in doc and emoji in doc
        # The caster caption text is reused verbatim, and the equity swing shown.
        assert caption(d)["text"] in doc
    assert "+15 pts" in doc and "-20 pts" in doc  # signed Δequity swings


def test_render_html_renders_board_from_fen():
    reel = build_reel(_ONE_OF_EACH)
    doc = render_html(reel)
    # The FEN carries through to the reel and a Unicode board is drawn.
    assert reel[0].fen is not None
    assert 'class="board"' in doc
    assert "♘" in doc  # the knight from the _BASE fixture FEN


def test_render_html_empty_reel_is_graceful():
    doc = render_html([])
    assert doc.startswith("<!doctype html>")
    assert "No highlight-worthy moments" in doc
    assert 'class="moment"' not in doc


# --- Static SVG poster per moment (task 0218) --------------------------------

def test_poster_svg_is_self_contained_and_shows_caption_and_bar():
    reel = build_reel(_ONE_OF_EACH)
    d = reel[0]  # the missed_win (|−20|) leads
    svg = build_poster_svg(d, 1)
    # Well-formed, standalone SVG: opens offline, no external deps / fetches / scripts.
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "http://www.w3.org/2000/svg" in svg  # the only http: permitted is the XML ns
    assert "https://" not in svg
    assert "<script" not in svg and "<image" not in svg
    # The drama kind's caster label/emoji and the 1-based rank surface.
    emoji, label = _KIND_LABEL[d.kind]
    assert label in svg and emoji in svg
    assert "#1 " in svg
    # The social caption is rendered (it may wrap across lines, so check its tokens are
    # all present rather than the contiguous string).
    for tok in social_caption(d).split():
        assert tok in svg, f"caption token missing from poster: {tok!r}"
    # White-POV equity surfaces in the header.
    assert f"White {d.equity:.0f}%" in svg


def test_poster_svg_renders_board_from_fen():
    reel = build_reel(_ONE_OF_EACH)
    svg = build_poster_svg(reel[0], 1)
    assert reel[0].fen is not None
    assert "♘" in svg  # the knight from the _BASE fixture FEN
    # The White-POV bar draws a white fill rect over a dark background.
    assert "#f2f3f5" in svg and "#14151a" in svg


def test_poster_svg_bar_height_tracks_white_equity():
    # Two real moments at different White-POV equity: escape (75%) fills taller than
    # scramble (58%). Both fire on the synthetic fixture, so neither is None.
    reel = build_reel(_ONE_OF_EACH)
    by_kind_ev = {d.kind: d for d in reel}
    high = build_poster_svg(by_kind_ev["escape"], 1)    # equity 75
    low = build_poster_svg(by_kind_ev["scramble"], 1)   # equity 58

    def fill_h(svg):
        # The <rect> carrying the white fill colour carries the fill height in its tag.
        m = re.search(r'<rect[^>]*fill="#f2f3f5"[^>]*/>', svg)
        return float(re.search(r'height="([0-9.]+)"', m.group(0)).group(1))

    assert by_kind_ev["escape"].equity > by_kind_ev["scramble"].equity
    assert fill_h(high) > fill_h(low)


def test_poster_svg_escapes_dynamic_text():
    d = score_event(ev(san="<b>&", equity=65.0, delta_equity=15.0))
    svg = build_poster_svg(d, 1)
    assert "<b>&" not in svg.replace("<svg", "")  # raw markup never leaks into the body
    assert "&amp;" in svg and "&lt;b&gt;" in svg


def test_poster_svg_handles_missing_fen():
    d = score_event(ev(fen=None, equity=55.0, delta_equity=15.0))
    svg = build_poster_svg(d, 1)
    assert "no board snapshot" in svg
    assert svg.startswith("<svg")


def test_write_posters_writes_one_svg_per_moment(tmp_path):
    reel = build_reel(_ONE_OF_EACH)
    out = tmp_path / "posters"
    paths = write_posters(reel, str(out))
    assert len(paths) == len(reel)
    for i, p in enumerate(paths, start=1):
        assert p.endswith(f"poster-{i:02d}-{reel[i - 1].kind}.svg")
        assert (out / f"poster-{i:02d}-{reel[i - 1].kind}.svg").read_text().startswith("<svg")


def test_write_posters_empty_reel_writes_nothing(tmp_path):
    out = tmp_path / "posters"
    assert write_posters([], str(out)) == []


def test_cli_reel_writes_posters(tmp_path):
    from chess_equity.cli import main

    # Run --posters alongside --out-dir so we can compare the poster count against the
    # JSON moment count (the muted baseline may surface few/zero moments on the sample).
    out = tmp_path / "reel"
    posters = tmp_path / "posters"
    rc = main(
        [
            "reel", "--pgn", "data/sample/sample_games.pgn",
            "--out-dir", str(out), "--posters", str(posters),
        ]
    )
    assert rc == 0
    payload = json.loads((out / "reel.json").read_text())
    svgs = sorted(posters.glob("poster-*.svg"))
    assert len(svgs) == len(payload["moments"])  # one SVG per ranked moment
    for svg in svgs:
        text = svg.read_text()
        assert text.startswith("<svg") and text.rstrip().endswith("</svg>")


def test_cli_reel_writes_posters_standalone(tmp_path):
    from chess_equity.cli import main

    posters = tmp_path / "posters"
    rc = main(
        ["reel", "--pgn", "data/sample/sample_games.pgn", "--posters", str(posters)]
    )
    assert rc == 0
    assert posters.is_dir()  # the directory is created even on a quiet (0-moment) game


def test_render_html_escapes_dynamic_text():
    # A title with HTML metacharacters must be escaped, not injected raw.
    doc = render_html([], title="<b>x</b> & y")
    assert "<b>x</b>" not in doc
    assert "&lt;b&gt;x&lt;/b&gt; &amp; y" in doc


def _vtt_seconds(stamp: str) -> float:
    """Parse an ``HH:MM:SS.mmm`` WebVTT timestamp into seconds."""
    h, m, rest = stamp.split(":")
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _vtt_cues(vtt: str):
    """Return [(start_s, end_s, text), ...] for every cue in a WebVTT document."""
    cues = []
    lines = vtt.splitlines()
    for i, line in enumerate(lines):
        if " --> " in line:
            start, end = line.split(" --> ")
            text = lines[i + 1] if i + 1 < len(lines) else ""
            cues.append((_vtt_seconds(start), _vtt_seconds(end), text))
    return cues


def test_build_webvtt_one_cue_per_clip_with_contiguous_timings():
    reel = build_reel(_ONE_OF_EACH)
    vtt = build_webvtt(reel)
    assert vtt.startswith("WEBVTT")
    cues = _vtt_cues(vtt)
    # One cue per clip, narrating the move-grade + signed swing (caster caption).
    assert len(cues) == len(reel)
    for (start, end, text), d in zip(cues, reel):
        assert text == caption(d)["text"]
        assert end > start
    # Cue timings line up with the clip boundaries: clips play back-to-back, each
    # for its caption dwell time, so cue i ends exactly where cue i+1 begins.
    durations = clip_durations(reel)
    expected_start = 0.0
    for (start, end, _), dur in zip(cues, durations):
        assert abs(start - expected_start) < 1e-6
        assert abs(end - (expected_start + dur)) < 1e-6
        expected_start = end


def _srt_seconds(stamp: str) -> float:
    """Parse an ``HH:MM:SS,mmm`` SRT timestamp into seconds."""
    h, m, rest = stamp.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _srt_cues(srt: str):
    """Return [(index, start_s, end_s, text), ...] for every block in an SRT document."""
    cues = []
    for block in srt.strip().split("\n\n"):
        lines = block.splitlines()
        index = int(lines[0])
        start, end = lines[1].split(" --> ")
        text = "\n".join(lines[2:])
        cues.append((index, _srt_seconds(start), _srt_seconds(end), text))
    return cues


def test_build_srt_matches_webvtt_cue_parity():
    reel = build_reel(_ONE_OF_EACH)
    srt = build_srt(reel)
    vtt_cues = _vtt_cues(build_webvtt(reel))
    srt_cues = _srt_cues(srt)
    # One SRT block per WebVTT cue per clip.
    assert len(srt_cues) == len(vtt_cues) == len(reel)
    # Sequential 1-based indices.
    assert [c[0] for c in srt_cues] == list(range(1, len(reel) + 1))
    # SRT timestamps use the comma decimal separator (HH:MM:SS,mmm), not a period.
    assert re.search(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", srt)
    assert "WEBVTT" not in srt
    for (vstart, vend, vtext), (_idx, sstart, send, stext), d in zip(
        vtt_cues, srt_cues, reel
    ):
        # Cue boundaries match the WebVTT track cue-for-cue.
        assert abs(sstart - vstart) < 1e-6
        assert abs(send - vend) < 1e-6
        # Same narration payload (WebVTT escapes & < >; SRT carries it raw).
        assert stext == caption(d)["text"]


def test_build_srt_empty_reel_is_empty():
    assert build_srt([]) == ""


def test_render_html_embeds_inline_webvtt_captions_track():
    reel = build_reel(_ONE_OF_EACH)
    doc = render_html(reel)
    assert '<track kind="captions"' in doc
    # The track is inline (a base64 data: URI) so the file stays self-contained.
    m = re.search(r'src="data:text/vtt;base64,([^"]+)"', doc)
    assert m is not None
    vtt = base64.b64decode(m.group(1)).decode("utf-8")
    assert vtt.startswith("WEBVTT")
    assert len(_vtt_cues(vtt)) == len(reel)


def test_render_html_empty_reel_has_no_track():
    doc = render_html([])
    assert "<track" not in doc and "data:text/vtt" not in doc


def test_cli_reel_writes_html_clip_player(tmp_path):
    from chess_equity.cli import main

    html_path = tmp_path / "clip.html"
    rc = main(
        ["reel", "--pgn", "data/sample/sample_games.pgn", "--html", str(html_path)]
    )
    assert rc == 0
    assert html_path.exists()
    doc = html_path.read_text()
    assert doc.startswith("<!doctype html>")
    # Opens offline — nothing fetched from the network.
    assert "http://" not in doc and "https://" not in doc


def test_cli_reel_html_alongside_out_dir(tmp_path):
    from chess_equity.cli import main

    out = tmp_path / "reel"
    rc = main(
        ["reel", "--pgn", "data/sample/sample_games.pgn", "--out-dir", str(out), "--html"]
    )
    assert rc == 0
    # With --out-dir and bare --html, reel.html lands next to reel.json/reel.md.
    assert (out / "reel.json").exists()
    assert (out / "reel.html").exists()
    assert (out / "reel.html").read_text().startswith("<!doctype html>")


def test_cli_reel_writes_srt(tmp_path):
    from chess_equity.cli import main

    # The committed sample's baseline swings are muted (no drama), so this only
    # asserts the SRT artifact lands; cue content/format parity is covered by
    # test_build_srt_matches_webvtt_cue_parity over synthesized drama events.
    srt_path = tmp_path / "clip.srt"
    rc = main(["reel", "--pgn", "data/sample/sample_games.pgn", "--srt", str(srt_path)])
    assert rc == 0
    assert srt_path.exists()


def test_cli_reel_srt_alongside_out_dir(tmp_path):
    from chess_equity.cli import main

    out = tmp_path / "reel"
    rc = main(
        ["reel", "--pgn", "data/sample/sample_games.pgn", "--out-dir", str(out), "--srt"]
    )
    assert rc == 0
    # With --out-dir and bare --srt, reel.srt lands next to reel.json/reel.md.
    assert (out / "reel.json").exists()
    assert (out / "reel.srt").exists()


# --- VOD chapter markers (task 0237) -----------------------------------------

_CHAPTER_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}) (\w+): (\S+) \(([+-]\d+)\)$")


def _chapter_seconds(stamp: str) -> int:
    """Parse an ``HH:MM:SS`` chapter marker into whole seconds."""
    h, m, s = (int(x) for x in stamp.split(":"))
    return h * 3600 + m * 60 + s


def test_build_chapters_starts_at_zero_and_is_monotonic():
    reel = build_reel(_ONE_OF_EACH)
    lines = build_chapters(reel).splitlines()
    # One chapter marker per ranked moment, parsing to the documented format.
    assert len(lines) == len(reel)
    parsed = [_CHAPTER_RE.match(line) for line in lines]
    assert all(parsed), [l for l, p in zip(lines, parsed) if p is None]
    stamps = [_chapter_seconds(m.group(0).split(" ", 1)[0]) for m in parsed]
    # YouTube requires the opening chapter at 00:00:00; stamps never go backwards.
    assert stamps[0] == 0
    assert stamps == sorted(stamps)


def test_build_chapters_line_names_kind_san_and_signed_swing():
    reel = build_reel(_ONE_OF_EACH)
    lines = build_chapters(reel).splitlines()
    for line, d in zip(lines, reel):
        m = _CHAPTER_RE.match(line)
        assert m.group(4) == d.kind
        assert m.group(5) == d.san
        assert m.group(6) == f"{d.delta_equity:+.0f}"


def test_build_chapters_timeline_matches_the_clip_timeline():
    # Chapters share the back-to-back clip timeline with the VTT/SRT exports: chapter i
    # starts at the floored cumulative dwell time of every earlier clip.
    reel = build_reel(_ONE_OF_EACH)
    lines = build_chapters(reel).splitlines()
    durations = clip_durations(reel)
    expected_start = 0.0
    for line, dur in zip(lines, durations):
        stamp = line.split(" ", 1)[0]
        assert _chapter_seconds(stamp) == int(expected_start)
        expected_start += dur


def test_build_chapters_empty_reel_is_empty():
    assert build_chapters([]) == ""


def test_cli_reel_writes_chapters(tmp_path):
    from chess_equity.cli import main

    # The committed sample's baseline swings are muted, so this only asserts the
    # chapters artifact lands; format/timeline parity is covered by the unit tests
    # over synthesized drama events.
    chapters_path = tmp_path / "chapters.txt"
    rc = main(
        ["reel", "--pgn", "data/sample/sample_games.pgn", "--chapters", str(chapters_path)]
    )
    assert rc == 0
    assert chapters_path.exists()


def test_cli_reel_chapters_alongside_out_dir(tmp_path):
    from chess_equity.cli import main

    out = tmp_path / "reel"
    rc = main(
        ["reel", "--pgn", "data/sample/sample_games.pgn", "--out-dir", str(out), "--chapters"]
    )
    assert rc == 0
    # With --out-dir and bare --chapters, reel.chapters.txt lands next to reel.json.
    assert (out / "reel.json").exists()
    assert (out / "reel.chapters.txt").exists()
