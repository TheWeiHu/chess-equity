"""The divergence-reel proof artifact: the practical bar disagrees with the engine (task 0139).

Three halves, mirroring the rating-sweep fixture pattern (``test_rating_sweep``):

* the committed bullet game produces **at least one ply where the human-edge divergence
  badge fires** (|practical equity − cp-implied| ≥ threshold) — the wedge's money shot is
  real, not hypothetical;
* the badge logic here matches ``overlay/overlay.js`` (same threshold, same side rule); and
* the committed ``reports/divergence_reel.json`` / ``.md`` are byte-for-byte what the
  generator produces (drift guard), so the artifact can never silently diverge from the
  pipeline it claims to demonstrate.

Everything runs offline on :class:`MaterialEngine` (no Stockfish, no torch), so the whole
fixture is green in CI without any engine binary — the task's UNATTENDED-OK requirement.
"""

from __future__ import annotations

from chess_equity.validate.divergence_reel import (
    JSON_ARTIFACT_PATH,
    MD_ARTIFACT_PATH,
    PGN_PATH,
    THRESHOLD,
    TOP_N,
    collect_events,
    divergences,
    generate_json_artifact,
    generate_md_artifact,
    human_edge,
    top_divergences,
)


def _events():
    return collect_events(PGN_PATH.read_text(encoding="utf-8"))[1]


def test_committed_pgn_exists_and_is_bullet():
    # The replayed input is the source of truth; it must carry the clocks that drive the
    # divergence (a clock-blind game would never diverge from the cp bar).
    assert PGN_PATH.exists(), f"committed PGN missing at {PGN_PATH}"
    text = PGN_PATH.read_text(encoding="utf-8")
    assert "[%clk" in text, "PGN must carry [%clk] tags for the clock-aware bar to warp"


def test_at_least_one_ply_fires_the_divergence_badge():
    # The whole point of the artifact: on this game the practical bar and the objective cp
    # disagree by >= threshold on at least one ply, so the human-edge badge actually fires.
    events = _events()
    fired = [d for d in divergences(events) if abs(d.gap) >= THRESHOLD]
    assert fired, "no ply diverged past the threshold — the wedge example would be empty"
    # The featured top-N must themselves all be real fires (not padded with non-divergent
    # plies), so the committed reel is wall-to-wall money shots.
    _, top = top_divergences(PGN_PATH.read_text(encoding="utf-8"))
    assert len(top) >= 1
    assert all(human_edge(d.event) is not None for d in top), "a featured ply doesn't fire"


def test_badge_matches_overlay_rule():
    # Mirror overlay/overlay.js: gap>0 => White's practical edge, gap<0 => Black's.
    events = _events()
    for d in divergences(events):
        edge = human_edge(d.event)
        if edge is None:
            assert abs(d.gap) < THRESHOLD
        else:
            assert abs(d.gap) >= THRESHOLD
            assert edge["side"] == ("white" if d.gap > 0 else "black")


def test_top_divergences_are_sorted_by_magnitude():
    # The reel features the LARGEST disagreements first (descending |gap|).
    _, top = top_divergences(PGN_PATH.read_text(encoding="utf-8"))
    gaps = [abs(d.gap) for d in top]
    assert gaps == sorted(gaps, reverse=True), f"not sorted by magnitude: {gaps}"
    assert len(top) <= TOP_N


def test_committed_json_artifact_matches_generator():
    # Drift guard: the committed JSON must be exactly what the generator produces, so it
    # can never silently diverge from the pipeline. If this fails, re-run
    # `python -m chess_equity.validate.divergence_reel` and commit the result.
    assert JSON_ARTIFACT_PATH.exists(), f"committed artifact missing at {JSON_ARTIFACT_PATH}"
    assert JSON_ARTIFACT_PATH.read_text(encoding="utf-8") == generate_json_artifact()


def test_committed_md_artifact_matches_generator():
    assert MD_ARTIFACT_PATH.exists(), f"committed artifact missing at {MD_ARTIFACT_PATH}"
    assert MD_ARTIFACT_PATH.read_text(encoding="utf-8") == generate_md_artifact()
