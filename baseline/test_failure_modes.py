#!/usr/bin/env python3
"""Tests for the 0003 failure-mode evidence set.

Runs with pytest *or* plain `python3 test_failure_modes.py` (stdlib only). Guards
the curated dataset (schema, FEN sanity) and asserts that the rating-blind
baseline actually mischaracterises each position — i.e. the 'before picture' holds.
"""
import os

from fen_lint import validate
from report import baseline_white_pct, load_positions, practical_field

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS = load_positions(os.path.join(HERE, "failure_modes.json"))

CATEGORIES = {"dead-draw-hard", "absurd-refutation"}


def test_has_at_least_six_positions():
    assert len(POSITIONS) >= 6, "acceptance criterion: >=6 annotated FENs"


def test_both_failure_modes_represented():
    seen = {p["category"] for p in POSITIONS}
    assert CATEGORIES <= seen, f"missing categories: {CATEGORIES - seen}"
    for cat in CATEGORIES:
        assert sum(1 for p in POSITIONS if p["category"] == cat) >= 2


def test_every_position_is_well_formed():
    for p in POSITIONS:
        for field in ("id", "category", "fen", "name", "engine_cp", "why_baseline_misleads"):
            assert p.get(field) not in (None, ""), f"{p.get('id')} missing {field}"
        assert p["category"] in CATEGORIES
        validate(p["fen"])  # raises on a malformed FEN
        practical, band = practical_field(p)
        assert practical is not None, f"{p['id']} needs a hypothesised practical score"
        assert 0.0 <= practical <= 1.0
        assert band


def test_baseline_demonstrably_misleads():
    """The whole point: baseline output diverges from the practical hypothesis.

    - absurd-refutation: the engine's eval inflates a side the move-finder won't
      realise, so the baseline must be far (>=20 pts) from the practical hypothesis.
    - dead-draw-hard: the engine reads ~0.00 -> baseline ~50%, yet the practical
      hypothesis is asymmetric for at least some positions (the metric is blind to
      who is playing). We require the SET to show that asymmetry, not every row
      (some draws are also practically ~50%).
    """
    asymmetric_draws = 0
    for p in POSITIONS:
        base = baseline_white_pct(p)
        practical_pct = practical_field(p)[0] * 100.0
        gap = abs(base - practical_pct)
        if p["category"] == "absurd-refutation":
            assert gap >= 20.0, f"{p['id']}: expected a big baseline gap, got {gap:.1f}"
        if p["category"] == "dead-draw-hard":
            assert abs(base - 50.0) < 1e-6, f"{p['id']}: drawn position should baseline to 50%"
            if gap >= 10.0:
                asymmetric_draws += 1
    assert asymmetric_draws >= 1, "at least one 'dead 0.00' position must skew in practice"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except AssertionError as exc:
            failures += 1
            print("FAIL", t.__name__, "-", exc)
    raise SystemExit(1 if failures else 0)
