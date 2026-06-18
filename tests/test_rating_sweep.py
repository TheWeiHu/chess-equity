"""The rating-sweep proof artifact: the bar moves with rating, and stays in sync (task 0137).

Two halves, mirroring the ground-truth fixture pattern (``test_gate_ground_truth``):

* the sweep on the committed ``wdl-a`` artifact is strictly monotone in the expected
  direction (a winning White position gets *more* decisive as both players strengthen),
  with a spread big enough that it can't be flat — i.e. the bar IS rating-conditioned;
* a rating-blind ``baseline`` sweep on the same position is flat (the control); and
* ``reports/rating_sweep.md`` is byte-for-byte what the generator produces (drift guard).
"""

from __future__ import annotations

import pytest

from chess_equity.cli import build_model
from chess_equity.validate.rating_sweep import (
    ARTIFACT_PATH,
    DEMO_FEN,
    DEMO_RUNGS,
    generate_artifact,
    parse_rungs,
    sweep,
)
from chess_equity.wdl_regression import build_wdl_a_equity


def test_parse_rungs_roundtrips_and_rejects_junk():
    assert parse_rungs("1000,1500, 2000 ,2500") == [1000, 1500, 2000, 2500]
    assert parse_rungs("1500") == [1500]
    with pytest.raises(ValueError):
        parse_rungs("")
    with pytest.raises(ValueError):
        parse_rungs("1000,abc")
    with pytest.raises(ValueError):
        parse_rungs("1000,-5")


def test_sweep_is_strictly_monotone_and_not_flat():
    # The whole point: same position, same cp input, only rating changes — and the
    # rating-conditioned bar climbs. For a crushing White advantage, stronger players
    # convert it, so equity must rise strictly down the ladder.
    rungs = sweep(build_wdl_a_equity(), DEMO_FEN, DEMO_RUNGS)
    equities = [r.equity_white for r in rungs]
    assert equities == sorted(equities), f"sweep not increasing: {equities}"
    assert all(b > a for a, b in zip(equities, equities[1:])), f"not strict: {equities}"
    # Big enough that it visibly moves the bar — not numerical jitter around a flat line.
    assert equities[-1] - equities[0] > 5.0, f"spread too small to be a proof: {equities}"


def test_rating_blind_baseline_is_flat_control():
    # The negative control: a rating-blind bar prints one number regardless of rating.
    # If this ever starts moving, "rating-conditioned" has lost its meaning.
    rungs = sweep(build_model("baseline"), DEMO_FEN, DEMO_RUNGS)
    equities = [r.equity_white for r in rungs]
    assert max(equities) - min(equities) < 1e-9, f"baseline should be flat: {equities}"


def test_committed_artifact_matches_generator():
    # Drift guard: the committed report must be exactly what the generator produces, so it
    # can never silently diverge from the model it claims to demonstrate. If this fails,
    # re-run `python -m chess_equity.validate.rating_sweep` and commit the result.
    assert ARTIFACT_PATH.exists(), f"committed artifact missing at {ARTIFACT_PATH}"
    assert ARTIFACT_PATH.read_text(encoding="utf-8") == generate_artifact()
