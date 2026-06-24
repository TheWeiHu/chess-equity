"""Out-of-distribution high-rating flag (task 0255).

Maia-2's top rating embedding is a single coarse ``">2000"`` bucket — it can't tell a
2200 from a 2800 (see the product-wedge-streaming gap #1). When BOTH players clear that
bucket the equity bar is a coarse-bucket read, so the overlay marks it lower-confidence.
This pins the pure boundary logic (``is_rating_ood``) and that the overlay event carries
the ``rating_ood`` boolean the overlay reads.
"""
from chess_equity.broadcast import (
    RATING_OOD_THRESHOLD,
    GameTracker,
    is_rating_ood,
)
from chess_equity.models import LichessBaselineModel

T = RATING_OOD_THRESHOLD  # 2000


def test_both_above_threshold_is_ood():
    assert is_rating_ood(2200, 2800) is True
    assert is_rating_ood(T + 1, T + 1) is True


def test_boundary_is_strict_in_distribution():
    # Exactly at the threshold is still IN distribution (strict ``>``), on either side.
    assert is_rating_ood(T, T) is False
    assert is_rating_ood(T, 2800) is False
    assert is_rating_ood(2800, T) is False


def test_one_side_below_is_in_distribution():
    # The bar can resolve the matchup as soon as one side is inside the fine-grained bins.
    assert is_rating_ood(2500, 1800) is False
    assert is_rating_ood(1800, 2500) is False
    assert is_rating_ood(1500, 1500) is False


def test_unknown_rating_is_not_ood():
    # OTB / anonymous feeds drop a rating; we only flag when BOTH are confirmed over bucket.
    assert is_rating_ood(None, 2800) is False
    assert is_rating_ood(2800, None) is False
    assert is_rating_ood(None, None) is False


def test_overlay_event_carries_rating_ood_boolean():
    # Two >2000 players: every position event is flagged ood and the value is a real bool.
    pgn = (
        '[Event "OOD Test"]\n[White "A"]\n[Black "B"]\n'
        '[WhiteElo "2700"]\n[BlackElo "2650"]\n[Result "*"]\n\n'
        "1. e4 e5 2. Nf3 *\n"
    )
    tracker = GameTracker(
        "ood-test", LichessBaselineModel(), white_elo=2700, black_elo=2650
    )
    events = [e.to_overlay_event() for e in tracker.ingest(pgn)]
    assert events, "fixture must yield at least one move event"
    for evt in events:
        assert evt["rating_ood"] is True


def test_overlay_event_clears_rating_ood_for_amateurs():
    pgn = (
        '[Event "InDist Test"]\n[White "A"]\n[Black "B"]\n'
        '[WhiteElo "1600"]\n[BlackElo "1550"]\n[Result "*"]\n\n'
        "1. e4 e5 2. Nf3 *\n"
    )
    tracker = GameTracker(
        "indist-test", LichessBaselineModel(), white_elo=1600, black_elo=1550
    )
    events = [e.to_overlay_event() for e in tracker.ingest(pgn)]
    assert events
    for evt in events:
        assert evt["rating_ood"] is False
