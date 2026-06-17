"""Tests for the game-level held-out split (task 0030)."""

import pytest

from chess_equity.data.schema import PositionRow
from chess_equity.validate.split import game_level_split


def _row(game_id, ply=1, result=1.0):
    """A minimal PositionRow; only game_id matters for the split."""
    return PositionRow(
        cp_eval=0.0,
        white_elo=1500,
        black_elo=1500,
        ply=ply,
        phase="opening",
        time_control="180+2",
        tc_bucket="blitz",
        clock_remaining=None,
        side_to_move="white",
        result=result,
        game_id=game_id,
    )


def _rows(n_games=10, plies_per_game=5):
    return [
        _row(f"g{g}", ply=p)
        for g in range(n_games)
        for p in range(plies_per_game)
    ]


def test_no_game_leaks_across_the_split():
    train, test = game_level_split(_rows(), test_fraction=0.3, seed=0)
    train_ids = {r.game_id for r in train}
    test_ids = {r.game_id for r in test}
    # The defining property: no game id appears on both sides.
    assert train_ids.isdisjoint(test_ids)
    # And nothing is dropped — every row lands on exactly one side.
    assert len(train) + len(test) == len(_rows())


def test_split_is_deterministic_for_a_seed():
    a_train, a_test = game_level_split(_rows(), test_fraction=0.3, seed=42)
    b_train, b_test = game_level_split(_rows(), test_fraction=0.3, seed=42)
    assert [r.game_id for r in a_test] == [r.game_id for r in b_test]
    assert [r.game_id for r in a_train] == [r.game_id for r in b_train]


def test_different_seeds_give_different_test_games():
    _, t0 = game_level_split(_rows(20), test_fraction=0.3, seed=0)
    _, t1 = game_level_split(_rows(20), test_fraction=0.3, seed=1)
    assert {r.game_id for r in t0} != {r.game_id for r in t1}


def test_test_fraction_picks_roughly_that_share_of_games():
    train, test = game_level_split(_rows(10, plies_per_game=3), test_fraction=0.2, seed=0)
    # 10 games * 0.2 = 2 test games -> 2*3 = 6 test rows, 8*3 = 24 train rows.
    assert len({r.game_id for r in test}) == 2
    assert len({r.game_id for r in train}) == 8


def test_small_split_keeps_both_sides_nonempty():
    # 2 games at fraction 0.2 rounds to 0 test games; we force at least one each.
    train, test = game_level_split(_rows(2, plies_per_game=4), test_fraction=0.2, seed=0)
    assert len({r.game_id for r in test}) == 1
    assert len({r.game_id for r in train}) == 1


def test_missing_game_id_raises():
    rows = _rows(3) + [_row(None)]
    with pytest.raises(ValueError, match="game_id"):
        game_level_split(rows, test_fraction=0.3, seed=0)


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
def test_invalid_fraction_raises(frac):
    with pytest.raises(ValueError, match="test_fraction"):
        game_level_split(_rows(), test_fraction=frac, seed=0)
