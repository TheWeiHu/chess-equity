"""Game-level held-out split for the validation gate (task 0030).

Positions from a single game are *not* independent — consecutive plies share the
same opening, players, ratings and (largely) the same evaluation. Splitting rows at
random would leak almost-identical positions across the train/test boundary and make
any predictor look better calibrated than it is. :func:`game_level_split` instead
partitions **whole games**: every position from a given game lands entirely in train
or entirely in test, so the test set measures true held-out generalisation.

Pure and dependency-free (stdlib ``random`` only), so it lives in the light test path
beside :mod:`chess_equity.validate.metrics`.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from chess_equity.data.schema import PositionRow


def game_level_split(
    rows: Sequence[PositionRow],
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Tuple[List[PositionRow], List[PositionRow]]:
    """Partition ``rows`` into ``(train, test)`` at the game level.

    Distinct ``game_id``s are shuffled deterministically by ``seed`` and the first
    ``test_fraction`` of them become the test games; *all* of a game's positions go to
    the same side, so no game spans the split. Returns row lists in their original
    order within each side.

    ``test_fraction`` is a fraction of *games*, not rows (the unit that must not leak).
    With very few games it is rounded so at least one game lands on each side whenever
    there are at least two games to split.

    Raises ``ValueError`` if ``test_fraction`` is not in (0, 1), or if any row has no
    ``game_id`` (a dataset built before the column existed) — the gap surfaces loudly
    rather than silently leaking, mirroring ``harness.model_predictor``'s FEN check.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    rows = list(rows)
    missing = sum(1 for r in rows if r.game_id is None)
    if missing:
        raise ValueError(
            f"game_level_split needs every row to have a game_id; {missing} row(s) "
            "have none — rebuild the dataset so the game_id column is populated"
        )

    # Distinct game ids in first-seen order, then shuffled deterministically so the
    # split is reproducible across runs and machines for a given seed.
    game_ids = list(dict.fromkeys(r.game_id for r in rows))
    random.Random(seed).shuffle(game_ids)

    n_test = int(round(len(game_ids) * test_fraction))
    # With >=2 games, never let a side be empty (rounding can zero out small splits).
    if len(game_ids) >= 2:
        n_test = max(1, min(n_test, len(game_ids) - 1))

    test_ids = set(game_ids[:n_test])
    train = [r for r in rows if r.game_id not in test_ids]
    test = [r for r in rows if r.game_id in test_ids]
    return train, test
