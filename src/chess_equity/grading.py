"""Move grading by Δequity — the "moves can be good" reframe (task 0008).

The classic centipawn-loss grade can only ever be ``<= 0``: the best you can do is
match perfect play, and any deviation loses centipawns. This module grades a move by
the change in the *mover's equity*, benchmarked against what a player of the mover's
rating was expected to do — so a move stronger than the rating-typical mix scores
**positive**. That is the whole pitch (see :doc:`concept-equity-bar`).

For a played move ``m`` in position ``p`` (mover rated ``r`` vs opponent ``opp``):

- ``equity_after``   = equity of ``p·m`` from the mover's POV.
- ``expected_equity``= ``Σ_move P(move | r) · equity(p·move)`` — the equity of the
  *rating-typical* move mix (``P`` from a :class:`~chess_equity.adapters.HumanPolicy`,
  i.e. Maia in task 0005; a uniform placeholder until then).
- ``equity_best``    = equity of the equity-maximizing legal move.

and the two grades:

- **``grade_peer = equity_after − expected_equity``** — the headline. Positive ⇒ you
  beat your rating peers (a genuinely good move).
- ``grade_best = equity_after − equity_best`` (``<= 0``) — the classic "how much did
  you leave on the table" view, on the equity scale.

A move can *lose centipawns yet gain equity* — a sound trap a rating-peer opponent is
likely to walk into. ``cp_loss`` is reported alongside so that case is visible.

Everything here depends only on the :class:`~chess_equity.adapters.EquityModel` /
:class:`~chess_equity.adapters.HumanPolicy` contracts, so the real Maia models drop
in unchanged. With the placeholder material model the *machinery* is exercised; the
flagship trap demo needs Maia (0005) on real data — but the synthetic test
``test_grading`` shows the machinery surfaces the cp-loss-but-equity-gain case.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import chess
import chess.pgn

from chess_equity.adapters import EquityModel, HumanPolicy

# Headline Δequity grade bands, in equity percentage points, peer-relative (mover POV).
# Positive = you beat the rating-typical mix. These are the *base* bands at ~2000; they
# widen for lower ratings, where the peer move-mix is noisier (see :func:`scaled_bands`).
BASE_BANDS = [
    (10.0, "brilliant"),
    (3.0, "good"),
    (-3.0, "ok"),
    (-8.0, "inaccuracy"),
    (-15.0, "mistake"),
]


def scaled_bands(elo: int) -> List[tuple]:
    """Rating-aware grade bands: wider tolerance at lower ratings.

    A 1200's rating-typical mix is noisier than a 2400's, so the same Δequity means
    less. We widen the bands below ~2000 (scale 1.0 at/above 2000, up to ~1.6 at 800)
    and leave strong play on the tight base bands. Calibration against real Maia
    spreads (task 0005 / validation 0009) can replace this heuristic.
    """
    scale = max(1.0, 1.0 + (2000 - elo) / 2000.0)
    return [(threshold * scale, label) for threshold, label in BASE_BANDS]


def grade_label(grade_peer: float, elo: int) -> str:
    """Label a peer-relative Δequity for a mover rated ``elo``."""
    for threshold, label in scaled_bands(elo):
        if grade_peer >= threshold:
            return label
    return "blunder"


class UniformPolicy(HumanPolicy):
    """Placeholder peer model: every legal move equally likely.

    Stands in for Maia-2 (task 0005) so grading runs end-to-end. With a uniform peer
    mix, ``expected_equity`` is the *average* legal-move equity, so any above-average
    move grades positive — enough to exercise and demonstrate the reframe. Maia
    replaces this behind the same :class:`HumanPolicy` interface with no other change.
    """

    def move_probs(self, fen: str, elo: int) -> Dict[str, float]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        if not moves:
            return {}
        p = 1.0 / len(moves)
        return {m.uci(): p for m in moves}


@dataclass(frozen=True)
class MoveGrade:
    """The grade of one played move, on the equity scale (mover POV)."""

    ply: int
    san: str
    uci: str
    mover_white: bool
    mover_elo: int
    equity_after: float
    expected_equity: float
    equity_best: float
    grade_peer: float  # headline: equity_after - expected_equity (positive = beat peers)
    grade_best: float  # equity_after - equity_best (<= 0)
    label: str
    best_uci: str
    cp_loss: Optional[float]  # classic centipawn loss (mover POV, >= 0), if available

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class EquityGrader:
    """Grades moves by Δequity using an :class:`EquityModel` + a peer policy."""

    def __init__(self, model: EquityModel, policy: Optional[HumanPolicy] = None) -> None:
        self.model = model
        self.policy = policy or UniformPolicy()

    def _mover_equity_after(
        self, board: chess.Board, move: chess.Move, mover_white: bool,
        white_elo: int, black_elo: int,
    ) -> tuple:
        """(equity, cp) of the position after ``move``, from the mover's POV."""
        board.push(move)
        try:
            eq = self.model.evaluate(board.fen(), white_elo, black_elo)
        finally:
            board.pop()
        equity = eq.equity_white if mover_white else 100.0 - eq.equity_white
        # eq.cp is side-to-move POV; after the move the opponent is to move, so negate
        # to express the position from the mover's POV (classic centipawn convention).
        cp = None if eq.cp is None else -eq.cp
        return equity, cp

    def grade_move(
        self, fen: str, move: chess.Move, white_elo: int, black_elo: int, *, ply: int = 0
    ) -> MoveGrade:
        """Grade a single ``move`` played in ``fen``."""
        board = chess.Board(fen)
        if move not in board.legal_moves:
            raise ValueError(f"{move.uci()} is not legal in {fen}")
        mover_white = board.turn == chess.WHITE
        mover_elo = white_elo if mover_white else black_elo
        san = board.san(move)

        # Equity (mover POV) and mover-POV cp of every legal move.
        equities: Dict[str, float] = {}
        cps: Dict[str, Optional[float]] = {}
        for legal in board.legal_moves:
            eq, cp = self._mover_equity_after(
                board, legal, mover_white, white_elo, black_elo
            )
            equities[legal.uci()] = eq
            cps[legal.uci()] = cp

        played = move.uci()
        equity_after = equities[played]
        best_uci = max(equities, key=lambda u: equities[u])
        equity_best = equities[best_uci]

        # Expected equity over the rating-typical move mix; renormalize the policy onto
        # the legal moves we actually evaluated (a policy may omit zero-prob moves).
        probs = self.policy.move_probs(fen, mover_elo)
        mass = sum(probs.get(u, 0.0) for u in equities)
        if mass > 0:
            expected_equity = sum(
                probs.get(u, 0.0) * equities[u] for u in equities
            ) / mass
        else:
            expected_equity = sum(equities.values()) / len(equities)

        # Classic centipawn loss (mover POV), for the cp-vs-equity contrast.
        cp_loss = None
        if all(v is not None for v in cps.values()):
            cp_best = max(cps.values())  # type: ignore[type-var]
            cp_loss = float(cp_best - cps[played])  # type: ignore[operator]

        grade_peer = equity_after - expected_equity
        return MoveGrade(
            ply=ply,
            san=san,
            uci=played,
            mover_white=mover_white,
            mover_elo=mover_elo,
            equity_after=equity_after,
            expected_equity=expected_equity,
            equity_best=equity_best,
            grade_peer=grade_peer,
            grade_best=equity_after - equity_best,
            label=grade_label(grade_peer, mover_elo),
            best_uci=best_uci,
            cp_loss=cp_loss,
        )

    def grade_game(self, game: chess.pgn.Game, white_elo: int, black_elo: int) -> List[MoveGrade]:
        """Grade every move of a parsed PGN game in order."""
        board = game.board()
        grades: List[MoveGrade] = []
        for ply, move in enumerate(game.mainline_moves(), start=1):
            grades.append(self.grade_move(board.fen(), move, white_elo, black_elo, ply=ply))
            board.push(move)
        return grades
