"""Single-game equity scorecard — the CLI-first way to interrogate the thesis.

The whole project is one claim: a *rating-conditioned* predictor beats the
*rating-blind* centipawn bar at predicting **real** human outcomes. The statistical
proof of that lives in :mod:`chess_equity.validate` (log-loss/Brier/ECE over thousands
of positions, with a bootstrap gate). This module is the other half — the *fast
iteration loop*: feed it one game and it answers, on the command line, the four
questions you actually ask of a single game:

1. **Here is a game** — the PGN's moves, players, and ratings.
2. **Here is the score** — the objective centipawn eval embedded in the PGN
   (``[%eval …]``), the rating-blind number a normal engine bar would show.
3. **What's the real score?** — the actual result (``1-0``/``0-1``/``1/2-1/2``),
   i.e. the realized White score in ``{1, 0.5, 0}``.
4. **What are we predicting?** — ``equity_white`` = P(White wins) + ½·P(draw) from a
   chosen rating-conditioned model, scored against the realized outcome.

A single game is **illustrative, never proof** — the per-game Brier here is a tracking
number, not a powered statistic. The headline gate stays ``chess-equity validate`` /
``headline`` on a real Lichess dump. This module exists so you can *look at one game*
without opening the browser demo.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional

import chess
import chess.pgn

from chess_equity.adapters import EquityModel
from chess_equity.types import lichess_win_percent

# Realized White score for each PGN result token; ``*`` (unfinished) -> unknown.
_RESULT_TO_WHITE_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


@dataclass(frozen=True)
class MoveScore:
    """One ply of the scorecard: the position after ``san`` was played.

    ``cp_white`` is the embedded objective eval from White's POV (None when the PGN
    carries no ``[%eval]`` for this ply). ``equity_white`` is the chosen model's
    rating-conditioned prediction in [0, 100]; ``baseline_white`` is the rating-blind
    Lichess Win% derived from ``cp_white`` (the number to beat), or None when there's
    no embedded eval to derive it from.
    """

    ply: int
    san: str
    fen: str
    cp_white: Optional[float]
    equity_white: float
    baseline_white: Optional[float]


@dataclass(frozen=True)
class Scorecard:
    """A whole game scored move-by-move: predictions vs the realized outcome."""

    white: str
    black: str
    white_elo: int
    black_elo: int
    result: str
    white_score: Optional[float]  # the "real score": 1.0 / 0.5 / 0.0, or None if "*"
    model: str
    moves: List[MoveScore]

    @property
    def equity_brier(self) -> Optional[float]:
        """Mean squared error of the model's equity vs the realized White score.

        Averaged over every ply (the prediction at each position is judged against the
        *final* result — early, genuinely-uncertain positions inflate this, which is why
        a single game's number is illustrative, not proof). None if the game is
        unfinished.
        """
        return self._brier([m.equity_white for m in self.moves])

    @property
    def baseline_brier(self) -> Optional[float]:
        """Same Brier, for the rating-blind centipawn baseline (Lichess Win%).

        Only plies that carry an embedded ``[%eval]`` contribute, so this is comparable
        to :attr:`equity_brier` only over that subset; None if no ply has an eval or the
        game is unfinished.
        """
        return self._brier([m.baseline_white for m in self.moves])

    def _brier(self, preds: List[Optional[float]]) -> Optional[float]:
        if self.white_score is None:
            return None
        errs = [
            (p / 100.0 - self.white_score) ** 2 for p in preds if p is not None
        ]
        if not errs:
            return None
        return sum(errs) / len(errs)


def _white_cp(node: chess.pgn.ChildNode) -> Optional[float]:
    """White-POV centipawns from a node's embedded ``[%eval]`` (mate -> ±10000)."""
    score = node.eval()
    if score is None:
        return None
    return float(score.white().score(mate_score=10000))


def build_scorecard(
    game: chess.pgn.Game,
    model: EquityModel,
    *,
    model_name: str = "model",
    white_elo: Optional[int] = None,
    black_elo: Optional[int] = None,
) -> Scorecard:
    """Walk one game and score every position with ``model`` vs the real result.

    Ratings come from the explicit ``white_elo``/``black_elo`` overrides, else the PGN's
    ``WhiteElo``/``BlackElo`` headers, else 1500 — so ``score --pgn game.pgn`` works with
    zero flags on a real Lichess game.
    """
    headers = game.headers
    we = white_elo if white_elo is not None else _header_elo(headers, "WhiteElo")
    be = black_elo if black_elo is not None else _header_elo(headers, "BlackElo")
    result = headers.get("Result", "*")

    moves: List[MoveScore] = []
    board = game.board()
    for ply, node in enumerate(game.mainline(), start=1):
        san = board.san(node.move)
        board.push(node.move)
        fen = board.fen()
        cp_white = _white_cp(node)
        equity = model.evaluate(fen, we, be)
        baseline = lichess_win_percent(cp_white) if cp_white is not None else None
        moves.append(
            MoveScore(
                ply=ply,
                san=san,
                fen=fen,
                cp_white=cp_white,
                equity_white=equity.equity_white,
                baseline_white=baseline,
            )
        )

    return Scorecard(
        white=headers.get("White", "?"),
        black=headers.get("Black", "?"),
        white_elo=we,
        black_elo=be,
        result=result,
        white_score=_RESULT_TO_WHITE_SCORE.get(result),
        model=model_name,
        moves=moves,
    )


def build_scorecard_from_pgn(
    pgn_text: str,
    model: EquityModel,
    *,
    model_name: str = "model",
    white_elo: Optional[int] = None,
    black_elo: Optional[int] = None,
) -> Scorecard:
    """:func:`build_scorecard` over the first game in a PGN string."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("no game found in PGN")
    return build_scorecard(
        game,
        model,
        model_name=model_name,
        white_elo=white_elo,
        black_elo=black_elo,
    )


def _header_elo(headers: chess.pgn.Headers, key: str) -> int:
    raw = headers.get(key, "")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 1500


def render_scorecard(card: Scorecard) -> List[str]:
    """Render a :class:`Scorecard` as the CLI's human-readable block of lines."""
    lines: List[str] = []
    lines.append(f"# {card.white} ({card.white_elo}) vs {card.black} ({card.black_elo})")
    lines.append(f"# predicting: P(White wins) + 0.5*P(draw), White POV, via {card.model}")
    lines.append("")

    # Per-move table: the score (cp), the rating-blind baseline, and our prediction.
    lines.append(f"{'ply':>4} {'move':<7} {'cp':>7} {'win%':>6} {'equity':>7}")
    lines.append(f"{'':>4} {'':<7} {'(score)':>7} {'(blind)':>6} {'(pred)':>7}")
    for m in card.moves:
        cp = "   --  " if m.cp_white is None else f"{m.cp_white / 100:+6.2f} "
        blind = "   -- " if m.baseline_white is None else f"{m.baseline_white:5.1f} "
        lines.append(
            f"{m.ply:>4} {m.san:<7} {cp:>7} {blind:>6} {m.equity_white:6.1f}%"
        )

    lines.append("")
    real = "unfinished" if card.white_score is None else f"{card.result}  (White scored {card.white_score})"
    lines.append(f"# real score: {real}")

    eb, bb = card.equity_brier, card.baseline_brier
    if eb is not None:
        lines.append(f"# Brier (lower=better, illustrative single-game tracking only):")
        lines.append(f"#   equity   {eb:.4f}")
        if bb is not None:
            verdict = "equity tracks the result better" if eb < bb else (
                "centipawns track the result better" if eb > bb else "tie"
            )
            lines.append(f"#   baseline {bb:.4f}   -> {verdict} on THIS game")
        lines.append(
            "# NB: one game is illustrative, not proof — the powered gate is "
            "`chess-equity validate`/`headline` on a real dump."
        )
    return lines
