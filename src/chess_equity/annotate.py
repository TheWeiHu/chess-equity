"""Equity-annotated PGN export — load the bar into any chess GUI / broadcast tool (task 0197).

``grade`` computes a per-move equity + Δpeer grade but only prints them. This module
writes those grades back into a *standard* PGN as move comments, in the de-facto
Lichess/SCID ``[%key value]`` format, so SCID, a Lichess study import, or a broadcast
overlay can read the equity bar without knowing anything about this project.

Per move we emit, appended to whatever comment the move already carried (so existing
``[%eval]``/``[%clk]`` tags survive):

- ``[%equity 0.63]`` — **White-POV** win-equity in ``[0, 1]`` (``P(win)+0.5·P(draw)``,
  rendered from White's POV so the bar is stable as turns alternate, exactly like the
  ``eval`` bar — see :doc:`concept-equity-bar`).
- ``[%grade good]`` — the peer-relative grade label.
- a standard NAG (``$1``/``$2``/``$3``/``$4``/``$6``) so GUIs that render ``!``/``?``
  glyphs show the grade without parsing the comment.

Everything routes through :class:`~chess_equity.grading.EquityGrader`, so it works with
``--model baseline`` (no torch) and drops in the real Maia models unchanged.
"""

from __future__ import annotations

import chess
import chess.pgn

from chess_equity.adapters import EquityModel
from chess_equity.grading import EquityGrader, MoveGrade

# Grade label -> python-chess NAG. "ok" gets no glyph (an unremarkable move).
_LABEL_NAG = {
    "brilliant": chess.pgn.NAG_BRILLIANT_MOVE,  # $3  !!
    "good": chess.pgn.NAG_GOOD_MOVE,            # $1  !
    "inaccuracy": chess.pgn.NAG_DUBIOUS_MOVE,   # $6  ?!
    "mistake": chess.pgn.NAG_MISTAKE,           # $2  ?
    "blunder": chess.pgn.NAG_BLUNDER,           # $4  ??
}


def white_pov_equity(grade: MoveGrade) -> float:
    """White-POV win-equity in ``[0, 1]`` for the position after ``grade``'s move.

    ``MoveGrade.equity_after`` is the *mover's* equity in ``[0, 100]``; flip it for
    Black moves so the annotation is always from White's POV (matching the ``eval`` bar).
    """
    white = grade.equity_after if grade.mover_white else 100.0 - grade.equity_after
    return white / 100.0


def _equity_comment(grade: MoveGrade) -> str:
    """The ``[%equity …] [%grade …]`` tags for one move (no surrounding braces)."""
    return f"[%equity {white_pov_equity(grade):.2f}] [%grade {grade.label}]"


def annotate_game(
    game: chess.pgn.Game,
    model: EquityModel,
    white_elo: int,
    black_elo: int,
) -> chess.pgn.Game:
    """Annotate every mainline move of ``game`` in place with its equity + grade, and return it.

    Existing comments (e.g. ``[%eval]``/``[%clk]``) are preserved — the new tags are
    appended with a single space. Returns the same ``game`` object for convenience.
    """
    grader = EquityGrader(model)
    board = game.board()
    node: chess.pgn.GameNode = game
    for ply, move in enumerate(game.mainline_moves(), start=1):
        grade = grader.grade_move(board.fen(), move, white_elo, black_elo, ply=ply)
        node = node.next()  # the node reached by playing `move`
        tags = _equity_comment(grade)
        existing = node.comment.strip()
        node.comment = f"{existing} {tags}".strip() if existing else tags
        nag = _LABEL_NAG.get(grade.label)
        if nag is not None:
            node.nags.add(nag)
        board.push(move)
    return game


def annotate_pgn_file(
    in_path: str,
    out_path: str,
    model: EquityModel,
    white_elo: int,
    black_elo: int,
) -> int:
    """Read the first game of ``in_path``, annotate it, write it to ``out_path``.

    Returns the number of mainline moves annotated.
    """
    with open(in_path, encoding="utf-8") as fh:
        game = chess.pgn.read_game(fh)
    if game is None:
        raise ValueError(f"no game found in {in_path}")
    annotate_game(game, model, white_elo, black_elo)
    n_moves = sum(1 for _ in game.mainline_moves())
    with open(out_path, "w", encoding="utf-8") as fh:
        exporter = chess.pgn.FileExporter(fh)
        game.accept(exporter)
    return n_moves
