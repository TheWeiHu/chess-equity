"""Command-line entry point: ``chess-equity``.

Modes:

    chess-equity eval "<fen>" --white-elo 1500 --black-elo 1500
    chess-equity eval --pgn game.pgn --white-elo 1500 --black-elo 1500
    chess-equity grade --pgn game.pgn --white-elo 1500 --black-elo 1500

The CLI depends only on :class:`~chess_equity.adapters.EquityModel`; it constructs
the placeholder :class:`~chess_equity.models.LichessBaselineModel` today, but a new
model would drop in with no other changes here.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import chess
import chess.pgn

from chess_equity.adapters import EquityModel
from chess_equity.bar import render_eval
from chess_equity.grading import EquityGrader
from chess_equity.models import LichessBaselineModel

START_FEN = chess.STARTING_FEN


def _eval_fen(model: EquityModel, fen: str, white_elo: int, black_elo: int) -> str:
    equity = model.evaluate(fen, white_elo, black_elo)
    return render_eval(equity)


def _eval_pgn(model: EquityModel, path: str, white_elo: int, black_elo: int) -> List[str]:
    """Annotate every position in the first game of a PGN with its equity bar."""
    with open(path, encoding="utf-8") as fh:
        game = chess.pgn.read_game(fh)
    if game is None:
        raise ValueError(f"no game found in {path}")
    board = game.board()
    lines = [f"start  {_eval_fen(model, board.fen(), white_elo, black_elo)}"]
    for move in game.mainline_moves():
        san = board.san(move)
        board.push(move)
        lines.append(f"{san:7s}{_eval_fen(model, board.fen(), white_elo, black_elo)}")
    return lines


def build_model() -> EquityModel:
    """The model the CLI uses. Swap this single line to change models."""
    return LichessBaselineModel()


def _grade_pgn(model: EquityModel, path: str, white_elo: int, black_elo: int) -> List[str]:
    """Annotate every move of a PGN with its peer-relative Δequity grade."""
    with open(path, encoding="utf-8") as fh:
        game = chess.pgn.read_game(fh)
    if game is None:
        raise ValueError(f"no game found in {path}")
    grader = EquityGrader(model)
    lines = []
    for g in grader.grade_game(game, white_elo, black_elo):
        cp = "" if g.cp_loss is None else f"  cp_loss {g.cp_loss:+.0f}"
        # +Δ vs peers is the headline; Δ vs best is the classic "left on the table".
        lines.append(
            f"{g.ply:3d}. {g.san:7s} {g.label:11s} "
            f"Δpeer {g.grade_peer:+5.1f}  Δbest {g.grade_best:+5.1f}{cp}"
        )
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="chess-equity", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="evaluate a position or a whole game")
    ev.add_argument("fen", nargs="?", default=START_FEN, help="FEN (default: startpos)")
    ev.add_argument("--pgn", help="annotate every move of a PGN file instead")
    ev.add_argument("--white-elo", type=int, default=1500)
    ev.add_argument("--black-elo", type=int, default=1500)

    gr = sub.add_parser("grade", help="grade every move of a PGN by Δequity vs rating peers")
    gr.add_argument("--pgn", required=True, help="PGN file to grade")
    gr.add_argument("--white-elo", type=int, default=1500)
    gr.add_argument("--black-elo", type=int, default=1500)

    args = parser.parse_args(argv)
    model = build_model()

    if args.command == "eval":
        try:
            if args.pgn:
                for line in _eval_pgn(model, args.pgn, args.white_elo, args.black_elo):
                    print(line)
            else:
                print(_eval_fen(model, args.fen, args.white_elo, args.black_elo))
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "grade":
        try:
            for line in _grade_pgn(model, args.pgn, args.white_elo, args.black_elo):
                print(line)
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
