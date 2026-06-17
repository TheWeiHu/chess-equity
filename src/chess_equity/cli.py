"""Command-line entry point: ``chess-equity``.

Commands:

    chess-equity eval "<fen>" --white-elo 1500 --black-elo 1500
    chess-equity eval --pgn game.pgn --white-elo 1500 --black-elo 1500
    chess-equity grade --pgn game.pgn --white-elo 1500 --black-elo 1500
    chess-equity data build --pgn dump.pgn.zst --sample 50000 --out data/
    chess-equity validate --data data/dataset.csv --models baseline

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


def build_model() -> EquityModel:
    """The model the CLI uses. Swap this single line to change models."""
    return LichessBaselineModel()


def _run_eval(args: argparse.Namespace) -> int:
    model = build_model()
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


def _run_grade(args: argparse.Namespace) -> int:
    model = build_model()
    try:
        for line in _grade_pgn(model, args.pgn, args.white_elo, args.black_elo):
            print(line)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_data(args: argparse.Namespace) -> int:
    # Imported lazily so the common ``eval`` path never pays for the data deps.
    from chess_equity.data.build import build_dataset, month_url

    pgn = args.pgn
    if pgn is None:
        if args.month is None:
            print("error: provide --pgn <file> (or --month with a downloaded dump)", file=sys.stderr)
            return 1
        print(
            f"error: --month is a convenience; download {month_url(args.month)} first, "
            "then pass it via --pgn",
            file=sys.stderr,
        )
        return 1
    try:
        out = build_dataset(pgn, args.out, sample=args.sample, fmt=args.format)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    # Lazy import: keeps the eval path free of the data loader.
    from chess_equity.data.build import load_rows
    from chess_equity.validate.harness import PREDICTORS, evaluate, format_report

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in requested if m not in PREDICTORS]
    if unknown:
        print(
            f"error: unknown model(s) {unknown}; available: {sorted(PREDICTORS)}",
            file=sys.stderr,
        )
        return 1
    try:
        rows = load_rows(args.data)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not rows:
        print(f"error: no rows in {args.data}", file=sys.stderr)
        return 1

    predictors = {name: PREDICTORS[name] for name in requested}
    reports = evaluate(rows, predictors)
    report = format_report(reports, title=f"Validation report — {args.data}")
    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


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

    data = sub.add_parser("data", help="build / manage the training+validation dataset")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    build = data_sub.add_parser("build", help="parse a Lichess PGN dump into a dataset")
    build.add_argument("--pgn", help="path to a PGN file (plain or .zst)")
    build.add_argument("--month", help="YYYY-MM Lichess month (prints the dump URL to fetch)")
    build.add_argument("--sample", type=int, default=None, help="cap the number of rows")
    build.add_argument("--out", default="data", help="output directory (default: data/)")
    build.add_argument("--format", choices=("csv", "parquet"), default="csv")

    val = sub.add_parser("validate", help="score predictors against real outcomes (task 0009)")
    val.add_argument("--data", required=True, help="path to a built dataset (csv/parquet)")
    val.add_argument("--models", default="baseline", help="comma-separated predictor names")
    val.add_argument("--out", help="write the Markdown report here (default: stdout)")

    args = parser.parse_args(argv)

    if args.command == "eval":
        return _run_eval(args)
    if args.command == "grade":
        return _run_grade(args)
    if args.command == "data":
        return _run_data(args)
    if args.command == "validate":
        return _run_validate(args)

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
