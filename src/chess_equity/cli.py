"""Command-line entry point: ``chess-equity``.

Modes:

    chess-equity eval "<fen>" --white-elo 1500 --black-elo 1500
    chess-equity eval --pgn game.pgn --white-elo 1500 --black-elo 1500
    chess-equity broadcast --round <id>            # live Lichess broadcast round
    chess-equity broadcast --pgn game.pgn          # replay a finished game as "live"

The CLI depends only on :class:`~chess_equity.adapters.EquityModel`; it constructs
the placeholder :class:`~chess_equity.models.LichessBaselineModel` today, but a new
model would drop in with no other changes here.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, TextIO

import chess
import chess.pgn

from chess_equity.adapters import EquityModel
from chess_equity.bar import render_eval
from chess_equity.broadcast import (
    BroadcastIngestor,
    LichessRoundFeed,
    LocalPgnFeed,
    MoveEvent,
    UrlPgnFeed,
)
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


def _event_line(event: MoveEvent) -> str:
    """One JSONL record the overlay (task 0019) can tail."""
    return json.dumps(event.to_dict())


def _run_broadcast(args: argparse.Namespace, model: EquityModel, out: TextIO) -> int:
    """Drive broadcast ingestion, writing one JSON event per line to ``out``."""
    if args.pgn:
        with open(args.pgn, encoding="utf-8") as fh:
            feed = LocalPgnFeed(fh.read(), moves_per_poll=args.moves_per_poll)
    elif args.round:
        feed = LichessRoundFeed(args.round, token=args.token)
    elif args.url:
        feed = UrlPgnFeed(args.url)
    else:
        raise ValueError("broadcast needs one of --pgn / --round / --url")

    ingestor = BroadcastIngestor(
        feed,
        model,
        white_elo=args.white_elo,
        black_elo=args.black_elo,
    )

    def emit(event: MoveEvent) -> None:
        out.write(_event_line(event) + "\n")
        out.flush()

    # A local replay terminates (max_idle_polls=1); a live feed runs until interrupted
    # (--max-polls caps it). interval=0 for replays keeps tests/CI instant.
    stats = ingestor.run(
        emit,
        interval=args.interval,
        max_polls=args.max_polls,
        max_idle_polls=1,
    )
    print(
        f"# {stats.events} events over {stats.polls} polls "
        f"({stats.errors} feed errors), max equity compute {stats.max_compute_ms:.1f} ms",
        file=sys.stderr,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="chess-equity", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="evaluate a position or a whole game")
    ev.add_argument("fen", nargs="?", default=START_FEN, help="FEN (default: startpos)")
    ev.add_argument("--pgn", help="annotate every move of a PGN file instead")
    ev.add_argument("--white-elo", type=int, default=1500)
    ev.add_argument("--black-elo", type=int, default=1500)

    bc = sub.add_parser(
        "broadcast",
        help="stream per-move equity events from a live (or replayed) broadcast",
    )
    src = bc.add_mutually_exclusive_group(required=True)
    src.add_argument("--round", help="Lichess broadcast round id (live feed)")
    src.add_argument("--url", help="arbitrary public PGN URL (generic feed)")
    src.add_argument("--pgn", help="local PGN file, replayed move-by-move as 'live'")
    bc.add_argument("--white-elo", type=int, default=None, help="override White rating")
    bc.add_argument("--black-elo", type=int, default=None, help="override Black rating")
    bc.add_argument("--interval", type=float, default=2.0, help="seconds between polls")
    bc.add_argument("--max-polls", type=int, default=None, help="stop after N polls")
    bc.add_argument(
        "--moves-per-poll", type=int, default=1, help="replay pacing (local --pgn only)"
    )
    bc.add_argument("--token", default=None, help="Lichess API token (optional)")

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

    if args.command == "broadcast":
        try:
            return _run_broadcast(args, model, sys.stdout)
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
