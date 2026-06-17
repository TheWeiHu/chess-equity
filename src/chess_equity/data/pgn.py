"""Stream a Lichess PGN dump into :class:`~chess_equity.data.schema.PositionRow`s.

Lichess monthly dumps are tens of GB; we never materialise one. ``iter_rows`` reads
one game at a time off a text handle (which may be a decompressing zstd stream — see
:func:`chess_equity.data.build.open_pgn`) and yields a row per *evaluated* position.

We keep only games that are usable as supervised data:

- both ratings present and integer (the conditioning signal),
- a decisive-or-drawn result (``*`` games — abandoned / still in progress — are dropped
  because they have no label),
- and, per position, an ``[%eval]`` tag (Lichess annotates ~6% of games; the rest are
  silently skipped position-by-position).

The eval/clock parsing lives in :mod:`chess_equity.data.schema` so it is unit-tested
in isolation; this module only walks the game tree and assembles rows.
"""

from __future__ import annotations

import re
from typing import IO, Iterator, Optional

import chess
import chess.pgn

from chess_equity.data.schema import (
    PositionRow,
    game_phase,
    parse_clock,
    parse_eval,
    tc_bucket,
)

_EVAL_RE = re.compile(r"\[%eval\s+([^\]]+)\]")
_CLK_RE = re.compile(r"\[%clk\s+([^\]]+)\]")

_RESULT_TO_WHITE_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def _int_header(game: "chess.pgn.Game", key: str) -> Optional[int]:
    try:
        return int(game.headers.get(key, ""))
    except (TypeError, ValueError):
        return None


def _non_king_pieces(board: chess.Board) -> int:
    """Count pieces on the board excluding the two kings (drives the phase heuristic)."""
    return chess.popcount(board.occupied) - 2


def _game_id(game: "chess.pgn.Game") -> Optional[str]:
    """The game's identity, for the game-level validation split (task 0030).

    Lichess stamps each game with ``[Site "https://lichess.org/<id>"]``; we keep just
    the trailing ``<id>`` slug. Falls back to ``GameId``, then the raw ``Site`` value,
    and finally ``None`` when nothing identifies the game (positions from such a game
    then can't be guaranteed leak-free and the split surfaces that loudly).
    """
    site = game.headers.get("Site", "").strip()
    if site:
        return site.rstrip("/").rsplit("/", 1)[-1] or site
    game_id = game.headers.get("GameId", "").strip()
    return game_id or None


def rows_from_game(
    game: "chess.pgn.Game", *, include_fen: bool = False
) -> Iterator[PositionRow]:
    """Yield one :class:`PositionRow` per evaluated position in ``game``.

    Returns nothing (the game is skipped) when ratings are missing or the result is
    not a finished W/D/L outcome. ``include_fen`` records each position's FEN (the
    board is already computed for the phase heuristic, so it is free to capture); it
    is off by default because the FEN ~triples row size and only board models use it.
    """
    result = _RESULT_TO_WHITE_SCORE.get(game.headers.get("Result", "*"))
    white_elo = _int_header(game, "WhiteElo")
    black_elo = _int_header(game, "BlackElo")
    if result is None or white_elo is None or black_elo is None:
        return

    time_control = game.headers.get("TimeControl", "-")
    bucket = tc_bucket(time_control)
    game_id = _game_id(game)

    node = game
    # Track BOTH players' clocks across the whole game (task 0026): a [%clk] is the
    # clock of the player who just moved, so we update it on every ply — including
    # plies with no [%eval] (which we skip emitting) — and carry the last-known value
    # forward. ``clock_remaining`` is then the *side-to-move's* clock at each emitted
    # position, fixing the earlier mislabelling (it used to be the mover's clock).
    white_clock: Optional[float] = None
    black_clock: Optional[float] = None
    while node.variations:
        node = node.variations[0]
        comment = node.comment or ""

        clk_match = _CLK_RE.search(comment)
        if clk_match is not None:
            clk = parse_clock(clk_match.group(1))
            if clk is not None:
                # Odd ply = White just moved; even ply = Black just moved.
                if node.ply() % 2 == 1:
                    white_clock = clk
                else:
                    black_clock = clk

        eval_match = _EVAL_RE.search(comment)
        if eval_match is None:
            continue
        cp_eval = parse_eval(eval_match.group(1))
        if cp_eval is None:
            continue

        board = node.board()  # position *after* this move
        stm_white = board.turn == chess.WHITE

        yield PositionRow(
            cp_eval=cp_eval,
            white_elo=white_elo,
            black_elo=black_elo,
            ply=node.ply(),
            phase=game_phase(node.ply(), _non_king_pieces(board)),
            time_control=time_control,
            tc_bucket=bucket,
            clock_remaining=white_clock if stm_white else black_clock,
            white_clock=white_clock,
            black_clock=black_clock,
            side_to_move="white" if stm_white else "black",
            result=result,
            game_id=game_id,
            fen=board.fen() if include_fen else None,
        )


def iter_rows(
    handle: IO[str], *, limit: Optional[int] = None, include_fen: bool = False
) -> Iterator[PositionRow]:
    """Stream rows from every game on ``handle``, stopping after ``limit`` rows.

    ``handle`` is any text stream of concatenated PGN games (a plain file, or a
    zstd-decompressing wrapper). ``limit`` caps the emitted row count so a caller can
    build a small sample without reading a whole multi-GB dump. ``include_fen``
    records each position's FEN (see :func:`rows_from_game`).
    """
    emitted = 0
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            return
        for row in rows_from_game(game, include_fen=include_fen):
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                return
