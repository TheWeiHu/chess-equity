#!/usr/bin/env python3
"""Build the web-demo JSON for an arbitrary game (task 0011).

The static web page (``app.js``) renders any file in the ``demo-game.json`` schema:
a rating grid of White-POV equity per ply, plus the classic centipawn line. This
module turns a real PGN into that schema so the demo works on **real games with real
ratings** — the importer (``import_game.py``) wraps it with a Lichess fetch.

Two evaluation sources, mirroring the bar the page draws:

* **classic centipawns** — reuse the PGN's own ``[%eval]`` annotation (Lichess'
  Stockfish) when present; otherwise fall back to material (``MaterialEngine``).
* **our equity** — any :class:`~chess_equity.adapters.EquityModel`, evaluated over a
  rating grid built around the two players' real ratings so the slider is meaningful.

The default ``baseline`` model is rating-blind (the grid is flat — honest, no heavy
deps); pass a Maia-2 model for real rating-conditioned numbers.
"""
from __future__ import annotations

import io
import os
import re
import sys
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess  # noqa: E402
import chess.pgn  # noqa: E402

from chess_equity.adapters import ObjectiveEngine  # noqa: E402
from chess_equity.models import MaterialEngine  # noqa: E402

# Standard ladder mixed in with the players' real ratings so the slider has a few
# stops to explore even when both players are close in rating.
DEFAULT_LADDER = (1100, 1500, 1900, 2300)
MATE_CP = 10000.0

# Same coarse grade bands as build_demo.py (mover-POV Δequity, percentage points).
GRADE_BANDS = [(10, "brilliant"), (3, "good"), (-3, "ok"), (-8, "inaccuracy"), (-15, "mistake")]
_EVAL_RE = re.compile(r"\[%eval\s+([^\]\s]+)")


def grade_label(delta: float) -> str:
    for threshold, label in GRADE_BANDS:
        if delta >= threshold:
            return label
    return "blunder"


def grade_move(prev_ref: float, ref_white: float, prev_fen: str) -> dict:
    """Mover-POV Δequity grade between two consecutive reference-band equities.

    ``prev_ref``/``ref_white`` are White-POV equity (%) before and after the move;
    ``prev_fen`` is the position the move was played from (its side-to-move is the
    mover, so the White-POV delta is flipped for Black). Shared by build_demo.py and
    build_game.py so the grade computation can't drift between the demo and import paths.
    """
    mover_white = chess.Board(prev_fen).turn == chess.WHITE
    delta = (ref_white - prev_ref) if mover_white else (prev_ref - ref_white)
    return {"label": grade_label(delta), "delta": round(delta, 1)}


def eval_to_cp_white(token: str) -> Optional[float]:
    """Parse a Lichess ``[%eval]`` token (White POV) to centipawns, or None.

    ``"0.24"`` -> 24.0 ; ``"-1.5"`` -> -150.0 ; ``"#3"`` -> +MATE ; ``"#-2"`` -> -MATE.
    """
    token = token.strip()
    if not token:
        return None
    if token.startswith("#"):
        rest = token[1:]
        if rest in ("", "-"):
            return None
        try:
            n = int(rest)
        except ValueError:
            return None
        # "#0" / "#-0": side to move is mated; treat sign as given (default White wins).
        return MATE_CP if n >= 0 else -MATE_CP
    try:
        return float(token) * 100.0
    except ValueError:
        return None


def _eval_in_comment(comment: str) -> Optional[float]:
    m = _EVAL_RE.search(comment or "")
    return eval_to_cp_white(m.group(1)) if m else None


def _material_cp_white(fen: str, engine: ObjectiveEngine) -> float:
    """White-POV centipawns from any objective engine (material or a real UCI engine).

    ``engine.eval`` is side-to-move-relative; flip it to White POV. A forced mate
    becomes a decisive ±MATE_CP: a *positive* mate count means the side to move is
    mating, a non-positive one means it is (being) mated. (For the bare
    :class:`MaterialEngine` the only mate is ``mate=0`` on a finished board, i.e. the
    side to move has just been mated — so this reduces to the old behaviour.)
    """
    obj = engine.eval(fen)
    white = chess.Board(fen).turn == chess.WHITE
    if obj.mate is not None:
        stm_decisive = MATE_CP if obj.mate > 0 else -MATE_CP
        return stm_decisive if white else -stm_decisive
    cp_stm = obj.cp or 0.0
    return cp_stm if white else -cp_stm


def _int_header(headers, key: str) -> Optional[int]:
    raw = (headers.get(key, "") or "").strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def rating_bands(white_elo: int, black_elo: int, ladder: Sequence[int] = DEFAULT_LADDER) -> List[int]:
    """A sorted, de-duplicated band list that always contains both real ratings."""
    return sorted(set(list(ladder) + [white_elo, black_elo]))


def build_game(
    pgn_text: str,
    *,
    model,
    ladder: Sequence[int] = DEFAULT_LADDER,
    name: Optional[str] = None,
) -> dict:
    """Parse one PGN game and assemble the web-demo JSON dict.

    ``model`` is any object with ``evaluate(fen, white_elo, black_elo) -> Equity``
    (``equity_white`` in [0, 100]). Raises ``ValueError`` if the PGN has no game.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("no game found in PGN")

    headers = game.headers
    white_elo = _int_header(headers, "WhiteElo") or 1500
    black_elo = _int_header(headers, "BlackElo") or 1500
    bands = rating_bands(white_elo, black_elo, ladder)
    engine = MaterialEngine()

    # Walk the mainline; a node's [%eval] is the eval of the position AFTER its move.
    board = game.board()
    plies = [{"san": "(start)", "fen": board.fen(), "eval_cp": None}]
    for node in game.mainline():
        san = board.san(node.move)
        board.push(node.move)
        plies.append({"san": san, "fen": board.fen(), "eval_cp": _eval_in_comment(node.comment)})

    ref_key = f"{white_elo}-{black_elo}"
    moves = []
    prev_ref = None
    for i, node in enumerate(plies):
        fen = node["fen"]
        cp_white = node["eval_cp"]
        if cp_white is None:
            cp_white = _material_cp_white(fen, engine)

        equity = {}
        for we in bands:
            for be in bands:
                equity[f"{we}-{be}"] = round(model.evaluate(fen, we, be).equity_white, 1)

        ref_white = equity[ref_key]
        grade = None
        if i > 0 and prev_ref is not None:
            grade = grade_move(prev_ref, ref_white, plies[i - 1]["fen"])
        prev_ref = ref_white

        moves.append(
            {"ply": i, "san": node["san"], "fen": fen, "cp": cp_white, "equity": equity, "grade": grade}
        )

    site = headers.get("Site", "")
    return {
        "_comment": (
            "Imported game for task 0011 (web/import_game.py). `cp` is White-POV "
            "centipawns from the PGN's [%eval] when present, else material. "
            "`equity[white-black]` is White-POV win-equity (%) over a rating grid "
            "built around the players' real ratings; with the baseline model it is "
            "rating-blind (flat grid)."
        ),
        "model": getattr(model, "name", model.__class__.__name__),
        "source": site,
        "game": {
            "name": name or headers.get("Event", "Imported game"),
            "white": headers.get("White", "?"),
            "black": headers.get("Black", "?"),
            "white_elo_default": white_elo,
            "black_elo_default": black_elo,
        },
        "rating_bands": bands,
        "moves": moves,
    }
