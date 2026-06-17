#!/usr/bin/env python3
"""Generate ``web/demo-game.json`` for the equity-bar web demo (task 0010).

The web demo is a static page; it reads a precomputed JSON so it needs no backend
(the task explicitly allows "precomputed JSON for a fixed demo game"). This script
produces that JSON for **Légal's Mate** — the textbook queen "sacrifice" where White
goes *material-down* (the classic centipawn bar plunges) while actually delivering a
forced mate (the rating-conditioned equity bar stays winning). That contradiction is
the flagship the demo exists to show.

Two equity sources:

* ``--model demo`` (default): a small, transparent **illustrative** rating skew over a
  hand-annotated practical equity per ply. It is NOT a trained model — it exists only
  so the committed demo renders the rating-slider "wow" with zero heavy deps. The
  equity is illustrative and labelled as such.
* ``--model maia2``: the real rating-conditioned bar — calls Maia-2's value head for
  every (fen, white_elo, black_elo) grid cell. Use this to replace the illustrative
  numbers with real ones once Maia is installed (``pip install maia2``).

And two **centipawn-bar** sources (the dashed objective line the equity is contrasted
against):

* ``--cp-engine material`` (default, committed): the dependency-free material count.
  Deterministic and reproducible regardless of Stockfish version — and, crucially, it
  is *dumb* enough to plunge on Légal's queen "sacrifice", which is the whole contrast
  the committed demo teaches.
* ``--cp-engine stockfish``: a real Stockfish eval at ``--depth`` per ply (errors with
  an install hint if no binary is found — it never silently degrades to material). NB:
  a real engine *solves* Légal's Mate (it sees the forced mate at the queen-grab), so
  with ``stockfish`` the dramatic green-equity / red-cp contradiction softens into a
  subtler practical-vs-objective gap at the unsound sac itself. Use it on positions a
  shallow material count misreads but a deep engine does not pre-solve.

Run from the repo root:
``python web/build_demo.py [--model demo|maia2] [--cp-engine material|stockfish] [--depth N]``
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running from the repo without an install: add ../src and this dir to the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import chess  # noqa: E402

from chess_equity.models import MaterialEngine  # noqa: E402
from chess_equity.types import lichess_win_percent  # noqa: E402

# Single source of truth for the cp/grade formulas the web demo shares with the
# import path (web/game_json.py) — keep them here and the two can't drift.
from game_json import _material_cp_white, drama_by_band, grade_label  # noqa: E402

# The demo game (SAN) and, per ply, the hand-annotated practical White-POV equity at
# the reference band (1500 vs 1500). These annotations encode the *known* truth about
# Légal's Mate — White's attack is winning throughout once Black grabs the queen — which
# pure material can't see. (ply 0 is the start position.)
MOVES = ["e4", "e5", "Nf3", "Nc6", "Bc4", "d6", "Nc3", "Bg4", "Nxe5", "Bxd1", "Bxf7+", "Ke7", "Nd5#"]
REF_EQUITY_WHITE = [52, 53, 52, 53, 52, 54, 53, 55, 56, 60, 92, 96, 98, 100]

REFERENCE_BAND = 1500
RATING_BANDS = [1100, 1500, 1900, 2300]

def _demo_equity(eq_ref: float, cp_white: float, white_elo: int, black_elo: int) -> float:
    """Illustrative rating skew: expand the reference-band equity across ratings.

    The *story* the slider tells: a stronger **Black** defends the trap better, so White's
    practical equity drifts back toward the objective (material) truth; a stronger
    **White** executes the attack more reliably, nudging equity up. Toy model — the real
    one is ``--model maia2``.
    """
    objective = lichess_win_percent(cp_white) / 100.0  # material-only White equity, [0,1]
    eq = eq_ref / 100.0
    w_exec = (white_elo - REFERENCE_BAND) / 1600.0  # >0 => stronger White
    b_def = (black_elo - REFERENCE_BAND) / 1600.0   # >0 => stronger Black
    eq = eq + w_exec * 0.12 * (1.0 - eq) - b_def * 0.30 * (eq - objective)
    return round(100.0 * min(0.99, max(0.01, eq)), 1)


def _build_cp_engine(name: str, depth: int):
    """The objective engine behind the centipawn bar — Stockfish or material.

    ``stockfish`` constructs a real :class:`StockfishEngine`, which raises
    :class:`StockfishNotFound` (with an install hint) when no binary is present; it
    never silently degrades to material, because a fake centipawn bar would
    misrepresent the very benchmark this demo contrasts equity against.
    """
    if name == "stockfish":
        from chess_equity.stockfish import StockfishEngine

        return StockfishEngine(depth=depth)
    return MaterialEngine()


def build(model_name: str, cp_engine: str = "material", depth: int = 12) -> dict:
    engine = _build_cp_engine(cp_engine, depth)
    real_model = None
    if model_name == "maia2":
        from chess_equity.cli import build_model

        real_model = build_model("maia2")

    board = chess.Board()
    plies = [{"san": "(start)", "fen": board.fen()}]
    for san in MOVES:
        board.push(board.parse_san(san))
        plies.append({"san": san, "fen": board.fen()})

    moves = []
    prev_white_eq = None
    for i, node in enumerate(plies):
        fen = node["fen"]
        # Material centipawns, always White-POV (mate => a decisive ±10000).
        cp_white = _material_cp_white(fen, engine)

        equity = {}
        for we in RATING_BANDS:
            for be in RATING_BANDS:
                if real_model is not None:
                    equity[f"{we}-{be}"] = round(real_model.evaluate(fen, we, be).equity_white, 1)
                else:
                    equity[f"{we}-{be}"] = _demo_equity(REF_EQUITY_WHITE[i], cp_white, we, be)

        ref_white = equity[f"{REFERENCE_BAND}-{REFERENCE_BAND}"]
        grade = None
        if i > 0 and prev_white_eq is not None:
            mover_white = chess.Board(plies[i - 1]["fen"]).turn == chess.WHITE
            # Δ from the mover's POV (White-POV delta flips for Black).
            delta = (ref_white - prev_white_eq) if mover_white else (prev_white_eq - ref_white)
            grade = {"label": grade_label(delta), "delta": round(delta, 1)}
        prev_white_eq = ref_white

        moves.append(
            {
                "ply": i,
                "san": node["san"],
                "fen": fen,
                "cp": cp_white,
                "equity": equity,
                "grade": grade,
            }
        )

    cp_desc = (
        f"real Stockfish evals at depth {depth}"
        if cp_engine == "stockfish"
        else "material count"
    )
    return {
        "_comment": (
            "Web-demo fixture for task 0010: Légal's Mate. `cp` is White-POV centipawns "
            f"from {cp_desc}; `equity[white-black]` is White-POV win-equity in percent "
            "over a rating grid. Generated by web/build_demo.py. With --model demo "
            "(default) the equity is an ILLUSTRATIVE rating skew, not a trained model; "
            "regenerate with --model maia2 for real rating-conditioned numbers."
        ),
        "model": model_name,
        "cp_engine": cp_engine,
        "game": {
            "name": "Légal's Mate",
            "white": "Légal",
            "black": "Saint Brie",
            "white_elo_default": REFERENCE_BAND,
            "black_elo_default": REFERENCE_BAND,
        },
        "rating_bands": RATING_BANDS,
        "moves": moves,
        "drama": drama_by_band(moves, RATING_BANDS),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=("demo", "maia2"), default="demo")
    ap.add_argument(
        "--cp-engine",
        choices=("material", "stockfish"),
        default="material",
        help="objective engine behind the centipawn bar (default: material)",
    )
    ap.add_argument(
        "--depth", type=int, default=12, help="Stockfish search depth (--cp-engine stockfish)"
    )
    ap.add_argument(
        "--out", default=os.path.join(os.path.dirname(__file__), "demo-game.json")
    )
    args = ap.parse_args(argv)
    data = build(args.model, cp_engine=args.cp_engine, depth=args.depth)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(
        f"wrote {args.out} ({len(data['moves'])} plies, model={args.model}, "
        f"cp_engine={args.cp_engine})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
