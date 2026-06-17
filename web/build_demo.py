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

Why the committed demo stays on ``material`` (and Stockfish is opt-in) is recorded in
``docs/web-demo-objective-bar-decision.md`` (task 0082).

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
from game_json import _material_cp_white, grade_label  # noqa: E402

REFERENCE_BAND = 1500
RATING_BANDS = [1100, 1500, 1900, 2300]

# The demo catalog. Each game carries, per node (the start position + one per ply), a
# hand-annotated practical White-POV equity at the reference band (1500 vs 1500) for the
# illustrative ``demo`` model — ``--model maia2`` ignores these and calls the real net.
# Every game is a short, famous mate where pure material misreads the position, so the
# equity-vs-centipawn divergence (the whole pitch) shows up somewhere. ``ref_equity``
# length must be ``len(moves) + 1`` (the leading entry is the start position).
GAMES = {
    # Légal's queen "sacrifice": White is material-down yet forcing mate — the flagship.
    "legals": {
        "name": "Légal's Mate", "white": "Légal", "black": "Saint Brie",
        "file": "demo-game.json",
        "moves": ["e4", "e5", "Nf3", "Nc6", "Bc4", "d6", "Nc3", "Bg4",
                  "Nxe5", "Bxd1", "Bxf7+", "Ke7", "Nd5#"],
        "ref_equity": [52, 53, 52, 53, 52, 54, 53, 55, 56, 60, 92, 96, 98, 100],
    },
    # Scholar's Mate: after 3...Nf6?? material is dead even but White has a forced mate.
    "scholars": {
        "name": "Scholar's Mate", "white": "Attacker", "black": "Beginner",
        "file": "scholars-game.json",
        "moves": ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"],
        "ref_equity": [52, 53, 52, 53, 52, 58, 90, 100],
    },
    # Fool's Mate: after 2.g4?? White is materially fine but objectively (and practically)
    # lost to ...Qh4# — the centipawn bar reads ~even while equity collapses.
    "fools": {
        "name": "Fool's Mate", "white": "Blunderer", "black": "Punisher",
        "file": "fools-game.json",
        "moves": ["f3", "e5", "g4", "Qh4#"],
        "ref_equity": [50, 47, 47, 30, 0],
    },
}
DEFAULT_GAME = "legals"

# Back-compat module aliases for the original single-game constants.
MOVES = GAMES[DEFAULT_GAME]["moves"]
REF_EQUITY_WHITE = GAMES[DEFAULT_GAME]["ref_equity"]

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


def build(
    model_name: str, cp_engine: str = "material", depth: int = 12, game: str = DEFAULT_GAME
) -> dict:
    spec = GAMES[game]
    moves_san = spec["moves"]
    ref_equity = spec["ref_equity"]
    assert len(ref_equity) == len(moves_san) + 1, (
        f"game {game!r}: ref_equity must have one entry per node "
        f"(start + {len(moves_san)} plies)"
    )

    engine = _build_cp_engine(cp_engine, depth)
    real_model = None
    if model_name == "maia2":
        from chess_equity.cli import build_model

        real_model = build_model("maia2")

    board = chess.Board()
    plies = [{"san": "(start)", "fen": board.fen()}]
    for san in moves_san:
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
                    equity[f"{we}-{be}"] = _demo_equity(ref_equity[i], cp_white, we, be)

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
            f"Web-demo fixture: {spec['name']}. `cp` is White-POV centipawns "
            f"from {cp_desc}; `equity[white-black]` is White-POV win-equity in percent "
            "over a rating grid. Generated by web/build_demo.py. With --model demo "
            "(default) the equity is an ILLUSTRATIVE rating skew, not a trained model; "
            "regenerate with --model maia2 for real rating-conditioned numbers."
        ),
        "model": model_name,
        "cp_engine": cp_engine,
        "game": {
            "key": game,
            "name": spec["name"],
            "white": spec["white"],
            "black": spec["black"],
            "white_elo_default": REFERENCE_BAND,
            "black_elo_default": REFERENCE_BAND,
        },
        "rating_bands": RATING_BANDS,
        "moves": moves,
    }


def build_manifest() -> dict:
    """The catalog the web page reads to populate its game selector.

    Lists every bundled game (key, display name, players, JSON file) so the front
    end can offer "scroll through multiple games" without hard-coding the list.
    """
    return {
        "_comment": (
            "Game catalog for the web demo's selector. Generated by web/build_demo.py "
            "(run with --all). `file` is the per-game JSON the page loads via ?game=."
        ),
        "default": DEFAULT_GAME,
        "games": [
            {
                "key": key,
                "name": spec["name"],
                "white": spec["white"],
                "black": spec["black"],
                "file": spec["file"],
            }
            for key, spec in GAMES.items()
        ],
    }


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


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
        "--game",
        choices=tuple(GAMES),
        default=DEFAULT_GAME,
        help="which catalog game to build (default: legals)",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="build every catalog game (to its own file) plus the games.json manifest",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output path (default: the selected game's catalog file). Ignored with --all.",
    )
    args = ap.parse_args(argv)
    here = os.path.dirname(__file__)

    games = list(GAMES) if args.all else [args.game]
    for key in games:
        data = build(args.model, cp_engine=args.cp_engine, depth=args.depth, game=key)
        out = (
            os.path.join(here, GAMES[key]["file"])
            if args.all or args.out is None
            else args.out
        )
        _write_json(out, data)
        print(
            f"wrote {out} ({len(data['moves'])} plies, game={key}, "
            f"model={args.model}, cp_engine={args.cp_engine})"
        )

    if args.all:
        manifest = os.path.join(here, "games.json")
        _write_json(manifest, build_manifest())
        print(f"wrote {manifest} ({len(GAMES)} games)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
