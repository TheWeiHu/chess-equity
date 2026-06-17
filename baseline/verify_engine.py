#!/usr/bin/env python3
"""Engine-check the curated 0003 failure-mode set (task 0028).

The ``engine_cp`` values and the "only move" claims in ``failure_modes.json`` were
hand-entered from endgame theory (see README "Deferred"). This tool wires task
0001's :class:`~chess_equity.adapters.ObjectiveEngine` to a real Stockfish (via
:class:`chess_equity.stockfish.StockfishEngine`), recomputes each position at a
fixed depth, and checks the curated values against the engine — turning the set
into *engine-checked fixtures*.

Two checks per position:

1. **cp agreement** — a drawn study (``engine_cp == 0``) must read ~drawn; a
   decisive study must keep the same sign and stay clearly decisive.
2. **only-move** — where a position carries an ``only_move_uci`` (the unique idea
   at the root, e.g. the knight fork's ``e7e8n``), the engine's chosen move must
   match it.

The *checking logic* is pure (:func:`check_position` takes an :class:`Analysis`),
so it is unit-tested with a fake engine and needs no binary. Run against a real
engine with::

    python3 baseline/verify_engine.py            # needs Stockfish on PATH / $STOCKFISH_PATH

Exits non-zero if any check fails (so it can gate a fixtures refresh). When no
Stockfish binary is found it prints a hint and exits 0 (nothing to verify).

Note: engines famously misjudge fortresses (the wrong-bishop and opposite-coloured
-bishop draws) and very deep underpromotions (Saavedra's move-6 ``c8=R``) at modest
depth — surfacing those is the point; the cp tolerance below is generous so only
gross disagreements flag.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SET = os.path.join(HERE, "failure_modes.json")

# Tolerances (centipawns, White POV). Generous on purpose — we flag gross
# disagreement, not depth-sensitive fortress nuance.
DRAW_TOL = 150            # |cp| this small still reads as "drawn"
DECISIVE_MIN = 200        # a decisive study must reach at least this magnitude


def load_positions(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["positions"] if isinstance(data, dict) else data


def _white_cp(analysis) -> Optional[float]:
    """Engine cp from White's POV (positions are all White-to-move, so POV == STM)."""
    return analysis.eval.cp


def _white_decisive(analysis) -> Optional[int]:
    """+1/-1 if the engine sees a forced result for White, else None (sign of mate)."""
    mate = analysis.eval.mate
    if mate is None:
        return None
    if mate == 0:
        return -1  # side to move (White) is mated
    return 1 if mate > 0 else -1


def check_position(pos: dict, analysis) -> dict:
    """Compare one curated position against an engine :class:`Analysis`.

    Returns a result dict: ``{id, claims, checks: [...], ok}`` where each check is
    ``{name, ok, detail}``. Pure — no engine call here, so it is trivially testable.
    """
    checks: List[dict] = []
    claimed_cp = float(pos["engine_cp"])
    cp = _white_cp(analysis)
    decisive = _white_decisive(analysis)

    if claimed_cp == 0:
        ok = decisive is None and cp is not None and abs(cp) <= DRAW_TOL
        detail = (
            f"claimed draw; engine mate={analysis.eval.mate} cp={cp} "
            f"(|cp|<= {DRAW_TOL} and no forced mate)"
        )
    else:
        want_sign = 1 if claimed_cp > 0 else -1
        if decisive is not None:
            ok = decisive == want_sign
            detail = f"claimed decisive sign {want_sign:+d}; engine mate sign {decisive:+d}"
        else:
            ok = cp is not None and (1 if cp > 0 else -1) == want_sign and abs(cp) >= DECISIVE_MIN
            detail = (
                f"claimed decisive sign {want_sign:+d} (>= {DECISIVE_MIN}cp); "
                f"engine cp={cp}"
            )
    checks.append({"name": "cp", "ok": bool(ok), "detail": detail})

    only_move = pos.get("only_move_uci")
    if only_move:
        got = analysis.best_move
        checks.append(
            {
                "name": "only-move",
                "ok": got == only_move,
                "detail": f"claimed {only_move}; engine chose {got}",
            }
        )

    return {
        "id": pos["id"],
        "name": pos.get("name", pos["id"]),
        "checks": checks,
        "ok": all(c["ok"] for c in checks),
    }


def verify(positions, engine) -> List[dict]:
    """Run every position through ``engine`` (any object with ``.analyse(fen)``)."""
    return [check_position(p, engine.analyse(p["fen"])) for p in positions]


def render(results: List[dict]) -> str:
    lines = ["Engine verification of the 0003 failure-mode set", "=" * 70]
    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        lines.append(f"[{mark}] {r['name']}  ({r['id']})")
        for c in r["checks"]:
            cmark = "ok " if c["ok"] else "!! "
            lines.append(f"    {cmark}{c['name']}: {c['detail']}")
    passed = sum(1 for r in results if r["ok"])
    lines.append("")
    lines.append(f"{passed}/{len(results)} positions agree with the engine.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Engine-check the 0003 failure-mode set.")
    ap.add_argument("--json", default=DEFAULT_SET)
    ap.add_argument("--depth", type=int, default=None, help="fixed search depth")
    args = ap.parse_args(argv)

    try:
        from chess_equity.stockfish import StockfishEngine, stockfish_path
    except Exception as exc:  # package not installed / import error
        print(f"chess_equity not importable ({exc}); install the package first.")
        return 0

    if not stockfish_path():
        print(
            "No Stockfish binary found (set $STOCKFISH_PATH or put `stockfish` on "
            "PATH). Nothing to verify; the curated values stay hand-entered."
        )
        return 0

    kwargs = {"depth": args.depth} if args.depth else {}
    engine = StockfishEngine(**kwargs)
    results = verify(load_positions(args.json), engine)
    print(render(results))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
