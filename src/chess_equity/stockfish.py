"""Stockfish adapter — a real :class:`ObjectiveEngine` (task 0028).

Task 0001 defined the :class:`~chess_equity.adapters.ObjectiveEngine` contract
(``fen -> centipawns/mate``) and shipped only the trivial
:class:`~chess_equity.models.MaterialEngine`. This wires a real UCI engine
(Stockfish by default) behind that same contract so the curated 0003 failure-mode
set can be *engine-checked* instead of hand-entered (see ``baseline/verify_engine.py``).

Like the Maia-2 adapter, the heavy/external dependency — here a Stockfish binary —
sits behind an **injectable backend** so the test suite and CI need no engine
installed. A backend maps ``(fen, depth) -> AnalysisResult``; the default backend
shells out to a UCI engine via ``python-chess``, and tests pass a fake one.

Evaluations are reported from the **side-to-move's POV**, matching
:class:`MaterialEngine` and the :class:`ObjectiveEval` docstring.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Callable, Optional

import chess

from chess_equity.adapters import ObjectiveEngine, ObjectiveEval

DEFAULT_DEPTH = 18

_INSTALL_HINT = (
    "No Stockfish (UCI) binary found. Install it (`brew install stockfish` / "
    "`apt-get install stockfish`) and either put it on PATH or set STOCKFISH_PATH "
    "to the binary. For tests/CI, inject a fake backend instead of a real engine."
)


class StockfishNotFound(RuntimeError):
    """Raised when the real backend is used but no UCI binary can be located."""


@dataclass(frozen=True)
class Analysis:
    """One engine verdict: the objective eval plus the engine's chosen move.

    ``eval`` is from the side-to-move's POV (like :class:`ObjectiveEval`).
    ``best_move`` is the first move of the principal variation, in UCI, or ``None``
    when the engine returns no PV (e.g. a terminal position).
    """

    eval: ObjectiveEval
    best_move: Optional[str]


# A backend maps (fen, depth) -> Analysis. This is the only seam the real engine
# touches, so the whole adapter is testable without a Stockfish binary.
Backend = Callable[[str, int], Analysis]


def stockfish_path(explicit: Optional[str] = None) -> Optional[str]:
    """Locate a UCI engine binary: explicit arg, then ``$STOCKFISH_PATH``, then PATH.

    Returns ``None`` if none is found (callers decide whether to skip or raise).
    """
    candidate = explicit or os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
    if candidate and (explicit or os.environ.get("STOCKFISH_PATH")):
        # An explicit/env path must actually exist to count.
        return candidate if os.path.exists(candidate) else None
    return candidate


def _eval_from_relative(cp: Optional[int], mate: Optional[int]) -> ObjectiveEval:
    """Convert python-chess relative (side-to-move POV) score parts to ObjectiveEval."""
    if mate is not None:
        return ObjectiveEval(mate=int(mate))
    return ObjectiveEval(cp=float(cp if cp is not None else 0.0))


class RealStockfishBackend:
    """Production backend: analyse each FEN with a UCI engine via ``python-chess``.

    A fresh engine process is opened and quit per call (via a context manager), so
    nothing is left running and there is no shared-state to manage. For the handful
    of positions the verifier checks this is plenty fast; batch users can supply a
    persistent backend instead.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        resolved = stockfish_path(path)
        if not resolved:
            raise StockfishNotFound(_INSTALL_HINT)
        self.path = resolved

    def __call__(self, fen: str, depth: int) -> Analysis:
        import chess.engine  # local import: only needed on the real path

        board = chess.Board(fen)
        with chess.engine.SimpleEngine.popen_uci(self.path) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
        relative = info["score"].relative
        ev = _eval_from_relative(relative.score(), relative.mate())
        pv = info.get("pv")
        best = pv[0].uci() if pv else None
        return Analysis(eval=ev, best_move=best)


class StockfishEngine(ObjectiveEngine):
    """A real :class:`ObjectiveEngine` backed by Stockfish (or any UCI engine).

    Construct with a fixed search ``depth`` (reproducible fixtures want a fixed
    depth, not time). The backend is injectable: omit it for the real engine
    (raising :class:`StockfishNotFound` if no binary is available), or pass a fake
    one in tests.
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        *,
        path: Optional[str] = None,
        depth: int = DEFAULT_DEPTH,
    ) -> None:
        self.depth = depth
        self._backend: Backend = backend or RealStockfishBackend(path=path)

    def analyse(self, fen: str) -> Analysis:
        """Full verdict for ``fen``: objective eval + the engine's best move (UCI)."""
        return self._backend(fen, self.depth)

    def eval(self, fen: str) -> ObjectiveEval:
        """:class:`ObjectiveEngine` contract — eval from the side-to-move's POV."""
        return self.analyse(fen).eval

    def best_move(self, fen: str) -> Optional[str]:
        """The engine's chosen move (first PV move) in UCI, or ``None``."""
        return self.analyse(fen).best_move
