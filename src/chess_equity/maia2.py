"""Maia-2 adapter — the first *principled* rating-conditioned equity bar (task 0005).

Maia-2 (`CSSLab/maia2`, NeurIPS 2024) is a single rating-conditioned model whose
released inference API returns, in one forward pass::

    move_probs, win_prob = inference.inference_each(model, prepared, fen, elo_self, elo_oppo)

- ``move_probs`` — P(move | position, rating): the :class:`HumanPolicy` (used by the
  search/rollout baselines 0006/0007, and to decide whether an "absurd refutation"
  actually gets found).
- ``win_prob`` — Maia-2's value head, trained on real Lichess outcomes with labels
  {win:+1, draw:0, loss:-1} and squashed to [0, 1]. That expected-score-in-[0,1] is
  **exactly our side-to-move equity** ``P(win) + 0.5·P(draw)`` — so the bar comes
  straight from Maia-2, conditioned on *both* players' ratings. This is the first
  real ``EquityModel``; baselines 0004/0006/0007 become comparisons against it.

Heavyweight by design: real Maia-2 needs ``torch`` + a downloaded checkpoint. To keep
the test suite (and CI) light, every class here takes an injectable *inference
backend* — a callable ``(fen, elo_self, elo_oppo) -> (move_probs, win_prob)``. The
default backend lazily imports ``maia2`` and loads the model on first use; tests pass
a fake backend, so nothing here requires torch to be installed.

Calibration caveat (feeds 0009): the value head is a *secondary* objective in the
Maia-2 paper with few reported numbers — verify it in the validation harness before
trusting it as the shipped bar, especially at extreme ratings and in endgames.
"""

from __future__ import annotations

import os
import pickle
from typing import Callable, Dict, Optional, Tuple

import chess

from chess_equity.adapters import EquityModel, HumanPolicy, white_to_move
from chess_equity.types import Equity, WDL

# A backend maps (fen, elo_self, elo_oppo) -> (move_probs, win_prob), where
# move_probs is {uci: prob} from the side-to-move's POV and win_prob is that side's
# equity in [0, 1]. This is the exact shape of maia2.inference.inference_each.
InferResult = Tuple[Dict[str, float], float]
Backend = Callable[[str, int, int], InferResult]

# Where the real model's evaluations are memoised across process restarts.
DEFAULT_CACHE_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "chess-equity", "maia2.pkl"
)

_INSTALL_HINT = (
    "Maia-2 is not installed. Install it with `pip install maia2` (pulls torch) and "
    "let it download the checkpoint on first use; see https://github.com/CSSLab/maia2. "
    "For tests/CI, pass a fake inference backend instead of using the real model."
)


class Maia2NotInstalled(RuntimeError):
    """Raised when the real Maia-2 backend is used but ``maia2`` can't be imported."""


def wdl_from_equity(equity: float, draw_scale: float = 0.5) -> WDL:
    """Split a scalar side-to-move equity in [0, 1] into a valid WDL triple.

    Maia-2's value head gives us the *scalar* equity directly but not the draw split,
    so we model the draw mass: it peaks at equity 0.5 and vanishes at 0/1, scaled by
    ``draw_scale``. By construction ``p_win + 0.5·p_draw == equity`` exactly, so the
    bar value is faithful to Maia-2 — only the win/loss-vs-draw partition is modelled.
    """
    e = min(max(equity, 0.0), 1.0)
    p_draw = draw_scale * 2.0 * min(e, 1.0 - e)
    p_win = e - 0.5 * p_draw
    p_loss = 1.0 - e - 0.5 * p_draw
    return WDL.from_unnormalized(p_win=p_win, p_draw=p_draw, p_loss=p_loss)


class RealMaia2Backend:
    """The production backend: lazily loads Maia-2 and calls ``inference_each``.

    The model (and its ~23M params) is loaded once, on the first call, so importing
    this module — or constructing the backend — stays cheap. If ``maia2`` isn't
    importable we raise :class:`Maia2NotInstalled` with an install hint rather than a
    bare ``ImportError``.
    """

    def __init__(self, model_type: str = "rapid", device: str = "cpu") -> None:
        self.model_type = model_type
        self.device = device
        self._model = None
        self._prepared = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from maia2 import inference, model  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without maia2
            raise Maia2NotInstalled(_INSTALL_HINT) from exc
        self._model = model.from_pretrained(type=self.model_type, device=self.device)
        self._prepared = inference.prepare()

    def __call__(self, fen: str, elo_self: int, elo_oppo: int) -> InferResult:  # pragma: no cover - needs torch + weights
        self._ensure_loaded()
        from maia2 import inference  # type: ignore

        move_probs, win_prob = inference.inference_each(
            self._model, self._prepared, fen, elo_self, elo_oppo
        )
        return dict(move_probs), float(win_prob)


class CachedBackend:
    """Memoise a backend by ``(fen, elo_self, elo_oppo)``, optionally on disk.

    Maia-2 calls are not free and the search/rollout baselines (0006/0007) hammer the
    same positions, so caching is part of the contract. With ``path`` set the cache is
    pickled to disk and reloaded on construction, so it survives process restarts
    (re-used for batching/precompute in task 0012).
    """

    def __init__(self, backend: Backend, path: Optional[str] = None) -> None:
        self._backend = backend
        self._path = path
        self._cache: Dict[Tuple[str, int, int], InferResult] = {}
        self.hits = 0
        self.misses = 0
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    self._cache = pickle.load(fh)
            except (OSError, pickle.UnpicklingError, EOFError):
                self._cache = {}

    def __call__(self, fen: str, elo_self: int, elo_oppo: int) -> InferResult:
        key = (fen, elo_self, elo_oppo)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        result = self._backend(fen, elo_self, elo_oppo)
        self._cache[key] = result
        if self._path:
            self._flush()
        return result

    def _flush(self) -> None:
        path = self._path
        assert path is not None  # only reached when a cache path is configured
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._cache, fh)


def _normalize_over_legal(fen: str, move_probs: Dict[str, float]) -> Dict[str, float]:
    """Keep only legal moves from a backend's distribution and renormalise to sum 1."""
    legal = {m.uci() for m in chess.Board(fen).legal_moves}
    kept = {uci: max(p, 0.0) for uci, p in move_probs.items() if uci in legal}
    total = sum(kept.values())
    if total <= 0:
        # Degenerate backend output — fall back to uniform over legal moves.
        if not legal:
            return {}
        return {uci: 1.0 / len(legal) for uci in legal}
    return {uci: p / total for uci, p in kept.items()}


class Maia2Policy(HumanPolicy):
    """``HumanPolicy`` backed by Maia-2: P(move | position, rating).

    The opponent rating defaults to the player's own (a peer game) since the policy
    interface only carries one Elo; the equity model below threads both explicitly.
    """

    def __init__(self, backend: Optional[Backend] = None) -> None:
        self._backend = backend if backend is not None else RealMaia2Backend()

    def move_probs(self, fen: str, elo: int) -> Dict[str, float]:
        raw, _ = self._backend(fen, elo, elo)
        return _normalize_over_legal(fen, raw)


class Maia2Equity(EquityModel):
    """The MVP equity model: Maia-2's ``win_prob`` as a rating-conditioned bar.

    ``win_prob`` is the side-to-move's equity in [0, 1]; we expand it to a WDL via
    :func:`wdl_from_equity` and render the White-POV bar so it stays stable as turns
    alternate. Changing either rating changes the equity for a fixed position — the
    whole point — and a stronger move *raises* the mover's equity.
    """

    SOURCE = "maia2"

    def __init__(self, backend: Optional[Backend] = None, draw_scale: float = 0.5) -> None:
        self._backend = backend if backend is not None else RealMaia2Backend()
        self.draw_scale = draw_scale

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        is_white = white_to_move(fen)
        elo_self, elo_oppo = (white_elo, black_elo) if is_white else (black_elo, white_elo)
        _, win_prob = self._backend(fen, elo_self, elo_oppo)
        wdl = wdl_from_equity(win_prob, draw_scale=self.draw_scale)
        return Equity.from_side_to_move(
            wdl,
            white_to_move=is_white,
            source=self.SOURCE,
        )


def build_maia2_equity(cache_path: Optional[str] = DEFAULT_CACHE_PATH) -> Maia2Equity:
    """Construct a production :class:`Maia2Equity` (real backend + on-disk cache)."""
    backend: Backend = RealMaia2Backend()
    if cache_path:
        backend = CachedBackend(backend, path=cache_path)
    return Maia2Equity(backend)
