"""A model-level equity cache + a precompute pass (task 0012).

The existing :class:`~chess_equity.maia2.CachedBackend` memoises Maia-2's *backend*
calls; this generalises one level up to **any** :class:`~chess_equity.adapters.EquityModel`,
so the same cache serves the CLI, the broadcast path, search/rollout baselines, and the
web demo. A live overlay (0019) needs sub-second-per-move equity, so caching is a
product requirement, not polish.

:class:`CachingEquityModel` wraps a base model and memoises by
``(model_key, fen, white_elo, black_elo)`` — the inputs the equity actually depends on.
It returns a result identical to the uncached model (verified in tests) and can persist
to a small JSON file so warm restarts are instant. Search-parameter keying (depth / k)
is deferred until the search baselines (0006/0007) land — they aren't in the
:meth:`EquityModel.evaluate` signature yet.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

from chess_equity.adapters import EquityModel
from chess_equity.types import Equity, WDL

# In-memory key: the inputs that determine an Equity. model_key keeps two models'
# evaluations of the same position from colliding in a shared cache.
_Key = Tuple[str, str, int, int]


def _equity_to_dict(eq: Equity) -> Dict[str, object]:
    return {
        "p_win": eq.wdl.p_win,
        "p_draw": eq.wdl.p_draw,
        "p_loss": eq.wdl.p_loss,
        "equity_white": eq.equity_white,
        "source": eq.source,
        "cp": eq.cp,
    }


def _equity_from_dict(d: Dict[str, object]) -> Equity:
    cp = d.get("cp")
    return Equity(
        wdl=WDL(p_win=float(d["p_win"]), p_draw=float(d["p_draw"]), p_loss=float(d["p_loss"])),
        equity_white=float(d["equity_white"]),
        source=str(d["source"]),
        cp=(float(cp) if cp is not None else None),
    )


class CachingEquityModel(EquityModel):
    """Memoise any :class:`EquityModel` by ``(model_key, fen, white_elo, black_elo)``.

    ``model_key`` defaults to the wrapped model's ``SOURCE`` (or class name), so a
    persistent cache can safely hold several models at once. With ``path`` set the
    cache is read on construction and written after each miss, surviving restarts.
    ``hits`` / ``misses`` are exposed for the hit-rate the task asks to report.
    """

    def __init__(
        self,
        base: EquityModel,
        *,
        model_key: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        self.base = base
        self.model_key = model_key or getattr(base, "SOURCE", type(base).__name__)
        self._path = path
        self._cache: Dict[_Key, Dict[str, object]] = {}
        self.hits = 0
        self.misses = 0
        if path and os.path.exists(path):
            self._load(path)

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        key: _Key = (self.model_key, fen, white_elo, black_elo)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return _equity_from_dict(cached)
        self.misses += 1
        eq = self.base.evaluate(fen, white_elo, black_elo)
        self._cache[key] = _equity_to_dict(eq)
        if self._path:
            self._flush()
        return eq

    @property
    def total(self) -> int:
        return self.hits + self.misses

    def hit_rate(self) -> float:
        """Fraction of lookups served from cache, in [0, 1] (0 when nothing asked)."""
        return self.hits / self.total if self.total else 0.0

    # --- persistence (JSON: diff-friendly, no pickle) -------------------------

    def _flush(self) -> None:
        path = self._path
        assert path is not None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        entries = [
            {
                "model_key": mk,
                "fen": fen,
                "white_elo": we,
                "black_elo": be,
                "equity": payload,
            }
            for (mk, fen, we, be), payload in self._cache.items()
        ]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "entries": entries}, fh)

    def _load(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
        except (OSError, ValueError):
            return
        for e in blob.get("entries", []):
            key: _Key = (
                str(e["model_key"]),
                str(e["fen"]),
                int(e["white_elo"]),
                int(e["black_elo"]),
            )
            self._cache[key] = e["equity"]
