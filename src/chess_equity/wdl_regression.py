"""Approach A — a rating-conditioned WDL model fit by multinomial logistic regression.

This is the first equity model that actually conditions on *who is playing*. It fits

    P(White W / D / L | cp_eval, white_elo, black_elo, ply, time_control)

from the Lichess dataset (task 0002) and exposes it two ways:

- as a validation **predictor** (``wdl-a`` in :mod:`chess_equity.validate.harness`),
  the natural drop-in for the 0009 gate — it maps a :class:`PositionRow` straight to
  a White expected-score, so it sits beside the rating-blind baseline in the report;
- as an :class:`~chess_equity.adapters.EquityModel` (``--model wdl-a``), which derives
  the same features from a FEN via an :class:`ObjectiveEngine` and renders a bar.

It is deliberately tiny and dependency-free: a pure-Python multinomial logistic
regression (no numpy / sklearn), so it stays in the light test path and the fitted
artifact is a small, diff-friendly JSON. Re-scoped (task 0004): Maia-2's value head
(0005) is the principled core; this is the transparent, cheap **baseline to compare
against it** in validation — and a Stockfish-only fallback when no learned value head
is available.

The key feature is the ``cp × skill`` interaction: a plain additive model can only
shift the bar up/down by rating, but the real effect is that the *same* engine eval
is more decisive between strong players than weak ones — so the eval's slope must
itself depend on skill. That interaction is what lets this beat the rating-blind
baseline in the off-2300 bands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import exp, tanh
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import chess

from chess_equity.adapters import EquityModel, ObjectiveEngine
from chess_equity.data.schema import PositionRow
from chess_equity.types import Equity, WDL

# Class order is fixed and shared by training, prediction, and serialization.
# Index 0 = White win, 1 = draw, 2 = White loss (all from White's POV, like the data).
N_CLASSES = 3

# The engineered feature vector, in a fixed order (the artifact stores weights aligned
# to this list, so changing it bumps FEATURE_VERSION).
FEATURE_NAMES = (
    "bias",
    "cp_sat",            # tanh(cp/400): saturating eval, so mate sentinels don't blow up
    "avg_skill",         # (avg_elo - 1500) / 400
    "rating_delta",      # (white_elo - black_elo) / 400, White's rating edge
    "cp_x_skill",        # cp_sat * avg_skill: the eval is more decisive at higher skill
    "ply_norm",          # min(ply, 80)/40 - 1
    "tc_bullet",
    "tc_rapid",
    "tc_classical",
    "tc_correspondence",
)
FEATURE_VERSION = 1
N_FEATURES = len(FEATURE_NAMES)

# blitz is the dropped reference time-control bucket (its one-hot is the all-zero case).
_TC_ONEHOT = {
    "bullet": "tc_bullet",
    "rapid": "tc_rapid",
    "classical": "tc_classical",
    "correspondence": "tc_correspondence",
}

# A mate is fed in as a large centipawn magnitude; tanh(3000/400) ~= 1 saturates it.
_MATE_CP = 3000.0


def features(
    cp_eval_white: float,
    white_elo: int,
    black_elo: int,
    ply: int,
    tc_bucket: str,
) -> List[float]:
    """Build the White-POV feature vector. ``cp_eval_white`` is White's centipawns.

    Shared by the predictor (which reads them off a :class:`PositionRow`) and the
    :class:`EquityModel` adapter (which derives them from a FEN), so the two paths can
    never drift out of sync.
    """
    cp_sat = tanh(cp_eval_white / 400.0)
    avg_skill = ((white_elo + black_elo) / 2.0 - 1500.0) / 400.0
    rating_delta = (white_elo - black_elo) / 400.0
    ply_norm = min(ply, 80) / 40.0 - 1.0
    onehot = {name: 0.0 for name in _TC_ONEHOT.values()}
    key = _TC_ONEHOT.get(tc_bucket)
    if key is not None:
        onehot[key] = 1.0
    return [
        1.0,
        cp_sat,
        avg_skill,
        rating_delta,
        cp_sat * avg_skill,
        ply_norm,
        onehot["tc_bullet"],
        onehot["tc_rapid"],
        onehot["tc_classical"],
        onehot["tc_correspondence"],
    ]


def _softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    exps = [exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _target(result: float) -> List[float]:
    """One-hot the White result into [win, draw, loss]. Lichess results are discrete."""
    if result >= 0.75:
        return [1.0, 0.0, 0.0]
    if result <= 0.25:
        return [0.0, 0.0, 1.0]
    return [0.0, 1.0, 0.0]


@dataclass
class WdlRegression:
    """A fitted multinomial-logistic WDL model: weights are ``[class][feature]``."""

    weights: List[List[float]]
    feature_version: int = FEATURE_VERSION
    meta: Optional[Dict[str, object]] = None

    def _proba(self, x: Sequence[float]) -> List[float]:
        logits = [sum(w * xj for w, xj in zip(self.weights[k], x)) for k in range(N_CLASSES)]
        return _softmax(logits)

    def predict_white_wdl(
        self,
        cp_eval_white: float,
        white_elo: int,
        black_elo: int,
        ply: int,
        tc_bucket: str,
    ) -> WDL:
        """Predicted White-POV W/D/L for the given engineered inputs."""
        p = self._proba(features(cp_eval_white, white_elo, black_elo, ply, tc_bucket))
        return WDL.from_unnormalized(p_win=p[0], p_draw=p[1], p_loss=p[2])

    def predict_white_equity(
        self,
        cp_eval_white: float,
        white_elo: int,
        black_elo: int,
        ply: int,
        tc_bucket: str,
    ) -> float:
        """White expected-score P(win)+0.5·P(draw) — the validation harness signature."""
        return self.predict_white_wdl(cp_eval_white, white_elo, black_elo, ply, tc_bucket).equity

    # --- serialization --------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature_version": self.feature_version,
            "feature_names": list(FEATURE_NAMES),
            "class_order": ["white_win", "draw", "white_loss"],
            "weights": self.weights,
            "meta": self.meta or {},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "WdlRegression":
        version = int(payload.get("feature_version", FEATURE_VERSION))
        if version != FEATURE_VERSION:
            raise ValueError(
                f"artifact feature_version {version} != code {FEATURE_VERSION}; retrain"
            )
        weights = [[float(w) for w in row] for row in payload["weights"]]  # type: ignore[index]
        if len(weights) != N_CLASSES or any(len(r) != N_FEATURES for r in weights):
            raise ValueError("artifact weight shape does not match the feature spec")
        meta = payload.get("meta") or {}
        return cls(weights=weights, feature_version=version, meta=dict(meta))  # type: ignore[arg-type]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "WdlRegression":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit(
    rows: Sequence[PositionRow],
    *,
    lr: float = 0.5,
    iters: int = 3000,
    l2: float = 1e-4,
) -> WdlRegression:
    """Fit a :class:`WdlRegression` by batch gradient descent on multinomial log-loss.

    Pure Python and deterministic (zero init, full-batch GD) so a training run is
    reproducible from the committed sample. The gradient of softmax cross-entropy is
    the clean ``(p_k - t_k) · x_j``; L2 keeps weights bounded on tiny datasets.
    """
    if not rows:
        raise ValueError("need at least one row to fit")
    X = [
        features(r.cp_eval, r.white_elo, r.black_elo, r.ply, r.tc_bucket) for r in rows
    ]
    T = [_target(r.result) for r in rows]
    n = len(rows)
    weights = [[0.0] * N_FEATURES for _ in range(N_CLASSES)]

    for _ in range(iters):
        grad = [[0.0] * N_FEATURES for _ in range(N_CLASSES)]
        for x, t in zip(X, T):
            logits = [
                sum(w * xj for w, xj in zip(weights[k], x)) for k in range(N_CLASSES)
            ]
            p = _softmax(logits)
            for k in range(N_CLASSES):
                d = p[k] - t[k]
                gk = grad[k]
                for j in range(N_FEATURES):
                    gk[j] += d * x[j]
        for k in range(N_CLASSES):
            for j in range(N_FEATURES):
                weights[k][j] -= lr * (grad[k][j] / n + l2 * weights[k][j])

    meta = {
        "n_train": n,
        "iters": iters,
        "lr": lr,
        "l2": l2,
        "final_log_loss": _train_log_loss(weights, X, T),
    }
    return WdlRegression(weights=weights, meta=meta)


def _train_log_loss(
    weights: List[List[float]], X: Sequence[Sequence[float]], T: Sequence[Sequence[float]]
) -> float:
    """Mean multinomial cross-entropy on the training set (for the artifact's meta)."""
    from math import log

    total = 0.0
    for x, t in zip(X, T):
        logits = [sum(w * xj for w, xj in zip(weights[k], x)) for k in range(N_CLASSES)]
        p = _softmax(logits)
        total += -sum(tk * log(max(pk, 1e-12)) for pk, tk in zip(p, t))
    return total / len(X)


# --- artifact location + EquityModel adapter ----------------------------------

# Shipped alongside the package so ``--model wdl-a`` and the ``wdl-a`` predictor work
# from an install with no extra data files. Retrain with ``chess-equity train``.
ARTIFACT_NAME = "wdl_a.json"


def default_artifact_path() -> Path:
    """Path to the committed ``wdl-a`` artifact inside the package tree."""
    return Path(__file__).resolve().parent / "artifacts" / ARTIFACT_NAME


def load_wdl_a_model(path: Optional[str] = None) -> WdlRegression:
    """Load the fitted ``wdl-a`` model (the committed artifact by default)."""
    return WdlRegression.load(path or str(default_artifact_path()))


class WdlRegressionModel(EquityModel):
    """``EquityModel`` over a fitted :class:`WdlRegression` + an objective engine.

    The model needs a centipawn eval, so it derives one from the FEN via an
    :class:`ObjectiveEngine` (the placeholder :class:`MaterialEngine` by default — swap
    in Stockfish behind the same interface). ``ply`` comes from the board; the time
    control is unknown from a bare FEN, so it falls back to the blitz reference bucket.
    """

    SOURCE = "wdl-a"

    def __init__(
        self,
        model: WdlRegression,
        engine: Optional[ObjectiveEngine] = None,
        *,
        default_tc_bucket: str = "blitz",
    ) -> None:
        # Imported here to avoid models.py <-> wdl_regression.py import cycles.
        from chess_equity.models import MaterialEngine

        self.model = model
        self.engine = engine or MaterialEngine()
        self.default_tc_bucket = default_tc_bucket

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        obj = self.engine.eval(fen)
        board = chess.Board(fen)
        wtm = board.turn == chess.WHITE
        if obj.mate is not None:
            # mate > 0: side to move is mating; mate <= 0: side to move is (being) mated.
            cp_stm = _MATE_CP if obj.mate > 0 else -_MATE_CP
            cp_display: Optional[float] = None
        else:
            cp_stm = obj.cp if obj.cp is not None else 0.0
            cp_display = cp_stm if wtm else -cp_stm
        cp_white = cp_stm if wtm else -cp_stm
        white_wdl = self.model.predict_white_wdl(
            cp_white, white_elo, black_elo, board.ply(), self.default_tc_bucket
        )
        stm_wdl = white_wdl if wtm else white_wdl.flipped()
        return Equity.from_side_to_move(
            stm_wdl, white_to_move=wtm, source=self.SOURCE, cp=cp_display
        )


def build_wdl_a_equity(
    engine: Optional[ObjectiveEngine] = None, *, path: Optional[str] = None
) -> WdlRegressionModel:
    """Construct the ``wdl-a`` :class:`EquityModel` from the committed artifact."""
    return WdlRegressionModel(load_wdl_a_model(path), engine)


# Silence "imported but unused" for the re-export some callers rely on.
__all__ = [
    "WdlRegression",
    "WdlRegressionModel",
    "features",
    "fit",
    "load_wdl_a_model",
    "build_wdl_a_equity",
    "default_artifact_path",
]
