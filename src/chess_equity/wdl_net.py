"""Approach D — an end-to-end, rating-conditioned WDL **network** (task 0013).

Where Approach A (:mod:`chess_equity.wdl_regression`) fits ``P(W/D/L | cp_eval,
ratings, …)`` — i.e. it still needs Stockfish to produce ``cp_eval`` at inference —
this model predicts the rating-conditioned outcome **straight from the board**:

    P(White W / D / L | board, white_elo, black_elo)

so at inference there is *no* engine call: one forward pass over the raw position
and the two ratings. That is the long-term answer the task asks for — "drop the
Stockfish dependency at inference by predicting rating-conditioned WDL directly," the
way Maia-2's value head does (concept: ``theweihu__chess-equity/concept-equity-bar``).

It deliberately mirrors the seams the rest of the project already has:

- a pure-Python **board encoder** (:func:`encode_board`, ``chess`` only, no torch) so
  the feature contract is unit-testable in the light test path;
- a small **torch** net (:class:`WDLNet`) and trainer (:func:`train_wdl_net`), kept
  behind lazy imports so importing this module — or the validation harness — never
  pulls torch unless you actually train or score the net;
- an :class:`~chess_equity.adapters.EquityModel` adapter (:class:`WdlNetModel`,
  ``--model wdl-net`` / the ``wdl-net`` board predictor) so it slots beside Maia-2 in
  the 0009 gate via :func:`chess_equity.validate.harness.model_predictor`.

This is an MVP of the stretch: a compact MLP (board branch + rating branch → 3-head
WDL), not a Maia-2-scale residual tower. Enough to answer the task's real question on
real data — *does end-to-end beat regression-on-Stockfish-eval, and is it worth the
complexity?* — and leave the architecture to grow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import chess

from chess_equity.adapters import EquityModel
from chess_equity.data.schema import PositionRow
from chess_equity.types import WDL, Equity

# ---------------------------------------------------------------------------
# Board encoder — pure Python (no torch / numpy), so it stays in the light path.
# ---------------------------------------------------------------------------

# 12 piece planes (6 types × 2 colours) over 64 squares = 768, in *absolute* terms
# (white pieces vs black pieces, White's POV), plus 5 board-state scalars. Ratings
# are encoded separately (:func:`rating_features`) so the net learns who is who from
# them rather than from a side-relative board flip — the label is always White-POV.
_PIECE_ORDER = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
N_BOARD_PLANES = 12 * 64  # 768
N_BOARD_STATE = 5  # white_to_move + 4 castling rights
N_BOARD_FEATURES = N_BOARD_PLANES + N_BOARD_STATE  # 773

N_RATING_FEATURES = 3  # avg_skill, rating_delta, |rating_delta| (asymmetry magnitude)

# Bumping the encoder layout (planes/scalars/ratings) invalidates a saved artifact;
# the version is stamped into the artifact and checked on load.
FEATURE_VERSION = 1

# Class order, shared by the trainer, the net head, and serialization.
# 0 = White win, 1 = draw, 2 = White loss — all White's POV, matching the data.
N_CLASSES = 3


def encode_board(fen: str) -> List[float]:
    """Encode ``fen`` into the fixed-length White-POV board feature vector.

    Pure ``chess`` — returns a plain ``list[float]`` so it is testable without torch
    or numpy. Layout: 12 piece planes (768) then ``[white_to_move, K, Q, k, q]``
    castling rights (5), total :data:`N_BOARD_FEATURES`.
    """
    board = chess.Board(fen)
    planes = [0.0] * N_BOARD_PLANES
    for square, piece in board.piece_map().items():
        plane = _PIECE_ORDER.index(piece.piece_type)
        if piece.color == chess.BLACK:
            plane += 6
        planes[plane * 64 + square] = 1.0
    state = [
        1.0 if board.turn == chess.WHITE else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
    ]
    return planes + state


def rating_features(white_elo: int, black_elo: int) -> List[float]:
    """The rating branch's inputs: average skill, White's edge, and its magnitude.

    Normalised on the same 1500/400 scale Approach A uses, so the two models'
    rating conditioning is directly comparable.
    """
    avg_skill = ((white_elo + black_elo) / 2.0 - 1500.0) / 400.0
    rating_delta = (white_elo - black_elo) / 400.0
    return [avg_skill, rating_delta, abs(rating_delta)]


def _white_result_class(result: float) -> int:
    """Map a White-POV result in {1.0, 0.5, 0.0} to a class index (win/draw/loss)."""
    if result >= 0.75:
        return 0
    if result <= 0.25:
        return 2
    return 1


# ---------------------------------------------------------------------------
# The net + trainer — torch, imported lazily so the light path never pays for it.
# ---------------------------------------------------------------------------


@dataclass
class NetConfig:
    """Everything needed to reconstruct a trained net for inference.

    Stamped into the saved artifact alongside the weights so a load can reject a
    feature-layout mismatch loudly instead of silently mis-scoring.
    """

    board_features: int = N_BOARD_FEATURES
    rating_features: int = N_RATING_FEATURES
    board_hidden: int = 256
    trunk_hidden: int = 128
    dropout: float = 0.3
    feature_version: int = FEATURE_VERSION
    # Provenance for the report header / leakage guard — set by the trainer.
    train_month: Optional[str] = None
    n_train: int = 0
    epochs: int = 0


def _build_module(cfg: "NetConfig"):
    """Construct the bare :class:`torch.nn.Module` for ``cfg`` (lazy torch import)."""
    import torch
    from torch import nn

    class WDLNet(nn.Module):
        """Board branch + rating branch → fused trunk → 3-logit White-POV WDL head."""

        def __init__(self, c: NetConfig) -> None:
            super().__init__()
            self.board = nn.Sequential(
                nn.Linear(c.board_features, c.board_hidden),
                nn.ReLU(),
                nn.Dropout(c.dropout),
                nn.Linear(c.board_hidden, c.board_hidden),
                nn.ReLU(),
                nn.Dropout(c.dropout),
            )
            self.trunk = nn.Sequential(
                nn.Linear(c.board_hidden + c.rating_features, c.trunk_hidden),
                nn.ReLU(),
                nn.Dropout(c.dropout),
                nn.Linear(c.trunk_hidden, N_CLASSES),
            )

        def forward(self, board_x, rating_x):  # type: ignore[no-untyped-def]
            b = self.board(board_x)
            return self.trunk(torch.cat([b, rating_x], dim=1))

    return WDLNet(cfg)


def _encode_rows(rows: Sequence[PositionRow]) -> Tuple[List[List[float]], List[List[float]], List[int]]:
    """Vectorise rows into (board features, rating features, class labels).

    Skips rows with no FEN (a board model can't score them) — the caller decides
    whether that is an error.
    """
    boards: List[List[float]] = []
    ratings: List[List[float]] = []
    labels: List[int] = []
    for row in rows:
        if row.fen is None:
            continue
        boards.append(encode_board(row.fen))
        ratings.append(rating_features(row.white_elo, row.black_elo))
        labels.append(_white_result_class(row.result))
    return boards, ratings, labels


def train_wdl_net(
    rows: Sequence[PositionRow],
    *,
    cfg: Optional[NetConfig] = None,
    epochs: int = 8,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    train_month: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> "TrainedNet":
    """Fit the end-to-end WDL net on ``rows`` (each needs a ``fen``).

    A small, deterministic Adam loop over cross-entropy on the White-POV W/D/L label.
    Returns a :class:`TrainedNet` (module + config) ready to save or wrap.
    """
    import torch
    from torch import nn

    cfg = cfg or NetConfig()
    boards, ratings, labels = _encode_rows(rows)
    if not boards:
        raise ValueError("train_wdl_net got no rows with a FEN; rebuild with --with-fen")
    cfg.n_train = len(boards)
    cfg.epochs = epochs
    cfg.train_month = train_month

    torch.manual_seed(seed)
    Xb = torch.tensor(boards, dtype=torch.float32)
    Xr = torch.tensor(ratings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    module = _build_module(cfg)
    opt = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    n = Xb.shape[0]
    gen = torch.Generator().manual_seed(seed)

    module.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            logits = module(Xb[idx], Xr[idx])
            loss = loss_fn(logits, y[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        if log is not None:
            log(f"epoch {epoch + 1}/{epochs}  train_logloss={total / n:.4f}")

    module.eval()
    return TrainedNet(module=module, cfg=cfg)


@dataclass
class TrainedNet:
    """A fitted net plus its config — the savable / wrappable unit."""

    module: Any  # torch.nn.Module
    cfg: NetConfig

    def predict_white_wdl(self, fen: str, white_elo: int, black_elo: int) -> WDL:
        """White-POV WDL for one position (no grad). Terminal positions resolve directly."""
        import torch

        board = chess.Board(fen)
        if board.is_checkmate():
            # Side to move is mated; translate to White's POV.
            return WDL(0.0, 0.0, 1.0) if board.turn == chess.WHITE else WDL(1.0, 0.0, 0.0)
        if board.is_stalemate() or board.is_insufficient_material():
            return WDL(0.0, 1.0, 0.0)
        xb = torch.tensor([encode_board(fen)], dtype=torch.float32)
        xr = torch.tensor([rating_features(white_elo, black_elo)], dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(self.module(xb, xr), dim=1)[0].tolist()
        return WDL.from_unnormalized(probs[0], probs[1], probs[2])

    def save(self, path: str) -> None:
        """Serialise weights + config to ``path`` (a torch ``.pt`` archive)."""
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg.__dict__, "state_dict": self.module.state_dict()}, path)

    @classmethod
    def load(cls, path: str) -> "TrainedNet":
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = NetConfig(**blob["cfg"])
        if cfg.feature_version != FEATURE_VERSION:
            raise ValueError(
                f"artifact feature_version {cfg.feature_version} != current "
                f"{FEATURE_VERSION}; retrain the net"
            )
        module = _build_module(cfg)
        module.load_state_dict(blob["state_dict"])
        module.eval()
        return cls(module=module, cfg=cfg)


# ---------------------------------------------------------------------------
# EquityModel adapter — what the bar / CLI / validation harness consume.
# ---------------------------------------------------------------------------


class WdlNetModel(EquityModel):
    """The ``wdl-net`` equity model: end-to-end board → rating-conditioned WDL bar."""

    SOURCE = "wdl-net"

    def __init__(self, net: TrainedNet) -> None:
        self._net = net

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        white_wdl = self._net.predict_white_wdl(fen, white_elo, black_elo)
        is_white = chess.Board(fen).turn == chess.WHITE
        # The net predicts White-POV; render the side-to-move WDL for the triple, and
        # the White-POV bar comes back out identical via ``from_side_to_move``.
        stm_wdl = white_wdl if is_white else white_wdl.flipped()
        return Equity.from_side_to_move(stm_wdl, white_to_move=is_white, source=self.SOURCE)


ARTIFACT_NAME = "wdl_net.pt"


def default_artifact_path() -> Path:
    """Path to the committed ``wdl-net`` artifact inside the package tree."""
    return Path(__file__).resolve().parent / "artifacts" / ARTIFACT_NAME


def build_wdl_net_equity(path: Optional[str] = None) -> WdlNetModel:
    """Construct the ``wdl-net`` :class:`EquityModel` from a saved artifact."""
    artifact = path or str(default_artifact_path())
    return WdlNetModel(TrainedNet.load(artifact))
