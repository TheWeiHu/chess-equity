"""Tests for Approach D — the end-to-end board → WDL net (task 0013).

Two layers:

- the **board encoder** is pure ``chess`` (no torch), so its feature contract is
  tested in the light path here;
- the **net / trainer / adapter** need torch, so those tests ``importorskip`` it.

The tiny ``PositionRow`` lists below are illustrative FIXTURES for exercising the
training/inference mechanics — never presented as thesis evidence (the real
head-to-head lives in ``reports/wdl_net_real.md``, built from a real Lichess dump).
"""

from __future__ import annotations

import chess
import pytest

from chess_equity.data.schema import PositionRow
from chess_equity.wdl_net import (
    N_BOARD_FEATURES,
    N_BOARD_PLANES,
    encode_board,
    rating_features,
    _white_result_class,
)


# --- encoder: pure, light path -------------------------------------------------


def test_encode_board_length_and_startpos_pieces():
    v = encode_board(chess.Board().fen())
    assert len(v) == N_BOARD_FEATURES
    # 32 pieces on the start board -> exactly 32 hot plane cells.
    assert sum(v[:N_BOARD_PLANES]) == 32.0
    # state scalars: white to move + all four castling rights available.
    assert v[N_BOARD_PLANES:] == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_encode_board_side_to_move_scalar_flips():
    after_e4 = chess.Board()
    after_e4.push_san("e4")
    v = encode_board(after_e4.fen())
    # Black to move now -> the white_to_move scalar is 0.
    assert v[N_BOARD_PLANES] == 0.0


def test_encode_board_is_all_binary():
    v = encode_board(chess.Board().fen())
    assert set(v) <= {0.0, 1.0}


def test_rating_features_sign_and_scale():
    feats = rating_features(1900, 1500)
    avg_skill, delta, abs_delta = feats
    assert avg_skill == pytest.approx((1700 - 1500) / 400.0)
    assert delta == pytest.approx(400 / 400.0)  # White's edge, positive
    assert abs_delta == pytest.approx(abs(delta))
    # symmetric position reads as flat skill, no edge.
    assert rating_features(1500, 1500) == [0.0, 0.0, 0.0]


def test_white_result_class_buckets():
    assert _white_result_class(1.0) == 0  # White win
    assert _white_result_class(0.5) == 1  # draw
    assert _white_result_class(0.0) == 2  # White loss


# --- net / trainer / adapter: torch-gated --------------------------------------


def _fixture_rows(n: int = 48):
    """Illustrative fixtures (NOT evidence): White wins the even games, loses the odd."""
    fen = chess.Board().fen()
    rows = []
    for i in range(n):
        rows.append(
            PositionRow(
                cp_eval=0.0,
                white_elo=1700,
                black_elo=1500,
                ply=10,
                phase="middlegame",
                time_control="600+0",
                tc_bucket="rapid",
                clock_remaining=None,
                side_to_move="white",
                result=1.0 if i % 2 == 0 else 0.0,
                game_id=f"g{i}",
                fen=fen,
            )
        )
    return rows


def test_train_and_predict_shapes():
    pytest.importorskip("torch")
    from chess_equity.wdl_net import WdlNetModel, train_wdl_net

    net = train_wdl_net(_fixture_rows(), epochs=2, batch_size=16, seed=0)
    assert net.cfg.n_train == 48
    eq = WdlNetModel(net).evaluate(chess.Board().fen(), 1700, 1500)
    assert eq.source == "wdl-net"
    assert 0.0 <= eq.equity_white <= 100.0
    triple = (eq.wdl.p_win, eq.wdl.p_draw, eq.wdl.p_loss)
    assert sum(triple) == pytest.approx(1.0, abs=1e-5)


def test_save_load_roundtrip(tmp_path):
    pytest.importorskip("torch")
    from chess_equity.wdl_net import TrainedNet, WdlNetModel, train_wdl_net

    net = train_wdl_net(_fixture_rows(), epochs=2, batch_size=16, seed=0)
    path = tmp_path / "wdl_net.pt"
    net.save(str(path))
    reloaded = TrainedNet.load(str(path))
    fen = chess.Board().fen()
    a = WdlNetModel(net).evaluate(fen, 1700, 1500).equity_white
    b = WdlNetModel(reloaded).evaluate(fen, 1700, 1500).equity_white
    assert a == pytest.approx(b, abs=1e-4)


def test_predict_is_deterministic_eval_mode():
    """Dropout must be off at inference: the same input gives the same equity twice."""
    pytest.importorskip("torch")
    from chess_equity.wdl_net import WdlNetModel, train_wdl_net

    model = WdlNetModel(train_wdl_net(_fixture_rows(), epochs=2, batch_size=16, seed=0))
    fen = chess.Board().fen()
    assert model.evaluate(fen, 1700, 1500).equity_white == pytest.approx(
        model.evaluate(fen, 1700, 1500).equity_white
    )


def test_checkmate_resolves_without_net():
    """A terminal position is decided directly (the net's preprocessing isn't called)."""
    pytest.importorskip("torch")
    from chess_equity.wdl_net import WdlNetModel, train_wdl_net

    model = WdlNetModel(train_wdl_net(_fixture_rows(), epochs=1, batch_size=16, seed=0))
    # Fool's mate: White is checkmated, Black to move would be... here White is mated.
    mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    eq = model.evaluate(mated.fen(), 1500, 1500)
    # White (to move) is mated -> White-POV equity is 0.
    assert eq.equity_white == pytest.approx(0.0)


def test_rating_conditioning_changes_equity():
    """Flipping which side is far stronger must move the bar (the whole point)."""
    pytest.importorskip("torch")
    from chess_equity.wdl_net import WdlNetModel, train_wdl_net

    model = WdlNetModel(train_wdl_net(_fixture_rows(), epochs=3, batch_size=16, seed=0))
    fen = chess.Board().fen()
    strong_white = model.evaluate(fen, 2200, 1400).equity_white
    strong_black = model.evaluate(fen, 1400, 2200).equity_white
    assert strong_white != pytest.approx(strong_black)
