"""Tests for the model-level equity cache and the precompute pass (task 0012).

The cache must be *transparent* — same result as the uncached model — while turning
repeated lookups into hits, surviving restarts via the on-disk JSON, and keeping two
models' evaluations of the same position apart. Precompute then drives it over a whole
game and reports the hit-rate / latency the task asks for.
"""

from __future__ import annotations

import chess
import pytest

from chess_equity.adapters import EquityModel
from chess_equity.cache import CachingEquityModel
from chess_equity.models import LichessBaselineModel
from chess_equity.precompute import precompute_game
from chess_equity.types import Equity, WDL

SAMPLE_PGN = (
    '[Event "T"]\n[White "a"]\n[Black "b"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0\n"
)


class CountingModel(EquityModel):
    """A model that records how many times the underlying evaluate ran."""

    SOURCE = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        self.calls += 1
        # Make the output depend on the inputs so we can assert transparency.
        e = 0.5 + (white_elo - black_elo) / 10000.0
        return Equity(
            wdl=WDL.from_unnormalized(e, 0.0, 1.0 - e),
            equity_white=100.0 * e,
            source=self.SOURCE,
            cp=float(len(fen)),
        )


# --- cache transparency + hit/miss --------------------------------------------

def test_cache_is_transparent():
    base = LichessBaselineModel()
    cached = CachingEquityModel(base)
    fen = chess.STARTING_FEN
    direct = base.evaluate(fen, 1600, 1400)
    via = cached.evaluate(fen, 1600, 1400)
    assert via.equity_white == pytest.approx(direct.equity_white)
    assert via.wdl.p_win == pytest.approx(direct.wdl.p_win)
    assert via.source == direct.source
    assert via.cp == direct.cp


def test_cache_counts_hits_and_misses():
    counter = CountingModel()
    cached = CachingEquityModel(counter)
    cached.evaluate(chess.STARTING_FEN, 1500, 1500)  # miss
    cached.evaluate(chess.STARTING_FEN, 1500, 1500)  # hit
    cached.evaluate(chess.STARTING_FEN, 1500, 1500)  # hit
    assert counter.calls == 1
    assert cached.misses == 1 and cached.hits == 2
    assert cached.hit_rate() == pytest.approx(2 / 3)


def test_cache_keys_on_ratings():
    counter = CountingModel()
    cached = CachingEquityModel(counter)
    cached.evaluate(chess.STARTING_FEN, 1500, 1500)
    cached.evaluate(chess.STARTING_FEN, 2000, 1500)  # different ratings -> miss
    assert counter.calls == 2


def test_hit_rate_zero_when_unused():
    assert CachingEquityModel(CountingModel()).hit_rate() == 0.0


# --- persistence ---------------------------------------------------------------

def test_cache_persists_to_disk(tmp_path):
    path = str(tmp_path / "cache.json")
    counter = CountingModel()
    warm = CachingEquityModel(counter, path=path)
    warm.evaluate(chess.STARTING_FEN, 1700, 1500)
    assert counter.calls == 1

    # A fresh model + fresh cache loaded from disk should serve the same key as a hit.
    counter2 = CountingModel()
    reloaded = CachingEquityModel(counter2, path=path)
    eq = reloaded.evaluate(chess.STARTING_FEN, 1700, 1500)
    assert counter2.calls == 0  # served from the persisted cache
    assert reloaded.hits == 1
    assert eq.equity_white == pytest.approx(100.0 * (0.5 + 200 / 10000.0))


# --- precompute ----------------------------------------------------------------

def test_precompute_covers_every_ply():
    result = precompute_game(LichessBaselineModel(), SAMPLE_PGN, white_elo=1500, black_elo=1480)
    # 7 half-moves -> 8 records (the start position plus one per move).
    assert len(result.plies) == 8
    assert result.plies[0].ply == 0 and result.plies[0].san is None
    assert result.plies[1].san == "e4"
    for p in result.plies:
        assert 0.0 <= p.equity_white <= 100.0
        assert p.p_win + p.p_draw + p.p_loss == pytest.approx(1.0)
    assert result.cache_misses == 8 and result.cache_hits == 0  # all cold the first pass


def test_precompute_uses_cache_on_rerun():
    cached = CachingEquityModel(LichessBaselineModel())
    precompute_game(cached, SAMPLE_PGN)
    misses_after_first = cached.misses
    precompute_game(cached, SAMPLE_PGN)  # same game again -> all hits
    assert cached.misses == misses_after_first
    assert cached.hits == misses_after_first


def test_precompute_to_dict_is_json_shaped():
    result = precompute_game(LichessBaselineModel(), SAMPLE_PGN)
    d = result.to_dict()
    assert set(d) >= {"source", "white_elo", "black_elo", "plies", "cache_hits", "cache_misses"}
    assert isinstance(d["plies"], list) and isinstance(d["plies"][0], dict)
    assert "equity_white" in d["plies"][0]


def test_precompute_rejects_empty_pgn():
    with pytest.raises(ValueError):
        precompute_game(LichessBaselineModel(), "   ")


# A position where White is in check with a single legal reply (Kg1); see task 0075.
FORCED_PGN = (
    '[SetUp "1"]\n'
    '[FEN "8/8/8/8/8/6k1/7r/7K w - - 0 1"]\n\n'
    "1. Kg1 *\n"
)


def test_precompute_flags_forced_moves():
    result = precompute_game(LichessBaselineModel(), FORCED_PGN)
    # ply 0 is the start position (no move led here) -> never forced.
    assert result.plies[0].forced is False
    # ply 1 is reached by Kg1, the only legal move from the start position -> forced.
    assert result.plies[1].san == "Kg1"
    assert result.plies[1].forced is True


def test_precompute_unforced_moves_are_not_flagged():
    # The sample game has normal opening choices, so no ply is forced.
    result = precompute_game(LichessBaselineModel(), SAMPLE_PGN)
    assert all(p.forced is False for p in result.plies)


def test_forced_flag_serialises():
    result = precompute_game(LichessBaselineModel(), FORCED_PGN)
    assert "forced" in result.to_dict()["plies"][1]
