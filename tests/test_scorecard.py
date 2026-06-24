"""Tests for the single-game CLI scorecard (`chess-equity score`).

The scorecard is the CLI-first iteration loop for the thesis: feed one game, see the
embedded objective score, the rating-blind baseline, the model's rating-conditioned
equity, and the real result. These tests drive the *pure* builder over the offline
sample PGN with a fake model, so they need no engine, no dump, and no network.
"""

import io

import chess
import chess.pgn
import pytest

from chess_equity.adapters import EquityModel
from chess_equity.scorecard import (
    build_scorecard,
    build_scorecard_from_pgn,
    render_scorecard,
    render_scorecard_svg,
)
from chess_equity.types import WDL, Equity, lichess_win_percent


# --------------------------------------------------------------------------- #
# Fakes / fixtures
# --------------------------------------------------------------------------- #


class ConstantModel(EquityModel):
    """A model that always predicts a fixed White-POV equity (in [0, 100])."""

    def __init__(self, equity_white: float) -> None:
        self._equity_white = equity_white

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        board = chess.Board(fen)
        # WDL POV doesn't matter for these tests; we only assert on equity_white.
        return Equity(
            wdl=WDL(1.0, 0.0, 0.0),
            equity_white=self._equity_white,
            source="constant",
        )


class RatingEchoModel(EquityModel):
    """Predicts White's rating as a percentage — to prove ratings reach the model."""

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        return Equity(wdl=WDL(1.0, 0.0, 0.0), equity_white=float(white_elo), source="echo")


# A real Scholar's-mate PGN with embedded evals and a decisive result.
WHITE_WINS_PGN = """[Event "Test"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "1600"]
[BlackElo "1550"]

1. e4 { [%eval 0.2] } e5 { [%eval 0.15] } 2. Bc4 { [%eval 0.25] } Nc6 { [%eval 0.2] } 3. Qh5 { [%eval 0.1] } Nf6 { [%eval #1] } 4. Qxf7# 1-0
"""

UNFINISHED_PGN = """[Event "Test"]
[White "carol"]
[Black "dave"]
[Result "*"]
[WhiteElo "2000"]
[BlackElo "2010"]

1. d4 { [%eval 0.1] } d5 { [%eval 0.0] } *
"""

NO_ELO_PGN = """[Event "Test"]
[White "erin"]
[Black "frank"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1
"""


def _read(pgn_text: str) -> chess.pgn.Game:
    return chess.pgn.read_game(io.StringIO(pgn_text))


# --------------------------------------------------------------------------- #
# "Here is a game" — ratings, players, moves
# --------------------------------------------------------------------------- #


def test_ratings_come_from_headers():
    card = build_scorecard(_read(WHITE_WINS_PGN), ConstantModel(50.0))
    assert (card.white, card.black) == ("alice", "bob")
    assert (card.white_elo, card.black_elo) == (1600, 1550)


def test_rating_overrides_beat_headers_and_reach_the_model():
    card = build_scorecard(
        _read(WHITE_WINS_PGN), RatingEchoModel(), white_elo=900, black_elo=2400
    )
    assert (card.white_elo, card.black_elo) == (900, 2400)
    # The model echoed White's rating into every equity_white -> ratings really flow in.
    assert all(m.equity_white == 900.0 for m in card.moves)


def test_missing_elo_headers_default_to_1500():
    card = build_scorecard(_read(NO_ELO_PGN), ConstantModel(50.0))
    assert (card.white_elo, card.black_elo) == (1500, 1500)


def test_one_movescore_per_ply_in_order():
    card = build_scorecard(_read(WHITE_WINS_PGN), ConstantModel(50.0))
    assert [m.san for m in card.moves] == ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]
    assert [m.ply for m in card.moves] == [1, 2, 3, 4, 5, 6, 7]


# --------------------------------------------------------------------------- #
# "Here is the score" + "what's the real score" + "what are we predicting"
# --------------------------------------------------------------------------- #


def test_embedded_eval_becomes_white_pov_cp_and_blind_baseline():
    card = build_scorecard(_read(WHITE_WINS_PGN), ConstantModel(50.0))
    first = card.moves[0]  # 1. e4 { [%eval 0.2] } -> +20 cp White POV
    assert first.cp_white == pytest.approx(20.0)
    assert first.baseline_white == pytest.approx(lichess_win_percent(20.0))
    # The mate eval (#1) saturates the cp and pushes the blind baseline toward 100%.
    mate = card.moves[5]  # ...Nf6 { [%eval #1] }
    assert mate.cp_white == pytest.approx(9999.0)  # Mate(+1) at mate_score=10000
    assert mate.baseline_white > 99.0


def test_missing_eval_leaves_cp_and_baseline_none():
    card = build_scorecard(_read(WHITE_WINS_PGN), ConstantModel(50.0))
    last = card.moves[-1]  # 3. Qxf7# carries no [%eval]
    assert last.cp_white is None
    assert last.baseline_white is None


def test_real_score_from_result():
    assert build_scorecard(_read(WHITE_WINS_PGN), ConstantModel(50.0)).white_score == 1.0
    assert build_scorecard(_read(NO_ELO_PGN), ConstantModel(50.0)).white_score == 0.0
    assert build_scorecard(_read(UNFINISHED_PGN), ConstantModel(50.0)).white_score is None


# --------------------------------------------------------------------------- #
# Scoring: Brier of prediction vs realized outcome
# --------------------------------------------------------------------------- #


def test_equity_brier_is_mean_squared_error_vs_white_score():
    # White won (y=1). A model that always says 80% has error (0.8-1)^2 = 0.04 every ply.
    card = build_scorecard(_read(WHITE_WINS_PGN), ConstantModel(80.0))
    assert card.equity_brier == pytest.approx(0.04)


def test_a_confident_correct_model_beats_the_blind_baseline_here():
    # White wins; a 99%-White model should track the result better than the lukewarm
    # opening evals the blind baseline reads off the embedded cp.
    card = build_scorecard(_read(WHITE_WINS_PGN), ConstantModel(99.0))
    assert card.equity_brier < card.baseline_brier


def test_brier_is_none_for_unfinished_game():
    card = build_scorecard(_read(UNFINISHED_PGN), ConstantModel(50.0))
    assert card.equity_brier is None
    assert card.baseline_brier is None


def test_baseline_brier_none_when_no_embedded_evals():
    card = build_scorecard(_read(NO_ELO_PGN), ConstantModel(50.0))
    assert card.equity_brier is not None  # equity always predicted
    assert card.baseline_brier is None  # ...but no cp to derive the blind baseline


# --------------------------------------------------------------------------- #
# from_pgn helper + rendering
# --------------------------------------------------------------------------- #


def test_build_from_pgn_string():
    card = build_scorecard_from_pgn(WHITE_WINS_PGN, ConstantModel(50.0), model_name="x")
    assert card.model == "x"
    assert len(card.moves) == 7


def test_build_from_pgn_rejects_empty():
    with pytest.raises(ValueError):
        build_scorecard_from_pgn("   ", ConstantModel(50.0))


def test_render_answers_the_four_questions():
    card = build_scorecard(_read(WHITE_WINS_PGN), ConstantModel(99.0), model_name="wdl-a")
    text = "\n".join(render_scorecard(card))
    assert "alice (1600) vs bob (1550)" in text  # here is a game
    assert "predicting" in text and "wdl-a" in text  # what are we predicting
    assert "Qxf7#" in text  # the moves
    assert "real score: 1-0" in text  # what's the real score
    assert "equity" in text and "baseline" in text  # the head-to-head
    assert "validate" in text  # honest pointer to the real gate


def test_render_handles_unfinished_without_brier():
    card = build_scorecard(_read(UNFINISHED_PGN), ConstantModel(50.0))
    text = "\n".join(render_scorecard(card))
    assert "unfinished" in text
    assert "Brier" not in text  # no score line for an unfinished game


def test_sample_pgn_real_fixture_runs_end_to_end():
    """The committed offline sample PGN flows through the whole builder."""
    with open("data/sample/sample_games.pgn", encoding="utf-8") as fh:
        card = build_scorecard(chess.pgn.read_game(fh), ConstantModel(60.0))
    assert card.white == "alice" and card.result == "1-0"
    assert card.moves and card.moves[0].cp_white == pytest.approx(20.0)


# --------------------------------------------------------------------------- #
# Shareable SVG scorecard (task 0253)
# --------------------------------------------------------------------------- #


def _graded(pgn_game, model):
    """Grade a parsed game with the offline grader (no engine, no network)."""
    from chess_equity.grading import EquityGrader

    return EquityGrader(model).grade_game(pgn_game, 1600, 1550)


def test_scorecard_svg_is_wellformed_and_has_key_fields():
    """The SVG parses as XML and carries every acceptance field (over the sample fixture)."""
    import xml.dom.minidom as minidom

    from chess_equity.models import LichessBaselineModel

    with open("data/sample/sample_games.pgn", encoding="utf-8") as fh:
        game = chess.pgn.read_game(fh)
    model = LichessBaselineModel()  # offline material model — no engine
    card = build_scorecard(game, model, model_name="baseline")
    svg = render_scorecard_svg(card, _graded(_read_first("data/sample/sample_games.pgn"), model))

    # Standalone, well-formed SVG document (minidom raises on malformed XML).
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    minidom.parseString(svg)

    # Players + Elo, result, the two-bar thesis contrast, swing, accuracy, sparkline.
    assert "alice (1500)" in svg and "bob (1480)" in svg
    assert "result: 1-0" in svg
    assert "Objective (blind)" in svg and "Equity (rating-aware)" in svg
    assert "Biggest swing" in svg
    assert "Accuracy" in svg and "White" in svg and "Black" in svg
    assert "Equity trajectory" in svg


def test_scorecard_svg_escapes_player_names():
    """A player name with XML metacharacters can't break the SVG document."""
    import xml.dom.minidom as minidom

    pgn = WHITE_WINS_PGN.replace('"alice"', '"a<b>&\\"c"')
    game = _read(pgn)
    model = ConstantModel(55.0)
    card = build_scorecard(game, model)
    svg = render_scorecard_svg(card, _graded(_read(pgn), model))

    minidom.parseString(svg)  # would raise if the raw < & " leaked through
    assert "a<b>" not in svg and "&amp;" in svg


def test_scorecard_svg_shows_no_eval_when_pgn_carries_no_objective():
    """With no embedded [%eval], the objective bar reads 'no eval', not a bogus fill."""
    import xml.dom.minidom as minidom

    pgn = """[Event "Test"]
[White "carol"]
[Black "dave"]
[Result "1-0"]
[WhiteElo "1700"]
[BlackElo "1700"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""
    game = _read(pgn)
    model = ConstantModel(72.0)
    card = build_scorecard(game, model)
    svg = render_scorecard_svg(card, _graded(_read(pgn), model))

    minidom.parseString(svg)
    assert "no eval" in svg


def _read_first(path):
    with open(path, encoding="utf-8") as fh:
        return chess.pgn.read_game(fh)
