"""Tests for Δequity move grading (task 0008)."""

import io

import chess
import chess.pgn
import pytest

from chess_equity.adapters import EquityModel
from chess_equity.grading import (
    BASE_BANDS,
    EquityGrader,
    UniformPolicy,
    grade_label,
    scaled_bands,
)
from chess_equity.models import LichessBaselineModel
from chess_equity.types import WDL, Equity


# --------------------------------------------------------------------------- #
# UniformPolicy
# --------------------------------------------------------------------------- #


def test_uniform_policy_is_uniform_over_legal_moves():
    probs = UniformPolicy().move_probs(chess.STARTING_FEN, 1500)
    assert len(probs) == 20  # 20 legal opening moves
    assert all(p == pytest.approx(1 / 20) for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0)


def test_uniform_policy_empty_on_terminal_position():
    # Stalemate: no legal moves.
    stalemate = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    assert UniformPolicy().move_probs(stalemate, 1500) == {}


# --------------------------------------------------------------------------- #
# Bands / labels — rating-aware
# --------------------------------------------------------------------------- #


def test_scaled_bands_widen_at_lower_ratings():
    strong = scaled_bands(2000)
    weak = scaled_bands(800)
    assert strong == BASE_BANDS  # base bands at/above 2000
    # Lower rating widens the magnitude of every threshold.
    assert abs(weak[0][0]) > abs(strong[0][0])
    assert abs(weak[-1][0]) > abs(strong[-1][0])


def test_grade_label_thresholds():
    assert grade_label(20.0, 2000) == "brilliant"
    assert grade_label(5.0, 2000) == "good"
    assert grade_label(0.0, 2000) == "ok"
    assert grade_label(-5.0, 2000) == "inaccuracy"
    assert grade_label(-10.0, 2000) == "mistake"
    assert grade_label(-50.0, 2000) == "blunder"


# --------------------------------------------------------------------------- #
# EquityGrader with the real placeholder model
# --------------------------------------------------------------------------- #


def test_capturing_a_hanging_queen_grades_positive():
    # White rook on d1, Black queen hanging on d4. Rxd4 is far above the average
    # legal move, so it must grade POSITIVE vs peers — the whole point of 0008.
    fen = "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"
    grader = EquityGrader(LichessBaselineModel())
    grade = grader.grade_move(fen, chess.Move.from_uci("d1d4"), 1500, 1500)
    assert grade.grade_peer > 0  # beats the rating-typical mix
    assert grade.grade_best == pytest.approx(0.0)  # it IS the best move
    assert grade.label in ("good", "brilliant")


def test_best_move_grade_best_is_zero_others_negative():
    fen = "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"
    grader = EquityGrader(LichessBaselineModel())
    # A move that ignores the free queen leaves equity below the best -> grade_best < 0.
    weak = grader.grade_move(fen, chess.Move.from_uci("e1e2"), 1500, 1500)
    assert weak.grade_best < 0


def test_grade_move_rejects_illegal():
    grader = EquityGrader(LichessBaselineModel())
    with pytest.raises(ValueError):
        grader.grade_move(chess.STARTING_FEN, chess.Move.from_uci("e2e5"), 1500, 1500)


# --------------------------------------------------------------------------- #
# Flagship demo: a centipawn-LOSING move with a POSITIVE equity grade
# --------------------------------------------------------------------------- #


class _MockModel(EquityModel):
    """Equity decoupled from centipawns, to stage the trap case.

    One target position (after the 'trap' move) is given high equity but a *losing*
    centipawn score; every other resulting position is even. This is exactly the
    shape Maia-2 (0005) produces on real traps — a move a rating-peer opponent likely
    refutes wrongly. Here we hand-set the numbers to prove the grader surfaces it.
    """

    def __init__(self, trap_fen: str) -> None:
        self.trap_fen = trap_fen

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        if fen == self.trap_fen:
            # mover-POV equity 80%, but mover-POV cp = -100 (material lost).
            # Equity.cp is opponent-POV after the move, so store +100.
            wdl = WDL.from_unnormalized(0.75, 0.1, 0.15)
            return Equity(wdl=wdl, equity_white=80.0, source="mock", cp=100.0)
        wdl = WDL.from_unnormalized(0.45, 0.1, 0.45)
        return Equity(wdl=wdl, equity_white=50.0, source="mock", cp=0.0)


def test_cp_losing_move_can_have_positive_equity_grade():
    # Pick a real position and treat one legal move as the trap.
    fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    board = chess.Board(fen)
    trap_move = chess.Move.from_uci("e2e4")
    board.push(trap_move)
    trap_fen = board.fen()

    grader = EquityGrader(_MockModel(trap_fen))
    grade = grader.grade_move(fen, trap_move, 1500, 1500)

    # The headline: positive peer-relative grade despite a centipawn LOSS.
    assert grade.grade_peer > 0, "a move stronger than peers must score positive"
    assert grade.cp_loss is not None and grade.cp_loss > 0, "and it lost centipawns"
    assert grade.equity_after == pytest.approx(80.0)


# --------------------------------------------------------------------------- #
# grade_game over a PGN
# --------------------------------------------------------------------------- #


def test_grade_game_grades_every_move():
    pgn = "1. e4 e5 2. Nf3 Nc6 *"
    game = chess.pgn.read_game(io.StringIO(pgn))
    grades = EquityGrader(LichessBaselineModel()).grade_game(game, 1500, 1500)
    assert [g.ply for g in grades] == [1, 2, 3, 4]
    assert [g.san for g in grades] == ["e4", "e5", "Nf3", "Nc6"]
    # Every grade is JSON-friendly and labelled.
    for g in grades:
        assert g.label
        assert "grade_peer" in g.to_dict()


def test_grader_mover_pov_alternates():
    # After 1.e4, it's Black to move; grading 1...e5 must be from Black's POV.
    pgn = "1. e4 e5 *"
    game = chess.pgn.read_game(io.StringIO(pgn))
    grades = EquityGrader(LichessBaselineModel()).grade_game(game, 1500, 1500)
    assert grades[0].mover_white is True
    assert grades[1].mover_white is False


# --------------------------------------------------------------------------- #
# Per-side scoreline — caster accuracy-style summary (task 0200)
# --------------------------------------------------------------------------- #


def _sample_grades():
    """Grade the first game of the committed sample PGN (real fixture, baseline model)."""
    from pathlib import Path

    pgn = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_games.pgn"
    with open(pgn, encoding="utf-8") as fh:
        game = chess.pgn.read_game(fh)
    return EquityGrader(LichessBaselineModel()).grade_game(game, 1500, 1500)


def test_scoreline_counts_sum_to_move_count_per_side():
    from chess_equity.grading import GRADE_LABELS, scoreline

    grades = _sample_grades()
    line = scoreline(grades)
    for side, sl in (("white", line.white), ("black", line.black)):
        # Every label is present, and the per-label counts sum to the side's move count.
        assert set(sl.label_counts) == set(GRADE_LABELS)
        assert sum(sl.label_counts.values()) == sl.n_moves
        assert sl.n_moves == sum(1 for g in grades if g.mover_white == (side == "white"))
    # And both sides together account for every move exactly once.
    assert line.white.n_moves + line.black.n_moves == len(grades)


def test_scoreline_worst_move_is_min_delta_peer_per_side():
    from chess_equity.grading import scoreline

    grades = _sample_grades()
    line = scoreline(grades)
    for white, sl in ((True, line.white), (False, line.black)):
        side_moves = [g for g in grades if g.mover_white == white]
        assert sl.worst is not None
        # The worst move's drop is the minimum grade_peer over that side's moves.
        assert sl.worst.grade_peer == min(g.grade_peer for g in side_moves)
        # And its label matches the move it points at.
        assert sl.worst.label in {g.label for g in side_moves}


def test_scoreline_round_trips_to_json_dict():
    from chess_equity.grading import scoreline

    line = scoreline(_sample_grades())
    d = line.to_dict()
    assert set(d) == {"white", "black"}
    assert d["white"]["worst"]["san"]  # nested MoveGrade dict is JSON-friendly
    assert isinstance(d["white"]["mean_peer"], float)


# --------------------------------------------------------------------------- #
# Round leaderboard — accuracy ranking across a multi-game PGN (task 0207)
# --------------------------------------------------------------------------- #


def _round_games():
    """Grade the committed 2-game round fixture; returns ``[(white, black, grades)]``.

    The fixture has ``alice`` on BOTH boards (White on board 1, Black on board 2) so the
    pooling-across-boards path is exercised. Real per-move math, baseline model — offline.
    """
    from pathlib import Path

    from chess_equity.broadcast import _parse_elo, split_games

    pgn = Path(__file__).resolve().parents[1] / "data" / "sample" / "round_games.pgn"
    text = pgn.read_text(encoding="utf-8")
    grader = EquityGrader(LichessBaselineModel())
    games = []
    for game_pgn in split_games(text):
        game = chess.pgn.read_game(io.StringIO(game_pgn))
        we = _parse_elo(game.headers, "WhiteElo") or 1500
        be = _parse_elo(game.headers, "BlackElo") or 1500
        games.append(
            (game.headers["White"], game.headers["Black"], grader.grade_game(game, we, be))
        )
    return games


def test_round_leaderboard_pools_a_player_across_boards():
    from chess_equity.grading import round_leaderboard

    games = _round_games()
    scores = round_leaderboard(games)
    by_name = {s.name: s for s in scores}
    # alice, bob, carol — every distinct player gets exactly one row.
    assert set(by_name) == {"alice", "bob", "carol"}
    # alice played 4 moves on board 1 (White) + 4 on board 2 (Black) = 8, pooled.
    alice = by_name["alice"]
    assert alice.n_moves == 8
    # Pooled move count == her moves across BOTH games (White on one, Black on the other).
    expected = sum(
        sum(1 for g in grades if (g.mover_white == (w == "alice")))
        for (w, b, grades) in games
        if "alice" in (w, b)
    )
    assert alice.n_moves == expected


def test_round_leaderboard_row_invariants():
    from chess_equity.grading import ACCURATE_LABELS, GRADE_LABELS, round_leaderboard

    for s in round_leaderboard(_round_games()):
        # label_counts cover every grade label and sum to the player's move count.
        assert set(s.label_counts) == set(GRADE_LABELS)
        assert sum(s.label_counts.values()) == s.n_moves
        # accuracy == share of ok-or-better moves; blunder/mistake counts mirror labels.
        accurate = sum(s.label_counts[label] for label in ACCURATE_LABELS)
        assert s.accuracy == pytest.approx(100.0 * accurate / s.n_moves)
        assert s.blunders == s.label_counts["blunder"]
        assert s.mistakes == s.label_counts["mistake"]
        assert 0.0 <= s.accuracy <= 100.0


def test_round_leaderboard_is_ranked_deterministically():
    from chess_equity.grading import _leaderboard_rank_key, round_leaderboard

    scores = round_leaderboard(_round_games())
    # The list is sorted by the documented key (accuracy desc, mean_peer desc, …).
    keys = [_leaderboard_rank_key(s) for s in scores]
    assert keys == sorted(keys)


def test_round_leaderboard_round_trips_to_json_rows():
    from chess_equity.grading import round_leaderboard

    scores = round_leaderboard(_round_games())
    rows = [s.to_dict() for s in scores]
    for r in rows:
        assert set(r) >= {
            "name", "n_moves", "accuracy", "blunders", "mistakes", "mean_peer", "worst"
        }
        assert isinstance(r["accuracy"], float)
        # worst is a nested MoveGrade dict (or None for an empty player — not here).
        assert r["worst"] is None or r["worst"]["san"]


def test_round_leaderboard_render_has_a_row_per_player():
    from chess_equity.grading import render_leaderboard, round_leaderboard

    scores = round_leaderboard(_round_games())
    lines = render_leaderboard(scores)
    # header + separator + one row per player.
    assert len(lines) == len(scores) + 2
    assert "player" in lines[0] and "acc%" in lines[0]
    for s in scores:
        assert any(s.name in line for line in lines[2:])


def test_round_leaderboard_player_carries_rating():
    from chess_equity.grading import round_leaderboard

    scores = round_leaderboard(_round_games())
    # The fixture sets WhiteElo/BlackElo per board; every row carries a positive rating
    # equal to the modal mover_elo of that player's pooled moves.
    for s in scores:
        assert isinstance(s.rating, int) and s.rating > 0


def test_leaderboard_export_rows_schema_and_rank():
    import csv

    from chess_equity.grading import (
        LEADERBOARD_COLUMNS,
        LEADERBOARD_CSV_COLUMNS,
        leaderboard_export_rows,
        render_leaderboard_csv,
        round_leaderboard,
    )

    scores = round_leaderboard(_round_games())
    rows = leaderboard_export_rows(scores)
    # One row per player: the stable broadcast columns plus the nested phase breakdown.
    assert len(rows) == len(scores)
    for r in rows:
        assert set(r) == set(LEADERBOARD_COLUMNS) | {"phases"}
        assert isinstance(r["accuracy"], float) and 0.0 <= r["accuracy"] <= 100.0
        assert isinstance(r["avg_delta"], float)
        assert isinstance(r["rating"], int)
    # rank is 1-based and monotonic, matching the already-ranked order.
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert [r["player"] for r in rows] == [s.name for s in scores]

    # CSV exports the base columns + flattened phase columns, header-first and parseable.
    csv_text = render_leaderboard_csv(scores)
    parsed = list(csv.DictReader(csv_text.splitlines()))
    assert [list(p.keys()) for p in parsed][0] == LEADERBOARD_CSV_COLUMNS
    # The base columns are a stable prefix of the CSV schema (back-compat for consumers).
    assert LEADERBOARD_CSV_COLUMNS[: len(LEADERBOARD_COLUMNS)] == LEADERBOARD_COLUMNS
    assert len(parsed) == len(rows)
    assert [p["player"] for p in parsed] == [r["player"] for r in rows]


def test_position_phase_heuristic():
    from chess_equity.grading import position_phase

    # Opening: starting position, full material, move 1.
    assert position_phase(chess.Board()) == "opening"
    # Middlegame: still plenty of material but past the opening cutoff (fullmove 15).
    mid = chess.Board()
    mid.set_fen("r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 15")
    assert position_phase(mid) == "middlegame"
    # Endgame: K+R vs K+R — only 2 non-king/non-pawn pieces, well under the threshold.
    end = chess.Board()
    end.set_fen("8/5k2/8/8/8/3r4/5K2/3R4 w - - 0 40")
    assert position_phase(end) == "endgame"
    # Endgame wins over opening: few pieces even at a low move number.
    early_end = chess.Board()
    early_end.set_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 5")
    assert position_phase(early_end) == "endgame"


def test_phase_breakdown_sums_and_bounds():
    from chess_equity.grading import PHASES, round_leaderboard

    for s in round_leaderboard(_round_games()):
        # Every phase bucket is present and the per-phase move counts sum to the total.
        assert set(s.phases) == set(PHASES)
        assert sum(p["n_moves"] for p in s.phases.values()) == s.n_moves
        for stat in s.phases.values():
            assert set(stat) == {"n_moves", "accuracy", "avg_delta"}
            assert 0.0 <= stat["accuracy"] <= 100.0
            # An empty phase reports zeroed accuracy/avg_delta, not an error.
            if stat["n_moves"] == 0:
                assert stat["accuracy"] == 0.0 and stat["avg_delta"] == 0.0


def test_phase_breakdown_in_json_and_csv():
    import csv

    from chess_equity.grading import (
        PHASES,
        leaderboard_export_rows,
        render_leaderboard_csv,
        round_leaderboard,
    )

    scores = round_leaderboard(_round_games())
    # JSON export carries the nested per-phase breakdown.
    for r in leaderboard_export_rows(scores):
        assert set(r["phases"]) == set(PHASES)
    # CSV export carries flat per-phase columns whose move counts sum to n_moves.
    parsed = list(csv.DictReader(render_leaderboard_csv(scores).splitlines()))
    for p in parsed:
        phase_moves = sum(int(p[f"{phase}_moves"]) for phase in PHASES)
        assert phase_moves == int(p["n_moves"])
        for phase in PHASES:
            assert 0.0 <= float(p[f"{phase}_acc"]) <= 100.0
