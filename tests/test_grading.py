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


def test_round_leaderboard_sort_modes_pick_the_primary_key():
    from chess_equity.grading import (
        LEADERBOARD_SORTS,
        _leaderboard_rank_key,
        round_leaderboard,
    )

    games = _round_games()
    assert LEADERBOARD_SORTS == ("accuracy", "lead", "blunders")
    # Every mode sorts by its own rank key — and the same membership, just reordered.
    base = {s.name for s in round_leaderboard(games)}
    for sort in LEADERBOARD_SORTS:
        scores = round_leaderboard(games, sort=sort)
        assert {s.name for s in scores} == base
        keys = [_leaderboard_rank_key(s, sort) for s in scores]
        assert keys == sorted(keys)
    # 'lead' leads with mean Δpeer desc; 'blunders' leads with fewest blunders first.
    lead = round_leaderboard(games, sort="lead")
    assert [s.mean_peer for s in lead] == sorted(
        (s.mean_peer for s in lead), reverse=True
    )
    blun = round_leaderboard(games, sort="blunders")
    assert [s.blunders for s in blun] == sorted(s.blunders for s in blun)
    # Default is unchanged (accuracy) and equals an explicit accuracy sort.
    assert [s.name for s in round_leaderboard(games)] == [
        s.name for s in round_leaderboard(games, sort="accuracy")
    ]


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


def test_round_leaderboard_render_shows_worst_move():
    from chess_equity.grading import _worst_cell, render_leaderboard, round_leaderboard

    scores = round_leaderboard(_round_games())
    lines = render_leaderboard(scores)
    assert "worst" in lines[0]
    # Each player's row carries their worst-move cell (SAN Δpeer, or '-' when none).
    for i, s in enumerate(scores):
        row = lines[2 + i]
        cell = _worst_cell(s.worst)
        assert cell in row
        if s.worst is not None:
            assert s.worst.san in row  # the SAN of the biggest drop is rendered


def test_worst_cell_formats_san_and_delta_or_dash():
    from chess_equity.grading import _worst_cell, round_leaderboard

    assert _worst_cell(None) == "-"
    # A real worst move renders as "SAN Δpeer" with a signed, 1-decimal delta.
    worst = next(s.worst for s in round_leaderboard(_round_games()) if s.worst is not None)
    cell = _worst_cell(worst)
    assert cell.startswith(worst.san + " ")
    assert cell.endswith(f"{worst.grade_peer:+.1f}")


def test_round_leaderboard_player_carries_rating():
    from chess_equity.grading import round_leaderboard

    scores = round_leaderboard(_round_games())
    # The fixture sets WhiteElo/BlackElo per board; every row carries a positive rating
    # equal to the modal mover_elo of that player's pooled moves.
    for s in scores:
        assert isinstance(s.rating, int) and s.rating > 0


# --------------------------------------------------------------------------- #
# Min-moves qualification floor — a cameo can't top the board (task 0227)
# --------------------------------------------------------------------------- #


def _mg(label, mover_white, grade_peer=0.0, ply=1):
    """A minimal MoveGrade fixture for pure-function leaderboard tests (not evidence).

    Only the fields the leaderboard math reads (label, mover_white, mover_elo, grade_peer,
    phase) carry meaning here; the rest are filler so the frozen dataclass is constructible.
    """
    from chess_equity.grading import MoveGrade

    return MoveGrade(
        ply=ply, san="e4", uci="e2e4", mover_white=mover_white, mover_elo=1500,
        phase="opening", equity_after=50.0, expected_equity=50.0 - grade_peer,
        equity_best=50.0, grade_peer=grade_peer, grade_best=0.0, label=label,
        best_uci="e2e4", cp_loss=0.0,
    )


def _cameo_round():
    """One board: a 1-move 100%-accuracy cameo (White) vs a full, lower-accuracy player."""
    cameo = [_mg("good", mover_white=True, grade_peer=5.0)]             # 100% acc, 1 move
    regular = [_mg("good", mover_white=False, grade_peer=2.0) for _ in range(6)]
    regular += [_mg("mistake", mover_white=False, grade_peer=-3.0) for _ in range(2)]  # 75%
    return [("cameo", "regular", cameo + regular)]


def test_round_leaderboard_min_moves_keeps_cameo_off_the_top():
    from chess_equity.grading import round_leaderboard

    scores = round_leaderboard(_cameo_round(), min_moves=5)
    by_name = {s.name: s for s in scores}
    # The cameo is the most accurate row, yet it must NOT rank first — it's unqualified.
    assert by_name["cameo"].accuracy > by_name["regular"].accuracy
    assert by_name["cameo"].n_moves == 1 and not by_name["cameo"].qualified
    assert by_name["regular"].qualified
    assert scores[0].name == "regular"   # a qualified player tops the board
    assert scores[-1].name == "cameo"    # the cameo is ranked below, never interleaved above


def test_round_leaderboard_min_moves_zero_is_a_no_op():
    from chess_equity.grading import round_leaderboard

    # The library default (0) imposes no floor: everyone qualifies and the cameo's higher
    # accuracy tops the board, exactly the pre-0227 behaviour.
    scores = round_leaderboard(_cameo_round())  # min_moves defaults to 0
    assert all(s.qualified for s in scores)
    assert scores[0].name == "cameo"


def test_min_moves_qualified_flag_in_export_and_render():
    from chess_equity.grading import (
        LEADERBOARD_COLUMNS,
        leaderboard_export_rows,
        render_leaderboard,
        round_leaderboard,
    )

    scores = round_leaderboard(_cameo_round(), min_moves=5)
    # JSON/CSV export carries the `qualified` bool (appended to the stable column list).
    assert "qualified" in LEADERBOARD_COLUMNS
    rows = {r["player"]: r for r in leaderboard_export_rows(scores)}
    assert rows["regular"]["qualified"] is True
    assert rows["cameo"]["qualified"] is False

    # Text render lists qualified players numbered, then an `unqualified` section.
    lines = render_leaderboard(scores, min_moves=5)
    body = lines[2:]  # drop header + separator
    heading_idx = next(i for i, l in enumerate(body) if l.startswith("unqualified"))
    assert "(< 5 moves)" in body[heading_idx]
    # The cameo sits inside the unqualified section, not above it.
    assert any("cameo" in l for l in body[heading_idx + 1:])
    assert all("cameo" not in l for l in body[:heading_idx])
    # Qualified rows are numbered 1..k; the cameo never gets rank "1".
    assert body[0].split()[0] == "1" and "regular" in body[0]


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


def test_leaderboard_md_is_a_valid_table_ordered_by_sort():
    from chess_equity.grading import (
        LEADERBOARD_MD_HEADERS,
        render_leaderboard_md,
        round_leaderboard,
    )

    for sort in ("accuracy", "lead", "blunders"):
        scores = round_leaderboard(_round_games(), sort=sort)
        md = render_leaderboard_md(scores)
        lines = md.splitlines()
        # Header row + separator row + one row per player; trailing newline present.
        assert md.endswith("\n")
        assert len(lines) == len(scores) + 2
        # Every line is a pipe-delimited markdown row with the same column count.
        ncols = len(LEADERBOARD_MD_HEADERS)
        for line in lines:
            assert line.startswith("| ") and line.endswith(" |")
            assert line.count("|") == ncols + 1
        # Header names and a GitHub-style separator row of dashes.
        header_cells = [c.strip() for c in lines[0].strip("| ").split(" | ")]
        assert header_cells == LEADERBOARD_MD_HEADERS
        assert set("".join(lines[1].split())) <= set("|-")
        # Rows are in ranked order: first column is 1..N, players match `scores`.
        body = lines[2:]
        ranks = [row.strip("| ").split(" | ")[0].strip() for row in body]
        assert ranks == [str(i) for i in range(1, len(scores) + 1)]
        players = [row.strip("| ").split(" | ")[1].strip() for row in body]
        assert players == [s.name for s in scores]


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


# --------------------------------------------------------------------------- #
# Equity-swing sparkline (task 0239)
# --------------------------------------------------------------------------- #


def _grade(equity_after, mover_white=True, ply=1):
    """A minimal MoveGrade fixture for the pure sparkline function."""
    from chess_equity.grading import MoveGrade

    return MoveGrade(
        ply=ply, san="e4", uci="e2e4", mover_white=mover_white, mover_elo=1500,
        phase="opening", equity_after=equity_after, expected_equity=50.0,
        equity_best=equity_after, grade_peer=0.0, grade_best=0.0,
        label="ok", best_uci="e2e4", cp_loss=0.0,
    )


def test_sparkline_has_one_block_per_graded_ply():
    from chess_equity.grading import equity_sparkline

    pgn = "1. e4 e5 2. Nf3 Nc6 *"
    game = chess.pgn.read_game(io.StringIO(pgn))
    grades = EquityGrader(LichessBaselineModel()).grade_game(game, 1500, 1500)
    spark = equity_sparkline(grades)
    assert len(spark) == len(grades) == 4


def test_sparkline_rises_for_a_winning_white_trajectory():
    from chess_equity.grading import SPARK_BLOCKS, equity_sparkline

    # White equity climbs 5 → 50 → 95 → 100: blocks must be non-decreasing and top out.
    grades = [_grade(e, mover_white=True, ply=i)
              for i, e in enumerate([5.0, 50.0, 95.0, 100.0], start=1)]
    spark = equity_sparkline(grades)
    assert len(spark) == 4
    idx = [SPARK_BLOCKS.index(c) for c in spark]
    assert idx == sorted(idx)              # non-decreasing
    assert spark[0] == SPARK_BLOCKS[0]     # near-empty board floor
    assert spark[-1] == SPARK_BLOCKS[-1]   # full block when White is fully winning


def test_sparkline_is_white_pov_for_black_movers():
    from chess_equity.grading import SPARK_BLOCKS, equity_sparkline

    # A Black mover sitting on equity_after=90 (Black winning) is White-POV ~10 → low block.
    spark = equity_sparkline([_grade(90.0, mover_white=False)])
    assert spark == SPARK_BLOCKS[0]


# --------------------------------------------------------------------------- #
# Graphical equity-trajectory SVG (task 0242)
# --------------------------------------------------------------------------- #


def test_trajectory_svg_has_one_polyline_point_per_graded_ply():
    import re

    from chess_equity.grading import equity_trajectory_svg

    pgn = "1. e4 e5 2. Nf3 Nc6 *"
    game = chess.pgn.read_game(io.StringIO(pgn))
    grades = EquityGrader(LichessBaselineModel()).grade_game(game, 1500, 1500)
    svg = equity_trajectory_svg(grades)
    # Standalone SVG document.
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    # The data polyline has exactly one "x,y" point per graded ply.
    poly = re.search(r'<polyline points="([^"]*)"', svg)
    assert poly is not None
    points = poly.group(1).split()
    assert len(points) == len(grades) == 4
    for p in points:
        x, _, y = p.partition(",")
        float(x), float(y)  # well-formed coordinates


def test_trajectory_svg_draws_the_50_percent_midline():
    from chess_equity.grading import equity_trajectory_svg

    svg = equity_trajectory_svg([_grade(50.0)])
    # The midline is a dashed horizontal <line>, labelled 50%.
    assert "stroke-dasharray" in svg
    assert ">50%<" in svg
    # A 50% equity point sits exactly on the midline y.
    import re

    line = re.search(r'<line x1="[^"]*" y1="([0-9.]+)" x2="[^"]*" y2="([0-9.]+)"', svg)
    assert line is not None and line.group(1) == line.group(2)


def test_trajectory_svg_rejects_empty_series():
    import pytest

    from chess_equity.grading import equity_trajectory_svg

    with pytest.raises(ValueError):
        equity_trajectory_svg([])
