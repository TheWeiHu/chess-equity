"""Tests for personal (per-player) equity — task 0014.

All profile-building logic is exercised offline from PGN text; the one network path
(:func:`fetch_user_games`) is tested through the monkeypatchable ``_urlopen`` seam.
"""

from __future__ import annotations

import io
import json

import chess

from chess_equity import personal
from chess_equity.adapters import EquityModel
from chess_equity.cli import main
from chess_equity.personal import (
    PhaseStats,
    PlayerProfile,
    PersonalEquityModel,
    build_profile,
    update_profile,
)
from chess_equity.types import WDL, Equity

# Ruy Lopez opening; White ("Alice") blunders on move 3 (eval crashes 0.2 -> -3.8).
# Alice's white moves: e4 (loss 0), Nf3 (loss 0), Bb5 (loss 400 = blunder), Ba4 (loss 20).
ALICE_GAME = """[Event "Test"]
[Site "https://lichess.org/abcd1234"]
[White "Alice"]
[Black "Bob"]
[WhiteElo "1600"]
[BlackElo "1600"]
[Result "1-0"]

1. e4 { [%eval 0.2] } e5 { [%eval 0.1] } 2. Nf3 { [%eval 0.3] } Nc6 { [%eval 0.2] }
3. Bb5 { [%eval -3.8] } a6 { [%eval -3.7] } 4. Ba4 { [%eval -3.9] } Nf6 { [%eval -4.0] } 1-0
"""

# A game Alice is not in — must be ignored entirely.
OTHER_GAME = """[Event "Test"]
[White "Carol"]
[Black "Dave"]
[WhiteElo "1800"]
[BlackElo "1800"]
[Result "1-0"]

1. e4 { [%eval 0.2] } e5 { [%eval 0.1] } 1-0
"""


class _RatingSensitiveBase(EquityModel):
    """A toy model whose equity moves with the rating delta, so offsets are observable."""

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        edge = (white_elo - black_elo) / 4000.0
        wdl = WDL.from_unnormalized(0.5 + edge, 0.0, 0.5 - edge)
        return Equity.from_side_to_move(
            wdl, white_to_move=chess.Board(fen).turn == chess.WHITE, source="fake"
        )


def test_phase_stats_aggregation():
    s = PhaseStats()
    s.add(0.0)
    s.add(100.0)
    s.add(400.0)  # blunder (>= BLUNDER_CP)
    assert s.n_moves == 3
    assert s.avg_cp_loss == 500.0 / 3
    assert s.n_blunders == 1
    assert s.blunder_rate == 1 / 3


def test_phase_offset_signs():
    p = PlayerProfile("x")
    # Opening strong (low loss), endgame weak (high loss), middlegame absent.
    for _ in range(4):
        p._phase("opening").add(0.0)
        p._phase("endgame").add(200.0)
    assert p.overall_acpl == 100.0
    assert p.phase_offset("opening") > 0  # better than own average
    assert p.phase_offset("endgame") < 0  # worse than own average
    assert p.phase_offset("middlegame") == 0.0  # no data -> no adjustment


def test_build_profile_attributes_only_target_moves():
    profile = build_profile(io.StringIO(ALICE_GAME), "Alice")
    assert profile.n_games == 1
    assert profile.rating == 1600
    opening = profile.phases["opening"]
    assert opening.n_moves == 4  # only Alice's (white) moves, not Bob's
    assert opening.n_blunders == 1
    assert opening.avg_cp_loss == 105.0  # (0 + 0 + 400 + 20) / 4
    assert opening.blunder_rate == 0.25


def test_build_profile_is_case_insensitive():
    profile = build_profile(io.StringIO(ALICE_GAME), "alice")
    assert profile.n_games == 1
    assert profile.total_moves == 4


def test_build_profile_ignores_unrelated_games():
    profile = build_profile(io.StringIO(OTHER_GAME), "Alice")
    assert profile.n_games == 0
    assert profile.total_moves == 0


def test_update_profile_returns_membership():
    p = PlayerProfile("Bob")
    game = chess.pgn.read_game(io.StringIO(ALICE_GAME))
    assert update_profile(p, game) is True  # Bob is Black in the game
    assert p.total_moves == 4  # Bob's four black moves (e5, Nc6, a6, Nf6)


def test_personal_model_shifts_equity_and_tags_source():
    base = _RatingSensitiveBase()
    weak_opening = PlayerProfile("w")
    for _ in range(4):
        weak_opening._phase("opening").add(200.0)
        weak_opening._phase("endgame").add(0.0)
    # Opening is this player's weak phase -> negative offset -> lower effective rating.
    assert weak_opening.phase_offset("opening") < 0

    fen = chess.STARTING_FEN
    band = base.evaluate(fen, 1600, 1600)
    personalized = PersonalEquityModel(base, white_profile=weak_opening).evaluate(fen, 1600, 1600)
    assert personalized.equity_white < band.equity_white
    assert personalized.source.endswith("+personal")


def test_same_position_two_opponents_differ():
    """Acceptance demo: one position, two profiled opponents -> two equities."""
    base = _RatingSensitiveBase()

    weak_bob = PlayerProfile("weak")
    strong_bob = PlayerProfile("strong")
    for _ in range(4):
        weak_bob._phase("opening").add(200.0)
        weak_bob._phase("middlegame").add(0.0)
        strong_bob._phase("opening").add(0.0)
        strong_bob._phase("middlegame").add(200.0)
    assert weak_bob.phase_offset("opening") < 0 < strong_bob.phase_offset("opening")

    fen = chess.STARTING_FEN
    vs_weak = PersonalEquityModel(base, black_profile=weak_bob).evaluate(fen, 1600, 1600)
    vs_strong = PersonalEquityModel(base, black_profile=strong_bob).evaluate(fen, 1600, 1600)
    # Facing the weaker opponent, White's equity is higher.
    assert vs_weak.equity_white > vs_strong.equity_white


def test_fetch_user_games_uses_seam(monkeypatch):
    captured = {}

    class _FakeResp:
        def read(self):
            return b"[Event \"x\"]\n"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_open(req):
        captured["url"] = req.full_url
        captured["accept"] = req.get_header("Accept")
        return _FakeResp()

    monkeypatch.setattr(personal, "_urlopen", fake_open)
    text = personal.fetch_user_games("Alice", max_games=7)
    assert "lichess.org/api/games/user/Alice" in captured["url"]
    assert "evals=true" in captured["url"]
    assert "max=7" in captured["url"]
    assert captured["accept"] == "application/x-chess-pgn"
    assert text.startswith("[Event")


def test_fetch_user_games_builds_profile(monkeypatch):
    class _FakeResp:
        def read(self):
            return ALICE_GAME.encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(personal, "_urlopen", lambda req: _FakeResp())
    profile = personal.build_profile_for_user("Alice", max_games=10)
    assert profile.n_games == 1
    assert profile.phases["opening"].n_blunders == 1


def _write(tmp_path, text):
    p = tmp_path / "game.pgn"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_cli_personal_pgn(tmp_path, capsys):
    rc = main(["personal", "--pgn", _write(tmp_path, ALICE_GAME), "--name", "Alice"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Alice" in out
    assert "opening" in out


def test_cli_personal_json(tmp_path, capsys):
    rc = main(["personal", "--pgn", _write(tmp_path, ALICE_GAME), "--name", "Alice", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["username"] == "Alice"
    assert data["games"] == 1
    phases = {row["phase"]: row for row in data["phases"]}
    assert phases["opening"]["blunder_rate"] == 0.25


def test_cli_personal_demo(tmp_path, capsys):
    rc = main(["personal", "--pgn", _write(tmp_path, ALICE_GAME), "--name", "Alice", "--demo"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "demo" in out
    assert "personalized" in out


def test_cli_personal_requires_source(capsys):
    rc = main(["personal"])
    assert rc == 1
    assert "error" in capsys.readouterr().err
