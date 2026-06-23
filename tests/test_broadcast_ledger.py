"""Per-game equity ledger CSV (``broadcast --ledger``, task 0204).

``grade --annotate-pgn`` (task 0197) exports the equity bar back into a chess GUI; this
is the flat tabular counterpart for spreadsheets / post-show graphics: ``broadcast
--ledger out.csv`` replays a finished local PGN and writes one row per published move
(ply, side, san, equity, delta_equity, grade, drama label/score, clocks) under a header.
It reuses the existing :class:`MoveEvent` fields — no live feed, no extra model calls.

Fixture: ``data/sample/sample_games.pgn`` — the sanctioned offline-smoke fixture
(illustrative, not evidence; see project CLAUDE.md).

The pure-function tests drive an *engine-free* :class:`LichessBaselineModel` so they are
deterministic regardless of whether Stockfish is installed; the CLI test exercises the
flag wiring and pins the header + row-count acceptance (row count is model-independent —
it equals the number of half-moves replayed).
"""
import csv
import io
import os

import chess
import chess.pgn

from chess_equity.broadcast import (
    LEDGER_COLUMNS,
    BroadcastIngestor,
    LocalPgnFeed,
    ledger_row,
    write_ledger,
)
from chess_equity.cli import main
from chess_equity.models import LichessBaselineModel

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PGN = os.path.join(HERE, "..", "data", "sample", "sample_games.pgn")


def _total_half_moves():
    """Total mainline half-moves across every game in the sample (the ledger row count)."""
    total = 0
    with open(SAMPLE_PGN, "r", encoding="utf-8") as fh:
        text = fh.read()
    stream = io.StringIO(text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        total += sum(1 for _ in game.mainline_moves())
    return total


def _replay_events():
    """Replay the sample's first game through the engine-free baseline model."""
    with open(SAMPLE_PGN, "r", encoding="utf-8") as fh:
        text = fh.read()
    feed = LocalPgnFeed(text)
    ingestor = BroadcastIngestor(
        feed, LichessBaselineModel(), white_elo=1800, black_elo=1800
    )
    events = []
    while True:
        snap = feed.poll()
        if snap is None:
            break
        events.extend(ingestor.ingest_snapshot(snap))
    return events


def test_write_ledger_header_and_one_row_per_move():
    events = _replay_events()
    assert events  # the fixture has moves

    buf = io.StringIO()
    rows = write_ledger(events, buf)
    assert rows == len(events)

    reader = list(csv.reader(io.StringIO(buf.getvalue())))
    assert reader[0] == LEDGER_COLUMNS  # header columns, exact order
    assert len(reader) == len(events) + 1  # header + one data row per graded move


def test_ledger_row_fields_track_the_event():
    events = _replay_events()
    sans = [e.san for e in events]

    rows = [ledger_row(e) for e in events]
    # Every move is graded on the baseline, so no grade cell is blank, and san/ply line up.
    for event, row in zip(events, rows):
        assert row["san"] == event.san
        assert row["ply"] == event.ply
        assert row["grade"], row
        # Mover side: in the post-move FEN the side to move is the opponent.
        assert row["side"] == ("white" if not event.white_to_move else "black")
        # Model attribution (task 0224): the ledger stamps which equity model
        # produced the row, sourced from event.source. The engine-free replay uses
        # the baseline model, so every row names it.
        assert row["model"] == event.source == "LichessBaselineModel"
    assert [r["san"] for r in rows] == sans

    # The scholar's-mate finisher is a clutch swing → its drama columns are populated.
    last = rows[-1]
    assert last["drama_label"] == "clutch"
    assert last["drama_score"] != ""


def test_cli_ledger_writes_csv(tmp_path):
    out = tmp_path / "ledger.csv"
    rc = main(
        [
            "broadcast",
            "--pgn",
            SAMPLE_PGN,
            "--ledger",
            str(out),
            "--white-elo",
            "1800",
            "--black-elo",
            "1800",
            "--interval",
            "0",
        ]
    )
    assert rc == 0
    assert out.exists()

    reader = list(csv.reader(out.open(encoding="utf-8")))
    assert reader[0] == LEDGER_COLUMNS
    # Row count == graded moves == every half-move replayed (model-independent).
    assert len(reader) - 1 == _total_half_moves()


def test_cli_ledger_requires_pgn(capsys):
    rc = main(["broadcast", "--round", "abcd1234", "--ledger", "out.csv"])
    assert rc == 2
    assert "requires --pgn" in capsys.readouterr().err
