"""Turn a Lichess PGN file into a committed tabular dataset, and load it back.

This is the thin I/O shell around :func:`chess_equity.data.pgn.iter_rows`:

- :func:`open_pgn` transparently decompresses ``.zst`` dumps (the format Lichess
  ships) so the parser only ever sees text, and never holds a whole dump in memory.
- :func:`build_dataset` streams rows out to CSV (always — stdlib, diff-friendly,
  what the committed sample uses) or Parquet (``--format parquet``, needs ``pyarrow``,
  for the full-scale builds 0004 trains on).
- :func:`load_rows` / :func:`load_dataframe` are the import surface other tasks use,
  so they never re-implement the schema.

CSV is the default precisely so the green-gate (pytest, no heavy deps) stays cheap;
Parquet stays an opt-in extra for when the data gets big.
"""

from __future__ import annotations

import contextlib
import csv
from pathlib import Path
from typing import IO, Iterable, Iterator, List, Optional, Sequence

from chess_equity.data.pgn import iter_rows
from chess_equity.data.schema import PositionRow, columns as schema_columns, rating_bucket

LICHESS_DB_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


@contextlib.contextmanager
def open_pgn(path: str) -> Iterator[IO[str]]:
    """Open a PGN file as a text stream, decompressing ``.zst`` on the fly.

    Streams in both cases — a multi-GB ``.zst`` is decompressed incrementally, never
    expanded to disk or memory.
    """
    p = Path(path)
    if p.suffix == ".zst":
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "reading .zst dumps needs the 'data' extra: pip install 'chess-equity[data]'"
            ) from exc
        import io

        fh = p.open("rb")
        try:
            reader = zstandard.ZstdDecompressor().stream_reader(fh)
            yield io.TextIOWrapper(reader, encoding="utf-8")
        finally:
            fh.close()
    else:
        with p.open("r", encoding="utf-8") as fh:
            yield fh


def month_url(month: str) -> str:
    """The canonical Lichess dump URL for a ``YYYY-MM`` month (e.g. ``2026-05``)."""
    return LICHESS_DB_URL.format(month=month)


def _write_csv(rows: Iterable[PositionRow], out: Path, cols: Sequence[str]) -> int:
    count = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        # ``extrasaction="ignore"`` drops the ``fen`` key when it is not selected, so
        # the row's full ``as_dict()`` can be written against either column set.
        writer = csv.DictWriter(fh, fieldnames=list(cols), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
            count += 1
    return count


def _write_parquet(rows: Iterable[PositionRow], out: Path, cols: Sequence[str]) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "writing Parquet needs the 'data' extra: pip install 'chess-equity[data]'"
        ) from exc
    materialised: List[PositionRow] = list(rows)
    table = pa.table({col: [getattr(r, col) for r in materialised] for col in cols})
    pq.write_table(table, out)
    return len(materialised)


def _write_partitioned(
    rows: Iterable[PositionRow], root: Path, cols: Sequence[str], fmt: str
) -> int:
    """Write a hive-partitioned tree ``root/tc_bucket=<b>/rating_bucket=<rb>/part.<fmt>``.

    Groups rows by (``tc_bucket``, rating band) so 0004/0009 can read just the slices
    they need. Rows are grouped in memory — fine for the committed sample and sampled
    builds; a full-dump streaming writer (pyarrow ``write_dataset``) is a follow-up.
    The partition keys live in the directory names; each row still carries
    ``tc_bucket`` and the ratings, so reads reconstruct identical rows.
    """
    groups: dict = {}
    for row in rows:
        key = (row.tc_bucket, rating_bucket(row.white_elo, row.black_elo))
        groups.setdefault(key, []).append(row)

    count = 0
    for (tcb, rb), group in groups.items():
        part_dir = root / f"tc_bucket={tcb}" / f"rating_bucket={rb}"
        part_dir.mkdir(parents=True, exist_ok=True)
        target = part_dir / f"part.{fmt}"
        if fmt == "csv":
            count += _write_csv(group, target, cols)
        else:
            count += _write_parquet(group, target, cols)
    return count


def build_dataset(
    pgn_path: str,
    out_dir: str,
    *,
    sample: Optional[int] = None,
    fmt: str = "csv",
    name: str = "dataset",
    include_fen: bool = False,
    partition: bool = False,
) -> Path:
    """Parse ``pgn_path`` into ``out_dir/<name>.<fmt>`` and return the written path.

    ``sample`` caps the row count (for the committed fixture / quick runs); ``None``
    consumes the whole file. ``fmt`` is ``"csv"`` or ``"parquet"``. ``include_fen``
    appends a ``fen`` column so board models (Maia, 0005) can be validated in 0009 —
    off by default because it ~triples row size.

    With ``partition=True`` the output is a hive-partitioned **directory**
    ``out_dir/<name>/tc_bucket=…/rating_bucket=…/part.<fmt>`` (and that dir is
    returned) so 0004/0009 can read only the rating/time-control slices they need;
    :func:`load_rows` reads such a directory transparently.
    """
    if fmt not in ("csv", "parquet"):
        raise ValueError(f"unknown format {fmt!r} (expected 'csv' or 'parquet')")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cols = schema_columns(include_fen=include_fen)
    with open_pgn(pgn_path) as handle:
        rows = iter_rows(handle, limit=sample, include_fen=include_fen)
        if partition:
            target = out_path / name
            target.mkdir(parents=True, exist_ok=True)
            _write_partitioned(rows, target, cols, fmt)
            return target
        target = out_path / f"{name}.{fmt}"
        if fmt == "csv":
            _write_csv(rows, target, cols)
        else:
            _write_parquet(rows, target, cols)
    return target


def _coerce_row(record: dict) -> PositionRow:
    clk = record.get("clock_remaining")
    # ``fen`` is optional: datasets built before it existed (or without it) have no
    # such key, so a missing/empty value loads back as ``None``.
    fen = record.get("fen")
    # ``game_id`` (task 0030) is keyed, not positional, so datasets predating it load
    # back as ``None`` rather than shifting columns.
    game_id = record.get("game_id")
    return PositionRow(
        cp_eval=float(record["cp_eval"]),
        white_elo=int(record["white_elo"]),
        black_elo=int(record["black_elo"]),
        ply=int(record["ply"]),
        phase=str(record["phase"]),
        time_control=str(record["time_control"]),
        tc_bucket=str(record["tc_bucket"]),
        clock_remaining=(float(clk) if clk not in (None, "") else None),
        side_to_move=str(record["side_to_move"]),
        result=float(record["result"]),
        game_id=(str(game_id) if game_id not in (None, "") else None),
        fen=(str(fen) if fen not in (None, "") else None),
    )


def _read_file(p: Path) -> List[PositionRow]:
    """Read one dataset part file (``.csv`` or ``.parquet``) into typed rows."""
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(p)
        return [_coerce_row(rec) for rec in table.to_pylist()]
    with p.open("r", encoding="utf-8") as fh:
        return [_coerce_row(rec) for rec in csv.DictReader(fh)]


def load_rows(path: str) -> List[PositionRow]:
    """Load a built dataset back into typed rows.

    Accepts a single CSV/Parquet file, or a **partitioned directory** written with
    ``partition=True`` (every ``part.*`` under the hive tree is read and concatenated;
    partition order is not significant). The import surface for downstream tasks — they
    get :class:`PositionRow`s and never re-derive the schema or the column names.
    """
    p = Path(path)
    if p.is_dir():
        parts = sorted(q for q in p.rglob("part.*") if q.suffix in (".csv", ".parquet"))
        rows: List[PositionRow] = []
        for part in parts:
            rows.extend(_read_file(part))
        return rows
    return _read_file(p)


def load_dataframe(path: str):
    """Load a built dataset as a pandas DataFrame (needs the 'data' extra).

    A convenience for the modelling tasks (0004) that want a frame; everything else
    should use :func:`load_rows` and stay pandas-free.
    """
    import pandas as pd

    return pd.DataFrame([r.as_dict() for r in load_rows(path)])
