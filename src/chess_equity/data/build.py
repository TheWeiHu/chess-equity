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
from typing import IO, Iterable, Iterator, List, Optional

from chess_equity.data.pgn import iter_rows
from chess_equity.data.schema import COLUMNS, PositionRow

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


def _write_csv(rows: Iterable[PositionRow], out: Path) -> int:
    count = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
            count += 1
    return count


def _write_parquet(rows: Iterable[PositionRow], out: Path) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "writing Parquet needs the 'data' extra: pip install 'chess-equity[data]'"
        ) from exc
    materialised: List[PositionRow] = list(rows)
    table = pa.table({col: [getattr(r, col) for r in materialised] for col in COLUMNS})
    pq.write_table(table, out)
    return len(materialised)


def build_dataset(
    pgn_path: str,
    out_dir: str,
    *,
    sample: Optional[int] = None,
    fmt: str = "csv",
    name: str = "dataset",
) -> Path:
    """Parse ``pgn_path`` into ``out_dir/<name>.<fmt>`` and return the written path.

    ``sample`` caps the row count (for the committed fixture / quick runs); ``None``
    consumes the whole file. ``fmt`` is ``"csv"`` or ``"parquet"``.
    """
    if fmt not in ("csv", "parquet"):
        raise ValueError(f"unknown format {fmt!r} (expected 'csv' or 'parquet')")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / f"{name}.{fmt}"
    with open_pgn(pgn_path) as handle:
        rows = iter_rows(handle, limit=sample)
        if fmt == "csv":
            _write_csv(rows, target)
        else:
            _write_parquet(rows, target)
    return target


def _coerce_row(record: dict) -> PositionRow:
    clk = record.get("clock_remaining")
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
    )


def load_rows(path: str) -> List[PositionRow]:
    """Load a built dataset (CSV or Parquet) back into typed rows.

    The import surface for downstream tasks — they get :class:`PositionRow`s and never
    re-derive the schema or the column names.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(p)
        return [_coerce_row(rec) for rec in table.to_pylist()]
    with p.open("r", encoding="utf-8") as fh:
        return [_coerce_row(rec) for rec in csv.DictReader(fh)]


def load_dataframe(path: str):
    """Load a built dataset as a pandas DataFrame (needs the 'data' extra).

    A convenience for the modelling tasks (0004) that want a frame; everything else
    should use :func:`load_rows` and stay pandas-free.
    """
    import pandas as pd

    return pd.DataFrame([r.as_dict() for r in load_rows(path)])
