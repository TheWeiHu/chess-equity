"""Stream a Lichess monthly dump to a local cache, with resume + optional checksum.

Task 0002 stubbed ``data build --month`` to merely print the dump URL; this implements
the fetch. The standard rated dumps are tens of GB, so the rules are:

- **stream, never materialise** — copy the HTTP body to disk in fixed-size chunks; the
  build step (:func:`chess_equity.data.build.open_pgn`) then streams the ``.zst`` too,
  so a full month never sits in memory or as a decompressed file.
- **resume** — a partial download lands in ``<name>.part``; a re-run sends an HTTP
  ``Range`` from where it stopped, so a dropped 30 GB transfer doesn't restart at 0. If
  the server ignores the range (responds ``200`` not ``206``), we restart cleanly.
- **checksum is opt-in** — Lichess does not publish a stable machine-readable checksum
  per dump, so we verify only when the caller supplies an expected SHA-256 (e.g. from
  the database page); otherwise completion is signalled by the atomic rename alone.

The HTTP fetch goes through the module-level :func:`_urlopen` (stdlib ``urlopen`` by
default) so tests inject a fake opener and never touch the network.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Optional
from urllib.request import Request, urlopen

from chess_equity.data.build import month_url

# Where dumps are cached between runs. Override per-call with ``dest_dir``.
DEFAULT_DUMP_DIR = os.path.join(os.path.expanduser("~"), ".cache", "chess-equity", "dumps")

# 1 MiB transfer chunks — big enough to amortise syscalls, small enough to stay streaming.
_CHUNK = 1 << 20

# Seam: tests monkeypatch this to avoid the network. Matches urllib.request.urlopen.
_urlopen = urlopen

# A progress callback receives (bytes_so_far, total_bytes_or_None).
Progress = Callable[[int, Optional[int]], None]


def dump_filename(month: str) -> str:
    """The canonical filename for a ``YYYY-MM`` dump (matches the URL's basename)."""
    return f"lichess_db_standard_rated_{month}.pgn.zst"


def dump_path(month: str, dest_dir: str = DEFAULT_DUMP_DIR) -> Path:
    return Path(dest_dir) / dump_filename(month)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download_month(
    month: str,
    dest_dir: str = DEFAULT_DUMP_DIR,
    *,
    resume: bool = True,
    expected_sha256: Optional[str] = None,
    url: Optional[str] = None,
    chunk_size: int = _CHUNK,
    progress: Optional[Progress] = None,
) -> Path:
    """Download the ``month`` dump into ``dest_dir`` and return its path.

    Idempotent: if the final file already exists it is returned as-is (after a checksum
    check when ``expected_sha256`` is given). A partial ``.part`` file is resumed via an
    HTTP ``Range`` request unless ``resume=False``.
    """
    target = dump_path(month, dest_dir)
    if target.exists():
        if expected_sha256 and _sha256(target) != expected_sha256:
            raise RuntimeError(f"checksum mismatch on cached {target}; delete it and retry")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    start = part.stat().st_size if (resume and part.exists()) else 0

    req = Request(url or month_url(month))
    if start:
        req.add_header("Range", f"bytes={start}-")
    resp = _urlopen(req)

    # If we asked to resume but the server sent the whole file (200, not 206), restart.
    status = getattr(resp, "status", None) or getattr(resp, "code", 200)
    if start and status != 206:
        start = 0

    total = _content_length(resp, start)
    mode = "ab" if start else "wb"
    done = start
    with open(part, mode) as fh:
        if progress:
            progress(done, total)
        while True:
            block = resp.read(chunk_size)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if progress:
                progress(done, total)

    if expected_sha256:
        actual = _sha256(part)
        if actual != expected_sha256:
            raise RuntimeError(
                f"checksum mismatch: got {actual}, expected {expected_sha256} "
                f"(partial left at {part})"
            )

    os.replace(part, target)  # atomic: a reader never sees a half-written final name
    return target


def _content_length(resp: object, start: int) -> Optional[int]:
    """Total expected bytes from the response headers, or ``None`` if unknown.

    On a ranged (206) response ``Content-Length`` is the *remaining* bytes, so add the
    bytes we already have to report the full size to a progress callback.
    """
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Content-Length")
    if raw is None:
        return None
    try:
        return int(raw) + start
    except (TypeError, ValueError):
        return None
