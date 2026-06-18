"""Tests for the Lichess dump downloader (task 0023) — no network.

A fake opener stands in for ``urllib.request.urlopen`` so the streaming, resume
(HTTP Range), atomic-rename, checksum, and idempotent-cache behaviours are all
exercised against a few bytes instead of a 30 GB dump.
"""

from __future__ import annotations

import hashlib
import io

import pytest

from chess_equity.data import download as dl

PAYLOAD = b"PGN-DUMP-CONTENTS-" * 100  # ~1800 bytes; the "dump" body
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResponse(io.BytesIO):
    """A urlopen-like response: a readable body plus ``status`` and ``headers``."""

    def __init__(self, body: bytes, status: int = 200, total: int | None = None):
        super().__init__(body)
        self.status = status
        remaining = len(body) if total is None else total
        self.headers = {"Content-Length": str(remaining)}


def _opener_for(body: bytes, *, honor_range: bool = True):
    """Build a fake _urlopen that serves ``body``, optionally honouring Range."""

    def opener(req):
        rng = req.get_header("Range")
        if rng and honor_range:
            start = int(rng.split("=")[1].split("-")[0])
            return FakeResponse(body[start:], status=206)
        return FakeResponse(body, status=200)

    return opener


def test_downloads_and_verifies_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_urlopen", _opener_for(PAYLOAD))
    path = dl.download_month("2026-05", str(tmp_path), expected_sha256=SHA)
    assert path.name == "lichess_db_standard_rated_2026-05.pgn.zst"
    assert path.read_bytes() == PAYLOAD
    # The .part scratch file is gone after the atomic rename.
    assert not path.with_name(path.name + ".part").exists()


def test_checksum_mismatch_raises_and_leaves_no_final(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_urlopen", _opener_for(PAYLOAD))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        dl.download_month("2026-05", str(tmp_path), expected_sha256="deadbeef")
    assert not dl.dump_path("2026-05", str(tmp_path)).exists()  # no bad final file


def test_resume_uses_range_and_appends(tmp_path, monkeypatch):
    # Pre-seed a .part with the first 500 bytes, as if a prior run stopped there.
    target = dl.dump_path("2026-05", str(tmp_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(PAYLOAD[:500])

    seen = {}

    def opener(req):
        seen["range"] = req.get_header("Range")
        start = int(seen["range"].split("=")[1].split("-")[0])
        return FakeResponse(PAYLOAD[start:], status=206)

    monkeypatch.setattr(dl, "_urlopen", opener)
    path = dl.download_month("2026-05", str(tmp_path))
    assert seen["range"] == "bytes=500-"  # resumed from the partial size
    assert path.read_bytes() == PAYLOAD  # 500 existing + remainder, intact


def test_server_ignores_range_restarts_cleanly(tmp_path, monkeypatch):
    # A stale .part exists, but the server replies 200 (whole file) — must not double-write.
    target = dl.dump_path("2026-05", str(tmp_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(b"GARBAGE-OLD-PARTIAL")

    monkeypatch.setattr(dl, "_urlopen", _opener_for(PAYLOAD, honor_range=False))
    path = dl.download_month("2026-05", str(tmp_path))
    assert path.read_bytes() == PAYLOAD  # restarted from scratch, no garbage prefix


def test_existing_file_is_idempotent(tmp_path, monkeypatch):
    target = dl.dump_path("2026-05", str(tmp_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(PAYLOAD)

    def boom(req):  # opener must never be called when the dump is already cached
        raise AssertionError("should not download an already-cached dump")

    monkeypatch.setattr(dl, "_urlopen", boom)
    assert dl.download_month("2026-05", str(tmp_path)) == target


def test_progress_reports_total_from_range(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_urlopen", _opener_for(PAYLOAD))
    seen = []
    dl.download_month("2026-05", str(tmp_path), progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (len(PAYLOAD), len(PAYLOAD))  # finished, total known


def test_cli_build_month_downloads_then_builds(tmp_path, monkeypatch):
    """`data build --month` fetches the dump, then builds from it (download wiring)."""
    from pathlib import Path

    from chess_equity import cli

    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_games.pgn"

    def fake_download(month, dest_dir, **kw):
        assert month == "2026-05"
        return sample  # stand in for the streamed dump (plain PGN, no zstandard needed)

    monkeypatch.setattr(dl, "download_month", fake_download)
    monkeypatch.setattr(dl, "data_extra_available", lambda: True)
    out_dir = tmp_path / "out"
    rc = cli.main(["data", "build", "--month", "2026-05", "--out", str(out_dir)])
    assert rc == 0
    assert (out_dir / "dataset.csv").exists()


def test_cli_build_month_missing_extra_fails_before_fetch(tmp_path, monkeypatch, capsys):
    """Without the 'data' extra, `--month` errors with the install hint and never fetches."""
    from chess_equity import cli

    monkeypatch.setattr(dl, "data_extra_available", lambda: False)

    def no_download(*args, **kwargs):  # the early exit must precede any download
        raise AssertionError("must not download when the data extra is missing")

    def no_network(req):  # ...and certainly never open a socket
        raise AssertionError("must not touch the network when the data extra is missing")

    monkeypatch.setattr(dl, "download_month", no_download)
    monkeypatch.setattr(dl, "_urlopen", no_network)

    rc = cli.main(["data", "build", "--month", "2026-05", "--out", str(tmp_path / "out")])
    assert rc == 1
    assert "chess-equity[data]" in capsys.readouterr().err
