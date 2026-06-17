#!/usr/bin/env python3
"""Import a Lichess game into the web demo (task 0011).

Fetch a game by URL or id, auto-fill both players' ratings, and emit a JSON the
static page already renders (``demo-game.json`` schema, via :mod:`game_json`). The
classic bar reuses Lichess' own ``[%eval]``; our equity is computed per ply.

Usage (from the repo root)::

    python web/import_game.py https://lichess.org/abcd1234        # -> web/imported-game.json
    python web/import_game.py abcd1234 --model maia2 --out web/g.json

Then open the page at ``index.html?game=imported-game.json``.

Lichess etiquette: a fetched PGN is cached (``--cache-dir``, default
``~/.cache/chess-equity/lichess``); a cached id never hits the network, so re-runs
and tests make zero requests. One request per uncached game, with a descriptive
User-Agent. A token (``--token`` / ``$LICHESS_TOKEN``) is optional — public games
need none — and raises the rate limit when supplied.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Callable, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from game_json import build_game  # noqa: E402

EXPORT_URL = "https://lichess.org/game/export/{id}"
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "chess-equity", "lichess")
USER_AGENT = "chess-equity/0.1 (https://github.com/TheWeiHu/chess-equity; game import demo)"

# A game id is exactly 8 base62 chars; URLs may append a color/ply (`/white`, `#33`).
_ID_RE = re.compile(r"(?:lichess\.org/)?([A-Za-z0-9]{8})")


class ImportError_(RuntimeError):
    """Raised when a game cannot be located or fetched."""


def extract_game_id(source: str) -> str:
    """Pull the 8-char game id out of a Lichess URL or accept a bare id."""
    source = source.strip()
    m = _ID_RE.search(source)
    if not m:
        raise ImportError_(f"could not find a Lichess game id in {source!r}")
    return m.group(1)


# An opener maps (url, headers, timeout) -> response text. Injectable for tests.
Opener = Callable[[str, dict, float], str]


def _urllib_opener(url: str, headers: dict, timeout: float) -> str:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ImportError_(f"fetching {url}: {exc}") from exc


def fetch_pgn(
    game_id: str,
    *,
    cache_dir: str = DEFAULT_CACHE,
    token: Optional[str] = None,
    timeout: float = 10.0,
    opener: Opener = _urllib_opener,
) -> str:
    """Return the PGN for ``game_id``, caching it so re-runs make no request."""
    cache_path = os.path.join(cache_dir, f"{game_id}.pgn")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            return fh.read()

    url = EXPORT_URL.format(id=game_id) + "?evals=true&clocks=false&literate=false"
    headers = {"Accept": "application/x-chess-pgn", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    pgn = opener(url, headers, timeout)
    if not pgn.strip():
        raise ImportError_(f"empty PGN for game {game_id} (private or not found?)")

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        fh.write(pgn)
    return pgn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Import a Lichess game into the web demo.")
    ap.add_argument("source", help="Lichess game URL or 8-char id")
    ap.add_argument("--model", choices=("baseline", "maia2"), default="baseline")
    ap.add_argument(
        "--out", default=os.path.join(os.path.dirname(__file__), "imported-game.json")
    )
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--token", default=os.environ.get("LICHESS_TOKEN"))
    args = ap.parse_args(argv)

    from chess_equity.cli import build_model

    game_id = extract_game_id(args.source)
    pgn = fetch_pgn(game_id, cache_dir=args.cache_dir, token=args.token)
    data = build_game(pgn, model=build_model(args.model))

    import json

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    rel = os.path.basename(args.out)
    print(
        f"wrote {args.out} ({len(data['moves'])} plies, model={args.model}); "
        f"open index.html?game={rel}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
