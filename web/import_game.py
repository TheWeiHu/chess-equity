#!/usr/bin/env python3
"""Import a Lichess game into the web demo (task 0011).

Fetch a game by URL or id, auto-fill both players' ratings, and emit a JSON the
static page already renders (``demo-game.json`` schema, via :mod:`game_json`). The
classic bar reuses Lichess' own ``[%eval]``; our equity is computed per ply.

Usage (from the repo root)::

    python web/import_game.py https://lichess.org/abcd1234        # -> web/imported-game.json
    python web/import_game.py abcd1234 --model maia2 --out web/g.json
    python web/import_game.py --user DrNykterstein                # the player's latest game

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
EXPORT_USER_URL = "https://lichess.org/api/games/user/{name}"
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "chess-equity", "lichess")
USER_AGENT = "chess-equity/0.1 (https://github.com/TheWeiHu/chess-equity; game import demo)"

# A game id is exactly 8 base62 chars; URLs may append a color/ply (`/white`, `#33`).
_ID_RE = re.compile(r"(?:lichess\.org/)?([A-Za-z0-9]{8})")
# The id inside a PGN's Site header — used to key the cache for a user-fetched game.
_SITE_ID_RE = re.compile(r'\[Site\s+"[^"]*lichess\.org/([A-Za-z0-9]{8})')


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


def fetch_latest_user_game(
    username: str,
    *,
    cache_dir: str = DEFAULT_CACHE,
    token: Optional[str] = None,
    timeout: float = 10.0,
    opener: Opener = _urllib_opener,
) -> str:
    """Resolve ``username``'s most recent game, seed the per-id cache, return its id.

    Hits ``/api/games/user/{name}?max=1&evals=true`` (one PGN, newest first), pulls the
    game id from the Site header, and writes the PGN into the same ``{id}.pgn`` cache
    :func:`fetch_pgn` uses — so the caller just does ``fetch_pgn(id)`` and gets a cache
    hit (no second request). Raises :class:`ImportError_` if the user has no public game.
    """
    url = EXPORT_USER_URL.format(name=username) + "?max=1&evals=true&clocks=false&literate=false"
    headers = {"Accept": "application/x-chess-pgn", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    pgn = opener(url, headers, timeout)
    if not pgn.strip():
        raise ImportError_(f"no public games found for user {username!r}")

    m = _SITE_ID_RE.search(pgn)
    if not m:
        raise ImportError_(f"could not find a game id in {username!r}'s latest game PGN")
    game_id = m.group(1)

    cache_path = os.path.join(cache_dir, f"{game_id}.pgn")
    if not os.path.exists(cache_path):
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            fh.write(pgn)
    return game_id


def main(argv=None, *, opener: Opener = _urllib_opener) -> int:
    ap = argparse.ArgumentParser(description="Import a Lichess game into the web demo.")
    ap.add_argument("source", nargs="?", help="Lichess game URL or 8-char id")
    ap.add_argument(
        "--user", help="import this user's most recent game (instead of a URL/id)"
    )
    ap.add_argument("--model", choices=("baseline", "maia2"), default="baseline")
    ap.add_argument(
        "--out", default=os.path.join(os.path.dirname(__file__), "imported-game.json")
    )
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--token", default=os.environ.get("LICHESS_TOKEN"))
    args = ap.parse_args(argv)

    if bool(args.source) == bool(args.user):
        ap.error("give exactly one of: a game URL/id, or --user <name>")

    from chess_equity.cli import build_model

    try:
        if args.user:
            game_id = fetch_latest_user_game(
                args.user, cache_dir=args.cache_dir, token=args.token, opener=opener
            )
        else:
            game_id = extract_game_id(args.source)
        pgn = fetch_pgn(game_id, cache_dir=args.cache_dir, token=args.token, opener=opener)
    except ImportError_ as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
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
