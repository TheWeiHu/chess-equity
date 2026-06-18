#!/usr/bin/env python3
"""Live analysis server for the interactive board (web/live.html).

The precomputed demo (web/build_demo.py + demo-game.json) is static, but a board you
can *move pieces on* needs the engines on every position — so this is a tiny backend.
Stdlib only (``http.server``): it serves the web/ folder AND exposes one JSON endpoint
that, for any FEN, returns the legal moves, game state, the centipawn bar (Stockfish),
and the rating-conditioned equity bar (Maia-2). Both models load once at startup.

Run it (real engines needed — see DEPENDENCIES.md):

    uv run python web/server.py            # http://localhost:8000/live.html
    PORT=9000 STOCKFISH_DEPTH=14 uv run python web/server.py

Legal-move generation, SAN, and game-over detection are python-chess; the browser
just renders and POSTs move intents, so the page needs no chess library of its own.
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import chess  # noqa: E402

MATE_CP = 10000.0
PORT = int(os.environ.get("PORT", "8000"))
DEPTH = int(os.environ.get("STOCKFISH_DEPTH", "12"))  # snappy for interactive play; bump via env


class Engines:
    """Lazily-loaded Stockfish (persistent process) + Maia-2 (loaded once)."""

    def __init__(self) -> None:
        self._sf = None        # chess.engine.SimpleEngine
        self._maia = None      # EquityModel
        self._lock = threading.Lock()   # one Stockfish process + Maia model, neither thread-safe

    def _ensure(self) -> None:
        if self._sf is None:
            import chess.engine
            from chess_equity.stockfish import stockfish_path, StockfishNotFound

            path = stockfish_path()
            if not path:
                raise StockfishNotFound(
                    "no Stockfish binary found — `brew install stockfish` or set "
                    "$STOCKFISH_PATH (see DEPENDENCIES.md)"
                )
            print(f"  · opening Stockfish ({path}) at depth {DEPTH} …", flush=True)
            self._sf = chess.engine.SimpleEngine.popen_uci(path)
        if self._maia is None:
            print("  · loading Maia-2 (first call downloads the checkpoint) …", flush=True)
            from chess_equity.cli import build_model

            self._maia = build_model("maia2")

    def analyse(self, fen: str, white_elo: int, black_elo: int) -> dict:
        """Full live verdict for a position: legality, state, both bars."""
        with self._lock:
            return self._analyse_locked(fen, white_elo, black_elo)

    def _analyse_locked(self, fen: str, white_elo: int, black_elo: int) -> dict:
        self._ensure()
        import chess.engine

        board = chess.Board(fen)
        legal: dict[str, list[str]] = {}
        for mv in board.legal_moves:
            legal.setdefault(chess.square_name(mv.from_square), []).append(
                chess.square_name(mv.to_square)
            )
        for k in legal:
            legal[k] = sorted(set(legal[k]))

        over = board.is_game_over()
        result = {
            "fen": board.fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "legal": legal,
            "check": board.is_check(),
            "checkmate": board.is_checkmate(),
            "stalemate": board.is_stalemate() or board.is_insufficient_material()
            or board.is_seventyfive_moves() or board.is_fivefold_repetition(),
            "game_over": over,
            "best_move": None,
        }

        if over:
            # Terminal: no engine/Maia call (a move-policy model can't score a position
            # with no legal moves). The result is decided.
            if board.is_checkmate():
                result["cp"] = -MATE_CP if board.turn == chess.WHITE else MATE_CP
                result["equity_white"] = 0.0 if board.turn == chess.WHITE else 100.0
            else:
                result["cp"] = 0.0
                result["equity_white"] = 50.0
            return result

        # Centipawn bar — Stockfish, converted to White's POV.
        info = self._sf.analyse(board, chess.engine.Limit(depth=DEPTH))
        ws = info["score"].white()
        if ws.is_mate():
            result["cp"] = MATE_CP if ws.mate() > 0 else -MATE_CP
        else:
            result["cp"] = float(ws.score() or 0)
        pv = info.get("pv")
        if pv:
            result["best_move"] = pv[0].uci()
            result["best_san"] = board.san(pv[0])

        # Equity bar — Maia-2 at the given ratings (already White-POV).
        result["equity_white"] = round(self._maia.evaluate(fen, white_elo, black_elo).equity_white, 1)
        return result


ENGINES = Engines()


def play(payload: dict) -> dict:
    """Apply an optional move to a FEN, then analyse the resulting position."""
    fen = payload.get("fen") or chess.STARTING_FEN
    board = chess.Board(fen)  # raises ValueError on a bad FEN
    san = None
    uci = payload.get("uci")
    if uci:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {uci} in this position")
        san = board.san(move)
        board.push(move)
    we = int(payload.get("white_elo", 1500))
    be = int(payload.get("black_elo", 1500))
    out = ENGINES.analyse(board.fen(), we, be)
    out["san"] = san
    return out


# ---- famous games (web/famous-games.json) -> FEN-per-ply for the board ----------

def _load_famous() -> list:
    try:
        with open(os.path.join(HERE, "famous-games.json"), encoding="utf-8") as fh:
            return json.load(fh).get("games", [])
    except (OSError, ValueError):
        return []


FAMOUS = _load_famous()
FAMOUS_BY_ID = {g["id"]: g for g in FAMOUS}
_MOVES_CACHE: dict = {}


def _moves_from_game(parsed) -> list:
    """A parsed python-chess game → [{ply, san, uci, fen}] along its MAINLINE.

    Only the mainline is walked, so PGN side-variations (the ``(14... Kf8 …)`` lines a
    pasted game may contain) are ignored — the board steps through the game as played.
    """
    board = parsed.board()                      # respects a [FEN] setup header if present
    out = [{"ply": 0, "san": "(start)", "uci": None, "fen": board.fen()}]
    for i, mv in enumerate(parsed.mainline_moves(), 1):
        san = board.san(mv)
        board.push(mv)
        out.append({"ply": i, "san": san, "uci": mv.uci(), "fen": board.fen()})
    return out


def game_moves(g: dict) -> list:
    """Parse a famous game's PGN into [{ply, san, uci, fen}] (start + one per ply)."""
    if g["id"] in _MOVES_CACHE:
        return _MOVES_CACHE[g["id"]]
    import chess.pgn

    out = _moves_from_game(chess.pgn.read_game(io.StringIO(g["pgn"])))
    _MOVES_CACHE[g["id"]] = out
    return out


def parse_pgn(text: str) -> dict:
    """Turn pasted PGN text into the same shape ``/api/game`` returns (name + moves).

    Accepts headerless movetext (``1. e4 c5 …``) or a full PGN with tags. Variations are
    dropped (mainline only). Raises ``ValueError`` on unparseable input or a game with no
    moves, which the POST handler surfaces as a 400.
    """
    import chess.pgn

    parsed = chess.pgn.read_game(io.StringIO((text or "").strip()))
    if parsed is None:
        raise ValueError("could not parse PGN — paste movetext like '1. e4 c5 2. Nf3 …'")
    moves = _moves_from_game(parsed)
    if len(moves) <= 1:
        raise ValueError("no legal moves found in that PGN")

    h = parsed.headers

    def _tag(key):
        v = (h.get(key) or "").strip()
        return v if v and v != "?" else None

    white, black = _tag("White"), _tag("Black")
    name = _tag("Event") or (f"{white} vs {black}" if white and black else "Pasted game")
    date = h.get("Date") or ""
    year = int(date[:4]) if date[:4].isdigit() else None
    return {"name": name, "white": white, "black": black, "year": year, "moves": moves}


class App(SimpleHTTPRequestHandler):
    """Serve the web/ folder for GET; handle POST /api/play for live evals."""

    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def end_headers(self) -> None:  # noqa: N802
        # Dev server: never let the browser cache the static assets, so editing
        # live.js / style.css / board.js always takes effect on reload (otherwise a
        # cached old script throws against markup that has since changed).
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/games":
            self._send_json({"games": [
                {"id": g["id"], "name": g["name"], "white": g.get("white"),
                 "black": g.get("black"), "year": g.get("year"),
                 "plies": len(game_moves(g)) - 1}
                for g in FAMOUS
            ]})
            return
        if parsed.path == "/api/game":
            gid = (parse_qs(parsed.query).get("id") or [None])[0]
            g = FAMOUS_BY_ID.get(gid)
            if not g:
                self._send_json({"error": f"unknown game {gid!r}"}, 404)
                return
            self._send_json({"id": g["id"], "name": g["name"], "white": g.get("white"),
                             "black": g.get("black"), "moves": game_moves(g)})
            return
        super().do_GET()

    def _send_json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) or b"{}"
        if path == "/api/play":
            try:
                self._send_json(play(json.loads(raw)))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, 400)
            except Exception as exc:  # surface engine/model failures cleanly
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            return
        if path == "/api/pgn":
            try:
                self._send_json(parse_pgn(json.loads(raw).get("pgn", "")))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, 400)
            except Exception as exc:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 400)
            return
        self._send_json({"error": "not found"}, 404)


def main() -> int:
    print(f"chess-equity live server → http://localhost:{PORT}/live.html", flush=True)
    print("  (engines load on the first /api/play call)", flush=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), App)
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down …")
    finally:
        if ENGINES._sf is not None:
            ENGINES._sf.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
