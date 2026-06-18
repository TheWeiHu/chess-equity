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

import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

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


class App(SimpleHTTPRequestHandler):
    """Serve the web/ folder for GET; handle POST /api/play for live evals."""

    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def _send_json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/api/play":
            self._send_json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send_json(play(payload))
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # surface engine/model failures cleanly
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)


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
