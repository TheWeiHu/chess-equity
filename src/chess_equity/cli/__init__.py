"""Command-line entry point: ``chess-equity``.

Commands:

    chess-equity eval "<fen>" --white-elo 1500 --black-elo 1500
    chess-equity eval --pgn game.pgn --white-elo 1500 --black-elo 1500
    chess-equity score --pgn game.pgn              # one-game scorecard: score vs real result
    chess-equity grade --pgn game.pgn --white-elo 1500 --black-elo 1500
    chess-equity broadcast --round <id>            # live Lichess broadcast round
    chess-equity broadcast --pgn game.pgn          # replay a finished game as "live"
    chess-equity highlights --pgn game.pgn         # auto-detect drama/clutch moments
    chess-equity personal --user <lichess-name>    # per-player phase profile + offsets
    chess-equity data build --pgn dump.pgn.zst --sample 50000 --out data/
    chess-equity validate --data data/dataset.csv --models baseline

The CLI depends only on :class:`~chess_equity.adapters.EquityModel`; it constructs
the placeholder :class:`~chess_equity.models.LichessBaselineModel` today, but a new
model would drop in with no other changes here.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from typing import List, Optional, TextIO

import chess
import chess.pgn

from chess_equity.adapters import EquityModel
from chess_equity.bar import render_eval
from chess_equity.broadcast import (
    BroadcastIngestor,
    GameEvent,
    LichessRoundFeed,
    LocalPgnFeed,
    MoveEvent,
    UrlPgnFeed,
)
from chess_equity.grading import EquityGrader
from chess_equity.models import LichessBaselineModel, placeholder_equity_warning
from chess_equity.rollout import MaiaRolloutModel, estimate_to_equity
from chess_equity.search import MaiaSearchModel
from chess_equity.search import estimate_to_equity as search_estimate_to_equity

START_FEN = chess.STARTING_FEN


def _eval_fen(model: EquityModel, fen: str, white_elo: int, black_elo: int) -> str:
    equity = model.evaluate(fen, white_elo, black_elo)
    return render_eval(equity)


def _eval_pgn(model: EquityModel, path: str, white_elo: int, black_elo: int) -> List[str]:
    """Annotate every position in the first game of a PGN with its equity bar."""
    with open(path, encoding="utf-8") as fh:
        game = chess.pgn.read_game(fh)
    if game is None:
        raise ValueError(f"no game found in {path}")
    board = game.board()
    lines = [f"start  {_eval_fen(model, board.fen(), white_elo, black_elo)}"]
    for move in game.mainline_moves():
        san = board.san(move)
        board.push(move)
        lines.append(f"{san:7s}{_eval_fen(model, board.fen(), white_elo, black_elo)}")
    return lines


def _grade_game(model: EquityModel, path: str, white_elo: int, black_elo: int):
    """Grade every move of the first game in ``path``; returns its MoveGrade list."""
    with open(path, encoding="utf-8") as fh:
        game = chess.pgn.read_game(fh)
    if game is None:
        raise ValueError(f"no game found in {path}")
    return EquityGrader(model).grade_game(game, white_elo, black_elo)


def _grade_round(model: EquityModel, path: str, white_elo: int, black_elo: int):
    """Grade every game of a multi-game PGN; returns ``[(white, black, grades), ...]``.

    Per-game ratings come from each game's ``WhiteElo``/``BlackElo`` headers (the round's
    players differ board to board), falling back to the CLI defaults when a header is
    missing/``?``. Players are named from the ``White``/``Black`` headers so the leaderboard
    can pool a player's moves across every board.
    """
    from chess_equity.broadcast import _parse_elo, split_games

    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    games = []
    grader = EquityGrader(model)
    for game_pgn in split_games(text):
        game = chess.pgn.read_game(io.StringIO(game_pgn))
        if game is None:
            continue
        we = _parse_elo(game.headers, "WhiteElo") or white_elo
        be = _parse_elo(game.headers, "BlackElo") or black_elo
        white = game.headers.get("White", "?")
        black = game.headers.get("Black", "?")
        games.append((white, black, grader.grade_game(game, we, be)))
    if not games:
        raise ValueError(f"no games found in {path}")
    return games


def _grade_lines(grades) -> List[str]:
    """One text line per graded move (peer-relative Δequity + classic Δbest/cp)."""
    lines = []
    for g in grades:
        cp = "" if g.cp_loss is None else f"  cp_loss {g.cp_loss:+.0f}"
        # +Δ vs peers is the headline; Δ vs best is the classic "left on the table".
        lines.append(
            f"{g.ply:3d}. {g.san:7s} {g.label:11s} "
            f"Δpeer {g.grade_peer:+5.1f}  Δbest {g.grade_best:+5.1f}{cp}"
        )
    return lines


def _event_line(event: MoveEvent) -> str:
    """One JSONL ``position`` record in the overlay's schema (task 0019).

    Emits :meth:`MoveEvent.to_overlay_event` (nested, White-POV equity in [0, 1]) so
    the broadcast stream is directly consumable by overlay.js — paired with the
    one-time ``game`` metadata event (player names) emitted via ``on_game``.
    """
    return json.dumps(event.to_overlay_event())


def build_model(
    name: str = "baseline",
    *,
    n: int = 500,
    seed: Optional[int] = None,
    depth: int = 2,
    k: int = 4,
) -> EquityModel:
    """Construct the requested equity model by name.

    ``baseline`` is the rating-blind placeholder (no extra deps); ``maia2`` is the
    real rating-conditioned bar and lazily loads Maia-2 on first evaluation (so the
    error, if it's not installed, surfaces only when actually used). ``maia-rollout``
    is the slow Monte Carlo self-play oracle (``n`` rollouts; non-interactive);
    ``maia-search`` is the Maia-weighted expectimax (``depth``/``k``; non-interactive).
    """
    if name == "maia2":
        # Lazy import so the common baseline path never pays for the maia2 module.
        from chess_equity.maia2 import build_maia2_equity

        return build_maia2_equity()
    if name == "wdl-a":
        # Lazy import + artifact load so the baseline path stays free of the model file.
        from chess_equity.wdl_regression import build_wdl_a_equity

        return build_wdl_a_equity()
    if name == "wdl-net":
        # Approach D (0013): the end-to-end board → WDL net. Lazy import so the
        # baseline path never pulls torch; loads the committed artifact.
        from chess_equity.wdl_net import build_wdl_net_equity

        return build_wdl_net_equity()
    if name == "maia-rollout":
        from chess_equity.rollout import build_maia_rollout

        return build_maia_rollout(n=n, seed=seed)
    if name == "maia-search":
        from chess_equity.search import build_maia_search

        return build_maia_search(depth=depth, k=k)
    if name == "baseline":
        # The shipped centipawn bar should be a real engine eval (Stockfish at the
        # given depth) when a binary is available, else fall back to material (0043).
        from chess_equity.stockfish import resolve_objective_engine

        return LichessBaselineModel(engine=resolve_objective_engine(depth=depth))
    raise ValueError(
        f"unknown model {name!r}; choose from: "
        "baseline, maia2, wdl-a, wdl-net, maia-rollout, maia-search"
    )


def _apply_profiles(model: EquityModel, args: argparse.Namespace) -> EquityModel:
    """Wrap ``model`` in a :class:`PersonalEquityModel` if profiles were requested.

    Reads ``--white-profile`` / ``--black-profile`` off ``args`` (each a Lichess
    username, or ``player@file.pgn`` for an offline profile — see
    :func:`chess_equity.personal.load_profile`). With neither set, returns ``model``
    unchanged, so the band-average bar is unaffected.
    """
    white_spec = getattr(args, "white_profile", None)
    black_spec = getattr(args, "black_profile", None)
    if not white_spec and not black_spec:
        return model
    from chess_equity.personal import PersonalEquityModel, load_profile

    max_games = getattr(args, "max_games", 50)
    token = getattr(args, "token", None)
    white = load_profile(white_spec, max_games=max_games, token=token) if white_spec else None
    black = load_profile(black_spec, max_games=max_games, token=token) if black_spec else None
    return PersonalEquityModel(model, white_profile=white, black_profile=black)


def _eval_rollout_fen(model: MaiaRolloutModel, fen: str, white_elo: int, black_elo: int) -> str:
    """Bar + 95% CI line for the Monte Carlo rollout oracle (task 0007)."""
    est = model.estimate(fen, white_elo, black_elo)
    equity = estimate_to_equity(est, fen, model.SOURCE)
    return (
        f"{render_eval(equity)}  95% CI [{est.ci_low:.1f}, {est.ci_high:.1f}]  "
        f"n={est.n} ({est.n_terminal} terminal, {est.mean_plies:.0f} avg plies)"
    )


def _eval_search_fen(model: MaiaSearchModel, fen: str, white_elo: int, black_elo: int) -> str:
    """Bar + search-shape line for the Maia-weighted expectimax (task 0006)."""
    est = model.estimate(fen, white_elo, black_elo)
    equity = search_estimate_to_equity(est, fen, model.SOURCE)
    return (
        f"{render_eval(equity)}  depth={est.depth} k={est.k}  "
        f"({est.n_leaves} leaves, {est.n_terminal} terminal, "
        f"trunc={est.truncated_mass:.2f})"
    )


def _equity_for_fen(model: EquityModel, fen: str, args: argparse.Namespace):
    """Resolve a single :class:`Equity` for ``fen`` across model kinds.

    The one place per-model dispatch lives: rollout/search go through their estimate
    → ``estimate_to_equity`` path, everything else through ``evaluate`` (profile-wrapped).
    Both the single-eval (``--svg``) and batch (``--fens``) paths call through here so
    they score identically.
    """
    if isinstance(model, MaiaRolloutModel):
        return estimate_to_equity(model.estimate(fen, args.white_elo, args.black_elo),
                                  fen, model.SOURCE)
    if isinstance(model, MaiaSearchModel):
        return search_estimate_to_equity(model.estimate(fen, args.white_elo, args.black_elo),
                                         fen, model.SOURCE)
    return _apply_profiles(model, args).evaluate(fen, args.white_elo, args.black_elo)


def _eval_equity(model: EquityModel, args: argparse.Namespace):
    """Resolve a single :class:`Equity` for ``args.fen`` (used by the ``--svg`` snapshot)."""
    return _equity_for_fen(model, args.fen, args)


def _equity_label(equity_white: float) -> str:
    """Favoured-side verdict for the White-POV bar value (matches :func:`bar.render_bar`)."""
    return "White" if equity_white >= 50.0 else "Black"


def _read_fen_lines(path: str) -> List[str]:
    """Read one FEN per line from ``path`` ( ``-`` = stdin); skip blank and ``#`` lines."""
    fh = sys.stdin if path == "-" else open(path, encoding="utf-8")
    try:
        fens = []
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                fens.append(line)
        return fens
    finally:
        if fh is not sys.stdin:
            fh.close()


def _eval_fens(model: EquityModel, args: argparse.Namespace) -> int:
    """Batch-score every FEN in ``args.fens`` through the shared single-eval path (0221).

    Emits a JSON array of ``{fen, white_equity, label}`` with ``--json``, else one text
    line per FEN (``<fen>\\t<bar>``). Reuses :func:`_equity_for_fen` so a batch score is
    bit-identical to the single-position score.
    """
    fens = _read_fen_lines(args.fens)
    records = []
    for fen in fens:
        equity = _equity_for_fen(model, fen, args)
        records.append({
            "fen": fen,
            "white_equity": round(equity.equity_white, 2),
            "label": _equity_label(equity.equity_white),
        })
    if getattr(args, "json", False):
        print(json.dumps(records, indent=2))
    else:
        for rec in records:
            print(f"{rec['fen']}\t{rec['white_equity']:.1f}% ({rec['label']})")
    return 0


def _run_eval(args: argparse.Namespace) -> int:
    model = build_model(args.model, n=args.n, seed=args.seed, depth=args.depth, k=args.k)
    try:
        if getattr(args, "fens", None):
            return _eval_fens(model, args)
        if getattr(args, "svg", None) and not args.pgn:
            from chess_equity.bar import render_svg

            equity = _eval_equity(model, args)
            white_to_move = chess.Board(args.fen).turn == chess.WHITE
            svg = render_svg(equity, white_to_move=white_to_move)
            with open(args.svg, "w", encoding="utf-8") as fh:
                fh.write(svg)
            print(f"wrote {args.svg}  ({render_eval(equity)})")
            return 0
        if args.pgn:
            for line in _eval_pgn(_apply_profiles(model, args), args.pgn, args.white_elo, args.black_elo):
                print(line)
        elif isinstance(model, MaiaRolloutModel):
            print(_eval_rollout_fen(model, args.fen, args.white_elo, args.black_elo))
        elif isinstance(model, MaiaSearchModel):
            print(_eval_search_fen(model, args.fen, args.white_elo, args.black_elo))
        else:
            print(_eval_fen(_apply_profiles(model, args), args.fen, args.white_elo, args.black_elo))
    except (ValueError, OSError, RuntimeError) as exc:
        # RuntimeError covers Maia2NotInstalled (a model failing to load at use time).
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_grade(args: argparse.Namespace) -> int:
    model = build_model(args.model, depth=args.depth)
    try:
        if args.round:
            from chess_equity.grading import (
                leaderboard_export_rows,
                render_leaderboard,
                render_leaderboard_csv,
                render_leaderboard_md,
                round_leaderboard,
            )

            # Pool every board's move grades by player and rank the round (task 0207).
            games = _grade_round(model, args.pgn, args.white_elo, args.black_elo)
            # A min-moves floor (task 0227) ranks brief cameos below qualified players;
            # `sort` (task 0234) picks the primary rank metric within each tier.
            scores = round_leaderboard(
                games,
                sort=getattr(args, "sort", "accuracy"),
                min_moves=getattr(args, "min_moves", 0),
            )
            # --json/--csv emit ONLY the machine-readable leaderboard to stdout (task 0214)
            # so it pipes cleanly into broadcast lower-third graphics; --json wins if both.
            if getattr(args, "json", False):
                print(json.dumps(leaderboard_export_rows(scores), indent=2))
            elif getattr(args, "csv", False):
                print(render_leaderboard_csv(scores), end="")
            else:
                for row in render_leaderboard(scores, min_moves=args.min_moves):
                    print(row)
            if args.summary_json:
                with open(args.summary_json, "w", encoding="utf-8") as fh:
                    json.dump({"players": [s.to_dict() for s in scores]}, fh, indent=2)
                print(f"wrote leaderboard JSON to {args.summary_json}")
            # --leaderboard-md (task 0244): a paste-ready markdown recap table written to
            # a file, independent of the stdout format (text/--json/--csv). Like
            # --summary-json/--trajectory-svg, it's an extra asset, not a stdout switch.
            if getattr(args, "leaderboard_md", None):
                with open(args.leaderboard_md, "w", encoding="utf-8") as fh:
                    fh.write(render_leaderboard_md(scores))
                print(f"wrote leaderboard markdown to {args.leaderboard_md}")
        elif args.annotate_pgn:
            from chess_equity.annotate import annotate_pgn_file

            n = annotate_pgn_file(
                args.pgn, args.annotate_pgn, model, args.white_elo, args.black_elo
            )
            print(f"wrote {n} annotated moves to {args.annotate_pgn}")
        else:
            from chess_equity.grading import (
                equity_sparkline,
                equity_trajectory_svg,
                render_scoreline,
                scoreline,
            )

            grades = _grade_game(model, args.pgn, args.white_elo, args.black_elo)
            for line in _grade_lines(grades):
                print(line)
            # One-line White-POV equity-swing sparkline (task 0239): the whole-game shape
            # at a glance, pure over the grades above — no extra model calls.
            if getattr(args, "sparkline", False) and grades:
                print()
                print(equity_sparkline(grades))
            # Graphical SVG trajectory (task 0242): the sparkline's overlay/VOD sibling,
            # a per-ply White-POV area/line chart with a 50% midline — pure over grades.
            if getattr(args, "trajectory_svg", None) and grades:
                with open(args.trajectory_svg, "w", encoding="utf-8") as fh:
                    fh.write(equity_trajectory_svg(grades))
                print(f"wrote equity-trajectory SVG to {args.trajectory_svg}")
            # Per-side caster scoreline (task 0200): a one-glance accuracy-style summary
            # aggregated only from the per-move grades — no extra model calls.
            line = scoreline(grades)
            print()
            for row in render_scoreline(line):
                print(row)
            if args.summary_json:
                with open(args.summary_json, "w", encoding="utf-8") as fh:
                    json.dump(line.to_dict(), fh, indent=2)
                print(f"wrote scoreline JSON to {args.summary_json}")
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_score(args: argparse.Namespace) -> int:
    """Scorecard one game: the score, the real result, and what we predict (task 0129)."""
    from chess_equity.scorecard import build_scorecard_from_pgn, render_scorecard

    model = build_model(args.model, n=args.n, seed=args.seed, depth=args.depth, k=args.k)
    model = _apply_profiles(model, args)
    try:
        with open(args.pgn, encoding="utf-8") as fh:
            pgn_text = fh.read()
        card = build_scorecard_from_pgn(
            pgn_text,
            model,
            model_name=args.model,
            white_elo=args.white_elo,
            black_elo=args.black_elo,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for line in render_scorecard(card):
        print(line)

    # Optional shareable SVG sibling (task 0253): per-side accuracy, the biggest
    # practical swing + drama label, and the equity sparkline need the per-move grades,
    # so grade the game with the card's resolved ratings and hand both to the renderer.
    if getattr(args, "svg", None):
        from chess_equity.scorecard import render_scorecard_svg

        grades = _grade_game(model, args.pgn, card.white_elo, card.black_elo)
        with open(args.svg, "w", encoding="utf-8") as fh:
            fh.write(render_scorecard_svg(card, grades))
        print(f"wrote scorecard SVG to {args.svg}")
    return 0


def _build_broadcast_feed(args: argparse.Namespace):
    """Construct the broadcast feed from --pgn / --round / --url.

    A fresh feed each call so the SSE path can give every overlay connection its own
    replay/stream (``LocalPgnFeed`` in particular is stateful — it advances per poll).
    """
    if args.pgn:
        with open(args.pgn, encoding="utf-8") as fh:
            return LocalPgnFeed(fh.read(), moves_per_poll=args.moves_per_poll)
    if args.round:
        return LichessRoundFeed(args.round, token=args.token)
    if args.url:
        return UrlPgnFeed(args.url)
    raise ValueError("broadcast needs one of --pgn / --round / --url")


def _overlay_static_dir() -> Optional[str]:
    """The repo's ``overlay/`` dir if present (so --serve-sse is a one-command overlay).

    Resolved relative to this package's repo checkout; ``None`` when running from an
    installed wheel without the overlay assets, in which case only ``/sse`` is served.
    """
    from pathlib import Path

    candidate = Path(__file__).resolve().parents[2] / "overlay"
    return str(candidate) if candidate.is_dir() else None


def _run_broadcast(args: argparse.Namespace, model: EquityModel, out: TextIO) -> int:
    """Drive broadcast ingestion, writing one JSON event per line to ``out``.

    With ``--serve-sse PORT`` the same overlay events are instead streamed over an
    HTTP Server-Sent-Events endpoint the overlay can ``EventSource`` onto (task 0094),
    collapsing the old capture-to-file → serve.py seam into one command.
    """
    model = _apply_profiles(model, args)

    # Objective engine for the centipawn fallback when the model carries no cp (e.g.
    # maia2 win-prob): keeps the overlay's classic ghost tick + human-edge divergence
    # badge alive on a maia2 feed (task 0103). Only consulted for cp-less models, so
    # warn=False — the model's own bar already warns when no engine is available.
    from chess_equity.stockfish import resolve_objective_engine

    cp_engine = resolve_objective_engine(depth=args.depth, warn=False)

    # Surface a visible 'reconnecting' state when a live feed drops: the ingestor keeps
    # retrying with bounded exponential backoff (resuming from the last seen move), and
    # this prints the attempt + wait to stderr so the streamer knows the overlay is
    # holding rather than dead (task 0175).
    def log_reconnect(attempt: int, delay: float) -> None:
        print(
            f"# broadcast feed error; reconnecting in {delay:.0f}s (attempt {attempt})",
            file=sys.stderr,
        )

    # A multi-board round (Titled Tuesday, blitz events) carries many simultaneous
    # games; --board narrows the stream to one (by player-name substring or board
    # index), defaulting to follow-all when unset (task 0182). The special value
    # --board auto follows ALL boards but auto-cuts the overlay focus to the liveliest
    # one via server-side drama scoring (task 0256), so it keeps the full follow-all
    # stream (selector None) and flips on the auto-follow director instead. The
    # --board auto:<player> form additionally biases that director toward a named
    # player's boards (a soft hybrid; task 0262).
    from chess_equity.broadcast import parse_auto_spec, parse_board_selector

    board_spec = getattr(args, "board", None)
    auto_follow, bias_player = parse_auto_spec(board_spec)
    selector = None if auto_follow else parse_board_selector(board_spec)

    # Human-vs-engine divergence caption callout threshold (task 0273); None on the flag
    # means "use the module default" so callers that never pass it are unaffected.
    from chess_equity.broadcast import DIVERGENCE_CAPTION_THRESHOLD

    div_threshold = getattr(args, "divergence_caption_threshold", None)
    if div_threshold is None:
        div_threshold = DIVERGENCE_CAPTION_THRESHOLD

    # --ledger: replay a finished local PGN into a flat per-move CSV for post-show stats
    # (task 0204). A finite snapshot export, not a live stream, so it requires --pgn and
    # short-circuits before the live serve/print paths.
    if getattr(args, "ledger", None) is not None:
        if not args.pgn:
            print("# broadcast --ledger requires --pgn (no live feed)", file=sys.stderr)
            return 2
        from chess_equity.broadcast import write_ledger

        ingestor = BroadcastIngestor(
            _build_broadcast_feed(args),
            model,
            white_elo=args.white_elo,
            black_elo=args.black_elo,
            clock_aware=args.clock_aware,
            engine=cp_engine,
            select=selector,
        )
        with open(args.pgn, encoding="utf-8") as fh:
            events = ingestor.ingest_snapshot(fh.read())
        with open(args.ledger, "w", encoding="utf-8", newline="") as fh:
            rows = write_ledger(events, fh)
        print(f"wrote {rows} move rows to {args.ledger}", file=sys.stderr)
        return 0

    # --captions-vtt: replay a finished local PGN into a timestamped WebVTT subtitle
    # track (one cue per graded caster line, task 0211). Like --ledger, a finite
    # snapshot export, not a live stream, so it requires --pgn and short-circuits.
    if getattr(args, "captions_vtt", None) is not None:
        if not args.pgn:
            print("# broadcast --captions-vtt requires --pgn (no live feed)", file=sys.stderr)
            return 2
        from chess_equity.broadcast import build_captions_vtt

        ingestor = BroadcastIngestor(
            _build_broadcast_feed(args),
            model,
            white_elo=args.white_elo,
            black_elo=args.black_elo,
            clock_aware=args.clock_aware,
            engine=cp_engine,
            select=selector,
        )
        with open(args.pgn, encoding="utf-8") as fh:
            events = ingestor.ingest_snapshot(fh.read())
        vtt = build_captions_vtt(
            events, auto_follow=auto_follow, divergence_threshold=div_threshold
        )
        with open(args.captions_vtt, "w", encoding="utf-8") as fh:
            fh.write(vtt)
        cues = vtt.count(" --> ")
        print(f"wrote {cues} caption cue(s) to {args.captions_vtt}", file=sys.stderr)
        return 0

    # --captions-srt: same finished-PGN replay as --captions-vtt, but writes the caster
    # cues as an SRT (SubRip) subtitle track for non-web editors (task 0229). Reuses the
    # exact same _caption_cues timeline, so the SRT is cue-for-cue identical to the VTT.
    if getattr(args, "captions_srt", None) is not None:
        if not args.pgn:
            print("# broadcast --captions-srt requires --pgn (no live feed)", file=sys.stderr)
            return 2
        from chess_equity.broadcast import build_captions_srt

        ingestor = BroadcastIngestor(
            _build_broadcast_feed(args),
            model,
            white_elo=args.white_elo,
            black_elo=args.black_elo,
            clock_aware=args.clock_aware,
            engine=cp_engine,
            select=selector,
        )
        with open(args.pgn, encoding="utf-8") as fh:
            events = ingestor.ingest_snapshot(fh.read())
        srt = build_captions_srt(
            events, auto_follow=auto_follow, divergence_threshold=div_threshold
        )
        with open(args.captions_srt, "w", encoding="utf-8") as fh:
            fh.write(srt)
        cues = srt.count(" --> ")
        print(f"wrote {cues} caption cue(s) to {args.captions_srt}", file=sys.stderr)
        return 0

    if args.serve_sse is not None:
        from chess_equity.broadcast import (
            FocusStatus,
            PinChannel,
            overlay_events,
            serve_sse,
        )

        # A live round (--round/--url) may be tuned into before its first move: keep
        # polling (no idle stop) and send keep-alive heartbeats so the connection
        # survives the quiet wait. A local --pgn replay is finite, so it still
        # terminates on idle (max_idle_polls=1, no heartbeat).
        is_live = bool(args.round or args.url)

        # Caster pin INPUT channel (task 0261): under `--board auto`, expose `POST /pin`
        # so a caster can hold focus on a board mid-stream. One shared channel feeds both
        # the handler (writer) and every overlay_events generator (reader/director).
        pin_channel = PinChannel() if auto_follow else None
        # Caster status OUTPUT channel (task 0265): under `--board auto`, expose
        # `GET /focus` so a caster control surface can read back which board is live and
        # why. Shared between the handler (reader) and the overlay_events director (writer).
        focus_status = FocusStatus() if auto_follow else None

        def make_events():
            ingestor = BroadcastIngestor(
                _build_broadcast_feed(args),
                model,
                white_elo=args.white_elo,
                black_elo=args.black_elo,
                clock_aware=args.clock_aware,
                engine=cp_engine,
                select=selector,
            )
            return overlay_events(
                ingestor,
                auto_follow=auto_follow,
                bias_player=bias_player,
                pin_channel=pin_channel,
                focus_status=focus_status,
                interval=args.interval,
                max_polls=args.max_polls,
                max_idle_polls=None if is_live else 1,
                heartbeat=is_live,
                on_reconnect=log_reconnect,
            )

        serve_sse(
            make_events,
            port=args.serve_sse,
            directory=_overlay_static_dir(),
            pin_channel=pin_channel,
            focus_status=focus_status,
            log=lambda msg: print(msg, file=sys.stderr),
        )
        return 0

    feed = _build_broadcast_feed(args)
    ingestor = BroadcastIngestor(
        feed,
        model,
        white_elo=args.white_elo,
        black_elo=args.black_elo,
        clock_aware=args.clock_aware,
        engine=cp_engine,
        select=selector,
    )

    # --captions: a human caster sentence per graded move (TTS/chat-ready) instead of
    # the machine JSONL stream (task 0190). The live counterpart to the offline reel.
    captions = getattr(args, "captions", False)

    # --board auto (task 0256): drive the JSONL stream through the overlay bridge so its
    # server-side `focus` cut events ride alongside the position events the overlay reads.
    # The live `--captions` stream is plain text (no routing metadata), so auto-follow is a
    # no-op there; the offline `--captions-srt/--captions-vtt` export DOES thread each cut's
    # director cue into the subtitle track (task 0263, handled in the export blocks above).
    if auto_follow and not captions:
        from chess_equity.broadcast import HEARTBEAT, overlay_events

        is_live = bool(args.round or args.url)
        for ev in overlay_events(
            ingestor,
            auto_follow=True,
            bias_player=bias_player,
            interval=args.interval,
            max_polls=args.max_polls,
            max_idle_polls=None if is_live else 1,
            on_reconnect=log_reconnect,
        ):
            if ev is HEARTBEAT:
                continue
            out.write(json.dumps(ev) + "\n")
            out.flush()
        stats = ingestor.stats
        print(
            f"# {stats.events} events over {stats.polls} polls "
            f"({stats.errors} feed errors, {stats.reconnects} reconnect(s), "
            f"max backoff {stats.max_backoff_s:.0f}s), "
            f"max equity compute {stats.max_compute_ms:.1f} ms",
            file=sys.stderr,
        )
        return 0

    def emit(event: MoveEvent) -> None:
        if captions:
            from chess_equity.broadcast import live_caption

            line = live_caption(event, divergence_threshold=div_threshold)
            if line is not None:
                out.write(line + "\n")
                out.flush()
            return
        out.write(_event_line(event) + "\n")
        out.flush()

    # Emit the overlay "game" metadata event (player names + ratings) once per game,
    # before its moves, so the overlay name-plates are populated (task 0047). In
    # --captions mode announce the pairing as a plain caster intro line instead.
    def emit_game(game: GameEvent) -> None:
        if captions:
            white = game.white_name or "White"
            black = game.black_name or "Black"
            we = f" ({game.white_elo})" if game.white_elo else ""
            be = f" ({game.black_elo})" if game.black_elo else ""
            out.write(f"🎙 {white}{we} vs {black}{be}\n")
            out.flush()
            return
        out.write(json.dumps(game.to_overlay()) + "\n")
        out.flush()

    ingestor.on_game = emit_game

    # A live feed runs until interrupted, so keep retrying a dropped feed forever
    # (max_idle_polls=None, --max-polls caps it); a finite --pgn replay still terminates
    # on idle (max_idle_polls=1). interval=0 for replays keeps tests/CI instant.
    is_live = bool(args.round or args.url)
    stats = ingestor.run(
        emit,
        interval=args.interval,
        max_polls=args.max_polls,
        max_idle_polls=None if is_live else 1,
        on_reconnect=log_reconnect,
    )
    print(
        f"# {stats.events} events over {stats.polls} polls "
        f"({stats.errors} feed errors, {stats.reconnects} reconnect(s), "
        f"max backoff {stats.max_backoff_s:.0f}s), "
        f"max equity compute {stats.max_compute_ms:.1f} ms",
        file=sys.stderr,
    )
    return 0


def _run_highlights(args: argparse.Namespace, model: EquityModel) -> int:
    """Detect drama/clutch moments in a game and print the highlight reel (task 0020)."""
    from chess_equity.drama import DramaEvent, detect, highlights

    with open(args.pgn, encoding="utf-8") as fh:
        pgn_text = fh.read()
    ingestor = BroadcastIngestor(
        feed=LocalPgnFeed(pgn_text),  # the feed is unused; events come from the snapshot
        model=model,
        white_elo=args.white_elo,
        black_elo=args.black_elo,
    )
    events = ingestor.ingest_snapshot(pgn_text)

    if args.json:
        reel = highlights(events, top=args.top)
        print(json.dumps([d.to_dict() for d in reel], indent=2))
        return 0

    in_order = detect(events)
    if not in_order:
        print("# no drama detected (a quiet game, or muted swings on the baseline model)")
        return 0
    print(f"# {len(in_order)} drama moment(s):")
    for d in in_order:
        print(f"{d.ply:3d}. [{d.kind:10s} {d.magnitude:.2f}] {d.headline}")
    reel: List[DramaEvent] = highlights(events, top=args.top)
    print(f"\n# top {len(reel)} highlight(s) by magnitude:")
    for d in reel:
        print(f"  {d.magnitude:.2f}  {d.kind:10s} ply {d.ply}: {d.headline}")
    return 0


def _run_reel(args: argparse.Namespace, model: EquityModel) -> int:
    """Export a ranked auto-highlight reel as JSON + markdown (task 0168).

    Replays a committed PGN through the broadcast pipeline, ranks the drama, and
    writes ``reel.json`` + ``reel.md`` to ``--out-dir`` (or prints markdown to stdout).
    """
    import os

    from chess_equity import reel as reel_mod

    with open(args.pgn, encoding="utf-8") as fh:
        pgn_text = fh.read()
    ingestor = BroadcastIngestor(
        feed=LocalPgnFeed(pgn_text),  # the feed is unused; events come from the snapshot
        model=model,
        white_elo=args.white_elo,
        black_elo=args.black_elo,
    )
    events = ingestor.ingest_snapshot(pgn_text)

    # --min-magnitude FLOOR (task 0240): drop trivial swings before ranking. Build the
    # full ranked reel first, filter below the 0..1 floor (logging how many were cut so
    # nothing is silently truncated), then apply --top to the qualified pool.
    reel = reel_mod.build_reel(events)
    floor = getattr(args, "min_magnitude", None)
    if floor is not None and not (0.0 <= floor <= 1.0):
        print(
            f"error: --min-magnitude must be in [0, 1] (got {floor:g})", file=sys.stderr
        )
        return 1
    if floor is not None:
        reel, dropped = reel_mod.drop_below_magnitude(reel, floor)
        if dropped:
            print(
                f"--min-magnitude {floor:g}: dropped {dropped} moment(s) below the floor",
                file=sys.stderr,
            )
    if args.top is not None:
        reel = reel[: args.top]

    # Human-vs-engine DIVERGENCE category (task 0272): a SEPARATE ranked list — the
    # moves where the rating-aware bar most disagrees with the engine (cp-implied) bar.
    # Computed from the raw events (drama events drop cp), independent of the drama
    # ranking + the --min-magnitude floor (a quiet move can be a huge divergence). It
    # rides alongside the drama reel in the json + markdown artifacts.
    divergence = reel_mod.detect_divergence(events, top=args.top)

    # --round (task 0198): a cross-game ROUND recap. The pooling is already done — a
    # multi-game PGN tags every event with its game_id and the drama detector is
    # stateless, so `reel` above already ranks moments across all boards. --round adds
    # the source labels (board # + pairing per moment) and a round-framed title so the
    # caster sees which board each pooled swing came from.
    title = args.title
    sources = None
    if getattr(args, "round_recap", False):
        sources = reel_mod.game_sources(pgn_text)
        if title == "Highlight reel":  # default unchanged by the user → frame as a round
            title = "Round recap"

    # --html PATH writes ONE self-contained HTML clip player (opens offline). It can
    # stand alone (print to stdout when PATH is "-") or sit alongside --out-dir's
    # json+md. Either flag may be used on its own.
    if args.html is not None and args.out_dir is None:
        html_doc = reel_mod.render_html(reel, title=title, sources=sources)
        if args.html == "-":
            print(html_doc, end="")
        else:
            with open(args.html, "w", encoding="utf-8") as fh:
                fh.write(html_doc)
            print(f"wrote {len(reel)} moment(s): {args.html}", file=sys.stderr)
        return 0

    # --srt PATH writes the narration as a standalone SRT subtitle file (or stdout
    # when PATH is "-"). Like --html, it can stand alone or sit alongside --out-dir.
    if args.srt is not None and args.out_dir is None:
        srt_doc = reel_mod.build_srt(reel)
        if args.srt == "-":
            print(srt_doc, end="")
        else:
            with open(args.srt, "w", encoding="utf-8") as fh:
                fh.write(srt_doc)
            print(f"wrote {len(reel)} moment(s): {args.srt}", file=sys.stderr)
        return 0

    # --chapters PATH writes the reel as VOD chapter markers (HH:MM:SS Title lines a
    # caster pastes into a YouTube/Twitch description). Like --srt it can stand alone
    # (stdout on "-") or sit alongside --out-dir's json+md.
    if args.chapters is not None and args.out_dir is None:
        chapters_doc = reel_mod.build_chapters(reel)
        if args.chapters == "-":
            print(chapters_doc, end="")
        else:
            with open(args.chapters, "w", encoding="utf-8") as fh:
                fh.write(chapters_doc)
            print(f"wrote {len(reel)} moment(s): {args.chapters}", file=sys.stderr)
        return 0

    # --posters DIR writes one static SVG poster per ranked moment (a shareable social
    # card). Like --html/--srt it can stand alone or sit alongside --out-dir's json+md.
    if args.posters is not None and args.out_dir is None:
        paths = reel_mod.write_posters(reel, args.posters, sources=sources)
        print(f"wrote {len(paths)} poster(s): {args.posters}", file=sys.stderr)
        return 0

    if args.out_dir is None:
        print(
            reel_mod.render_markdown(
                reel, title=title, sources=sources, divergence=divergence
            )
        )
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "reel.json")
    md_path = os.path.join(args.out_dir, "reel.md")
    written = [json_path, md_path]
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(
            reel_mod.render_json(
                reel, title=title, sources=sources, divergence=divergence
            )
            + "\n"
        )
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(
            reel_mod.render_markdown(
                reel, title=title, sources=sources, divergence=divergence
            )
        )
    if args.html is not None:
        html_path = args.html if args.html != "-" else os.path.join(args.out_dir, "reel.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(reel_mod.render_html(reel, title=title, sources=sources))
        written.append(html_path)
    if args.srt is not None:
        srt_path = args.srt if args.srt != "-" else os.path.join(args.out_dir, "reel.srt")
        with open(srt_path, "w", encoding="utf-8") as fh:
            fh.write(reel_mod.build_srt(reel))
        written.append(srt_path)
    if args.chapters is not None:
        chapters_path = (
            args.chapters if args.chapters != "-"
            else os.path.join(args.out_dir, "reel.chapters.txt")
        )
        with open(chapters_path, "w", encoding="utf-8") as fh:
            fh.write(reel_mod.build_chapters(reel))
        written.append(chapters_path)
    if args.posters is not None:
        poster_dir = args.posters
        written.extend(reel_mod.write_posters(reel, poster_dir, sources=sources))
    print(f"wrote {len(reel)} moment(s): {', '.join(written)}", file=sys.stderr)
    return 0


def _run_data_stamp(args: argparse.Namespace) -> int:
    """Backfill the source-month sidecar on an already-built dataset (task 0127)."""
    from pathlib import Path

    from chess_equity.data.source_month import write_source_month

    if not Path(args.path).exists():
        print(f"error: dataset not found: {args.path}", file=sys.stderr)
        return 1
    try:
        side = write_source_month(args.path, args.month)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"stamped {args.month} -> {side}")
    return 0


def _run_data(args: argparse.Namespace) -> int:
    if args.data_command == "stamp":
        return _run_data_stamp(args)

    # Imported lazily so the common ``eval`` path never pays for the data deps.
    from chess_equity.data.build import build_dataset, month_url

    pgn = args.pgn
    if pgn is None:
        if args.month is None:
            print("error: provide --pgn <file> (or --month to auto-download)", file=sys.stderr)
            return 1
        from urllib.error import URLError

        from chess_equity.data.download import (
            APPROX_DUMP_SIZE_GB,
            DEFAULT_DUMP_DIR,
            data_extra_available,
            download_month,
        )

        # Fail fast, before any network I/O: a month dump is a ~30 GB ``.zst`` that
        # only the 'data' extra can read, so a missing extra should error in seconds,
        # not after the download finishes (task 0071).
        if not data_extra_available():
            print(
                "error: reading .zst month dumps needs the 'data' extra: "
                "pip install 'chess-equity[data]'",
                file=sys.stderr,
            )
            return 1
        dump_dir = args.dump_dir or DEFAULT_DUMP_DIR
        print(
            f"note: the {args.month} dump is ~{APPROX_DUMP_SIZE_GB} GB compressed; "
            f"streaming to {dump_dir} (resumable, cached between runs)",
            file=sys.stderr,
        )

        def _progress(done: int, total: Optional[int]) -> None:
            mb = done / 1e6
            if total:
                print(f"\rdownloading {month_url(args.month)}: {mb:.0f}/{total / 1e6:.0f} MB",
                      end="", file=sys.stderr)
            else:
                print(f"\rdownloading {month_url(args.month)}: {mb:.0f} MB", end="", file=sys.stderr)

        try:
            dump = download_month(args.month, dest_dir=dump_dir, progress=_progress)
        except (URLError, OSError, RuntimeError) as exc:
            print(f"\nerror: downloading {args.month} dump: {exc}", file=sys.stderr)
            return 1
        print(f"\nfetched {dump}", file=sys.stderr)
        pgn = str(dump)
    # Clock-coverage diagnostic (task 0249): tally the fraction of rows carrying
    # [%clk] in the single streaming write pass, so a candidate dump can be vetted
    # for clock coverage before the expensive attended validation run.
    from chess_equity.clock_coverage import ClockCoverage, format_coverage

    coverage = ClockCoverage()
    try:
        out = build_dataset(
            pgn,
            args.out,
            sample=args.sample,
            fmt=args.format,
            include_fen=args.with_fen,
            partition=args.partition,
            # Stamp the source month when the dump came from --month, so the leakage
            # guard can read it back from the sidecar (task 0127).
            source_month=args.month,
            clock_coverage=coverage,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    print(format_coverage(coverage))
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    # Evidence-index drift guard (task 0219): a zero-data cross-check of reports/SUMMARY.md
    # against the committed real-data report headers. Runs before anything touches a
    # dataset, so it needs no --data.
    if getattr(args, "check_index", False):
        from chess_equity.validate.index_guard import check_index

        result = check_index()
        if result.ok:
            print("SUMMARY.md evidence index is consistent with the committed reports.")
            return 0
        print("SUMMARY.md evidence-index drift:", file=sys.stderr)
        for problem in result.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    if not args.data:
        print("error: validate needs --data (or use --check-index)", file=sys.stderr)
        return 2

    # Lazy import: keeps the eval path free of the data loader.
    from chess_equity.data.build import load_rows

    # Clock-coverage diagnostic (task 0249): a cheap, model-free slice that reports the
    # built dataset's [%clk] coverage and per-clock_band distribution over clock-bearing
    # rows. Runs before any predictor is built — vetting a candidate dump for clock
    # coverage shouldn't pay for model scoring or a holdout split.
    if getattr(args, "slice", None) == "clock":
        from chess_equity.clock_coverage import coverage_of, format_coverage

        try:
            rows = load_rows(args.data)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if not rows:
            print(f"error: no rows in {args.data}", file=sys.stderr)
            return 1
        cov = coverage_of(rows)
        report = format_coverage(cov, clock_bearing_only=True)
        if args.out:
            from pathlib import Path

            Path(args.out).write_text(report + "\n", encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(report)
        return 0
    from chess_equity.validate.harness import (
        PREDICTORS,
        build_predictors,
        compare_ece_to_baseline,
        compare_to_baseline,
        evaluate,
        format_baseline_comparison,
        format_ece_comparison,
        format_report,
        gate_verdicts,
    )

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    try:
        predictors = build_predictors(requested)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 1

    # Custom wdl-a artifact (task 0164): score wdl-a from a refit artifact (e.g. fit on a
    # different month) so an eval dump can be a genuine held-out test for it. The leakage
    # guard below reads the SAME artifact's fit_month, so a cross-dump refit reads as
    # held-out rather than tripping the in-distribution warning.
    wdl_a_artifact = getattr(args, "wdl_a_artifact", None)
    if wdl_a_artifact and "wdl-a" in predictors:
        from chess_equity.wdl_regression import load_wdl_a_model

        _custom_wdl_a = load_wdl_a_model(wdl_a_artifact)

        def _wdl_a_from_artifact(row, _m=_custom_wdl_a):
            return _m.predict_white_equity(
                row.cp_eval, row.white_elo, row.black_elo, row.ply, row.tc_bucket
            )

        predictors["wdl-a"] = _wdl_a_from_artifact
    try:
        rows = load_rows(args.data)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not rows:
        print(f"error: no rows in {args.data}", file=sys.stderr)
        return 1

    # Seed-stability (task 0156): --seeds re-runs the gate under each seed so a PASS can be
    # shown to survive re-sampling, not just hold at the committed seed 0. Parse the list
    # now (before the holdout split reassigns `rows`) and keep the full row set for it.
    seed_list: List[int] = []
    if getattr(args, "seeds", None):
        try:
            seed_list = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
        except ValueError:
            print(
                f"error: --seeds must be a comma-separated list of integers, got "
                f"{args.seeds!r}",
                file=sys.stderr,
            )
            return 1
        if not seed_list:
            print("error: --seeds was empty", file=sys.stderr)
            return 1
    full_rows = list(rows)

    # N-aware shrinkage knob (task 0163): with --shrink-wdl-a-k > 0, swap the wdl-a
    # predictor for one that blends toward the rating-blind baseline by per-cell weight
    # n/(n+k), so sparse high-rating cells (the 2000-2399 ECE blowup) fall back to the
    # baseline. Counts come from the full dataset (the support), computed before any
    # holdout split. K=0 leaves wdl-a untouched, so the default run is byte-identical.
    shrink_k = getattr(args, "shrink_wdl_a_k", 0.0) or 0.0
    if shrink_k > 0 and "wdl-a" in predictors:
        from chess_equity.validate.shrinkage import make_shrunk_predictor

        predictors["wdl-a"] = make_shrunk_predictor(full_rows, shrink_k)
        title_shrink = f" [wdl-a shrunk k={shrink_k:g}]"
    else:
        title_shrink = ""

    # The dataset's own source month, read from its sidecar (task 0127): the recorded
    # truth of which Lichess month --data was drawn from. It is what the leakage guard
    # uses to default --eval-month (so the operator can't silently get it wrong), and is
    # surfaced in the report title.
    from chess_equity.data.source_month import read_source_month

    data_month = read_source_month(args.data)

    # Leakage guard (task 0112): if the eval dataset's source month is a model's own
    # training month, its scores are memorization, not held-out evidence. Eval-month
    # precedence: explicit --eval-month, else the dataset's stamped source month (task
    # 0127), else inferred from the dataset path; --strict refuses. Resolved once so the
    # detection and the report's warning block agree.
    from chess_equity.validate.leakage import (
        detect_leakage,
        format_leakage_warning,
        infer_month_from_path,
        leakage_line,
        model_fit_months,
    )

    eval_month = (
        getattr(args, "eval_month", None)
        or data_month
        or infer_month_from_path(args.data)
    )
    leaks = detect_leakage(
        eval_month, model_fit_months(requested, wdl_a_path=wdl_a_artifact)
    )
    if leaks:
        print("warning: " + leakage_line(leaks, eval_month), file=sys.stderr)
        if getattr(args, "strict", False):
            print(
                "error: refusing (--strict) — eval month overlaps a model's training "
                "month; re-run on a held-out month",
                file=sys.stderr,
            )
            return 2

    title = f"Validation report — {args.data}{title_shrink}"
    if data_month:
        title += f" (data month: {data_month})"

    train: List = []
    if args.holdout is not None:
        from chess_equity.validate.split import game_level_split

        try:
            train, rows = game_level_split(
                rows, test_fraction=args.holdout, seed=args.seed
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        title += (
            f" (held-out test: {len(rows)} rows / "
            f"{len({r.game_id for r in rows})} games; "
            f"train: {len(train)} rows, seed {args.seed})"
        )

    # Post-hoc Platt recalibration of maia2 (task 0166): with --recalibrate-maia2, fit a
    # two-parameter logistic on the logit of maia2's prediction over the --holdout TRAIN
    # split (game-disjoint from the eval rows) and apply it at eval, to repair maia2's
    # high-rating ECE blowup without re-ordering predictions. Off by default, so the
    # committed run is byte-identical. Requires --holdout so the calibration set is genuinely
    # held-out from the eval rows (fitting on the eval rows would leak).
    if getattr(args, "recalibrate_maia2", False):
        if args.holdout is None:
            print(
                "error: --recalibrate-maia2 needs --holdout so the recalibrator is fit on "
                "a held-out train split, not the eval rows",
                file=sys.stderr,
            )
            return 1
        if "maia2" not in predictors:
            print(
                "error: --recalibrate-maia2 needs maia2 in --models (nothing to recalibrate)",
                file=sys.stderr,
            )
            return 1
        from chess_equity.validate.recalibration import make_recalibrated_predictor

        predictors["maia2"] = make_recalibrated_predictor(train, predictors["maia2"])
        title += " [maia2 Platt-recalibrated]"

    # Board models (maia2) read row.fen and may need torch — surface those failures as
    # a clean message rather than a traceback. ValueError = dataset built without
    # --with-fen; Maia2NotInstalled = no torch/checkpoint.
    from chess_equity.maia2 import Maia2NotInstalled

    try:
        reports = evaluate(rows, predictors, bins=args.ece_bins)
    except (ValueError, Maia2NotInstalled) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    # Significance: paired-bootstrap CI on each model's metric delta vs the baseline,
    # so the report says whether a win clears zero or is just noise (task 0060). Needs
    # the baseline plus at least one other predictor; --bootstrap 0 turns it off.
    # Computed before the report so the gate verdict can consume the CIs and require a
    # *significant* win on the headline metric, not just a point delta (task 0069).
    baseline_name = "baseline"
    comparisons = None
    if args.bootstrap > 0 and baseline_name in predictors and len(predictors) > 1:
        comparisons = compare_to_baseline(
            rows,
            predictors,
            baseline=baseline_name,
            n_resamples=args.bootstrap,
            seed=args.seed,
        )

    # Per-slice significance CIs (task 0068), computed up front so the report's worst-slice
    # line can carry a clears-zero / straddles-zero caveat (task 0161) and so the per-slice
    # CI section below can reuse the same object. Needs the baseline + a challenger.
    h2h_ci = None
    if args.bootstrap > 0 and baseline_name in predictors and len(predictors) > 1:
        from chess_equity.validate.harness import head_to_head_slice_cis

        h2h_ci = head_to_head_slice_cis(
            rows,
            predictors,
            baseline_name=baseline_name,
            n_resamples=args.bootstrap,
            seed=args.seed,
        )

    report = format_report(
        reports, title=title, comparisons=comparisons, head_to_head_cis=h2h_ci
    )
    leak_block = format_leakage_warning(leaks, eval_month)
    if leak_block:
        # Insert just below the H1 so the warning leads the artifact, above the gate verdict.
        head, _, body = report.partition("\n")
        report = f"{head}\n\n{leak_block.rstrip()}\n{body}"

    if comparisons:
        section = format_baseline_comparison(comparisons)
        if section:
            report = report + "\n" + section

    # Calibration error bars: a bin-resampling CI on each predictor's ECE plus the ECE
    # delta vs baseline (task 0072). ECE has no per-row term, so this is a separate
    # bootstrap from the significance section above; same --bootstrap budget / seed.
    if args.bootstrap > 0 and baseline_name in predictors:
        ece_cis = compare_ece_to_baseline(
            rows,
            predictors,
            baseline=baseline_name,
            bins=args.ece_bins,
            n_resamples=args.bootstrap,
            seed=args.seed,
        )
        ece_section = format_ece_comparison(ece_cis)
        if ece_section:
            report = report + "\n" + ece_section

    # Per-slice significance: a paired-bootstrap CI on the head-to-head log-loss delta
    # within each rating / clock / phase slice (task 0068), so the report can tell a real
    # band-level equity win from small-n noise — the overall CI (0060) can't. Same
    # --bootstrap budget / seed; needs the baseline plus at least one challenger.
    if h2h_ci is not None and h2h_ci.slices:
        from chess_equity.validate.harness import format_head_to_head_cis

        report = report + "\n" + format_head_to_head_cis(h2h_ci)

    # By time-control bucket (task 0155): the torch-free step toward the streaming /
    # time-pressure north star — does equity still beat the centipawn baseline within each
    # bullet/blitz/rapid/classical class? A point gate (no bootstrap), so it's emitted
    # whenever there's a baseline + a challenger; small buckets are flagged underpowered,
    # not silently passed.
    from chess_equity.validate.harness import format_tc_bucket_gate, tc_bucket_gate

    tc_gate = tc_bucket_gate(rows, predictors, baseline_name=baseline_name)
    if tc_gate is not None and tc_gate.buckets:
        report = report + "\n" + format_tc_bucket_gate(tc_gate)

    # 'Good moves read as good' (task 0117): the positive half of the thesis. Per
    # consecutive ply-pair, compare each predictor's mover-POV equity swing to the
    # engine's cp swing — does the bar give engine-approved moves visible upside, not a
    # saturated ~0? Cheap (no bootstrap); skipped when the dataset has no adjacent
    # ply-pairs to pair into moves.
    from chess_equity.validate.goodmoves import format_good_moves, measure_good_moves

    good_moves = measure_good_moves(rows, predictors)
    good_section = format_good_moves(
        good_moves, baseline=baseline_name, see_also="reports/goodmoves_real.md"
    )
    if good_section:
        report = report + "\n" + good_section

    # Cutoff-robustness sweep (task 0157): the good/blunder cutoffs in the section above
    # are arbitrary defaults, so re-measure the Δgood > Δblunder direction across a grid
    # of good × blunder cutoffs and report whether it holds in every cell (or names where
    # it breaks). Cheap (no bootstrap); skipped when there are no move-pairs to score.
    from chess_equity.validate.goodmoves import (
        format_good_moves_sweep,
        sweep_good_moves,
    )

    sweeps = sweep_good_moves(rows, predictors)
    sweep_section = format_good_moves_sweep(sweeps)
    if sweep_section:
        report = report + "\n" + sweep_section

    # Seed stability (task 0156): if --seeds was given, re-run the gate under each seed and
    # append a stability section so the committed-seed PASS is shown to survive re-sampling
    # (not a cherry-picked draw). Uses the full pre-split rows, re-drawing the --holdout
    # split and bootstrap per seed; same --bootstrap budget, --ece-bins, and --min-n floor.
    if seed_list:
        from chess_equity.validate.seed_stability import (
            format_seed_stability,
            reseed_stability,
        )

        stability = reseed_stability(
            full_rows,
            predictors,
            seeds=seed_list,
            holdout=args.holdout,
            baseline_name=baseline_name,
            n_resamples=args.bootstrap,
            ece_bins=args.ece_bins,
            min_n=args.min_n,
        )
        stability_section = format_seed_stability(stability)
        if stability_section:
            report = report + "\n" + stability_section

    if args.out:
        from pathlib import Path

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)

    if args.calibration:
        # Per-rating-band reliability curves for the first requested predictor (task 0027).
        from pathlib import Path

        from chess_equity.validate.calibration import band_reliability, format_calibration_report

        name = requested[0]
        # --bootstrap > 0 (the default) adds a bin-resampling ECE CI per band (task 0076)
        # so the band-level calibration claims carry error bars; --bootstrap 0 turns it off.
        # When the report's predictor isn't the baseline itself, thread the baseline in as a
        # second predictor (task 0089) so each band also shows the paired ECE delta vs baseline.
        # Use the local `predictors` dict so any --shrink-wdl-a-k override (task 0163) is
        # reflected here too; fall back to the global registry for a baseline not in --models.
        cal_predictor = predictors.get(name) or PREDICTORS[name]
        cal_baseline = (
            (predictors.get(baseline_name) or PREDICTORS[baseline_name])
            if name != baseline_name
            else None
        )
        bands = band_reliability(
            rows,
            cal_predictor,
            baseline=cal_baseline,
            bins=args.ece_bins,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
        cal = format_calibration_report(
            bands, predictor_name=name, title=f"Calibration by rating band — {args.data}"
        )
        Path(args.calibration).parent.mkdir(parents=True, exist_ok=True)
        Path(args.calibration).write_text(cal + "\n", encoding="utf-8")
        print(f"wrote {args.calibration}")

    if args.plots:
        # Render the same per-band reliability data as a calibration-curve PNG (task 0036).
        from pathlib import Path

        from chess_equity.validate.calibration import band_reliability
        from chess_equity.validate.plots import MatplotlibNotInstalled, save_reliability_plot

        name = requested[0]
        bands = band_reliability(rows, predictors[name], bins=args.ece_bins)
        Path(args.plots).parent.mkdir(parents=True, exist_ok=True)
        try:
            save_reliability_plot(
                bands, args.plots, title=f"Reliability by rating band — {name}"
            )
        except (MatplotlibNotInstalled, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {args.plots}")

    # Machine-checkable gate (task 0115): with --gate, the prose PASS/FAIL verdict drives
    # the *exit code* so CI / the autonomous loop can assert the thesis programmatically.
    # Exit 0 only if every rating-conditioned predictor beats the baseline on log-loss AND
    # Brier; nonzero (2) if any FAILS; 3 (misuse) if there is no challenger to gate. With
    # --bootstrap > 0 the gate is also significance-aware (task 0069): PASS requires the
    # log-loss delta CI to clear zero, so a noisy point win FAILs. --bootstrap 0 keeps the
    # point-only gate.
    if args.gate:
        verdicts = gate_verdicts(
            reports,
            baseline_name=baseline_name,
            comparisons=comparisons,
            min_n=args.min_n,
        )
        if not verdicts:
            print(
                "error: --gate needs a rating-conditioned predictor besides "
                f"'{baseline_name}' to gate against",
                file=sys.stderr,
            )
            return 3
        # Underpowered (task 0132): the held-out sample is below the n floor, so a PASS
        # would be untrustworthy. Distinct exit code 4 (INCONCLUSIVE) so CI / the loop can
        # tell "couldn't conclude" apart from PASS (0) and FAIL (2).
        if verdicts[0].underpowered:
            print(
                f"GATE: INCONCLUSIVE — held-out n={verdicts[0].held_out_n} is below the "
                f"n>={args.min_n} floor; a tiny-n win is not proof (pass --min-n 0 to "
                "override)",
                file=sys.stderr,
            )
            return 4
        gated = bool(comparisons)
        criterion = (
            "log-loss AND Brier with a significant (CI-clears-zero) log-loss win"
            if gated
            else "log-loss AND Brier"
        )
        failed = [v for v in verdicts if not v.passed]
        if failed:
            names = ", ".join(v.name for v in failed)
            print(
                f"GATE: FAIL — {names} did not beat '{baseline_name}' on {criterion}",
                file=sys.stderr,
            )
            return 2
        passed = ", ".join(v.name for v in verdicts)
        print(f"GATE: PASS — {passed} beat '{baseline_name}' on {criterion}")
    return 0


def _run_divergence(args: argparse.Namespace) -> int:
    """Measure how far the equity bar diverges from the Stockfish bar (task 0171)."""
    from chess_equity.data.build import load_rows
    from chess_equity.data.source_month import read_source_month
    from chess_equity.validate.divergence import format_divergence, measure_divergence
    from chess_equity.validate.harness import build_predictors

    try:
        predictors = build_predictors([args.equity, args.stockfish])
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 1
    try:
        rows = load_rows(args.data)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not rows:
        print(f"error: no rows in {args.data}", file=sys.stderr)
        return 1

    report = measure_divergence(
        rows,
        predictors[args.equity],
        equity_name=args.equity,
        stockfish=predictors[args.stockfish],
        stockfish_name=args.stockfish,
    )
    month = read_source_month(args.data) or "unknown"
    header = (
        f"# Divergence — real Lichess dump `{month}`, n={len(rows)} "
        f"(equity=`{args.equity}` vs Stockfish=`{args.stockfish}`)"
    )
    text = format_divergence(report, header=header)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def _run_precompute(args: argparse.Namespace) -> int:
    """Evaluate a whole game's equity in one cache-backed pass → UI-ready JSON (0012)."""
    from chess_equity.cache import CachingEquityModel
    from chess_equity.precompute import precompute_game

    try:
        with open(args.pgn, encoding="utf-8") as fh:
            pgn_text = fh.read()
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    base_model = build_model(args.model, depth=args.depth)
    # Be honest about what the bar is: the default 'baseline' is the rating-blind
    # material/centipawn placeholder, not Maia-2 (task 0081). Warn so the web demo's
    # equity isn't mistaken for the real rating-conditioned model.
    warning = placeholder_equity_warning(base_model)
    if warning:
        print(warning, file=sys.stderr)
    try:
        personalized = _apply_profiles(base_model, args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    model = CachingEquityModel(personalized, path=args.cache)
    try:
        result = precompute_game(
            model, pgn_text, white_elo=args.white_elo, black_elo=args.black_elo
        )
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    payload = json.dumps(result.to_dict(), indent=2)
    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(payload)
    print(
        f"# {len(result.plies)} plies, {result.compute_ms:.1f} ms total "
        f"({result.compute_ms / max(len(result.plies), 1):.2f} ms/ply), "
        f"cache {result.cache_hits} hit / {result.cache_misses} miss",
        file=sys.stderr,
    )
    return 0


def _run_train(args: argparse.Namespace) -> int:
    """Fit the wdl-a rating-conditioned WDL model and write the artifact (task 0004)."""
    from chess_equity.data.build import load_rows
    from chess_equity.wdl_regression import default_artifact_path, fit

    try:
        rows = load_rows(args.data)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not rows:
        print(f"error: no rows in {args.data}", file=sys.stderr)
        return 1
    model = fit(
        rows,
        lr=args.lr,
        iters=args.iters,
        l2=args.l2,
        source_month=getattr(args, "train_month", None),
    )
    out = args.out or str(default_artifact_path())
    model.save(out)
    meta = model.meta or {}
    print(
        f"wrote {out} (n_train={meta.get('n_train')}, "
        f"iters={meta.get('iters')}, final_log_loss={meta.get('final_log_loss'):.4f})"
    )
    return 0


def _run_train_net(args: argparse.Namespace) -> int:
    """Fit the end-to-end board → WDL net (Approach D, task 0013) and save it."""
    from chess_equity.data.build import load_rows
    from chess_equity.wdl_net import default_artifact_path, train_wdl_net

    try:
        rows = load_rows(args.data)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not rows:
        print(f"error: no rows in {args.data}", file=sys.stderr)
        return 1
    if all(r.fen is None for r in rows):
        print(
            f"error: {args.data} has no FEN column; rebuild with --with-fen "
            "(wdl-net learns straight from the board)",
            file=sys.stderr,
        )
        return 1
    net = train_wdl_net(
        rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        train_month=args.train_month,
        log=lambda m: print(m, file=sys.stderr),
    )
    out = args.out or str(default_artifact_path())
    net.save(out)
    print(f"wrote {out} (n_train={net.cfg.n_train}, epochs={net.cfg.epochs})")
    return 0


def _run_personal(args: argparse.Namespace) -> int:
    """Mine a player's profile and (optionally) show how it bends the equity bar."""
    import io

    from chess_equity.personal import (
        PHASES,
        PersonalEquityModel,
        build_profile,
        fetch_user_games,
        phase_of,
    )

    try:
        if args.pgn:
            target = args.name or args.user
            if not target:
                raise ValueError("reading a profile from --pgn needs --name <player> (or --user)")
            with open(args.pgn, encoding="utf-8") as fh:
                profile = build_profile(fh, target, max_games=args.max_games)
        elif args.user:
            pgn = fetch_user_games(args.user, max_games=args.max_games, token=args.token)
            profile = build_profile(io.StringIO(pgn), args.user, max_games=args.max_games)
        else:
            raise ValueError("personal needs --user <name> (live) or --pgn FILE (--name <player>)")
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rows = []
    for phase in PHASES:
        stats = profile.phases.get(phase)
        rows.append(
            {
                "phase": phase,
                "moves": stats.n_moves if stats else 0,
                "acpl": round(stats.avg_cp_loss, 1) if stats else 0.0,
                "blunder_rate": round(stats.blunder_rate, 3) if stats else 0.0,
                "elo_offset": round(profile.phase_offset(phase), 1),
            }
        )

    if args.json:
        print(
            json.dumps(
                {
                    "username": profile.username,
                    "rating": profile.rating,
                    "games": profile.n_games,
                    "moves": profile.total_moves,
                    "overall_acpl": round(profile.overall_acpl, 1),
                    "phases": rows,
                },
                indent=2,
            )
        )
        return 0

    print(f"profile: {profile.username}  (rating ~{profile.rating}, {profile.n_games} games, "
          f"{profile.total_moves} moves, overall ACPL {profile.overall_acpl:.0f})")
    print(f"  {'phase':11s} {'moves':>6s} {'ACPL':>7s} {'blunder%':>9s} {'Elo±':>7s}")
    for r in rows:
        print(f"  {r['phase']:11s} {r['moves']:6d} {r['acpl']:7.1f} "
              f"{100 * r['blunder_rate']:8.1f}% {r['elo_offset']:+7.0f}")

    if args.demo:
        base = build_model("baseline")
        nominal = profile.rating or 1500
        # The profiled player sits at White; the demo shows band-average vs personalized
        # equity for the FEN, exposing the phase-wise gap the offset introduces.
        personal = PersonalEquityModel(base, white_profile=profile)
        phase = phase_of(chess.Board(args.fen))
        band = base.evaluate(args.fen, nominal, nominal)
        mine = personal.evaluate(args.fen, nominal, nominal)
        print()
        print(f"demo ({phase}, both nominally {nominal}):")
        print(f"  band-average  {render_eval(band)}")
        print(f"  personalized  {render_eval(mine)}  (Elo offset {profile.phase_offset(phase):+.0f})")
    return 0



def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="chess-equity", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    import chess_equity.cli.broadcast as _broadcast_cmd
    import chess_equity.cli.data as _data_cmd
    import chess_equity.cli.divergence as _divergence_cmd
    import chess_equity.cli.doctor as _doctor_cmd
    import chess_equity.cli.eval as _eval_cmd
    import chess_equity.cli.grade as _grade_cmd
    import chess_equity.cli.headline as _headline_cmd
    import chess_equity.cli.highlights as _highlights_cmd
    import chess_equity.cli.personal as _personal_cmd
    import chess_equity.cli.precompute as _precompute_cmd
    import chess_equity.cli.reel as _reel_cmd
    import chess_equity.cli.score as _score_cmd
    import chess_equity.cli.train as _train_cmd
    import chess_equity.cli.train_net as _train_net_cmd
    import chess_equity.cli.validate as _validate_cmd

    # Each subcommand's parser is built by its own cli/<command>.py module; the
    # registration ORDER below defines `--help` listing order — keep it stable.
    _eval_cmd.add_parser(sub)
    _grade_cmd.add_parser(sub)
    _score_cmd.add_parser(sub)
    _broadcast_cmd.add_parser(sub)
    _highlights_cmd.add_parser(sub)
    _reel_cmd.add_parser(sub)
    _data_cmd.add_parser(sub)
    _validate_cmd.add_parser(sub)
    _precompute_cmd.add_parser(sub)
    _headline_cmd.add_parser(sub)
    _divergence_cmd.add_parser(sub)
    _train_cmd.add_parser(sub)
    _train_net_cmd.add_parser(sub)
    _doctor_cmd.add_parser(sub)
    _personal_cmd.add_parser(sub)

    args = parser.parse_args(argv)

    if args.command == "eval":
        return _run_eval(args)
    if args.command == "grade":
        return _run_grade(args)
    if args.command == "score":
        return _run_score(args)
    if args.command == "broadcast":
        try:
            return _run_broadcast(args, build_model(args.model, depth=args.depth), sys.stdout)
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.command == "highlights":
        try:
            return _run_highlights(args, build_model(args.model, depth=args.depth))
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.command == "reel":
        try:
            return _run_reel(args, build_model(args.model, depth=args.depth))
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.command == "data":
        return _run_data(args)
    if args.command == "validate":
        return _run_validate(args)
    if args.command == "headline":
        from chess_equity.validate.headline import run_headline

        return run_headline(args.data, out=args.out, bootstrap=args.bootstrap, seed=args.seed)
    if args.command == "divergence":
        return _run_divergence(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "train-net":
        return _run_train_net(args)
    if args.command == "precompute":
        return _run_precompute(args)
    if args.command == "doctor":
        from chess_equity.doctor import (
            doctor,
            probe_broadcast,
            probe_evidence,
            probe_model,
            probe_overlay,
            probe_serve_sse,
        )

        broadcast_probe = None
        if args.broadcast:
            from chess_equity.broadcast import feed_from_spec

            feed = feed_from_spec(args.broadcast, token=args.token)
            broadcast_probe = lambda: probe_broadcast(feed)  # noqa: E731
        overlay_probe = probe_overlay if args.overlay else None
        serve_sse_probe = probe_serve_sse if args.serve_sse else None
        evidence_probe = probe_evidence if args.evidence else None
        model_probe = (lambda: probe_model(args.model)) if args.model else None  # noqa: E731
        return doctor(
            engines=args.engine,
            broadcast_probe=broadcast_probe,
            overlay_probe=overlay_probe,
            serve_sse_probe=serve_sse_probe,
            evidence_probe=evidence_probe,
            model_probe=model_probe,
        )
    if args.command == "personal":
        return _run_personal(args)

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
