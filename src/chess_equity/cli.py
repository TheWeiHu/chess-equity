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


def _grade_pgn(model: EquityModel, path: str, white_elo: int, black_elo: int) -> List[str]:
    """Annotate every move of a PGN with its peer-relative Δequity grade."""
    with open(path, encoding="utf-8") as fh:
        game = chess.pgn.read_game(fh)
    if game is None:
        raise ValueError(f"no game found in {path}")
    grader = EquityGrader(model)
    lines = []
    for g in grader.grade_game(game, white_elo, black_elo):
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


def _run_eval(args: argparse.Namespace) -> int:
    model = build_model(args.model, n=args.n, seed=args.seed, depth=args.depth, k=args.k)
    try:
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
        for line in _grade_pgn(model, args.pgn, args.white_elo, args.black_elo):
            print(line)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_score(args: argparse.Namespace) -> int:
    """Scorecard one game: the score, the real result, and what we predict (task 0129)."""
    from chess_equity.scorecard import build_scorecard_from_pgn, render_scorecard

    model = build_model(args.model, n=args.n, seed=args.seed, depth=args.depth, k=args.k)
    try:
        with open(args.pgn, encoding="utf-8") as fh:
            pgn_text = fh.read()
        card = build_scorecard_from_pgn(
            pgn_text,
            _apply_profiles(model, args),
            model_name=args.model,
            white_elo=args.white_elo,
            black_elo=args.black_elo,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for line in render_scorecard(card):
        print(line)
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

    if args.serve_sse is not None:
        from chess_equity.broadcast import overlay_events, serve_sse

        # A live round (--round/--url) may be tuned into before its first move: keep
        # polling (no idle stop) and send keep-alive heartbeats so the connection
        # survives the quiet wait. A local --pgn replay is finite, so it still
        # terminates on idle (max_idle_polls=1, no heartbeat).
        is_live = bool(args.round or args.url)

        def make_events():
            ingestor = BroadcastIngestor(
                _build_broadcast_feed(args),
                model,
                white_elo=args.white_elo,
                black_elo=args.black_elo,
                clock_aware=args.clock_aware,
                engine=cp_engine,
            )
            return overlay_events(
                ingestor,
                interval=args.interval,
                max_polls=args.max_polls,
                max_idle_polls=None if is_live else 1,
                heartbeat=is_live,
            )

        serve_sse(
            make_events,
            port=args.serve_sse,
            directory=_overlay_static_dir(),
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
    )

    def emit(event: MoveEvent) -> None:
        out.write(_event_line(event) + "\n")
        out.flush()

    # Emit the overlay "game" metadata event (player names + ratings) once per game,
    # before its moves, so the overlay name-plates are populated (task 0047).
    def emit_game(game: GameEvent) -> None:
        out.write(json.dumps(game.to_overlay()) + "\n")
        out.flush()

    ingestor.on_game = emit_game

    # A local replay terminates (max_idle_polls=1); a live feed runs until interrupted
    # (--max-polls caps it). interval=0 for replays keeps tests/CI instant.
    stats = ingestor.run(
        emit,
        interval=args.interval,
        max_polls=args.max_polls,
        max_idle_polls=1,
    )
    print(
        f"# {stats.events} events over {stats.polls} polls "
        f"({stats.errors} feed errors), max equity compute {stats.max_compute_ms:.1f} ms",
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
    reel = reel_mod.build_reel(events, top=args.top)

    if args.out_dir is None:
        print(reel_mod.render_markdown(reel, title=args.title))
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "reel.json")
    md_path = os.path.join(args.out_dir, "reel.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(reel_mod.render_json(reel, title=args.title) + "\n")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(reel_mod.render_markdown(reel, title=args.title))
    print(f"wrote {len(reel)} moment(s): {json_path}, {md_path}", file=sys.stderr)
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
        )
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    # Lazy import: keeps the eval path free of the data loader.
    from chess_equity.data.build import load_rows
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

    def add_model_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--model",
            choices=("baseline", "maia2", "wdl-a", "maia-rollout", "maia-search"),
            default="baseline",
            help=(
                "equity model: rating-blind baseline (default), rating-conditioned "
                "maia2, the wdl-a regression, the maia-rollout Monte Carlo oracle, or "
                "the maia-search expectimax (last two slow, non-interactive)"
            ),
        )

    def add_profile_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--white-profile",
            help="personalize White: a Lichess username (mined live), or 'player@game.pgn' "
            "to profile from a local PGN offline (task 0086)",
        )
        p.add_argument(
            "--black-profile",
            help="personalize Black: same forms as --white-profile",
        )

    ev = sub.add_parser("eval", help="evaluate a position or a whole game")
    ev.add_argument("fen", nargs="?", default=START_FEN, help="FEN (default: startpos)")
    ev.add_argument("--pgn", help="annotate every move of a PGN file instead")
    ev.add_argument("--white-elo", type=int, default=1500)
    ev.add_argument("--black-elo", type=int, default=1500)
    ev.add_argument(
        "--n", type=int, default=500, help="rollout count for --model maia-rollout"
    )
    ev.add_argument(
        "--seed", type=int, default=None, help="RNG seed for --model maia-rollout (reproducible)"
    )
    ev.add_argument(
        "--depth", type=int, default=2, help="ply budget for --model maia-search"
    )
    ev.add_argument(
        "--k", type=int, default=4, help="top Maia moves kept per node for --model maia-search"
    )
    add_profile_args(ev)
    add_model_arg(ev)

    gr = sub.add_parser("grade", help="grade every move of a PGN by Δequity vs rating peers")
    gr.add_argument("--pgn", required=True, help="PGN file to grade")
    gr.add_argument("--white-elo", type=int, default=1500)
    gr.add_argument("--black-elo", type=int, default=1500)
    gr.add_argument(
        "--depth", type=int, default=2,
        help="Stockfish baseline search depth (also the maia-search ply budget)",
    )
    add_model_arg(gr)

    sc = sub.add_parser(
        "score",
        help="scorecard one game: the score, the real result, and what equity predicts",
    )
    sc.add_argument("--pgn", required=True, help="PGN file (uses its [%%eval] + result)")
    sc.add_argument(
        "--white-elo", type=int, default=None,
        help="override White rating (default: the PGN's WhiteElo, else 1500)",
    )
    sc.add_argument(
        "--black-elo", type=int, default=None,
        help="override Black rating (default: the PGN's BlackElo, else 1500)",
    )
    sc.add_argument(
        "--n", type=int, default=500, help="rollout count for --model maia-rollout"
    )
    sc.add_argument(
        "--seed", type=int, default=None, help="RNG seed for --model maia-rollout"
    )
    sc.add_argument(
        "--depth", type=int, default=2,
        help="Stockfish baseline search depth (also the maia-search ply budget)",
    )
    sc.add_argument(
        "--k", type=int, default=4, help="top Maia moves kept per node for --model maia-search"
    )
    add_profile_args(sc)
    add_model_arg(sc)

    bc = sub.add_parser(
        "broadcast",
        help="stream per-move equity events from a live (or replayed) broadcast",
    )
    src = bc.add_mutually_exclusive_group(required=True)
    src.add_argument("--round", help="Lichess broadcast round id (live feed)")
    src.add_argument("--url", help="arbitrary public PGN URL (generic feed)")
    src.add_argument("--pgn", help="local PGN file, replayed move-by-move as 'live'")
    bc.add_argument("--white-elo", type=int, default=None, help="override White rating")
    bc.add_argument("--black-elo", type=int, default=None, help="override Black rating")
    bc.add_argument("--interval", type=float, default=2.0, help="seconds between polls")
    bc.add_argument("--max-polls", type=int, default=None, help="stop after N polls")
    bc.add_argument(
        "--moves-per-poll", type=int, default=1, help="replay pacing (local --pgn only)"
    )
    bc.add_argument("--token", default=None, help="Lichess API token (optional)")
    bc.add_argument(
        "--clock-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="warp the published equity by the side-to-move's time pressure when the feed "
        "carries [%%clk] clocks (task 0097); --no-clock-aware emits the clock-blind bar",
    )
    bc.add_argument(
        "--depth", type=int, default=2,
        help="Stockfish baseline search depth (also the maia-search ply budget)",
    )
    bc.add_argument(
        "--serve-sse",
        type=int,
        default=None,
        metavar="PORT",
        help="stream overlay events as Server-Sent-Events on this port instead of "
        "printing JSON Lines — point an OBS browser source at "
        "http://localhost:PORT/?src=/sse (task 0094)",
    )
    add_profile_args(bc)
    add_model_arg(bc)

    hl = sub.add_parser(
        "highlights",
        help="detect drama/clutch moments in a game (task 0020)",
    )
    hl.add_argument("--pgn", required=True, help="PGN file to scan for drama")
    hl.add_argument("--white-elo", type=int, default=None, help="override White rating")
    hl.add_argument("--black-elo", type=int, default=None, help="override Black rating")
    hl.add_argument("--top", type=int, default=5, help="size of the highlight reel (default 5)")
    hl.add_argument("--json", action="store_true", help="emit the reel as JSON")
    hl.add_argument(
        "--depth", type=int, default=2,
        help="Stockfish baseline search depth (also the maia-search ply budget)",
    )
    add_model_arg(hl)

    rl = sub.add_parser(
        "reel",
        help="export a ranked highlight reel (JSON + markdown) from a replayed game (task 0168)",
    )
    rl.add_argument(
        "--pgn",
        default="data/sample/sample_games.pgn",
        help="PGN file to replay (default: committed sample fixture)",
    )
    rl.add_argument("--white-elo", type=int, default=None, help="override White rating")
    rl.add_argument("--black-elo", type=int, default=None, help="override Black rating")
    rl.add_argument("--top", type=int, default=None, help="cap the reel to the top N moments")
    rl.add_argument(
        "--out-dir",
        default=None,
        help="write reel.json + reel.md here (otherwise print markdown to stdout)",
    )
    rl.add_argument("--title", default="Highlight reel", help="reel title")
    rl.add_argument(
        "--depth", type=int, default=2,
        help="Stockfish baseline search depth (also the maia-search ply budget)",
    )
    add_model_arg(rl)

    data = sub.add_parser("data", help="build / manage the training+validation dataset")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    build = data_sub.add_parser("build", help="parse a Lichess PGN dump into a dataset")
    build.add_argument("--pgn", help="path to a PGN file (plain or .zst)")
    build.add_argument(
        "--month", help="YYYY-MM Lichess month — streams + caches the dump, then builds"
    )
    build.add_argument("--sample", type=int, default=None, help="cap the number of rows")
    build.add_argument("--out", default="data", help="output directory (default: data/)")
    build.add_argument("--format", choices=("csv", "parquet"), default="csv")
    build.add_argument(
        "--with-fen",
        action="store_true",
        help="record each position's FEN (needed to validate board models like Maia; ~3x size)",
    )
    build.add_argument(
        "--partition",
        action="store_true",
        help="write a hive-partitioned dir (tc_bucket=…/rating_bucket=…) for efficient slicing",
    )
    build.add_argument(
        "--dump-dir",
        default=None,
        help="cache dir for downloaded --month dumps (default: ~/.cache/chess-equity/dumps)",
    )

    stamp = data_sub.add_parser(
        "stamp",
        help="backfill the source-month sidecar on an existing dataset (task 0127)",
    )
    stamp.add_argument("path", help="path to a built dataset (csv/parquet file or partitioned dir)")
    stamp.add_argument("month", help="the YYYY-MM Lichess month the dataset was drawn from")

    val = sub.add_parser("validate", help="score predictors against real outcomes (task 0009)")
    # The underpowered-sample floor's default (task 0132) — imported here so the help text
    # shows the real number without pulling the heavy validate package at startup.
    from chess_equity.validate.harness import MIN_GATE_N
    val.add_argument("--data", required=True, help="path to a built dataset (csv/parquet)")
    val.add_argument(
        "--models",
        default="baseline",
        help="comma-separated predictors: baseline, baseline+clock, or the board model "
        "maia2 (needs a --with-fen dataset, and `pip install maia2` for real numbers)",
    )
    val.add_argument("--out", help="write the Markdown report here (default: stdout)")
    val.add_argument(
        "--holdout",
        type=float,
        metavar="FRACTION",
        help="score only a held-out test split (this fraction of GAMES, leak-free); "
        "needs a dataset with game_id (task 0030)",
    )
    val.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the --holdout game split and the bootstrap resampling",
    )
    val.add_argument(
        "--seeds",
        metavar="S1,S2,...",
        help="seed-stability check (task 0156): re-run the gate under each of these "
        "comma-separated seeds (e.g. 0,1,2,3,4) and append a stability section reporting "
        "the fraction of seeds that PASS and the spread of the log-loss delta + CI. Hardens "
        "the proof by showing PASS survives re-sampling, not just the committed seed 0",
    )
    val.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        metavar="N",
        help="paired-bootstrap resamples for the 95%% CI on each model-vs-baseline "
        "metric delta (task 0060; 0 disables; needs `baseline` + another model)",
    )
    val.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        metavar="N",
        help="reliability-bin count for ECE and the calibration tables (default 10); "
        "raise on large dumps / lower on small samples to sensitivity-check the ECE CIs",
    )
    val.add_argument(
        "--shrink-wdl-a-k",
        type=float,
        default=0.0,
        metavar="K",
        help="n-aware shrinkage of wdl-a toward the rating-blind baseline (task 0163): "
        "blend each prediction toward baseline by per-cell weight n/(n+K), so sparse "
        "high-rating cells (where wdl-a over-predicts and the 2000-2399 ECE blows up) "
        "fall back to the baseline while well-populated cells are unchanged. K=0 (the "
        "default) is a no-op, so committed numbers don't move unless you opt in",
    )
    val.add_argument(
        "--eval-month",
        metavar="YYYY-MM",
        help="the Lichess source month of --data, for the leakage guard (task 0112); "
        "if omitted it is inferred from the dataset path. When it equals a model's "
        "training month (e.g. wdl-a's 2016-05) the run is memorization, not held-out "
        "evidence: validate warns loudly (or refuses, with --strict)",
    )
    val.add_argument(
        "--wdl-a-artifact",
        metavar="PATH",
        help="score wdl-a from a custom artifact instead of the committed one (task 0164). "
        "Lets a held-out run use a wdl-a refit on a *different* month than the eval dump — "
        "the leakage guard reads this artifact's meta['fit_month'] too, so a genuine "
        "cross-dump refit reads as held-out, not in-distribution",
    )
    val.add_argument(
        "--strict",
        action="store_true",
        help="refuse the run (nonzero exit) instead of merely warning when the leakage "
        "guard (task 0112) finds the eval month overlaps a model's training month",
    )
    val.add_argument(
        "--gate",
        action="store_true",
        help="make the thesis gate machine-checkable (task 0115): exit 0 only if every "
        "rating-conditioned predictor beats `baseline` on log-loss AND Brier, exit 2 if "
        "any FAILS, exit 3 if no challenger to gate, exit 4 if INCONCLUSIVE (held-out n "
        "below --min-n; task 0132). For CI / the autonomous loop",
    )
    val.add_argument(
        "--min-n",
        type=int,
        default=MIN_GATE_N,
        help="underpowered-sample floor for the gate (task 0132): when the held-out n is "
        f"below this, --gate reads INCONCLUSIVE (exit 4) instead of PASS so a lucky tiny-n "
        f"win can't read green (default {MIN_GATE_N}; 0 disables the guard)",
    )
    val.add_argument(
        "--calibration",
        help="also write a per-rating-band reliability report (task 0027) here",
    )
    val.add_argument(
        "--plots",
        metavar="PATH",
        help="also render per-rating-band reliability curves to this PNG (task 0036; "
        "needs matplotlib: `pip install chess-equity[plots]`)",
    )

    pc = sub.add_parser(
        "precompute",
        help="evaluate a whole game's equity into a UI-ready JSON (task 0012)",
    )
    pc.add_argument("--pgn", required=True, help="PGN file to precompute")
    pc.add_argument("--white-elo", type=int, default=1500)
    pc.add_argument("--black-elo", type=int, default=1500)
    pc.add_argument("--out", help="write the JSON here (default: stdout)")
    pc.add_argument(
        "--cache", help="persistent cache path for warm restarts (omit = in-memory only)"
    )
    pc.add_argument(
        "--depth", type=int, default=2,
        help="Stockfish baseline search depth (also the maia-search ply budget)",
    )
    add_profile_args(pc)
    add_model_arg(pc)

    from chess_equity.validate.headline import HEADLINE_OUT, SMOKE_DATA

    hd = sub.add_parser(
        "headline",
        help="run the pinned headline thesis comparison (baseline,wdl-a,maia2 -> "
        f"{HEADLINE_OUT}; needs a --with-fen dataset for the maia2 leg)",
    )
    hd.add_argument(
        "--data",
        default=SMOKE_DATA,
        help=f"path to a --with-fen dataset to score (default: {SMOKE_DATA}, the "
        "committed dry-run sample; the real run points this at a full built dump)",
    )
    hd.add_argument("--out", default=HEADLINE_OUT, help=f"report path (default: {HEADLINE_OUT})")
    hd.add_argument(
        "--bootstrap", type=int, default=2000, metavar="N",
        help="paired-bootstrap resamples for the significance CIs (0 disables)",
    )
    hd.add_argument("--seed", type=int, default=0, help="RNG seed for the bootstrap")

    dv = sub.add_parser(
        "divergence",
        help="measure how far the equity bar DIVERGES from the Stockfish bar (task 0171)",
    )
    dv.add_argument("--data", required=True, help="path to a built dataset (csv/parquet)")
    dv.add_argument(
        "--equity",
        default="wdl-a",
        help="the rating-aware equity predictor to compare (default wdl-a; any "
        "validate --models name that reads cp_eval, e.g. baseline+clock)",
    )
    dv.add_argument(
        "--stockfish",
        default="baseline",
        help="the classic Stockfish-bar predictor to diverge from (default baseline: "
        "Lichess Win%% of cp_eval)",
    )
    dv.add_argument("--out", help="write the Markdown report here (default: stdout)")

    tr = sub.add_parser("train", help="fit the wdl-a rating-conditioned WDL model (task 0004)")
    tr.add_argument("--data", required=True, help="path to a built dataset (csv/parquet)")
    tr.add_argument("--out", help="artifact path (default: the packaged wdl_a.json)")
    tr.add_argument("--iters", type=int, default=3000, help="gradient-descent iterations")
    tr.add_argument("--lr", type=float, default=0.5, help="learning rate")
    tr.add_argument("--l2", type=float, default=1e-4, help="L2 regularisation strength")
    tr.add_argument(
        "--train-month",
        default=None,
        help="YYYY-MM the dataset came from, stamped into meta['fit_month'] for the "
        "leakage guard (task 0112) — set it so a held-out eval on a different month "
        "isn't mistaken for in-distribution",
    )

    tn = sub.add_parser(
        "train-net",
        help="fit the end-to-end board → WDL net (Approach D, task 0013; needs torch + a --with-fen dataset)",
    )
    tn.add_argument("--data", required=True, help="path to a --with-fen dataset (csv/parquet)")
    tn.add_argument("--out", help="artifact path (default: the packaged wdl_net.pt)")
    tn.add_argument("--epochs", type=int, default=8, help="training epochs (default 8)")
    tn.add_argument("--batch-size", type=int, default=512, help="minibatch size (default 512)")
    tn.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (default 1e-3)")
    tn.add_argument("--seed", type=int, default=0, help="RNG seed for shuffling + init")
    tn.add_argument(
        "--train-month",
        default=None,
        help="YYYY-MM the dataset came from, stamped into the artifact for the leakage guard",
    )

    dr = sub.add_parser(
        "doctor",
        help="check the optional engines (Stockfish, Maia-2) are installed and working",
    )
    dr.add_argument(
        "--engine",
        action="append",
        choices=["stockfish", "maia2"],
        help="check only this engine (repeatable); default checks all. Use "
        "`--engine stockfish` on a binary-only runner with no torch/Maia-2.",
    )

    pp = sub.add_parser(
        "personal",
        help="mine a player's per-phase quality profile and personalize the bar (task 0014)",
    )
    pp.add_argument("--user", help="Lichess username to mine over the network")
    pp.add_argument("--pgn", help="local PGN file to profile from instead (offline)")
    pp.add_argument("--name", help="which player in --pgn to profile (defaults to --user)")
    pp.add_argument("--max-games", type=int, default=50, help="cap mined games (default 50)")
    pp.add_argument("--token", default=None, help="Lichess API token (optional, raises rate limit)")
    pp.add_argument("--json", action="store_true", help="emit the profile as JSON")
    pp.add_argument(
        "--demo",
        action="store_true",
        help="also show band-average vs personalized equity for --fen",
    )
    pp.add_argument("--fen", default=START_FEN, help="position for --demo (default: startpos)")

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
        from chess_equity.doctor import doctor

        return doctor(engines=args.engine)
    if args.command == "personal":
        return _run_personal(args)

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
