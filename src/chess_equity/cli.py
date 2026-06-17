"""Command-line entry point: ``chess-equity``.

Commands:

    chess-equity eval "<fen>" --white-elo 1500 --black-elo 1500
    chess-equity eval --pgn game.pgn --white-elo 1500 --black-elo 1500
    chess-equity grade --pgn game.pgn --white-elo 1500 --black-elo 1500
    chess-equity broadcast --round <id>            # live Lichess broadcast round
    chess-equity broadcast --pgn game.pgn          # replay a finished game as "live"
    chess-equity highlights --pgn game.pgn         # auto-detect drama/clutch moments
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
    LichessRoundFeed,
    LocalPgnFeed,
    MoveEvent,
    UrlPgnFeed,
)
from chess_equity.grading import EquityGrader
from chess_equity.models import LichessBaselineModel
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
    """One JSONL record the overlay (task 0019) can tail."""
    return json.dumps(event.to_dict())


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
    if name == "maia-rollout":
        from chess_equity.rollout import build_maia_rollout

        return build_maia_rollout(n=n, seed=seed)
    if name == "maia-search":
        from chess_equity.search import build_maia_search

        return build_maia_search(depth=depth, k=k)
    if name == "baseline":
        return LichessBaselineModel()
    raise ValueError(
        f"unknown model {name!r}; choose from: "
        "baseline, maia2, wdl-a, maia-rollout, maia-search"
    )


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
            for line in _eval_pgn(model, args.pgn, args.white_elo, args.black_elo):
                print(line)
        elif isinstance(model, MaiaRolloutModel):
            print(_eval_rollout_fen(model, args.fen, args.white_elo, args.black_elo))
        elif isinstance(model, MaiaSearchModel):
            print(_eval_search_fen(model, args.fen, args.white_elo, args.black_elo))
        else:
            print(_eval_fen(model, args.fen, args.white_elo, args.black_elo))
    except (ValueError, OSError, RuntimeError) as exc:
        # RuntimeError covers Maia2NotInstalled (a model failing to load at use time).
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_grade(args: argparse.Namespace) -> int:
    model = build_model(args.model)
    try:
        for line in _grade_pgn(model, args.pgn, args.white_elo, args.black_elo):
            print(line)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_broadcast(args: argparse.Namespace, model: EquityModel, out: TextIO) -> int:
    """Drive broadcast ingestion, writing one JSON event per line to ``out``."""
    if args.pgn:
        with open(args.pgn, encoding="utf-8") as fh:
            feed = LocalPgnFeed(fh.read(), moves_per_poll=args.moves_per_poll)
    elif args.round:
        feed = LichessRoundFeed(args.round, token=args.token)
    elif args.url:
        feed = UrlPgnFeed(args.url)
    else:
        raise ValueError("broadcast needs one of --pgn / --round / --url")

    ingestor = BroadcastIngestor(
        feed,
        model,
        white_elo=args.white_elo,
        black_elo=args.black_elo,
    )

    def emit(event: MoveEvent) -> None:
        out.write(_event_line(event) + "\n")
        out.flush()

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


def _run_data(args: argparse.Namespace) -> int:
    # Imported lazily so the common ``eval`` path never pays for the data deps.
    from chess_equity.data.build import build_dataset, month_url

    pgn = args.pgn
    if pgn is None:
        if args.month is None:
            print("error: provide --pgn <file> (or --month to auto-download)", file=sys.stderr)
            return 1
        from urllib.error import URLError

        from chess_equity.data.download import DEFAULT_DUMP_DIR, download_month

        def _progress(done: int, total: Optional[int]) -> None:
            mb = done / 1e6
            if total:
                print(f"\rdownloading {month_url(args.month)}: {mb:.0f}/{total / 1e6:.0f} MB",
                      end="", file=sys.stderr)
            else:
                print(f"\rdownloading {month_url(args.month)}: {mb:.0f} MB", end="", file=sys.stderr)

        try:
            dump = download_month(
                args.month, dest_dir=args.dump_dir or DEFAULT_DUMP_DIR, progress=_progress
            )
        except (URLError, OSError, RuntimeError) as exc:
            print(f"\nerror: downloading {args.month} dump: {exc}", file=sys.stderr)
            return 1
        print(f"\nfetched {dump}", file=sys.stderr)
        pgn = str(dump)
    try:
        out = build_dataset(
            pgn, args.out, sample=args.sample, fmt=args.format, include_fen=args.with_fen
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
        evaluate,
        format_report,
    )

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    try:
        predictors = build_predictors(requested)
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

    title = f"Validation report — {args.data}"
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
        reports = evaluate(rows, predictors)
    except (ValueError, Maia2NotInstalled) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report = format_report(reports, title=title)
    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)

    if args.calibration:
        # Per-rating-band reliability curves for the first requested predictor (task 0027).
        from pathlib import Path

        from chess_equity.validate.calibration import band_reliability, format_calibration_report

        name = requested[0]
        bands = band_reliability(rows, PREDICTORS[name])
        cal = format_calibration_report(
            bands, predictor_name=name, title=f"Calibration by rating band — {args.data}"
        )
        Path(args.calibration).parent.mkdir(parents=True, exist_ok=True)
        Path(args.calibration).write_text(cal + "\n", encoding="utf-8")
        print(f"wrote {args.calibration}")
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
    model = CachingEquityModel(build_model(args.model), path=args.cache)
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
    model = fit(rows, lr=args.lr, iters=args.iters, l2=args.l2)
    out = args.out or str(default_artifact_path())
    model.save(out)
    meta = model.meta or {}
    print(
        f"wrote {out} (n_train={meta.get('n_train')}, "
        f"iters={meta.get('iters')}, final_log_loss={meta.get('final_log_loss'):.4f})"
    )
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
    add_model_arg(ev)

    gr = sub.add_parser("grade", help="grade every move of a PGN by Δequity vs rating peers")
    gr.add_argument("--pgn", required=True, help="PGN file to grade")
    gr.add_argument("--white-elo", type=int, default=1500)
    gr.add_argument("--black-elo", type=int, default=1500)
    add_model_arg(gr)

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
    add_model_arg(hl)

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
        "--dump-dir",
        default=None,
        help="cache dir for downloaded --month dumps (default: ~/.cache/chess-equity/dumps)",
    )

    val = sub.add_parser("validate", help="score predictors against real outcomes (task 0009)")
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
        "--seed", type=int, default=0, help="RNG seed for the --holdout game split"
    )
    val.add_argument(
        "--calibration",
        help="also write a per-rating-band reliability report (task 0027) here",
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
    add_model_arg(pc)

    tr = sub.add_parser("train", help="fit the wdl-a rating-conditioned WDL model (task 0004)")
    tr.add_argument("--data", required=True, help="path to a built dataset (csv/parquet)")
    tr.add_argument("--out", help="artifact path (default: the packaged wdl_a.json)")
    tr.add_argument("--iters", type=int, default=3000, help="gradient-descent iterations")
    tr.add_argument("--lr", type=float, default=0.5, help="learning rate")
    tr.add_argument("--l2", type=float, default=1e-4, help="L2 regularisation strength")

    args = parser.parse_args(argv)

    if args.command == "eval":
        return _run_eval(args)
    if args.command == "grade":
        return _run_grade(args)
    if args.command == "broadcast":
        try:
            return _run_broadcast(args, build_model(args.model), sys.stdout)
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.command == "highlights":
        try:
            return _run_highlights(args, build_model(args.model))
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.command == "data":
        return _run_data(args)
    if args.command == "validate":
        return _run_validate(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "precompute":
        return _run_precompute(args)

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
