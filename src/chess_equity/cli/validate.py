"""``chess-equity validate`` parser builder."""

from __future__ import annotations

import argparse


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    val = sub.add_parser("validate", help="score predictors against real outcomes (task 0009)")
    # The underpowered-sample floor's default (task 0132) — imported here so the help text
    # shows the real number without pulling the heavy validate package at startup.
    from chess_equity.validate.harness import MIN_GATE_N
    val.add_argument(
        "--data",
        help="path to a built dataset (csv/parquet); required unless --check-index",
    )
    val.add_argument(
        "--check-index",
        action="store_true",
        help="evidence-index drift guard (task 0219): cross-check reports/SUMMARY.md "
        "against the committed reports/*_real*.md headers + '## Gate verdict' lines and "
        "exit non-zero listing any row whose dump/n/verdict disagrees or any report "
        "missing from the table. Reads no data and computes no numbers — fully unattended",
    )
    val.add_argument(
        "--models",
        default="baseline",
        help="comma-separated predictors: baseline, baseline+clock, or the board model "
        "maia2 (needs a --with-fen dataset, and `pip install maia2` for real numbers)",
    )
    val.add_argument(
        "--slice",
        choices=("clock",),
        help="run a focused diagnostic instead of the full gate (task 0249): "
        "`clock` prints the dataset's [%%clk] coverage and a per-clock_band breakdown "
        "over clock-bearing rows only — cheap, model-free vetting of a candidate dump's "
        "clock coverage before the expensive attended validation run",
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
        "--recalibrate-maia2",
        action="store_true",
        help="post-hoc Platt recalibration of maia2 (task 0166): fit a two-parameter "
        "logistic on the logit of maia2's prediction over the --holdout TRAIN split and "
        "apply it at eval, to repair maia2's high-rating (2000+) ECE blowup without "
        "re-ordering predictions. Needs --holdout (fit must be held-out from eval) and "
        "maia2 in --models. Off by default, so committed numbers don't move unless you opt in",
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
    return val
