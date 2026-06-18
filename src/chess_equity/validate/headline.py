"""The canonical headline-run recipe: one pinned command -> the thesis report (task 0114).

The *real* thesis proof (tasks 0087 / 0098) is the comparison the whole project hangs
on: does a rating-conditioned predictor (``wdl-a``) and Maia-2's value head (``maia2``)
beat the rating-blind centipawn baseline at predicting real Lichess outcomes? That run
is HELD on torch + Maia weights + a full Lichess dump, so it can't run unattended — but
the *recipe* (which models, which slicers, where the report lands) should be pinned
once, not re-derived flag-by-flag the moment a human approves it.

This module is that pin. :data:`HEADLINE_MODELS` / :data:`HEADLINE_OUT` fix the exact
comparison, and :func:`run_headline` is a one-call wrapper over the ``validate`` command
(``chess-equity headline --data <dump>``) so the approved run is copy-paste. The slicers
are the gate's full :data:`~chess_equity.validate.harness.SLICERS` set (rating /
high_rating / phase / clock / …), so the report breaks the headline number down on every
axis the thesis cares about.

It stays import-cheap and torch-free: the Maia-2 leg is built lazily by the harness and
needs a ``--with-fen`` dataset, so the smoke test injects a fake backend (see
``tests/test_headline.py``) and the real run only pulls torch when a human supplies real
weights.
"""

from __future__ import annotations

import argparse

# The three predictors the headline comparison pins: the rating-blind baseline to beat,
# the rating-conditioned WDL regression (Approach A), and Maia-2's value head.
HEADLINE_MODELS = "baseline,wdl-a,maia2"

# Where the headline report lands — one fixed path so the artifact is always in the same
# place for the README / PR to point at.
HEADLINE_OUT = "reports/validation_headline.md"

# The committed fen-bearing sample the dry-run/smoke path scores against (the Maia-2 leg
# needs row.fen). The real run overrides --data with a full built dump.
SMOKE_DATA = "data/sample/dataset_fen.csv"


def headline_namespace(
    data: str, *, out: str = HEADLINE_OUT, bootstrap: int = 2000, seed: int = 0
) -> argparse.Namespace:
    """The pinned ``validate`` arguments for the headline run, as a ready Namespace.

    Everything except ``data`` (the dump to score) is fixed: the three headline models,
    the fixed ``--out`` path, the full default slicer set (the harness applies them), and
    the default ECE binning. Exposed separately from :func:`run_headline` so a test can
    assert the recipe without executing it.
    """
    return argparse.Namespace(
        command="validate",
        data=data,
        models=HEADLINE_MODELS,
        out=out,
        holdout=None,
        seed=seed,
        bootstrap=bootstrap,
        ece_bins=10,
        calibration=None,
        plots=None,
        # The headline run produces the thesis *report*; the machine-checkable PASS/FAIL
        # gate (task 0115) is a separate opt-in concern, so it stays off here. _run_validate
        # reads args.gate, so the attribute must exist on the Namespace.
        gate=False,
    )


def run_headline(
    data: str = SMOKE_DATA, *, out: str = HEADLINE_OUT, bootstrap: int = 2000, seed: int = 0
) -> int:
    """Run the pinned headline comparison and write the report to ``out``.

    Thin wrapper over :func:`chess_equity.cli._run_validate` with the headline recipe, so
    the report — overall, per-slice, significance, and ECE sections — is byte-identical to
    what ``chess-equity validate --models baseline,wdl-a,maia2 --out reports/validation_headline.md``
    produces. Returns the process exit code (0 on success). The ``_run_validate`` import is
    lazy to keep this module free of the CLI's import graph until the run actually fires.
    """
    from chess_equity.cli import _run_validate

    return _run_validate(headline_namespace(data, out=out, bootstrap=bootstrap, seed=seed))
