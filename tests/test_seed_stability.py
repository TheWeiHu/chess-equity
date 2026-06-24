"""Seed-stability of the gate verdict (task 0156): does PASS survive re-sampling?

The committed headline gate runs at one seed (``validation_real.md`` header: "(seed 0)"),
so a skeptic can call the PASS a cherry-picked draw. ``reseed_stability`` re-runs the same
gate under K seeds and reports verdict stability; ``--seeds`` exposes it on the CLI. These
tests pin the aggregation on small synthetic n (no real-data download, no torch): a strictly
point-separable challenger PASSes every seed (stable), a tie-with-baseline predictor PASSes
no seed (fails), and a tiny-n run reads underpowered rather than a lucky PASS.
"""

from pathlib import Path

from chess_equity.cli import main
from chess_equity.data.schema import PositionRow
from chess_equity.validate.harness import MIN_GATE_N
from chess_equity.validate.seed_stability import (
    format_seed_stability,
    reseed_stability,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset.csv"


def _row(result, *, game_id):
    return PositionRow(
        cp_eval=0.0,
        white_elo=1500,
        black_elo=1500,
        ply=20,
        phase="middlegame",
        time_control="600+0",
        tc_bucket="rapid",
        clock_remaining=None,
        side_to_move="white",
        result=result,
        game_id=game_id,
    )


def _separable_rows(n_games, plies_per_game=4):
    """A point-separable, game-id'd fixture: alternating results keep labels balanced, and
    each game gets a distinct ``game_id`` so the game-level holdout split has games to draw.
    A challenger that predicts the actual result strictly beats a constant-0.5 baseline on
    log-loss and Brier on every draw."""
    rows = []
    for g in range(n_games):
        for p in range(plies_per_game):
            idx = g * plies_per_game + p
            rows.append(_row(1.0 if idx % 2 == 0 else 0.0, game_id=f"g{g}"))
    return rows


# --- reseed_stability aggregation ----------------------------------------------------


def test_reseed_stability_passes_every_seed_for_separable_challenger():
    # >= MIN_GATE_N games so each held-out draw clears the underpowered floor, and a
    # perfectly separable challenger PASSes on all of them.
    rows = _separable_rows(1200)  # >= MIN_GATE_N held-out rows at holdout 0.5
    predictors = {"baseline": lambda r: 0.5, "challenger": lambda r: r.result}
    stability = reseed_stability(
        rows,
        predictors,
        seeds=[0, 1, 2, 3, 4],
        holdout=0.5,
        n_resamples=0,  # point-only gate keeps the test fast
        min_n=MIN_GATE_N,
    )
    assert stability.seeds == [0, 1, 2, 3, 4]
    m = stability.models[0]
    assert m.name == "challenger"
    assert m.n_seeds == 5
    assert m.n_pass == 5 and m.all_pass is True
    assert m.pass_fraction == 1.0
    # Every per-seed delta is a win (negative) and the draws are not identical, so the
    # spread is real (min <= mean <= max).
    assert m.delta_max < 0
    assert m.delta_min <= m.delta_mean <= m.delta_max


def test_reseed_stability_fails_every_seed_for_tied_predictor():
    rows = _separable_rows(1200)  # >= MIN_GATE_N held-out rows at holdout 0.5
    # A challenger identical to the baseline never strictly beats it -> 0/K seeds PASS.
    predictors = {"baseline": lambda r: 0.5, "tied": lambda r: 0.5}
    stability = reseed_stability(
        rows, predictors, seeds=[0, 1, 2], holdout=0.5, n_resamples=0, min_n=MIN_GATE_N
    )
    m = stability.models[0]
    assert m.n_pass == 0 and m.all_pass is False
    assert m.any_underpowered is False


def test_reseed_stability_flags_underpowered_below_floor():
    # Tiny n: every seed is underpowered, so the verdict is INCONCLUSIVE, not a lucky PASS.
    rows = _separable_rows(20)
    predictors = {"baseline": lambda r: 0.5, "challenger": lambda r: r.result}
    stability = reseed_stability(
        rows, predictors, seeds=[0, 1], holdout=0.2, n_resamples=0, min_n=MIN_GATE_N
    )
    m = stability.models[0]
    assert m.any_underpowered is True
    assert m.n_pass == 0  # underpowered can't read PASS


def test_reseed_stability_requires_seeds():
    rows = _separable_rows(10)
    predictors = {"baseline": lambda r: 0.5, "challenger": lambda r: r.result}
    try:
        reseed_stability(rows, predictors, seeds=[], holdout=0.2)
    except ValueError:
        pass
    else:  # pragma: no cover - the call must raise
        raise AssertionError("empty seeds should raise ValueError")


# --- format_seed_stability rendering -------------------------------------------------


def test_format_seed_stability_renders_summary_and_per_seed():
    rows = _separable_rows(1200)  # >= MIN_GATE_N held-out rows at holdout 0.5
    predictors = {"baseline": lambda r: 0.5, "challenger": lambda r: r.result}
    stability = reseed_stability(
        rows, predictors, seeds=[0, 1, 2], holdout=0.5, n_resamples=0, min_n=MIN_GATE_N
    )
    block = format_seed_stability(stability)
    assert "## Seed stability" in block
    assert "seeds PASS" in block
    assert "3/3" in block  # all three seeds PASS
    assert "stable" in block
    assert "Per seed:" in block


# --- CLI: --seeds appends the section ------------------------------------------------


def test_cli_seeds_flag_appends_stability_section(capsys):
    # End-to-end over the committed sample fixture: --seeds adds the section to the report.
    rc = main(
        [
            "validate",
            "--data",
            str(SAMPLE),
            "--models",
            "baseline,wdl-a",
            "--holdout",
            "0.5",
            "--bootstrap",
            "0",
            "--seeds",
            "0,1,2",
        ]
    )
    out, _err = capsys.readouterr()
    assert rc == 0
    assert "## Seed stability" in out
    assert "K=3 seeds" in out


def test_cli_seeds_rejects_non_integer(capsys):
    rc = main(
        [
            "validate",
            "--data",
            str(SAMPLE),
            "--models",
            "baseline,wdl-a",
            "--seeds",
            "0,foo,2",
        ]
    )
    _out, err = capsys.readouterr()
    assert rc == 1
    assert "comma-separated list of integers" in err
