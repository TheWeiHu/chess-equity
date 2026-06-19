"""The underpowered-sample guard (task 0132): a tiny-n run reads INCONCLUSIVE, not green.

The committed proof artifact is the 15-row `validation_sample.md`, and the harness itself
notes the sample barely reaches any size that powers a bootstrap. With n that small a lucky
point win plus a barely-non-straddling CI can read PASS by chance, overstating the thesis.
The gate now refuses to call a PASS below a documented floor (`MIN_GATE_N`): it reports a
distinct INCONCLUSIVE state, surfaced in `format_verdict` and as a 4th `--gate` exit code (4).

These tests pin both directions: the 15-row sample is flagged underpowered (not PASS), and a
synthetic n>=2000 fixture — a strict point win on every metric — still PASSes.
"""

from pathlib import Path

from chess_equity.cli import main
from chess_equity.data.schema import PositionRow
from chess_equity.validate.harness import (
    MIN_GATE_N,
    evaluate,
    format_verdict,
    gate_verdicts,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "dataset.csv"


# --- CLI exit code: 4 = INCONCLUSIVE on the underpowered 15-row sample ---------------


def test_gate_underpowered_sample_exits_four(capsys):
    # The 15-row sample is far below MIN_GATE_N, so the default --gate reads INCONCLUSIVE
    # (exit 4) rather than PASS — a lucky tiny-n win is not proof.
    rc = main(["validate", "--data", str(SAMPLE), "--bootstrap", "0", "--models", "baseline,wdl-a", "--gate"])
    out, err = capsys.readouterr()
    assert rc == 4
    assert "GATE: INCONCLUSIVE" in err
    assert "GATE: PASS" not in out  # underpowered must not read green


def test_gate_underpowered_is_distinct_from_pass_and_fail(capsys):
    # Exit 4 must not collide with PASS (0), FAIL (2), or misuse (3): the same sample that
    # would PASS with the guard off (--min-n 0) reads INCONCLUSIVE with the guard on.
    rc_off = main(
        ["validate", "--data", str(SAMPLE), "--bootstrap", "0", "--min-n", "0", "--models", "baseline,wdl-a", "--gate"]
    )
    assert rc_off == 0
    capsys.readouterr()
    rc_on = main(["validate", "--data", str(SAMPLE), "--bootstrap", "0", "--models", "baseline,wdl-a", "--gate"])
    assert rc_on == 4


# --- gate_verdicts / format_verdict: the underpowered state -------------------------


def _row(white_elo, black_elo, result, *, phase="middlegame"):
    return PositionRow(
        cp_eval=0.0,
        white_elo=white_elo,
        black_elo=black_elo,
        ply=20,
        phase=phase,
        time_control="600+0",
        tc_bucket="rapid",
        clock_remaining=None,
        side_to_move="white",
        result=result,
    )


def _synthetic_rows(n):
    """A point-separable fixture: the baseline cp predictor is fixed at 0.5, while the
    challenger predicts the actual result, so it strictly beats the baseline on log-loss and
    Brier. Alternating results keep the labels balanced so the baseline isn't accidentally
    perfect on a constant outcome.
    """
    return [_row(1500, 1500, 1.0 if i % 2 == 0 else 0.0) for i in range(n)]


def test_gate_verdicts_flags_underpowered_below_floor():
    rows = _synthetic_rows(50)  # well below any sane floor
    reports = evaluate(rows, {"baseline": lambda r: 0.5, "challenger": lambda r: r.result})
    verdicts = gate_verdicts(reports, min_n=MIN_GATE_N)
    v = verdicts[0]
    assert v.underpowered is True
    assert v.passed is False  # a tiny-n point win must not read PASS
    assert v.held_out_n == 50 and v.min_n == MIN_GATE_N


def test_format_verdict_renders_inconclusive():
    rows = _synthetic_rows(50)
    reports = evaluate(rows, {"baseline": lambda r: 0.5, "challenger": lambda r: r.result})
    block = "\n".join(format_verdict(gate_verdicts(reports, min_n=MIN_GATE_N)))
    assert "INCONCLUSIVE" in block and "underpowered" in block
    assert "-> **PASS**" not in block


def test_gate_verdicts_passes_when_n_clears_floor():
    # The complement: a synthetic fixture at the floor (n>=MIN_GATE_N) with a strict point
    # win still PASSes — the guard only suppresses underpowered runs, not legitimate ones.
    rows = _synthetic_rows(MIN_GATE_N)
    reports = evaluate(rows, {"baseline": lambda r: 0.5, "challenger": lambda r: r.result})
    verdicts = gate_verdicts(reports, min_n=MIN_GATE_N)
    v = verdicts[0]
    assert v.underpowered is False
    assert v.passed is True
    assert v.held_out_n == MIN_GATE_N


def test_min_n_zero_disables_the_guard():
    rows = _synthetic_rows(50)
    reports = evaluate(rows, {"baseline": lambda r: 0.5, "challenger": lambda r: r.result})
    v = gate_verdicts(reports, min_n=0)[0]
    assert v.underpowered is False
    assert v.passed is True  # tiny n, but the guard is off
