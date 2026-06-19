"""Regression guard for the committed real-data headline evidence (task 0129).

The real-evidence report (`reports/validation_real.md`, n>=8000 — task 0087/0128) was
lost once already: a cleanup closed-not-merged its PR and nothing in CI asserted the
file's existence or that it was the *real* run rather than the 15-row illustrative
stand-in (`reports/validation_sample.md`). This test locks the artifact in.

It is deliberately a CONDITIONAL guard, not an existence check:
- while `reports/validation_real.md` is absent (e.g. before task 0128 lands), it
  SKIPS cleanly so the suite stays green;
- once the file is committed, it asserts the file is a genuine real run — a real-sized
  sample (n>=8000), a PASS gate verdict, and a paired-bootstrap CI section — and that it
  is NOT the illustrative sample fixture sneaked in under the real name.

Task 0142 extends the same conditional pattern to `reports/calibration_real.md` (the
per-rating-band calibration report, task 0129): while it is absent the guard stays green;
once committed it asserts a real run — a real-sized sample (n>=8000 summed across bands)
with bootstrap CIs — and not the `baseline_calibration_sample.md` smoke fixture (15 rows,
`data/sample/`, the "SMOKE TEST, NOT EVIDENCE" banner) smuggled in under the real name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_REPORT = REPO_ROOT / "reports" / "validation_real.md"
CALIBRATION_REPORT = REPO_ROOT / "reports" / "calibration_real.md"

# n>=8000 is the headline-evidence floor (task 0128 acceptance): big enough that the
# paired-bootstrap CIs can actually clear zero, and far above the 15-row sample fixture.
MIN_REAL_N = 8000


def _overall_n(report: str) -> int:
    """Largest sample count in the `## Overall` table.

    The Overall block is one `| predictor | n | log-loss | Brier | ECE |` table; the only
    integer cells are the n column (the metrics are sub-1.0 decimals), so the max integer
    in the block is the run's n. Returns 0 if the section/table can't be found.
    """
    m = re.search(r"^##\s+Overall\s*$(.*?)(?=^##\s|\Z)", report, re.MULTILINE | re.DOTALL)
    if not m:
        return 0
    ns: list[int] = []
    for line in m.group(1).splitlines():
        for cell in (c.strip() for c in line.split("|")):
            if re.fullmatch(r"\d+", cell):
                ns.append(int(cell))
    return max(ns) if ns else 0


@pytest.mark.skipif(
    not REAL_REPORT.exists(),
    reason="reports/validation_real.md not committed yet (task 0128); guard stays green until it lands",
)
def test_committed_real_evidence_is_a_real_run() -> None:
    report = REAL_REPORT.read_text(encoding="utf-8")

    # Not the illustrative stand-in smuggled in under the real name.
    assert "data/sample/" not in report, (
        "validation_real.md points at the data/sample fixture — that's the illustrative "
        "stand-in, not real evidence"
    )
    assert "SMOKE TEST, NOT EVIDENCE" not in report, (
        "validation_real.md carries the smoke-test banner — it is not a real run"
    )

    # Real-sized sample.
    n = _overall_n(report)
    assert n >= MIN_REAL_N, (
        f"validation_real.md Overall n={n} is below the real-evidence floor {MIN_REAL_N}"
    )

    # A PASS gate verdict. The Gate-verdict prose defines what "**PASS**" means, so the
    # bare word appears even in a FAIL report; the per-model verdict is the `-> **PASS**`
    # arrow, which is what proves the thesis held.
    assert "-> **PASS**" in report, (
        "validation_real.md has no '-> **PASS**' gate verdict — the headline thesis did "
        "not pass on this run"
    )

    # A paired-bootstrap CI section — the significance evidence, not just point deltas.
    assert "Paired bootstrap" in report and "## Significance vs baseline" in report, (
        "validation_real.md is missing the paired-bootstrap significance section"
    )


def _calibration_total_n(report: str) -> int:
    """Total sample count across the `## ECE by rating band` table.

    That table is `| rating band | n | log-loss | Brier | ECE | ECE 95% CI |`; per data row
    the band cell holds a dash range (`1200-1599`), the metrics are sub-1.0 decimals, and the
    CI cell is a `[lo, hi]` bracket — so the only bare-integer cell is the n column. Summing
    those across the section gives the run's total n (bands partition the rows). Returns 0 if
    the section/table can't be found.
    """
    m = re.search(
        r"^##\s+ECE by rating band.*?$(.*?)(?=^##\s|\Z)", report, re.MULTILINE | re.DOTALL
    )
    if not m:
        return 0
    total = 0
    for line in m.group(1).splitlines():
        for cell in (c.strip() for c in line.split("|")):
            if re.fullmatch(r"\d+", cell):
                total += int(cell)
    return total


@pytest.mark.skipif(
    not CALIBRATION_REPORT.exists(),
    reason="reports/calibration_real.md not committed yet (task 0129); guard stays green until it lands",
)
def test_committed_calibration_evidence_is_a_real_run() -> None:
    report = CALIBRATION_REPORT.read_text(encoding="utf-8")

    # Not the illustrative smoke fixture smuggled in under the real name.
    assert "data/sample/" not in report, (
        "calibration_real.md points at the data/sample fixture — that's the illustrative "
        "smoke stand-in, not real evidence"
    )
    assert "SMOKE TEST, NOT EVIDENCE" not in report, (
        "calibration_real.md carries the smoke-test banner — it is not a real run"
    )

    # Real-sized sample (summed across rating bands).
    n = _calibration_total_n(report)
    assert n >= MIN_REAL_N, (
        f"calibration_real.md total n={n} is below the real-evidence floor {MIN_REAL_N}"
    )

    # Bootstrap CIs on each band's ECE — the per-band uncertainty evidence that the real
    # run carries and the tiny smoke fixture (one game per band) cannot.
    assert "95% CI" in report, (
        "calibration_real.md is missing the bootstrap ECE 95% CI column — without it a band's "
        "drift can't be told from small-sample noise, so it isn't real evidence"
    )
