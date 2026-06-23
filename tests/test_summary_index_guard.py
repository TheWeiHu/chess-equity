"""Guard: keep `reports/SUMMARY.md` (the gate index) honest against the reports it quotes.

`reports/SUMMARY.md` is the real-data gate index — one row per committed real-Lichess
evidence report, with each row hand-quoting that report's verdict word (PASS/FAIL/info)
and `n`. SUMMARY itself says "regenerate this table from their headers"; this test makes
that an *enforced* invariant so the index can't silently drift when a report is
regenerated, renamed, added, or removed (task 0194).

It is pure text parsing — reads no data, computes no numbers, so it stays in the fast
`tests/` path (no dump, no torch). Three checks, mirroring the task acceptance:

  (a) Coverage — every `reports/*real*.md` evidence report has exactly one SUMMARY row,
      and every SUMMARY row points at a real file on disk (bidirectional).
  (b) Verdict — the verdict word in each SUMMARY row matches the verdict the report itself
      states (classified from the report's own text, see ``_report_verdict``).
  (c) n — every integer the SUMMARY row quotes in its `n` cell appears (comma-normalised)
      in the report body, so a changed n can't sit un-propagated in the index.

Allowlist: the only report whose verdict is FAIL is the deliberate negative result
``wdl_net_real.md`` (Approach D kept on purpose). ``KNOWN_FAIL`` pins that — a *surprise*
FAIL on any other report (a real regression in the evidence) trips this test instead of
quietly riding in the index as "FAIL".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
SUMMARY = REPORTS_DIR / "SUMMARY.md"

# The committed real-data gate reports are exactly the `*real*.md` files (the cross-dump
# ones — validation_real_2016-05.md, _high.md, _xdump_refit.md — do NOT end in `_real.md`,
# so a `*_real.md` glob would miss them). The `*_sample.md` artifacts are illustrative
# smoke, intentionally excluded from the index, and don't match `*real*.md`.
REAL_GLOB = "*real*.md"

# Only this report may carry a FAIL verdict — Approach D's end-to-end net is a negative
# result kept on purpose (SUMMARY's "one deliberate FAIL"). Any other FAIL is a regression.
KNOWN_FAIL = {"wdl_net_real.md"}


def _real_report_files() -> set[str]:
    """Filenames of the committed real-data gate reports (excludes SUMMARY itself)."""
    return {p.name for p in REPORTS_DIR.glob(REAL_GLOB) if p.name != SUMMARY.name}


def _summary_rows() -> list[tuple[str, str, str]]:
    """Parse SUMMARY's index table into ``(report_filename, n_cell, verdict_word)`` rows.

    The table is ``| Report | Dump | n | Verdict |``. The Report cell is a
    ``[label](file.md)`` link; the Verdict cell leads with a bolded ``**PASS**`` /
    ``**FAIL**`` / ``**info**`` word. Header and separator rows (and any non-report row)
    are skipped because they carry no ``(file.md)`` link.
    """
    rows: list[tuple[str, str, str]] = []
    for line in SUMMARY.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        link = re.search(r"\(([^)]+\.md)\)", cells[0])
        if not link:  # header / separator / non-report row
            continue
        verdict = re.search(r"\*\*(PASS|FAIL|info)", cells[3])
        if not verdict:
            continue
        rows.append((link.group(1), cells[2], verdict.group(1)))
    return rows


def _report_verdict(text: str) -> str:
    """Classify a report's own stated gate verdict as ``PASS`` / ``FAIL`` / ``info``.

    The reports state their verdict heterogeneously, so the cues are matched in priority
    order (verified to reproduce every current SUMMARY verdict word):

    - **FAIL** if the report bolds ``**FAIL**`` — today only ``wdl_net_real.md``, which
      headlines a deliberate negative result. (Checked first: that report also carries a
      ``-> **PASS**`` line for the *baseline* comparison, so PASS cues alone would mislabel
      it.)
    - **PASS** if it carries any pass cue: a ``## Gate verdict`` section (validation runs),
      a bolded ``**PASS**``, the ``Gate PASS`` prose (recalibration), or a ``✅`` thesis
      tick (goodmoves, which never writes the word "PASS").
    - **info** otherwise — a measurement report (calibration / failure modes / divergence /
      drama thresholds) that states no gate.
    """
    if "**FAIL**" in text:
        return "FAIL"
    pass_cues = ("## Gate verdict", "**PASS**", "Gate PASS", "✅")
    if any(cue in text for cue in pass_cues):
        return "PASS"
    return "info"


def _ints(s: str) -> set[str]:
    """Integer tokens in ``s``, comma-grouping removed (``49,269`` -> ``49269``)."""
    return {m.replace(",", "") for m in re.findall(r"\d[\d,]*", s)}


@pytest.mark.skipif(not SUMMARY.exists(), reason="reports/SUMMARY.md not committed")
def test_summary_covers_exactly_the_real_reports() -> None:
    """(a) The index and the on-disk real reports are the same set — no row points at a
    missing/renamed file, and no committed real report is missing from the index."""
    on_disk = _real_report_files()
    indexed = {fname for fname, _, _ in _summary_rows()}

    missing_from_index = on_disk - indexed
    assert not missing_from_index, (
        f"real reports on disk with no SUMMARY.md row (index is stale): "
        f"{sorted(missing_from_index)}"
    )
    stale_rows = indexed - on_disk
    assert not stale_rows, (
        f"SUMMARY.md rows pointing at files that don't exist (renamed/removed?): "
        f"{sorted(stale_rows)}"
    )


@pytest.mark.skipif(not SUMMARY.exists(), reason="reports/SUMMARY.md not committed")
def test_summary_verdict_and_n_match_each_report() -> None:
    """(b) each row's verdict word matches the report's own stated verdict, and
    (c) every integer the row quotes in its n cell appears in the report body."""
    for fname, n_cell, summary_verdict in _summary_rows():
        report_path = REPORTS_DIR / fname
        if not report_path.exists():
            continue  # coverage test owns the missing-file failure; don't double-report
        text = report_path.read_text(encoding="utf-8")

        # (b) verdict word agreement
        actual = _report_verdict(text)
        assert actual == summary_verdict, (
            f"{fname}: SUMMARY.md says verdict '{summary_verdict}' but the report states "
            f"'{actual}' — the index drifted from the evidence"
        )

        # allowlist: only the deliberate negative result may be FAIL
        if summary_verdict == "FAIL":
            assert fname in KNOWN_FAIL, (
                f"{fname}: unexpected FAIL verdict (only {sorted(KNOWN_FAIL)} may be a "
                f"deliberate FAIL) — looks like an evidence regression, not a kept negative"
            )

        # (c) every n the index quotes is actually in the report
        report_ints = _ints(text)
        for token in _ints(n_cell):
            assert token in report_ints, (
                f"{fname}: SUMMARY.md n cell quotes '{token}' (from '{n_cell}') but that "
                f"number does not appear in the report — the index's n drifted"
            )
