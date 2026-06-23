"""Evidence-index drift guard for reports/SUMMARY.md (task 0219).

Two layers, both data-free (they only read committed markdown — no dataset, no numbers):

1. ``test_summary_index_is_consistent`` — the real guard. Cross-checks the committed
   ``reports/SUMMARY.md`` against every ``reports/*_real*.md`` header + ``## Gate verdict``
   line and asserts there is no drift. This is what keeps the proof index honest when a
   report is regenerated or a new one lands.

2. The targeted unit tests below prove the detection logic catches each drift class —
   an uncovered report, a dangling row, and mismatched month/n/verdict — so a future
   refactor can't silently turn the guard into a no-op.
"""

from __future__ import annotations

import textwrap

from chess_equity.validate.index_guard import (
    REPORTS_DIR,
    SUMMARY,
    check_index,
    parse_report,
    parse_summary_table,
)


def test_summary_index_is_consistent():
    """The committed SUMMARY.md matches the committed real-data reports — no drift."""
    result = check_index()
    assert result.ok, "SUMMARY.md drifted from the reports:\n" + "\n".join(result.problems)


def test_real_reports_all_parse():
    """Every committed real report exposes a parseable header (sanity for the guard)."""
    reports = sorted(REPORTS_DIR.glob("*_real*.md"))
    assert reports, "expected committed *_real*.md evidence reports"
    rows = {r.target for r in parse_summary_table(SUMMARY.read_text(encoding="utf-8"))}
    for path in reports:
        assert path.name in rows, f"{path.name} missing from SUMMARY table"
        parse_report(path)  # must not raise


# --- detection-logic unit tests (synthetic markdown fixtures, no real data) ---------

_GOOD_REPORT = textwrap.dedent(
    """\
    # Validation report — real Lichess dump — lichess_db_standard_rated_2013-01, n=12000 (seed 0)

    ## Gate verdict

    - **wdl-a** beats baseline: logloss -0.34 ... CI [-0.39, -0.30] -> **PASS**
    """
)

_GOOD_SUMMARY = textwrap.dedent(
    """\
    # index

    | Report | Dump (month) | n | Verdict |
    |---|---|--:|---|
    | [r_real.md](r_real.md) — headline | 2013-01 | 12,000 | **PASS** — wins |
    """
)


def _setup(tmp_path, report_text=_GOOD_REPORT, summary_text=_GOOD_SUMMARY, report_name="r_real.md"):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / report_name).write_text(report_text, encoding="utf-8")
    summary = reports / "SUMMARY.md"
    summary.write_text(summary_text, encoding="utf-8")
    return reports, summary


def test_clean_fixture_has_no_problems(tmp_path):
    reports, summary = _setup(tmp_path)
    assert check_index(reports, summary).ok


def test_uncovered_report_is_flagged(tmp_path):
    reports, summary = _setup(tmp_path)
    (reports / "ghost_real.md").write_text("# x — 2013-01, n=5000\n", encoding="utf-8")
    result = check_index(reports, summary)
    assert not result.ok
    assert any("ghost_real.md" in p and "not listed" in p for p in result.problems)


def test_dangling_row_is_flagged(tmp_path):
    summary = _GOOD_SUMMARY.replace("[r_real.md](r_real.md)", "[gone_real.md](gone_real.md)")
    reports, summary_path = _setup(tmp_path, summary_text=summary)
    result = check_index(reports, summary_path)
    assert not result.ok
    assert any("gone_real.md" in p and "not on disk" in p for p in result.problems)


def test_month_mismatch_is_flagged(tmp_path):
    summary = _GOOD_SUMMARY.replace("2013-01", "2016-05")
    reports, summary_path = _setup(tmp_path, summary_text=summary)
    result = check_index(reports, summary_path)
    assert not result.ok
    assert any("dump month mismatch" in p for p in result.problems)


def test_n_mismatch_is_flagged(tmp_path):
    summary = _GOOD_SUMMARY.replace("12,000", "99,999")
    reports, summary_path = _setup(tmp_path, summary_text=summary)
    result = check_index(reports, summary_path)
    assert not result.ok
    assert any("n=12000" in p for p in result.problems)


def test_verdict_mismatch_is_flagged(tmp_path):
    summary = _GOOD_SUMMARY.replace("**PASS**", "**FAIL**")
    reports, summary_path = _setup(tmp_path, summary_text=summary)
    result = check_index(reports, summary_path)
    assert not result.ok
    assert any("gate verdict mismatch" in p for p in result.problems)


def test_verdict_caveat_token_leads_with_pass(tmp_path):
    """A '**PASS (caveat)**' verdict cell reads as PASS, not a mismatch."""
    summary = _GOOD_SUMMARY.replace("**PASS** — wins", "**PASS (caveat)** — wins")
    reports, summary_path = _setup(tmp_path, summary_text=summary)
    assert check_index(reports, summary_path).ok
