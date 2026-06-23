"""Evidence-index drift guard for ``reports/SUMMARY.md`` (task 0219).

``reports/SUMMARY.md`` is the real-data evidence index: one table row per committed
``reports/*_real*.md`` artifact, with the dump month, ``n``, and a PASS/FAIL/info verdict
**quoted or parsed from that report's own header / ``## Gate verdict`` section**. Its own
text says "When new real-data reports land, regenerate this table from their headers" —
but nothing enforced it, so a regenerated report (or a freshly-landed one) could silently
diverge from the index: a new ``*_real*.md`` never added to the table, a row pointing at a
deleted file, or a header whose month/n/verdict no longer matches its row. That weakens
the proof artifact exactly where it claims rigour.

This module is the *index* guard — the structural complement to
:mod:`chess_equity.validate.drift_guard` (which re-runs ONE report's numbers from the
cached dump). It **reads no dataset and computes no numbers** (honoring the real-data-only
policy in ``CLAUDE.md``): it parses the markdown of each report header + ``## Gate
verdict`` line and of the SUMMARY table, then lists every disagreement. Fully unattended —
wired into the test suite via ``tests/test_index_guard.py``.

What it checks (and, deliberately, what it skips to avoid false positives):

* **Coverage (both directions).** Every ``reports/*_real*.md`` must appear as a linked
  row, and every row's link must resolve to a file on disk. ``*_sample.md`` are
  illustrative, not evidence, so they are excluded (SUMMARY's own Scope section says so).
* **Dump month.** Every ``YYYY-MM`` token in the report H1 must appear in the row's Dump
  cell and vice-versa. Reports whose H1 carries no month token (e.g. ``goodmoves_real``)
  skip this check rather than mis-fire.
* **n.** Each ``n=``/``n_high=`` value in the report H1 must appear (comma-grouping
  ignored) in the row's n cell. Reports with no H1 ``n`` skip this check.
* **Verdict.** Only for reports that carry a ``## Gate verdict`` section with explicit
  ``-> **PASS**`` / ``-> **FAIL**`` lines: the report verdict (FAIL if any line FAILs,
  else PASS) must equal the row's leading bold verdict token. Reports with no gate section
  (``info`` rows, plus the prose-PASS ``goodmoves``/``recalibration`` rows) are not
  machine-derivable, so their verdict is left unchecked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO_ROOT / "reports"
SUMMARY = REPORTS_DIR / "SUMMARY.md"

# Real-data evidence reports follow ``*_real*.md`` (note ``validation_real_2016-05.md`` and
# friends do NOT end in ``_real.md``). ``*_sample.md`` are illustrative-only and excluded.
REPORT_GLOB = "*_real*.md"

# A ``YYYY-MM`` token. Digit lookarounds (not ``\b``) so it still matches when glued to a
# word char, e.g. ``rated_2013-01`` — a ``\b`` between ``_`` and ``2`` doesn't exist.
_MONTH_RE = re.compile(r"(?<!\d)(\d{4}-\d{2})(?!\d)")
# H1 ``n``: ``n=12000``, ``n_high=49269``, ``n=12,000``, ``(n=100000)`` — capture the digits
# (with optional comma grouping) right after an ``n``/``n_high`` ``=``.
_N_RE = re.compile(r"\bn(?:_high)?\s*=\s*([\d,]+)")
# A gate-verdict line's terminal token, e.g. ``... -> **PASS**`` / ``-> **FAIL**``.
_GATE_LINE_RE = re.compile(r"->\s*\*\*(PASS|FAIL)\*\*")


@dataclass(frozen=True)
class ReportFacts:
    """What the guard can parse out of one ``*_real*.md`` report (header + gate line)."""

    name: str
    months: Set[str]
    ns: Set[int]
    # "PASS"/"FAIL" if the report has a ``## Gate verdict`` section with verdict lines;
    # None if it has no machine-readable gate (an info / prose-verdict report).
    gate_verdict: Optional[str]


@dataclass(frozen=True)
class SummaryRow:
    """One parsed row of the SUMMARY table (the index's claim about a report)."""

    target: str  # the linked filename, e.g. "validation_real.md"
    months: Set[str]
    n_cell: str  # raw n cell text (kept raw — n is matched by substring, not parsed)
    verdict: Optional[str]  # leading bold token: PASS / FAIL / info / None


@dataclass
class IndexReport:
    """The outcome of an index check: a list of human-readable problem strings."""

    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _h1(text: str) -> str:
    """The report's H1 line (first ``# `` heading), or ``""`` if none."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line
    return ""


def _gate_verdict(text: str) -> Optional[str]:
    """Parse the ``## Gate verdict`` section into FAIL (any FAIL line) / PASS / None."""
    m = re.search(
        r"^##\s+Gate verdict\s*$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL
    )
    if not m:
        return None
    verdicts = _GATE_LINE_RE.findall(m.group(1))
    if not verdicts:
        return None
    return "FAIL" if "FAIL" in verdicts else "PASS"


def parse_report(path: Path) -> ReportFacts:
    """Extract the index-relevant facts (month/n/verdict) from one report file."""
    text = path.read_text(encoding="utf-8")
    h1 = _h1(text)
    months = set(_MONTH_RE.findall(h1))
    ns = {int(g.replace(",", "")) for g in _N_RE.findall(h1)}
    return ReportFacts(name=path.name, months=months, ns=ns, gate_verdict=_gate_verdict(text))


def parse_summary_table(text: str) -> List[SummaryRow]:
    """Parse the ``| Report | Dump (month) | n | Verdict |`` table into rows.

    Recognizes data rows by the linked Report cell (``[text](target)``); skips the header
    and separator rows. Months come from the Dump cell; the verdict is the leading word of
    the first ``**bold**`` token in the Verdict cell (so ``**PASS (caveat)**`` -> PASS).
    """
    rows: List[SummaryRow] = []
    link_re = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    bold_re = re.compile(r"\*\*([^*]+)\*\*")
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        link = link_re.search(cells[0])
        if not link:  # header row / separator / non-data table line
            continue
        target = link.group(1).split("#")[0].split("/")[-1].strip()
        verdict_match = bold_re.search(cells[3])
        verdict = verdict_match.group(1).strip().split()[0] if verdict_match else None
        rows.append(
            SummaryRow(
                target=target,
                months=set(_MONTH_RE.findall(cells[1])),
                n_cell=cells[2],
                verdict=verdict,
            )
        )
    return rows


def _n_in_cell(n: int, cell: str) -> bool:
    """Is integer ``n`` present in the row's n cell, ignoring comma grouping?"""
    digits = re.findall(r"[\d,]+", cell)
    return any(d.replace(",", "") == str(n) for d in digits)


def check_index(
    reports_dir: Path = REPORTS_DIR, summary_path: Path = SUMMARY
) -> IndexReport:
    """Compare ``SUMMARY.md`` against the committed ``*_real*.md`` reports.

    Returns an :class:`IndexReport`; ``problems`` is empty iff the index is consistent.
    Reads no dataset and computes no numbers — purely a markdown cross-check.
    """
    report = IndexReport()

    if not summary_path.exists():
        report.problems.append(f"{summary_path} is missing")
        return report

    rows = parse_summary_table(summary_path.read_text(encoding="utf-8"))
    rows_by_target = {r.target: r for r in rows}

    on_disk = {p.name: p for p in sorted(reports_dir.glob(REPORT_GLOB))}

    # --- Coverage, both directions -------------------------------------------------
    for name in on_disk:
        if name not in rows_by_target:
            report.problems.append(f"{name}: real-data report is not listed in SUMMARY.md")
    for r in rows:
        # Only rows that point at a *_real* report are this guard's concern; a row linking
        # something else (e.g. REPRODUCE.md) is out of scope, not an error.
        if not Path(r.target).match(REPORT_GLOB):
            continue
        if r.target not in on_disk:
            report.problems.append(
                f"{r.target}: SUMMARY.md row points to a report that is not on disk"
            )

    # --- Per-report field consistency ----------------------------------------------
    for name, path in on_disk.items():
        row = rows_by_target.get(name)
        if row is None:
            continue  # already flagged as uncovered above
        facts = parse_report(path)

        if facts.months and facts.months != row.months:
            report.problems.append(
                f"{name}: dump month mismatch — report header {sorted(facts.months)} "
                f"vs SUMMARY {sorted(row.months)}"
            )

        for n in sorted(facts.ns):
            if not _n_in_cell(n, row.n_cell):
                report.problems.append(
                    f"{name}: n={n} from report header not found in SUMMARY n cell "
                    f"{row.n_cell!r}"
                )

        if facts.gate_verdict is not None and facts.gate_verdict != row.verdict:
            report.problems.append(
                f"{name}: gate verdict mismatch — report says {facts.gate_verdict} "
                f"vs SUMMARY {row.verdict}"
            )

    return report
