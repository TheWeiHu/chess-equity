"""Numeric drift guard for the committed real-data headline report (task 0159).

``reports/validation_real.md`` carries hardcoded headline numbers — the per-model gate
deltas, the bootstrap CI, and the ``## Overall`` log-loss/Brier — committed as the
thesis evidence. The existing :mod:`tests.test_real_evidence_guard` is a *structural*
guard: it asserts the file exists, is real-sized (n>=8000), and has a PASS verdict +
a CI section. Nothing, until now, checked that those numbers still match a fresh regen
from the cached dump — so a refactor in the metrics / baseline / wdl-a code could leave
the committed numbers stale while CI stays green.

This module is the *numeric* guard. :func:`regen_headline_gate` rebuilds the exact
n=12000 dataset from the cached ``2013-01`` dump (deterministically — ``iter_rows`` takes
the first-N evaluated positions, and ``cp_eval`` comes from the PGN ``[%eval]`` tag, so no
Stockfish is needed) and re-runs the torch-free ``baseline``/``wdl-a`` gate at the
committed seed. :func:`parse_committed_headline` pulls the same numbers back out of the
markdown, and :func:`compare_headline` reports any field that has drifted beyond
tolerance. The Maia-2 leg is deliberately out of scope — it needs torch + weights (held),
so it can't run unattended; the guard covers the torch-free headline numbers, which *are*
the gate verdict the thesis hangs on.

The dump is ~30 GB-class to download but only ~18 MB cached, so the guard runs only when
the dump is already on disk (see :func:`cached_dump`) — the dump-gated test skips cleanly
otherwise, exactly like the engine-smoke jobs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from chess_equity.data.download import DEFAULT_DUMP_DIR, dump_path

# The committed headline run's pinned parameters (the report header reads
# ``... 2013-01, n=12000, --with-fen (seed 0)`` and its significance section says
# "Paired bootstrap (1000 resamples)"). These reproduce the committed numbers EXACTLY,
# so the guard can assert near-equality rather than a loose band.
HEADLINE_MONTH = "2013-01"
HEADLINE_N = 12000
HEADLINE_SEED = 0
HEADLINE_RESAMPLES = 1000
# baseline + the one torch-free challenger the gate verdict is about. maia2 is excluded
# (torch/weights held), so the guard covers exactly the numbers it can reproduce offline.
HEADLINE_MODELS = ("baseline", "wdl-a")

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_REPORT = REPO_ROOT / "reports" / "validation_real.md"

# Tolerances. The point metrics (overall log-loss/Brier, gate deltas) have no RNG, so the
# regen matches the committed 4-dp values exactly; ABS_TOL=5e-4 catches any injected change
# of >=0.0005 (5 units in the last printed digit) while staying immune to float-formatting
# noise. The CI bounds come from a seeded bootstrap that also reproduces exactly at the
# pinned seed/resamples; a slightly looser CI_TOL absorbs any cross-platform BLAS jitter.
ABS_TOL = 5e-4
CI_TOL = 2e-3


@dataclass(frozen=True)
class ModelHeadline:
    """One predictor's headline numbers (committed or regenerated)."""

    log_loss: float
    brier: float


@dataclass(frozen=True)
class DeltaHeadline:
    """A challenger's gate verdict line: deltas vs baseline, the headline CI, PASS/FAIL."""

    log_loss_delta: float
    brier_delta: float
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    passed: bool


@dataclass(frozen=True)
class Headline:
    """The headline numbers of a run — what the guard compares across committed vs regen.

    ``overall`` maps predictor name -> its ``## Overall`` log-loss/Brier; ``deltas`` maps
    each challenger -> its gate verdict line. ``n``/``seed`` come from the report header on
    the committed side and from the run parameters on the regen side, so the guard can
    assert the regen was done at the same pinned settings the report claims.
    """

    n: int
    seed: Optional[int]
    overall: Dict[str, ModelHeadline]
    deltas: Dict[str, DeltaHeadline]


# --- Parsing the committed markdown -------------------------------------------------

_FLOAT = r"[-+]?\d+\.\d+"
# Gate-verdict line, e.g.:
#   - **wdl-a** beats baseline: logloss -0.3403, brier -0.0355; log_loss 95% CI
#     [-0.3881, -0.2978] (CI clears zero) -> **PASS**
_VERDICT_RE = re.compile(
    r"\*\*(?P<name>[\w.-]+)\*\*[^\n]*?logloss\s+(?P<ll>" + _FLOAT + r")"
    r",\s*brier\s+(?P<br>" + _FLOAT + r")"
    r"[^\n]*?CI\s*\[\s*(?P<lo>" + _FLOAT + r")\s*,\s*(?P<hi>" + _FLOAT + r")\s*\]"
    r"[^\n]*?->\s*\*\*(?P<verdict>PASS|FAIL)\*\*",
)


def _parse_header_n_seed(report: str) -> Tuple[int, Optional[int]]:
    """Pull ``n`` and ``seed`` from the report's H1, e.g. ``... n=12000 ... (seed 0)``."""
    n_match = re.search(r"\bn=(\d+)", report)
    seed_match = re.search(r"\(seed\s+(\d+)\)", report)
    n = int(n_match.group(1)) if n_match else 0
    seed = int(seed_match.group(1)) if seed_match else None
    return n, seed


def _parse_overall(report: str) -> Dict[str, ModelHeadline]:
    """Parse the ``## Overall`` table: ``| predictor | n | log-loss | Brier | ECE |``."""
    m = re.search(r"^##\s+Overall\s*$(.*?)(?=^##\s|\Z)", report, re.MULTILINE | re.DOTALL)
    out: Dict[str, ModelHeadline] = {}
    if not m:
        return out
    for line in m.group(1).splitlines():
        cells = [c.strip() for c in line.split("|") if c.strip()]
        # Data rows are: name, n (int), log-loss, Brier, ECE — skip the header/separator.
        if len(cells) != 5 or not re.fullmatch(r"\d+", cells[1]):
            continue
        name = cells[0]
        try:
            out[name] = ModelHeadline(log_loss=float(cells[2]), brier=float(cells[3]))
        except ValueError:
            continue
    return out


def parse_committed_headline(report: str) -> Headline:
    """Extract the committed headline numbers from a ``validation_real.md`` body."""
    n, seed = _parse_header_n_seed(report)
    deltas: Dict[str, DeltaHeadline] = {}
    for m in _VERDICT_RE.finditer(report):
        deltas[m.group("name")] = DeltaHeadline(
            log_loss_delta=float(m.group("ll")),
            brier_delta=float(m.group("br")),
            ci_lo=float(m.group("lo")),
            ci_hi=float(m.group("hi")),
            passed=m.group("verdict") == "PASS",
        )
    return Headline(n=n, seed=seed, overall=_parse_overall(report), deltas=deltas)


# --- Regenerating from the cached dump ----------------------------------------------

def cached_dump(month: str = HEADLINE_MONTH, dump_dir: str = DEFAULT_DUMP_DIR) -> Optional[Path]:
    """The cached dump for ``month`` if it is on disk, else ``None`` (so callers skip)."""
    p = dump_path(month, dump_dir)
    return p if p.exists() else None


def regen_headline_gate(
    dump: Path,
    out_dir: Path,
    *,
    sample: int = HEADLINE_N,
    seed: int = HEADLINE_SEED,
    n_resamples: int = HEADLINE_RESAMPLES,
    models: Sequence[str] = HEADLINE_MODELS,
) -> Headline:
    """Rebuild the headline dataset from ``dump`` and re-run the torch-free gate.

    Deterministic: ``build_dataset`` takes the first ``sample`` evaluated positions and
    ``cp_eval`` is read from the PGN, so the same dump yields the same rows. ``out_dir``
    is where the temporary CSV lands (a pytest ``tmp_path`` in the test). Returns the
    regenerated headline numbers in the same shape as :func:`parse_committed_headline`.
    """
    from chess_equity.data.build import build_dataset, load_rows
    from chess_equity.validate.harness import (
        build_predictors,
        compare_to_baseline,
        evaluate,
        gate_verdicts,
    )

    ds = build_dataset(str(dump), str(out_dir), sample=sample, fmt="csv", name="drift_ds")
    rows = load_rows(str(ds))
    predictors = build_predictors(list(models))
    reports = evaluate(rows, predictors, bins=10)
    comparisons = (
        compare_to_baseline(rows, predictors, baseline="baseline", n_resamples=n_resamples, seed=seed)
        if n_resamples > 0 and len(predictors) > 1
        else None
    )
    verdicts = gate_verdicts(reports, baseline_name="baseline", comparisons=comparisons)

    overall = {r.name: ModelHeadline(log_loss=r.overall.log_loss, brier=r.overall.brier) for r in reports}
    deltas = {
        v.name: DeltaHeadline(
            log_loss_delta=v.log_loss_delta,
            brier_delta=v.brier_delta,
            ci_lo=v.headline_ci.lo if v.headline_ci else None,
            ci_hi=v.headline_ci.hi if v.headline_ci else None,
            passed=v.passed,
        )
        for v in verdicts
    }
    n = reports[0].overall.n if reports else 0
    return Headline(n=n, seed=seed, overall=overall, deltas=deltas)


# --- Comparing the two ---------------------------------------------------------------

@dataclass(frozen=True)
class Drift:
    """One field whose committed value no longer matches the fresh regen."""

    field: str
    committed: float
    regen: float

    @property
    def delta(self) -> float:
        return self.regen - self.committed

    def __str__(self) -> str:
        return f"{self.field}: committed={self.committed:+.4f} regen={self.regen:+.4f} (Δ={self.delta:+.4f})"


def _drifts(field_name: str, committed: float, regen: float, tol: float) -> List[Drift]:
    return [Drift(field_name, committed, regen)] if abs(committed - regen) > tol else []


def compare_headline(
    committed: Headline,
    regen: Headline,
    *,
    abs_tol: float = ABS_TOL,
    ci_tol: float = CI_TOL,
) -> List[Drift]:
    """List every committed headline field that the fresh regen no longer reproduces.

    Compares only the predictors present on both sides (so the maia2 row in the committed
    report, which the torch-free regen omits, is not spuriously flagged). An empty list
    means the committed numbers still hold; a non-empty list is drift — the committed
    evidence has silently rotted relative to the code.
    """
    findings: List[Drift] = []
    for name in committed.overall.keys() & regen.overall.keys():
        c, r = committed.overall[name], regen.overall[name]
        findings += _drifts(f"overall[{name}].log_loss", c.log_loss, r.log_loss, abs_tol)
        findings += _drifts(f"overall[{name}].brier", c.brier, r.brier, abs_tol)
    for name in committed.deltas.keys() & regen.deltas.keys():
        c, r = committed.deltas[name], regen.deltas[name]
        findings += _drifts(f"delta[{name}].log_loss", c.log_loss_delta, r.log_loss_delta, abs_tol)
        findings += _drifts(f"delta[{name}].brier", c.brier_delta, r.brier_delta, abs_tol)
        if c.ci_lo is not None and r.ci_lo is not None:
            findings += _drifts(f"delta[{name}].ci_lo", c.ci_lo, r.ci_lo, ci_tol)
        if c.ci_hi is not None and r.ci_hi is not None:
            findings += _drifts(f"delta[{name}].ci_hi", c.ci_hi, r.ci_hi, ci_tol)
        if c.passed != r.passed:
            findings.append(Drift(f"delta[{name}].passed", float(c.passed), float(r.passed)))
    return findings
