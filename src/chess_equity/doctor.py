"""``chess-equity doctor`` — verify the optional external engines actually run (task 0073).

The core path needs none of this (see ``DEPENDENCIES.md``): the baseline CLI, tests, and
CI run on ``python-chess`` alone. But two bars depend on heavyweight, externally-provisioned
engines:

* the **classic centipawn bar** → a real **Stockfish** binary (``StockfishEngine``), and
* the **rating-conditioned equity bar** → **Maia-2** (``pip install maia2``, pulls torch,
  downloads a checkpoint on first use).

"Make Stockfish work, and install Maia" (task 0073) is really *provision + verify*. This
turns the verify half into one command: ``chess-equity doctor`` resolves Stockfish and runs
a real eval, imports Maia-2 and runs a real inference, and reports PASS/FAIL per engine with
the same install hint the adapters raise. Exit code is non-zero if any checked engine is
missing or broken, so it can gate a provisioning step.

The *reporting* logic (:func:`run_doctor`) is pure and the engine probes are injectable, so
the unit tests exercise it with fakes — no binary, no torch, no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Callable, List, Optional, TextIO

import chess

START_FEN = chess.STARTING_FEN


@dataclass
class Check:
    """The outcome of probing one optional engine.

    ``warn`` flags a *soft* problem: the check still passes (``ok`` stays True, exit code
    unaffected) but something is off enough to surface as ``WARN`` rather than ``PASS``
    (e.g. a model artifact that works but lacks leakage-guard provenance — task 0199).
    """

    name: str
    ok: bool
    detail: str
    warn: bool = False


class DoctorWarning(Exception):
    """A soft preflight failure: the probe's subject works, but a non-fatal caveat should
    surface as ``WARN`` (passing) instead of ``PASS``. :func:`check` maps it to a passing
    :class:`Check` with ``warn=True``; any *other* exception is a hard ``FAIL``."""


# A probe runs the real engine and returns a human-readable "it works" detail string,
# or raises (missing install, or installed-but-broken). Injectable so tests use fakes.
Probe = Callable[[], str]


def _probe_stockfish() -> str:
    """Resolve a real Stockfish and evaluate the start position."""
    from chess_equity.stockfish import StockfishEngine, StockfishNotFound, stockfish_path

    path = stockfish_path()
    if path is None:  # be explicit rather than relying on the engine to raise
        raise StockfishNotFound(
            "no Stockfish binary on PATH or $STOCKFISH_PATH — "
            "`brew install stockfish` / `apt-get install stockfish` (see DEPENDENCIES.md)"
        )
    ev = StockfishEngine(depth=8).eval(START_FEN)
    return f"{path}: startpos eval cp={ev.cp}"


def _probe_maia2() -> str:
    """Build the real Maia-2 model and run one rating-conditioned inference."""
    from chess_equity.cli import build_model

    model = build_model("maia2")
    eq = round(model.evaluate(START_FEN, 1500, 1500).equity_white, 1)
    return f"startpos equity(1500/1500) = {eq}% White"


# `doctor` verifies the optional engines, but not the project's actual headline claim:
# that the committed real-data gate reports are present and still passing. Without this a
# repo could ship with a missing or regressed proof and doctor would stay green.
# `probe_evidence` reads `reports/SUMMARY.md` — the canonical gate index, whose verdicts
# are quoted/parsed from each report's own header — and confirms every listed report
# exists on disk and corroborates its stated verdict.

# The only report SUMMARY may legitimately mark FAIL: the end-to-end board→WDL net, kept on
# purpose as a negative result (Approach D loses to the centipawn baseline). Any *other* FAIL
# is a regression doctor must catch.
EVIDENCE_FAIL_ALLOWLIST = frozenset({"wdl_net_real.md"})

# A row whose verdict is PASS must have its report corroborate it. Markers differ across
# reports: most say "PASS"; goodmoves_real states its pass in prose with a "✅". Accept either.
_PASS_MARKERS = ("PASS", "✅")


def reports_dir() -> Optional[Path]:
    """The repo's ``reports/`` dir, or ``None`` from an installed wheel without assets."""
    candidate = Path(__file__).resolve().parents[2] / "reports"
    return candidate if candidate.is_dir() else None


def _parse_summary_rows(summary_text: str) -> list:
    """Extract ``(filename, verdict)`` for each report row in ``SUMMARY.md``'s table.

    ``verdict`` is normalised to ``PASS`` / ``FAIL`` / ``info`` (``PASS (caveat)`` → ``PASS``).
    Only table rows that link a ``*.md`` report are returned; prose and the header row are
    skipped. Raises if the table has no parseable rows (a gutted/renamed SUMMARY).
    """
    import re

    rows = []
    for line in summary_text.splitlines():
        line = line.strip()
        if not line.startswith("| ["):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 4:
            continue
        link = re.search(r"\(([^)]+\.md)\)", cols[0])
        if not link:
            continue
        filename = link.group(1)
        verdict_col = cols[-1].upper()
        if "**FAIL**" in verdict_col or verdict_col.startswith("FAIL"):
            verdict = "FAIL"
        elif "**PASS**" in verdict_col or "PASS" in verdict_col:
            verdict = "PASS"
        else:
            verdict = "info"
        rows.append((filename, verdict))
    if not rows:
        raise ValueError("SUMMARY.md has no parseable report rows (gate index empty or renamed?)")
    return rows


def probe_evidence(directory: Optional[Path] = None) -> str:
    """Assert the committed real-data gate reports are present and still passing (task 0195).

    Reads ``reports/SUMMARY.md`` (the gate index) and, for every report it lists:

    * confirms the linked ``*_real.md`` file exists on disk — a missing proof fails here;
    * for a **PASS** verdict, confirms the report itself states a pass (a ``PASS`` token or
      the prose ``✅`` goodmoves uses) — a report that regressed to no-longer-passing while
      SUMMARY still claims PASS is caught;
    * for a **FAIL** verdict, confirms the file is the one allowlisted deliberate negative
      result (:data:`EVIDENCE_FAIL_ALLOWLIST`) and that the report states ``FAIL`` — any
      *other* FAIL is an unintended regression;
    * **info** rows (calibration/disagreement/threshold reports that state no gate) are
      existence-checked only.

    Reads report text but no datasets — safe to run unattended. Raises on the first problem
    so :func:`check` reports a FAIL with the offending detail.
    """
    directory = directory or reports_dir()
    if directory is None or not directory.is_dir():
        raise ValueError("reports/ dir not found (running from a wheel without assets?)")
    summary = directory / "SUMMARY.md"
    if not summary.is_file():
        raise ValueError("missing reports/SUMMARY.md (the gate index)")

    rows = _parse_summary_rows(summary.read_text(encoding="utf-8"))
    pass_n = fail_n = info_n = 0
    for filename, verdict in rows:
        report = directory / filename
        if not report.is_file():
            raise ValueError(f"gate report listed in SUMMARY.md is missing on disk: {filename}")
        text = report.read_text(encoding="utf-8")
        if verdict == "PASS":
            if not any(marker in text for marker in _PASS_MARKERS):
                raise ValueError(
                    f"{filename}: SUMMARY.md marks it PASS but the report states no pass "
                    "(regressed proof?)"
                )
            pass_n += 1
        elif verdict == "FAIL":
            if filename not in EVIDENCE_FAIL_ALLOWLIST:
                raise ValueError(
                    f"{filename}: gate report regressed to FAIL (only "
                    f"{sorted(EVIDENCE_FAIL_ALLOWLIST)} is an allowed deliberate FAIL)"
                )
            if "FAIL" not in text.upper():
                raise ValueError(f"{filename}: SUMMARY.md marks it FAIL but the report says no FAIL")
            fail_n += 1
        else:
            info_n += 1

    return (
        f"SUMMARY.md gate index OK — {len(rows)} report(s) present: "
        f"{pass_n} PASS, {fail_n} deliberate FAIL, {info_n} info"
    )


# --- active equity-model preflight (task 0199) -----------------------------------------
#
# `doctor`'s engine checks prove Stockfish/Maia-2 *can* run and the bundle/feed are
# shippable, but not that the model the overlay is configured to use will actually
# produce a bar. `probe_model` loads the selected `--model` and evaluates one fixture FEN
# so a missing/garbled artifact (or a NaN/out-of-range bar) turns doctor red *before* air,
# while a model that works but lacks leakage-guard provenance surfaces as WARN.

# A wdl-a artifact fit on this few rows or fewer is the committed tiny smoke-test seed,
# not a real fit — the bar still renders, but the numbers aren't trustworthy on air (WARN).
# The real shipped artifact is n_train=50000, well clear of this floor.
_WDL_A_SEED_MAX_TRAIN = 1000

# Fixture inputs for the "does it produce a sane bar?" probe — startpos at a mid rating.
_MODEL_FIXTURE_FEN = START_FEN
_MODEL_FIXTURE_ELO = 1500


def _wdl_a_provenance_warnings(meta: dict) -> List[str]:
    """Soft caveats on a wdl-a artifact's fit metadata (absent → empty list = clean PASS).

    A missing ``fit_month`` means the 0112 leakage guard can't refuse an eval set that *is*
    the training month; an ``n_train`` at/under the seed floor means it's the committed
    overfit smoke seed. Either is a WARN, not a FAIL — the model still evaluates.
    """
    warnings: List[str] = []
    if not meta.get("fit_month"):
        warnings.append("artifact has no fit_month (the 0112 leakage guard can't run)")
    n_train = meta.get("n_train")
    if isinstance(n_train, (int, float)) and not isinstance(n_train, bool):
        if n_train <= _WDL_A_SEED_MAX_TRAIN:
            warnings.append(
                f"n_train={n_train} looks like the tiny overfit seed, not a real fit"
            )
    return warnings


def _default_build_model(model_name: str):
    """Construct the named model via the CLI registry (lazy import avoids a cycle)."""
    from chess_equity.cli import build_model

    return build_model(model_name)


def probe_model(
    model_name: str = "baseline",
    build: Optional[Callable[[str], Any]] = None,
    artifact_path: Optional[Path] = None,
) -> str:
    """Assert the ACTIVE equity model loads and produces a sane bar before going live (0199).

    The reliability gap doctor's engine checks leave: the overlay reads *one* configured
    ``--model``, and nothing verifies it actually works until the first live position. This
    closes it:

    * the model **constructs** — ``--model wdl-a`` loads + parses its committed artifact;
      ``--model baseline`` builds its objective engine. A missing/garbled artifact FAILs;
    * it evaluates one fixture FEN to a **finite White-POV bar in [0,100]** — a NaN or
      out-of-range equity FAILs (the overlay would render a broken bar);
    * (wdl-a only) the artifact carries **fit provenance**: a missing ``fit_month`` (the
      leakage guard can't run) or a seed-sized ``n_train`` is a WARN, not a FAIL.

    Loads the model but no datasets/network — safe to run unattended for ``baseline`` and
    ``wdl-a``. ``build``/``artifact_path`` are injectable for tests. Raises ``ValueError``
    on a hard problem (FAIL); raises :class:`DoctorWarning` for a soft one (WARN).
    """
    build = build or _default_build_model
    provenance: Optional[str] = None
    warnings: List[str] = []

    if model_name == "wdl-a":
        from chess_equity.wdl_regression import default_artifact_path, load_wdl_a_model

        path = Path(artifact_path) if artifact_path else default_artifact_path()
        if not path.is_file():
            raise ValueError(f"--model wdl-a artifact missing on disk: {path}")
        try:
            fitted = load_wdl_a_model(str(path))
        except Exception as exc:  # noqa: BLE001 - report the parse/shape failure as a FAIL
            raise ValueError(f"--model wdl-a artifact unreadable ({path.name}): {exc}") from exc
        meta = fitted.meta or {}
        warnings = _wdl_a_provenance_warnings(meta)
        provenance = f"n_train={meta.get('n_train')}, fit_month={meta.get('fit_month') or 'absent'}"

    model = build(model_name)  # unknown model / failed load → FAIL via check()
    equity = model.evaluate(_MODEL_FIXTURE_FEN, _MODEL_FIXTURE_ELO, _MODEL_FIXTURE_ELO)
    bar = equity.equity_white
    if not isinstance(bar, (int, float)) or isinstance(bar, bool) or not isfinite(bar):
        raise ValueError(f"--model {model_name} produced a non-finite bar: {bar!r}")
    if not 0.0 <= float(bar) <= 100.0:
        raise ValueError(f"--model {model_name} bar {bar} is outside [0,100]% White-POV")

    win = float(bar) / 100.0
    detail = f"--model {model_name} loads; startpos win-equity {win:.2f} (0..1)"
    if provenance is not None:
        detail += f"; {provenance}"
    if warnings:
        raise DoctorWarning(detail + " — WARN: " + "; ".join(warnings))
    return detail


def check(name: str, probe: Probe) -> Check:
    """Run one probe, mapping success/exception to a :class:`Check`.

    A :class:`DoctorWarning` becomes a *passing* check flagged ``warn`` (a soft caveat,
    exit code unaffected). A clean failure exception (e.g. ``StockfishNotFound`` /
    ``Maia2NotInstalled``) becomes a failed check carrying its install hint; any other
    exception is reported as installed-but-broken so the message distinguishes the two.
    """
    try:
        return Check(name, True, probe())
    except DoctorWarning as warn:
        return Check(name, True, str(warn) or warn.__class__.__name__, warn=True)
    except Exception as exc:  # noqa: BLE001 - the whole point is to report, not crash
        return Check(name, False, str(exc) or exc.__class__.__name__)


def run_doctor(checks: List[Check], out: Optional[TextIO] = None) -> int:
    """Print each check and return 0 iff every checked engine works."""
    import sys

    out = out if out is not None else sys.stdout
    failures = 0
    for c in checks:
        mark = "FAIL" if not c.ok else ("WARN" if c.warn else "PASS")
        print(f"[{mark}] {c.name}: {c.detail}", file=out)
        if not c.ok:
            failures += 1
    summary = "all engines OK" if failures == 0 else f"{failures} engine(s) need attention"
    print(f"\n{len(checks) - failures}/{len(checks)} engines OK — {summary}", file=out)
    return 1 if failures else 0


def doctor(
    out: Optional[TextIO] = None,
    probes: Optional[dict] = None,
    engines: Optional[List[str]] = None,
    evidence_probe: Optional[Probe] = None,
    model_probe: Optional[Probe] = None,
) -> int:
    """Probe the optional engines with the real backends (override ``probes`` in tests).

    ``engines`` restricts the probes to a subset (e.g. ``["stockfish"]`` for a
    binary-only CI runner that never installs torch/Maia-2); ``None`` checks all.

    ``evidence_probe`` (set by ``doctor --evidence``) appends a check that the committed
    real-data gate reports listed in ``reports/SUMMARY.md`` are present and still state
    their expected verdict (task 0195) — so a missing/regressed proof turns doctor red.
    Reads report text but no datasets — safe to run unattended.

    ``model_probe`` (set by ``doctor --model NAME``) appends a preflight that the active
    equity model loads and produces a finite in-range bar (task 0199): a missing/garbled
    artifact FAILs, missing wdl-a fit provenance WARNs. Torch-free for baseline/wdl-a.
    """
    probes = probes or {"stockfish": _probe_stockfish, "maia2": _probe_maia2}
    if engines:
        probes = {name: probes[name] for name in engines if name in probes}
    checks = [check(name, probe) for name, probe in probes.items()]
    if evidence_probe is not None:
        checks.append(check("evidence", evidence_probe))
    if model_probe is not None:
        checks.append(check("model", model_probe))
    return run_doctor(checks, out=out)
