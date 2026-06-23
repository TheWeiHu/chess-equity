"""Numeric drift guard for reports/validation_real.md (task 0159).

Two layers:

1. ``test_committed_headline_matches_fresh_regen`` — the real guard. When the cached
   ``2013-01`` dump is on disk it rebuilds the exact n=12000 dataset, re-runs the
   torch-free ``baseline``/``wdl-a`` gate at the committed seed, and asserts the committed
   headline numbers (Overall log-loss/Brier, gate deltas, bootstrap CI, PASS) still match.
   Absent the dump it SKIPS with a clear message — exactly like the engine-smoke jobs — so
   CI stays green without the ~30 GB-class download.

2. ``test_comparator_flags_injected_number_change`` — a dump-free unit test proving the
   detection logic: mutate one committed number and assert ``compare_headline`` flags it.
   This is the acceptance check ("detects an injected number change in validation_real.md")
   and runs everywhere, dump or not.
"""

from __future__ import annotations

import pytest

from chess_equity.validate.drift_guard import (
    HEADLINE_MONTH,
    HEADLINE_N,
    HEADLINE_SEED,
    REAL_REPORT,
    cached_dump,
    compare_headline,
    parse_committed_headline,
    regen_headline_gate,
)

try:
    import zstandard  # noqa: F401  -- reading the .zst dump needs the data extra

    _HAVE_ZSTD = True
except ImportError:
    _HAVE_ZSTD = False

_DUMP = cached_dump()


@pytest.mark.skipif(
    not REAL_REPORT.exists(),
    reason="reports/validation_real.md not committed — nothing to drift-check",
)
@pytest.mark.skipif(
    _DUMP is None or not _HAVE_ZSTD,
    reason=(
        f"cached {HEADLINE_MONTH} dump absent (or no zstandard) — drift guard needs the "
        "real dump to regenerate; skipping like the engine-smoke jobs"
    ),
)
def test_committed_headline_matches_fresh_regen(tmp_path) -> None:
    committed = parse_committed_headline(REAL_REPORT.read_text(encoding="utf-8"))

    # Sanity: the report is the pinned run this guard knows how to reproduce. If the
    # committed report is ever regenerated at a different n/seed, update the drift_guard
    # constants rather than letting the guard compare against the wrong recipe.
    assert committed.n == HEADLINE_N, (
        f"validation_real.md n={committed.n} != guard's pinned {HEADLINE_N}; "
        "update HEADLINE_N in drift_guard.py if the headline run changed"
    )
    assert committed.seed == HEADLINE_SEED, (
        f"validation_real.md seed={committed.seed} != guard's pinned {HEADLINE_SEED}"
    )

    assert _DUMP is not None  # guaranteed by the skipif; narrows the type for the call
    regen = regen_headline_gate(_DUMP, tmp_path)

    # The regen reproduces the torch-free models; the committed report also carries maia2,
    # which compare_headline ignores (intersection only). What we DO check must be present.
    assert "wdl-a" in committed.deltas, "validation_real.md has no wdl-a gate verdict to check"

    drifts = compare_headline(committed, regen)
    assert not drifts, (
        "committed validation_real.md headline numbers have drifted from a fresh regen on "
        "the cached dump:\n  " + "\n  ".join(str(d) for d in drifts)
    )


def test_comparator_flags_injected_number_change() -> None:
    """The comparator catches a tampered/rotted committed number (dump not required)."""
    text = REAL_REPORT.read_text(encoding="utf-8")
    clean = parse_committed_headline(text)
    assert "wdl-a" in clean.deltas and "wdl-a" in clean.overall, (
        "validation_real.md is missing the wdl-a headline numbers the guard parses"
    )

    # Inject a clearly-out-of-tolerance change into the committed wdl-a log-loss delta,
    # exactly the kind of silent rot the guard exists to catch, and confirm comparing the
    # tampered parse against the pristine one surfaces it.
    tampered_text = text.replace("logloss -0.3403", "logloss -0.2000")
    assert tampered_text != text, "expected the committed wdl-a logloss delta -0.3403 in the report"
    tampered = parse_committed_headline(tampered_text)

    drifts = compare_headline(tampered, clean)
    assert any(d.field == "delta[wdl-a].log_loss" for d in drifts), (
        f"comparator failed to flag the injected wdl-a log-loss change; drifts={drifts}"
    )
