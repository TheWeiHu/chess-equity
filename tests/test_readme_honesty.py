"""README honesty guard (task 0134).

`reports/validation_sample.md` is a 15-row *illustrative* run, not statistically powered
proof. The README's "Results" and "Does equity beat centipawns?" sections cite it, so the
risk is the doc drifting into reading as if the sample IS the headline evidence. The real
proof comes from the `chess-equity headline` recipe on a real dump (and the committed
artifact that task 0128 lands), never from the sample fixture.

This test locks that honesty in: every README mention of the sample report must sit next to
an illustrative / not-powered disclaimer, and the README must point at the headline recipe
as the route to real evidence. Docs-only guard — no engine, no network, no torch.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"

SAMPLE_REPORT = "validation_sample.md"

# Phrases that, near a sample-report mention, make clear it is illustrative — not proof.
DISCLAIMERS = (
    "illustrative",
    "not statistically powered",
    "never the headline",
    "isn't yet proof",
    "smoke test",
)

# The recipe that produces real, statistically-powered evidence.
HEADLINE_RECIPE = "chess-equity headline"

# Max line distance between a sample-report mention and its disclaimer. The README cites the
# file once per section inside a single prose block; 8 lines covers a block + its code fence
# without reaching into an unrelated section.
DISCLAIMER_WINDOW = 8


def _lines() -> list[str]:
    return README.read_text(encoding="utf-8").splitlines()


def test_every_sample_report_mention_is_marked_illustrative() -> None:
    lines = _lines()
    mention_idx = [i for i, ln in enumerate(lines) if SAMPLE_REPORT in ln]
    assert mention_idx, "README no longer mentions the sample report — update this guard"

    disclaimer_idx = [
        i for i, ln in enumerate(lines) if any(d in ln.lower() for d in DISCLAIMERS)
    ]
    for i in mention_idx:
        nearest = min((abs(i - d) for d in disclaimer_idx), default=DISCLAIMER_WINDOW + 1)
        assert nearest <= DISCLAIMER_WINDOW, (
            f"README line {i + 1} cites {SAMPLE_REPORT} with no illustrative/not-proof "
            f"disclaimer within {DISCLAIMER_WINDOW} lines:\n  {lines[i].strip()}"
        )


def test_readme_points_to_the_headline_recipe_for_real_evidence() -> None:
    text = README.read_text(encoding="utf-8")
    # Collapse runs of whitespace so a phrase that wraps across a line break still matches.
    flat = " ".join(text.lower().split())
    assert HEADLINE_RECIPE in text, (
        "README must point readers at the `chess-equity headline` recipe as the route to "
        "real, statistically-powered evidence"
    )
    # The strong honesty claim must be present verbatim somewhere: the sample is never the proof.
    assert "never the headline" in flat, (
        "README must state outright that the sample report is never the headline proof"
    )


def test_readme_names_the_real_evidence_followup_task() -> None:
    # Acceptance: the doc points at the route to a *committed* real artifact (task 0128),
    # so a reader knows where real evidence will live, not just how to regenerate it.
    assert "0128" in README.read_text(encoding="utf-8"), (
        "README must point at task 0128 as the route to a committed real-evidence artifact"
    )
