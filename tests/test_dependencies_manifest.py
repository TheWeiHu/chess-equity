"""CI guard: every pyproject extra is documented in DEPENDENCIES.md (task 0057).

DEPENDENCIES.md (task 0045) declared an upkeep rule — "any task that introduces a
new external dependency adds its row here" — but nothing enforced it, so the manifest
would silently rot. This test fails CI when a `[project.optional-dependencies]` extra
has no matching row, the moment a new extra is added.

Scope (deliberately narrow, per the 0045 follow-up): extras only, by exact name. We do
NOT grep prose for `brew install` / binary references — that's too brittle to gate on.
"""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
MANIFEST = ROOT / "DEPENDENCIES.md"


def _extras():
    with open(PYPROJECT, "rb") as fh:
        data = tomllib.load(fh)
    return sorted(data.get("project", {}).get("optional-dependencies", {}))


def test_pyproject_has_extras():
    # Guard the guard: if this ever returns nothing, the parse broke (or the table
    # moved) and the check below would pass vacuously.
    assert _extras(), "expected at least one [project.optional-dependencies] extra"


def test_every_extra_has_a_dependencies_md_row():
    text = MANIFEST.read_text(encoding="utf-8")
    missing = []
    for name in _extras():
        # The manifest refers to an extra as `name` extra or **name** extra; accept
        # either, but require the literal extra name adjacent to the word "extra" so a
        # short name like "dev" can't match incidental prose.
        pattern = rf"(`{re.escape(name)}`|\*\*{re.escape(name)}\*\*)\s+extra"
        if not re.search(pattern, text):
            missing.append(name)
    assert not missing, (
        "DEPENDENCIES.md is missing a row for pyproject extra(s): "
        + ", ".join(missing)
        + " — add a manifest row (see the 'Upkeep rule' section)."
    )
