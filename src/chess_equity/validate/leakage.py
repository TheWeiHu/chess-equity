"""Leakage guard: refuse/warn when the eval set is a model's own training month (0112).

`reports/validation_sample.md` hand-warns that `wdl-a`'s edge there is memorization —
it was fit on those very rows. Nothing *enforces* that warning, so a real headline run
could silently report memorized numbers as evidence. This module makes the overlap a
first-class check: every rating-conditioned model records the Lichess month it was fit
on (its artifact ``meta["fit_month"]``), and ``validate`` compares that against the eval
dataset's source month. Same month → the gate's PASS is memorization, not held-out skill.

Pure + dependency-free: the harness/CLI supplies the eval month (declared via
``--eval-month`` or inferred from the dataset path) and the per-model fit months; this
module just detects collisions and renders the warning. The fit month for ``wdl-a`` is
``2016-05`` (see the validation-real-evidence note); held-out runs must use a *different*
month — ``2013-01`` was chosen for exactly that (PR #89).
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional, Sequence

# A ``YYYY-MM`` token not glued to other digits (so ``2016-05`` matches, ``12016-050``
# does not), with a 19xx/20xx year and a real 01–12 month.
_MONTH_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})-(0[1-9]|1[0-2])(?!\d)")


class Leak(NamedTuple):
    """One model whose recorded training month equals the eval dataset's month."""

    model: str
    month: str  # the shared Lichess YYYY-MM month


def infer_month_from_path(path: str) -> Optional[str]:
    """Best-effort ``YYYY-MM`` for a dataset path, e.g. ``data/dataset_2016-05.csv``.

    Used only when the caller did not declare ``--eval-month``: a dataset named after
    its source month gets the leakage check for free. Returns the *last* month-like
    token in the path (the filename usually carries the month, not a parent dir), or
    ``None`` when the path encodes no month — in which case the guard simply stays
    silent rather than guessing.
    """
    matches = _MONTH_RE.findall(path or "")
    if not matches:
        return None
    year, month = matches[-1]
    return f"{year}-{month}"


def model_fit_months(
    names: Sequence[str], wdl_a_path: Optional[str] = None
) -> Dict[str, str]:
    """Map each selected predictor to the Lichess month it was trained on, when recorded.

    Only models with a committed artifact carry provenance; today that is ``wdl-a``,
    whose ``meta["fit_month"]`` records its training dump. Rating-blind predictors
    (``baseline``, ``baseline+clock``) have no training month and cannot leak, so they
    are simply absent from the result. Loading ``wdl-a`` here is cheap (it just reads
    the JSON artifact, no torch).

    ``wdl_a_path`` overrides which artifact's provenance is read (task 0164) — when a run
    scores wdl-a from a refit artifact (``validate --wdl-a-artifact``), the guard must
    check *that* artifact's ``fit_month``, not the committed one's, or a genuine cross-dump
    refit would still trip (or silently miss) the in-distribution check.
    """
    months: Dict[str, str] = {}
    if "wdl-a" in names:
        from chess_equity.wdl_regression import load_wdl_a_model

        fit_month = (load_wdl_a_model(wdl_a_path).meta or {}).get("fit_month")
        if fit_month:
            months["wdl-a"] = str(fit_month)
    return months


def detect_leakage(
    eval_month: Optional[str], fit_months: Dict[str, str]
) -> List[Leak]:
    """The models whose training month equals ``eval_month`` (sorted by name).

    Empty when ``eval_month`` is unknown (nothing to compare) or no selected model was
    trained on it — i.e. a genuinely held-out run. A non-empty result means the eval set
    overlaps that model's training data, so its scores measure memorization, not skill.
    """
    if not eval_month:
        return []
    return [
        Leak(model=name, month=month)
        for name, month in sorted(fit_months.items())
        if month == eval_month
    ]


def leakage_line(leaks: Sequence[Leak], eval_month: Optional[str]) -> str:
    """A one-line summary of the overlap, for stderr (and the ``--strict`` refusal)."""
    models = ", ".join(f"`{lk.model}`" for lk in leaks)
    return (
        f"LEAKAGE: eval month {eval_month} is the training month of {models} — "
        "these scores measure memorization, not held-out skill."
    )


def format_leakage_warning(leaks: Sequence[Leak], eval_month: Optional[str]) -> str:
    """Render a loud Markdown blockquote prepended to the report when leakage is found.

    Returns the empty string when there is nothing to warn about, so the caller can
    unconditionally prepend it.
    """
    if not leaks:
        return ""
    models = ", ".join(f"`{lk.model}`" for lk in leaks)
    return (
        f"> ⚠️ **LEAKAGE — NOT HELD-OUT EVIDENCE.** The eval dataset's source month "
        f"(`{eval_month}`) is the very month {models} was trained on, so its apparent "
        "edge here is memorization, not held-out skill — the **PASS** below cannot be "
        "trusted as proof of the thesis. Re-run on a *different* month (the committed "
        "evidence uses `2013-01`; `wdl-a` was fit on `2016-05`), or pass `--strict` to "
        "refuse the run outright.\n"
    )
