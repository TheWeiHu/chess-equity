"""The dataset *source-month* sidecar — a one-line "data stamp" recording which
Lichess month a built dataset was drawn from.

Why this exists: the validation gate must never quietly validate the thesis on the
**same** month the rating-conditioned model (wdl-a) was *fit* on — that is leakage,
and a PASS earned that way is meaningless. The leakage guard (task 0126) compares the
model's recorded fit-month against the eval dataset's month; this sidecar is how the
dataset's month is carried alongside it, so the guard's ``--eval-month`` can default to
the truth instead of relying on the operator to remember and retype it.

The sidecar is a tiny JSON file written *next to* the dataset (``<dataset>.source.json``)
rather than a column inside it, so it works identically for a flat CSV/Parquet file and
a hive-partitioned directory, and so datasets built before this existed simply have no
sidecar (``read_source_month`` returns ``None``) — backward compatible by construction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# A dataset's sidecar is its own path with this suffix appended, e.g.
# ``data/dataset.csv`` -> ``data/dataset.csv.source.json`` and the partitioned
# directory ``data/dataset`` -> ``data/dataset.source.json`` (a sibling). Appending
# (rather than replacing the suffix) keeps it unambiguous and collision-free.
SIDECAR_SUFFIX = ".source.json"

# Lichess months are ``YYYY-MM`` (the dump filename's month token). We validate the
# shape so a typo'd stamp can't silently defeat the leakage guard later.
_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def normalize_month(month: str) -> str:
    """Validate a ``YYYY-MM`` month string and return it stripped.

    Raises :class:`ValueError` on anything that isn't a real ``YYYY-MM`` token, so a
    bad stamp fails loudly at write time rather than reading back as a plausible-looking
    month the leakage guard would trust.
    """
    m = (month or "").strip()
    if not _MONTH_RE.match(m):
        raise ValueError(f"source month must be YYYY-MM (e.g. 2016-05), got {month!r}")
    return m


def sidecar_path(dataset_path: "str | Path") -> Path:
    """The sidecar location for a dataset path (file or partitioned directory)."""
    return Path(str(dataset_path) + SIDECAR_SUFFIX)


def write_source_month(dataset_path: "str | Path", month: str) -> Path:
    """Write the source-month sidecar next to ``dataset_path`` and return its path.

    ``month`` is validated first, so the sidecar never records a malformed month.
    """
    month = normalize_month(month)
    side = sidecar_path(dataset_path)
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps({"source_month": month}) + "\n", encoding="utf-8")
    return side


def read_source_month(dataset_path: "str | Path") -> Optional[str]:
    """Read the source month stamped beside ``dataset_path``, or ``None`` if absent.

    Returns ``None`` (never raises) when there is no sidecar, so callers can treat an
    unstamped legacy dataset as "month unknown" rather than an error. A sidecar that is
    present but malformed is also treated as absent — the stamp is advisory metadata, so
    a corrupt one shouldn't crash a validation run.
    """
    side = sidecar_path(dataset_path)
    if not side.is_file():
        return None
    try:
        payload = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    month = payload.get("source_month") if isinstance(payload, dict) else None
    return str(month) if month else None
