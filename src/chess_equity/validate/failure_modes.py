"""Slice the validation gate by the two named 0003 failure modes (task 0111).

Objective 0003 names two ways the rating-blind centipawn baseline mischaracterises
practical chances, curated as annotated FENs in ``baseline/failure_modes.json``:

- **absurd-refutation** — the engine reads a side *winning* on a line almost no
  human finds (e.g. the Saavedra underpromotion), so the baseline inflates White
  while the practical result is near a draw.
- **hard-0.00** (the json's ``dead-draw-hard``) — the engine reads ``0.00`` so the
  baseline says 50/50, yet the draw hinges on technique weaker players miss, so the
  real result is asymmetric.

The gate's other slicers (rating / high_rating / phase / clock) never report
equity-vs-baseline *on these modes*, which is the most direct proof of the headline
claim. This module turns the curated set into a row :data:`Predictor`-compatible
slicer: a dataset row is tagged with a failure mode when its ``cp_eval`` sits within
``CP_WINDOW`` of that mode's curated engine eval — the **same ±75cp matching** the
calibration code already uses to measure these classes (see
:func:`chess_equity.validate.calibration.measure_position_classes`). Rows near no
anchor get ``"none"``. Registered in :data:`chess_equity.validate.harness.SLICERS`,
so the validation report grows a per-mode baseline-vs-model section for free.

Pure once loaded: the curated anchors are read lazily and cached, so importing this
module stays free of the json file (a missing file degrades to "everything is
``none``" rather than crashing the gate).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from chess_equity.data.schema import PositionRow

# The ±centipawn band a dataset row must fall inside to count as "in" a curated class.
# Matches calibration.measure_position_classes's default so the two agree on what a
# row "near" an anchor means.
CP_WINDOW = 75.0

# The "no failure mode" bucket — rows whose cp_eval is near no curated anchor.
NONE = "none"

# Curated json category -> the report label. We surface the two modes under the names
# objective 0003 / task 0111 use ("absurd-refutation", "hard-0.00"); the json's drawn
# class is stored as "dead-draw-hard".
_CATEGORY_LABEL = {
    "absurd-refutation": "absurd-refutation",
    "dead-draw-hard": "hard-0.00",
}

# (engine_cp, label) anchors loaded from baseline/failure_modes.json; None until loaded.
_ANCHORS: Optional[List[Tuple[float, str]]] = None


def _failure_modes_path() -> Path:
    """Repo-root ``baseline/failure_modes.json`` (this file lives at src/chess_equity/validate/)."""
    return Path(__file__).resolve().parents[3] / "baseline" / "failure_modes.json"


def _load_anchors() -> List[Tuple[float, str]]:
    """Read the curated (engine_cp, label) anchors, caching after the first call.

    A missing or malformed file yields an empty anchor list, so the slicer degrades to
    tagging every row ``"none"`` instead of breaking a validation run that doesn't care
    about this slice.
    """
    global _ANCHORS
    if _ANCHORS is not None:
        return _ANCHORS
    anchors: List[Tuple[float, str]] = []
    try:
        raw = json.loads(_failure_modes_path().read_text(encoding="utf-8"))
        for pos in raw.get("positions", []):
            label = _CATEGORY_LABEL.get(pos.get("category"))
            cp = pos.get("engine_cp")
            if label is not None and cp is not None:
                anchors.append((float(cp), label))
    except (OSError, ValueError, TypeError):
        anchors = []
    _ANCHORS = anchors
    return _ANCHORS


def failure_mode(row: PositionRow, *, cp_window: float = CP_WINDOW) -> str:
    """Tag a row with the 0003 failure mode whose curated eval its ``cp_eval`` is nearest.

    Returns the label of the nearest curated anchor within ``cp_window`` centipawns, or
    :data:`NONE` when the row sits near no anchor. ``cp_eval`` is always present (unlike
    ``fen``), so this slice is non-empty on any dataset — the report shows how the
    rating-conditioned model fares exactly in the cp regions the failure modes live in.
    """
    anchors = _load_anchors()
    best_label = NONE
    best_dist = cp_window
    for cp, label in anchors:
        dist = abs(row.cp_eval - cp)
        if dist <= best_dist:
            best_dist = dist
            best_label = label
    return best_label
