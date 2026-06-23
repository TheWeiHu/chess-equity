"""Failure-mode slicer — score the gate ON the two named failure modes (task 0111).

Objective 0003 names two failure modes the thesis must fix: drawn positions the engine
reads ~0.00 that aren't 50/50 in practice (``dead-draw-hard``), and engine-decisive
positions whose win hinges on a move club players miss (``absurd-refutation``). Slicing
the validation gate ON those modes is the most *direct* proof of the headline claim — the
per-rating/phase slices only get at it obliquely — so this module turns the curated
``baseline/failure_modes.json`` anchors into a :data:`failure_mode` slicer for
``harness.SLICERS``.

A row is tagged with a mode when its ``cp_eval`` lands within the **same ±75cp window the
calibration code uses** (:func:`chess_equity.validate.calibration.measure_position_classes`,
``cp_window=75.0``) around any curated anchor of that mode. The match is taken
**symmetrically** (``±anchor``) so a Black-side absurd refutation (``cp≈-1000``) counts as
the same mode as its White-side mirror — the failure mode is colour-symmetric. Rows that
match no anchor fall in the :data:`UNTAGGED` bucket. Registering the slicer in
``harness.SLICERS`` makes the validation report grow a ``## By failure_mode`` section and
head-to-head deltas automatically.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Dict, FrozenSet

from chess_equity.data.schema import PositionRow

# Same window the calibration class-measurement uses (validate/calibration.py): a row is
# "on" an anchor when |cp_eval - engine_cp| <= 75cp. Keep these in lockstep.
CP_WINDOW = 75.0

# Bucket label for rows that don't fall on any curated failure-mode anchor.
UNTAGGED = "none"

# baseline/failure_modes.json, resolved repo-root-relative (parents: [0]=validate,
# [1]=chess_equity, [2]=src, [3]=repo root). `baseline/` is a repo asset, matching the
# overlay path convention in cli.py — it is not packaged.
_FAILURE_MODES_JSON = (
    Path(__file__).resolve().parents[3] / "baseline" / "failure_modes.json"
)


@functools.lru_cache(maxsize=1)
def _anchors() -> Dict[str, FrozenSet[float]]:
    """``{category -> {engine_cp anchors}}`` from the curated JSON, loaded once.

    The slicer keys off the curated ``category`` (``dead-draw-hard`` / ``absurd-refutation``)
    and the distinct ``engine_cp`` values within it, so the modes stay tied to the data — a
    drift-guard test asserts these match the committed JSON.
    """
    data = json.loads(_FAILURE_MODES_JSON.read_text())
    anchors: Dict[str, set] = {}
    for pos in data["positions"]:
        anchors.setdefault(pos["category"], set()).add(float(pos["engine_cp"]))
    return {cat: frozenset(cps) for cat, cps in anchors.items()}


def failure_mode(row: PositionRow) -> str:
    """Tag ``row`` with the curated failure mode its ``cp_eval`` falls on, else ``"none"``.

    Returns the ``category`` of the first anchor (over sorted modes for determinism) whose
    window the row's White-POV ``cp_eval`` lands in — matched symmetrically (``±anchor``) so
    the colour-mirror of a mode counts too. The committed anchors don't overlap (0 vs ±1000),
    so the first match is unambiguous in practice.
    """
    cp = row.cp_eval
    for category in sorted(_anchors()):
        for anchor in _anchors()[category]:
            if abs(abs(cp) - abs(anchor)) <= CP_WINDOW:
                return category
    return UNTAGGED
