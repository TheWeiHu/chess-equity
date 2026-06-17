"""The tabular schema a Lichess game decomposes into, plus the small pure helpers
that derive each field.

One :class:`PositionRow` per *evaluated* position (Lichess only annotates ~6% of
games with ``[%eval]``; we keep exactly those). Every column here is something a
downstream task needs:

- ``cp_eval`` + ``result`` are the objective signal and the label that task 0009
  asks "does rating-conditioned equity beat plain centipawns at predicting?".
- ``white_elo`` / ``black_elo`` are what makes the model in 0004 *rating-conditioned*.
- ``phase`` / ``time_control`` / ``tc_bucket`` / ``clock_remaining`` let later work
  slice by where the rating-blind bar fails worst (endgames, time pressure).

These helpers are deliberately dependency-free and pure so they are trivially
testable without touching a real PGN dump (see ``tests/test_data_pgn.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

# Mate scores have no centipawn value; Lichess renders them as ``#N``. We map a mate
# in ``n`` to a large centipawn magnitude, nudged by ``n`` so "mate in 1" outranks
# "mate in 8". Clamped well above any real eval so it reads as decisive everywhere.
MATE_CP = 10_000

# Estimated game duration = base + 40 * increment seconds, bucketed Lichess-style.
# (Lichess uses ~40 expected moves to classify a time control.)
_TC_BULLET_MAX = 179
_TC_BLITZ_MAX = 479
_TC_RAPID_MAX = 1499

# Phase heuristic thresholds (see :func:`game_phase`).
_OPENING_MAX_PLY = 20
_ENDGAME_MAX_PIECES = 6  # non-king pieces left on the board

# Width (Elo points) of a rating-partition band. 200 keeps buckets coarse enough that
# each holds plenty of rows yet fine enough to slice calibration by strength (0025).
RATING_BUCKET_WIDTH = 200

# The ordered column list — the contract loaders and the Parquet/CSV writer share.
# ``game_id`` is the game each position came from; it is what lets validation split
# train/test at the *game* level so positions from one game never leak across the
# split (task 0030). Always written going forward; datasets built before it existed
# load it back as ``None`` (the column is keyed, not positional — see ``_coerce_row``).
COLUMNS = (
    "cp_eval",
    "white_elo",
    "black_elo",
    "ply",
    "phase",
    "time_control",
    "tc_bucket",
    "clock_remaining",
    "side_to_move",
    "result",
    "game_id",
)

# The FEN is opt-in: it roughly triples a row's on-disk size, and the
# (cp_eval, ratings)-only models (0003 baseline, 0004 regression) never read it.
# Board-needing models (Maia, 0005) DO need it to be scored in 0009 — see
# :func:`chess_equity.validate.harness.model_predictor`. Kept as a separate, appended
# column so datasets built without it load unchanged (backward compatible).
FEN_COLUMN = "fen"


def columns(*, include_fen: bool = False) -> tuple:
    """The dataset's column order, with the optional ``fen`` column appended last."""
    return COLUMNS + (FEN_COLUMN,) if include_fen else COLUMNS


@dataclass(frozen=True)
class PositionRow:
    """One evaluated position, flattened for tabular storage.

    ``cp_eval`` is always from White's POV (matching Lichess's ``[%eval]`` and our
    bar). ``result`` is the game's final outcome, also White's POV, in {1.0, 0.5,
    0.0} — the prediction label. ``clock_remaining`` is seconds for the side to
    move, or ``None`` when the game carries no ``[%clk]`` tags.
    """

    cp_eval: float
    white_elo: int
    black_elo: int
    ply: int
    phase: str
    time_control: str
    tc_bucket: str
    clock_remaining: Optional[float]
    side_to_move: str
    result: float
    # The id of the game this position came from (the Lichess game slug). Lets the
    # validation split partition whole games into train/test so no game's positions
    # leak across the split (task 0030). ``None`` for datasets built before it existed.
    game_id: Optional[str] = None
    # The position itself, White-POV FEN. Optional: ``None`` unless the dataset was
    # built with ``include_fen`` (it is what lets board models be scored in 0009).
    fen: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def mate_to_cp(mate_in: int) -> float:
    """Clamp a mate score (``#mate_in``, White POV) to a large signed centipawn value.

    ``mate_in`` is positive when White is mating, negative when Black is. Shorter
    mates get a slightly larger magnitude so ordering is preserved.
    """
    if mate_in == 0:
        # ``#0`` means the side to move is checkmated *now*; treat as a delivered mate
        # against them. Sign is filled in by the caller from board context.
        return float(MATE_CP)
    sign = 1 if mate_in > 0 else -1
    return float(sign * (MATE_CP - min(abs(mate_in), MATE_CP - 1)))


def parse_eval(raw: str) -> Optional[float]:
    """Parse a Lichess ``[%eval ...]`` payload into White-POV centipawns.

    Accepts a pawn-unit float (``"2.35"`` -> 235.0) or a mate (``"#-4"`` -> clamped).
    Returns ``None`` for anything unparseable.
    """
    raw = raw.strip()
    if not raw:
        return None
    if raw[0] == "#":
        try:
            return mate_to_cp(int(raw[1:]))
        except ValueError:
            return None
    try:
        return round(float(raw) * 100.0, 1)
    except ValueError:
        return None


def parse_clock(raw: str) -> Optional[float]:
    """Parse a ``[%clk H:MM:SS]`` payload into seconds remaining. ``None`` if invalid."""
    parts = raw.strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    seconds = 0.0
    for n in nums:
        seconds = seconds * 60.0 + n
    return seconds


def tc_seconds(time_control: str) -> Optional[int]:
    """Estimated game length in seconds from a ``base+increment`` time control.

    ``"600+5"`` -> 600 + 40*5 = 800. ``"-"`` (correspondence) -> ``None``.
    """
    time_control = time_control.strip()
    if not time_control or time_control == "-":
        return None
    base, _, inc = time_control.partition("+")
    try:
        base_s = int(base)
        inc_s = int(inc) if inc else 0
    except ValueError:
        return None
    return base_s + 40 * inc_s


def tc_bucket(time_control: str) -> str:
    """Bucket a time control into bullet/blitz/rapid/classical/correspondence."""
    est = tc_seconds(time_control)
    if est is None:
        return "correspondence"
    if est <= _TC_BULLET_MAX:
        return "bullet"
    if est <= _TC_BLITZ_MAX:
        return "blitz"
    if est <= _TC_RAPID_MAX:
        return "rapid"
    return "classical"


def game_phase(ply: int, non_king_pieces: int) -> str:
    """Coarse opening/middlegame/endgame label.

    A heuristic, not a definition: the first ``_OPENING_MAX_PLY`` plies are
    "opening"; once few pieces remain it is "endgame"; otherwise "middlegame".
    Material-count based so it does not need engine analysis.
    """
    if ply <= _OPENING_MAX_PLY:
        return "opening"
    if non_king_pieces <= _ENDGAME_MAX_PIECES:
        return "endgame"
    return "middlegame"


def rating_bucket(white_elo: int, black_elo: int, width: int = RATING_BUCKET_WIDTH) -> str:
    """Partition label for a game's rating band: the floored mean of both ratings.

    Lichess pairs similar ratings, so the mean is a faithful single label. ``1690`` ->
    ``"1600"`` (the band ``[1600, 1800)``) at the default 200-wide bands. The label is
    the band's lower bound as a string, so it sorts and reads cleanly in a hive path.
    """
    mean = (white_elo + black_elo) / 2.0
    return str(int(mean // width) * width)
