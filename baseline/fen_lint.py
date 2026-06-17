"""Minimal stdlib FEN structural validator.

Not a legality engine (no move generation) — it catches the typos that matter for
a committed position set: wrong square counts, missing/duplicate kings, adjacent
kings, and pawns on the back ranks. Used by test_failure_modes.py so the curated
FENs can't silently rot. Drop this once python-chess is a project dependency (it
ships a real validator).
"""
from __future__ import annotations

from typing import List, Tuple


class FenError(ValueError):
    pass


def _expand_rank(rank: str) -> List[str]:
    out: List[str] = []
    for ch in rank:
        if ch.isdigit():
            out.extend(["."] * int(ch))
        else:
            out.append(ch)
    return out


def parse_board(fen: str) -> List[List[str]]:
    """Return an 8x8 grid (rank 8 first) of piece chars or '.'. Raises FenError."""
    fields = fen.split()
    if len(fields) < 2:
        raise FenError("FEN needs at least a board and side-to-move field")
    ranks = fields[0].split("/")
    if len(ranks) != 8:
        raise FenError(f"expected 8 ranks, got {len(ranks)}")
    grid = []
    for i, rank in enumerate(ranks):
        cells = _expand_rank(rank)
        if len(cells) != 8:
            raise FenError(f"rank {8 - i} has {len(cells)} squares, not 8: {rank!r}")
        grid.append(cells)
    if fields[1] not in ("w", "b"):
        raise FenError(f"side to move must be w or b, got {fields[1]!r}")
    return grid


def _king_squares(grid: List[List[str]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    white = [(r, c) for r in range(8) for c in range(8) if grid[r][c] == "K"]
    black = [(r, c) for r in range(8) for c in range(8) if grid[r][c] == "k"]
    if len(white) != 1:
        raise FenError(f"need exactly one white king, found {len(white)}")
    if len(black) != 1:
        raise FenError(f"need exactly one black king, found {len(black)}")
    return white[0], black[0]


def validate(fen: str) -> None:
    """Raise FenError if the FEN is structurally bad; return None if it looks sane."""
    grid = parse_board(fen)

    # No pawns on rank 1 or rank 8 (grid row 0 == rank 8, row 7 == rank 1).
    for c in range(8):
        if grid[0][c] in ("P", "p") or grid[7][c] in ("P", "p"):
            raise FenError("pawn on a back rank")

    (wr, wc), (br, bc) = _king_squares(grid)
    if max(abs(wr - br), abs(wc - bc)) <= 1:
        raise FenError("kings are adjacent")

    # Sane piece counts (no more than a plausible maximum per side).
    flat = [cell for row in grid for cell in row]
    if flat.count("P") > 8 or flat.count("p") > 8:
        raise FenError("too many pawns")
    if sum(1 for x in flat if x.isupper()) > 16:
        raise FenError("too many white pieces")
    if sum(1 for x in flat if x.islower()) > 16:
        raise FenError("too many black pieces")
