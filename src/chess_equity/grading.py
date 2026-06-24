"""Move grading by Δequity — the "moves can be good" reframe (task 0008).

The classic centipawn-loss grade can only ever be ``<= 0``: the best you can do is
match perfect play, and any deviation loses centipawns. This module grades a move by
the change in the *mover's equity*, benchmarked against what a player of the mover's
rating was expected to do — so a move stronger than the rating-typical mix scores
**positive**. That is the whole pitch (see :doc:`concept-equity-bar`).

For a played move ``m`` in position ``p`` (mover rated ``r`` vs opponent ``opp``):

- ``equity_after``   = equity of ``p·m`` from the mover's POV.
- ``expected_equity``= ``Σ_move P(move | r) · equity(p·move)`` — the equity of the
  *rating-typical* move mix (``P`` from a :class:`~chess_equity.adapters.HumanPolicy`,
  i.e. Maia in task 0005; a uniform placeholder until then).
- ``equity_best``    = equity of the equity-maximizing legal move.

and the two grades:

- **``grade_peer = equity_after − expected_equity``** — the headline. Positive ⇒ you
  beat your rating peers (a genuinely good move).
- ``grade_best = equity_after − equity_best`` (``<= 0``) — the classic "how much did
  you leave on the table" view, on the equity scale.

A move can *lose centipawns yet gain equity* — a sound trap a rating-peer opponent is
likely to walk into. ``cp_loss`` is reported alongside so that case is visible.

Everything here depends only on the :class:`~chess_equity.adapters.EquityModel` /
:class:`~chess_equity.adapters.HumanPolicy` contracts, so the real Maia models drop
in unchanged. With the placeholder material model the *machinery* is exercised; the
flagship trap demo needs Maia (0005) on real data — but the synthetic test
``test_grading`` shows the machinery surfaces the cp-loss-but-equity-gain case.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import chess
import chess.pgn

from chess_equity.adapters import EquityModel, HumanPolicy

# Headline Δequity grade bands, in equity percentage points, peer-relative (mover POV).
# Positive = you beat the rating-typical mix. These are the *base* bands at ~2000; they
# widen for lower ratings, where the peer move-mix is noisier (see :func:`scaled_bands`).
BASE_BANDS = [
    (10.0, "brilliant"),
    (3.0, "good"),
    (-3.0, "ok"),
    (-8.0, "inaccuracy"),
    (-15.0, "mistake"),
]


# Game phases for the per-phase accuracy breakdown (task 0220). Order is opening →
# middlegame → endgame; every breakdown reports a bucket for each so the schema is stable.
PHASES = ["opening", "middlegame", "endgame"]

# Endgame threshold: when both sides combined have this few non-king, non-pawn pieces
# (knights/bishops/rooks/queens) left, the position is materially an endgame regardless
# of move number. The standard starting position has 14 such pieces.
_ENDGAME_MINOR_MAJOR_MAX = 6
# Opening cutoff: the development phase is the first ~10 full moves while material is high.
_OPENING_FULLMOVE_MAX = 10

# Non-king, non-pawn piece types that count toward the endgame material test.
_MINOR_MAJOR = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def position_phase(board: chess.Board) -> str:
    """Classify a position as ``opening`` / ``middlegame`` / ``endgame`` (simple heuristic).

    A deliberately-simple, documented ply+material rule (the task asks for exactly this,
    not engine-grade phase detection):

    - **endgame** when both sides together have ``<= 6`` non-king, non-pawn pieces left
      (queens/rooks/minors) — materially an endgame however early it happens;
    - else **opening** for the first 10 full moves (the development phase, material high);
    - else **middlegame**.

    The endgame test wins over the opening test, so an early massacre is still an endgame.
    """
    minor_major = sum(
        len(board.pieces(piece_type, color))
        for piece_type in _MINOR_MAJOR
        for color in (chess.WHITE, chess.BLACK)
    )
    if minor_major <= _ENDGAME_MINOR_MAJOR_MAX:
        return "endgame"
    if board.fullmove_number <= _OPENING_FULLMOVE_MAX:
        return "opening"
    return "middlegame"


def scaled_bands(elo: int) -> List[tuple]:
    """Rating-aware grade bands: wider tolerance at lower ratings.

    A 1200's rating-typical mix is noisier than a 2400's, so the same Δequity means
    less. We widen the bands below ~2000 (scale 1.0 at/above 2000, up to ~1.6 at 800)
    and leave strong play on the tight base bands. Calibration against real Maia
    spreads (task 0005 / validation 0009) can replace this heuristic.
    """
    scale = max(1.0, 1.0 + (2000 - elo) / 2000.0)
    return [(threshold * scale, label) for threshold, label in BASE_BANDS]


def grade_label(grade_peer: float, elo: int) -> str:
    """Label a peer-relative Δequity for a mover rated ``elo``."""
    for threshold, label in scaled_bands(elo):
        if grade_peer >= threshold:
            return label
    return "blunder"


class UniformPolicy(HumanPolicy):
    """Placeholder peer model: every legal move equally likely.

    Stands in for Maia-2 (task 0005) so grading runs end-to-end. With a uniform peer
    mix, ``expected_equity`` is the *average* legal-move equity, so any above-average
    move grades positive — enough to exercise and demonstrate the reframe. Maia
    replaces this behind the same :class:`HumanPolicy` interface with no other change.
    """

    def move_probs(self, fen: str, elo: int) -> Dict[str, float]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        if not moves:
            return {}
        p = 1.0 / len(moves)
        return {m.uci(): p for m in moves}


@dataclass(frozen=True)
class MoveGrade:
    """The grade of one played move, on the equity scale (mover POV)."""

    ply: int
    san: str
    uci: str
    mover_white: bool
    mover_elo: int
    phase: str  # game phase the move was played in: opening/middlegame/endgame (task 0220)
    equity_after: float
    expected_equity: float
    equity_best: float
    grade_peer: float  # headline: equity_after - expected_equity (positive = beat peers)
    grade_best: float  # equity_after - equity_best (<= 0)
    label: str
    best_uci: str
    cp_loss: Optional[float]  # classic centipawn loss (mover POV, >= 0), if available

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# Eighth-block ramp (U+2581..U+2588), low → high, for the equity sparkline.
SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def white_pov_equity(grade: MoveGrade) -> float:
    """White-POV equity (0..100) after ``grade``'s move, flipping Black-mover POV."""
    return grade.equity_after if grade.mover_white else 100.0 - grade.equity_after


def equity_sparkline(grades: List[MoveGrade]) -> str:
    """A one-line eighth-block sparkline of the per-ply White-POV equity series.

    One block per graded ply (``len(result) == len(grades)``), each cell mapping the
    move's White-POV equity over the full bar range [0, 100]% to one of the eight
    ramp glyphs. Absolute scaling (not per-series min/max) so a genuinely winning
    trajectory climbs toward █ and a flat game sits mid-height — the swing *shape* a
    caster reads at a glance. Pure over already-computed grades; no model calls.
    """
    cells = []
    n = len(SPARK_BLOCKS)
    for g in grades:
        frac = max(0.0, min(1.0, white_pov_equity(g) / 100.0))
        idx = min(n - 1, int(frac * n))
        cells.append(SPARK_BLOCKS[idx])
    return "".join(cells)


def equity_trajectory_svg(
    grades: List[MoveGrade],
    *,
    width: int = 480,
    height: int = 140,
) -> str:
    """A self-contained SVG win-equity trajectory chart for a graded game.

    The graphical sibling of :func:`equity_sparkline`: per-ply White-POV win-equity
    drawn as a filled area + polyline over the game, with a 50% midline so a caster
    sees at a glance who's ahead and where the swings are. One polyline point per
    graded ply (``len(points) == len(grades)``), x left→right over the game, y mapped
    over the full [0, 100]% bar range (top = White winning). Renderable as a static
    asset or an OBS browser-source widget. Pure string generation over already-computed
    grades: no torch, no data, no XML library, no model calls.
    """
    if not grades:
        raise ValueError("equity_trajectory_svg needs at least one graded ply")

    pad = 12
    plot_x = pad
    plot_y = 26
    plot_w = width - 2 * pad
    plot_h = height - plot_y - 22  # leave room for the caption strip below

    def _x(i: int) -> float:
        # Spread points across the plot; a single ply sits at the left edge.
        if len(grades) == 1:
            return round(plot_x, 2)
        return round(plot_x + plot_w * i / (len(grades) - 1), 2)

    def _y(equity_white: float) -> float:
        frac = max(0.0, min(1.0, equity_white / 100.0))
        return round(plot_y + plot_h * (1.0 - frac), 2)  # high equity → top

    pts = [(_x(i), _y(white_pov_equity(g))) for i, g in enumerate(grades)]
    poly = " ".join(f"{x},{y}" for x, y in pts)
    # Area = the polyline closed down to the baseline and back.
    area = f"{poly} {pts[-1][0]},{plot_y + plot_h} {pts[0][0]},{plot_y + plot_h}"
    mid_y = _y(50.0)
    final_pct = white_pov_equity(grades[-1])
    favour = "White" if final_pct >= 50.0 else "Black"

    aria = (
        f"Win-equity trajectory over {len(grades)} plies; "
        f"final White {final_pct:.1f}% ({favour} ahead)"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{aria}">\n'
        f'  <title>{aria}</title>\n'
        f'  <rect width="{width}" height="{height}" fill="#1b1b1b"/>\n'
        f'  <text x="{pad}" y="18" font-family="sans-serif" font-size="13" '
        f'font-weight="bold" fill="#f0f0f0">Win-equity trajectory</text>\n'
        f'  <text x="{width - pad}" y="18" font-family="sans-serif" font-size="12" '
        f'fill="#9a9a9a" text-anchor="end">White {final_pct:.1f}% ({favour})</text>\n'
        f'  <polygon points="{area}" fill="#3a6ea5" fill-opacity="0.35"/>\n'
        f'  <line x1="{plot_x}" y1="{mid_y}" x2="{plot_x + plot_w}" y2="{mid_y}" '
        f'stroke="#888" stroke-width="1" stroke-dasharray="4 3"/>\n'
        f'  <polyline points="{poly}" fill="none" stroke="#7fb2e5" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>\n'
        f'  <text x="{width - pad}" y="{mid_y - 3}" font-family="sans-serif" font-size="9" '
        f'fill="#6a6a6a" text-anchor="end">50%</text>\n'
        f'</svg>\n'
    )


class EquityGrader:
    """Grades moves by Δequity using an :class:`EquityModel` + a peer policy."""

    def __init__(self, model: EquityModel, policy: Optional[HumanPolicy] = None) -> None:
        self.model = model
        self.policy = policy or UniformPolicy()

    def _mover_equity_after(
        self, board: chess.Board, move: chess.Move, mover_white: bool,
        white_elo: int, black_elo: int,
    ) -> tuple:
        """(equity, cp) of the position after ``move``, from the mover's POV."""
        board.push(move)
        try:
            eq = self.model.evaluate(board.fen(), white_elo, black_elo)
        finally:
            board.pop()
        equity = eq.equity_white if mover_white else 100.0 - eq.equity_white
        # eq.cp is side-to-move POV; after the move the opponent is to move, so negate
        # to express the position from the mover's POV (classic centipawn convention).
        cp = None if eq.cp is None else -eq.cp
        return equity, cp

    def grade_move(
        self, fen: str, move: chess.Move, white_elo: int, black_elo: int, *, ply: int = 0
    ) -> MoveGrade:
        """Grade a single ``move`` played in ``fen``."""
        board = chess.Board(fen)
        if move not in board.legal_moves:
            raise ValueError(f"{move.uci()} is not legal in {fen}")
        mover_white = board.turn == chess.WHITE
        mover_elo = white_elo if mover_white else black_elo
        san = board.san(move)
        phase = position_phase(board)  # the position the move is played in (before the push)

        # Equity (mover POV) and mover-POV cp of every legal move.
        equities: Dict[str, float] = {}
        cps: Dict[str, Optional[float]] = {}
        for legal in board.legal_moves:
            eq, cp = self._mover_equity_after(
                board, legal, mover_white, white_elo, black_elo
            )
            equities[legal.uci()] = eq
            cps[legal.uci()] = cp

        played = move.uci()
        equity_after = equities[played]
        best_uci = max(equities, key=lambda u: equities[u])
        equity_best = equities[best_uci]

        # Expected equity over the rating-typical move mix; renormalize the policy onto
        # the legal moves we actually evaluated (a policy may omit zero-prob moves).
        probs = self.policy.move_probs(fen, mover_elo)
        mass = sum(probs.get(u, 0.0) for u in equities)
        if mass > 0:
            expected_equity = sum(
                probs.get(u, 0.0) * equities[u] for u in equities
            ) / mass
        else:
            expected_equity = sum(equities.values()) / len(equities)

        # Classic centipawn loss (mover POV), for the cp-vs-equity contrast.
        cp_loss = None
        if all(v is not None for v in cps.values()):
            cp_best = max(cps.values())  # type: ignore[type-var]
            cp_loss = float(cp_best - cps[played])  # type: ignore[operator]

        grade_peer = equity_after - expected_equity
        return MoveGrade(
            ply=ply,
            san=san,
            uci=played,
            mover_white=mover_white,
            mover_elo=mover_elo,
            phase=phase,
            equity_after=equity_after,
            expected_equity=expected_equity,
            equity_best=equity_best,
            grade_peer=grade_peer,
            grade_best=equity_after - equity_best,
            label=grade_label(grade_peer, mover_elo),
            best_uci=best_uci,
            cp_loss=cp_loss,
        )

    def grade_game(self, game: chess.pgn.Game, white_elo: int, black_elo: int) -> List[MoveGrade]:
        """Grade every move of a parsed PGN game in order."""
        board = game.board()
        grades: List[MoveGrade] = []
        for ply, move in enumerate(game.mainline_moves(), start=1):
            grades.append(self.grade_move(board.fen(), move, white_elo, black_elo, ply=ply))
            board.push(move)
        return grades


# --------------------------------------------------------------------------- #
# Per-side scoreline — a caster-facing "accuracy"-style summary (task 0200)
# --------------------------------------------------------------------------- #

# All grade labels, best → worst. Drives the column order in the scoreline table
# and guarantees every side reports a count for every label (zero if unused).
GRADE_LABELS = [label for _, label in BASE_BANDS] + ["blunder"]


@dataclass(frozen=True)
class SideScoreline:
    """One side's aggregate move quality over a graded game (derived, no model calls)."""

    white: bool
    n_moves: int
    label_counts: Dict[str, int]  # keyed by every GRADE_LABELS entry (0 if unused)
    mean_peer: float  # mean grade_peer (signed; positive = beat rating peers overall)
    worst: Optional[MoveGrade]  # the move with the minimum grade_peer (biggest drop)

    def to_dict(self) -> Dict[str, object]:
        return {
            "white": self.white,
            "n_moves": self.n_moves,
            "label_counts": dict(self.label_counts),
            "mean_peer": self.mean_peer,
            "worst": None if self.worst is None else self.worst.to_dict(),
        }


@dataclass(frozen=True)
class GameScoreline:
    """Per-side scoreline for a whole game (White vs Black)."""

    white: SideScoreline
    black: SideScoreline

    def to_dict(self) -> Dict[str, object]:
        return {"white": self.white.to_dict(), "black": self.black.to_dict()}


def _side_scoreline(grades: List[MoveGrade], *, white: bool) -> SideScoreline:
    moves = [g for g in grades if g.mover_white == white]
    counts = {label: 0 for label in GRADE_LABELS}
    for g in moves:
        # Unknown labels (shouldn't happen) still count, so the sum invariant holds.
        counts[g.label] = counts.get(g.label, 0) + 1
    mean_peer = sum(g.grade_peer for g in moves) / len(moves) if moves else 0.0
    worst = min(moves, key=lambda g: g.grade_peer) if moves else None
    return SideScoreline(
        white=white, n_moves=len(moves), label_counts=counts, mean_peer=mean_peer, worst=worst
    )


def scoreline(grades: List[MoveGrade]) -> GameScoreline:
    """Aggregate a :class:`MoveGrade` list into a per-side scoreline.

    Pure reduction over the grades the game already produced — no new model calls.
    Each side's ``label_counts`` sum to its move count and ``worst`` is the move with
    the minimum (most negative) ``grade_peer``.
    """
    return GameScoreline(
        white=_side_scoreline(grades, white=True),
        black=_side_scoreline(grades, white=False),
    )


# --------------------------------------------------------------------------- #
# Round leaderboard — accuracy ranking across a multi-game broadcast (task 0207)
# --------------------------------------------------------------------------- #

# Labels that count as an "accurate" move (ok-or-better). The accuracy % is the share
# of a player's moves in this set, pooled across every board they played this round.
ACCURATE_LABELS = {"brilliant", "good", "ok"}


def _accuracy(grades: List[MoveGrade]) -> float:
    """Share of ``grades`` graded ok-or-better (:data:`ACCURATE_LABELS`), 0..100."""
    if not grades:
        return 0.0
    accurate = sum(1 for g in grades if g.label in ACCURATE_LABELS)
    return 100.0 * accurate / len(grades)


def phase_breakdown(grades: List[MoveGrade]) -> Dict[str, Dict[str, object]]:
    """Split a player's moves by game phase and score each phase (task 0220).

    Returns a stable dict keyed by every :data:`PHASES` entry (a phase with no moves
    reports ``n_moves=0``, ``accuracy``/``avg_delta`` ``0.0``), where each value carries
    that phase's move count, accuracy % (same ok-or-better definition as the overall
    leaderboard), and mean Δpeer. Pure reduction over already-computed grades — no model
    calls. The per-phase ``n_moves`` sum to ``len(grades)``.
    """
    by_phase: Dict[str, List[MoveGrade]] = {phase: [] for phase in PHASES}
    for g in grades:
        by_phase.setdefault(g.phase, []).append(g)
    out: Dict[str, Dict[str, object]] = {}
    for phase in PHASES:
        gs = by_phase[phase]
        mean_peer = sum(g.grade_peer for g in gs) / len(gs) if gs else 0.0
        out[phase] = {
            "n_moves": len(gs),
            "accuracy": round(_accuracy(gs), 1),
            "avg_delta": round(mean_peer, 2),
        }
    return out


@dataclass(frozen=True)
class PlayerScore:
    """One player's pooled move quality across every game they played this round."""

    name: str
    n_moves: int
    label_counts: Dict[str, int]  # keyed by every GRADE_LABELS entry (0 if unused)
    accuracy: float  # % of moves graded ok-or-better (ACCURATE_LABELS), 0..100
    blunders: int  # count of "blunder" moves (== label_counts["blunder"])
    mistakes: int  # count of "mistake" moves (== label_counts["mistake"])
    mean_peer: float  # mean grade_peer (signed Δequity; positive = beat rating peers)
    rating: int  # the player's rating this round (modal mover_elo of their pooled moves)
    worst: Optional[MoveGrade]  # the move with the minimum grade_peer (biggest drop)
    # Per-phase accuracy/Δpeer split (task 0220), keyed by every PHASES entry. Each value:
    # {"n_moves", "accuracy", "avg_delta"}; the per-phase n_moves sum to n_moves.
    phases: Dict[str, Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "n_moves": self.n_moves,
            "label_counts": dict(self.label_counts),
            "accuracy": self.accuracy,
            "blunders": self.blunders,
            "mistakes": self.mistakes,
            "mean_peer": self.mean_peer,
            "rating": self.rating,
            "worst": None if self.worst is None else self.worst.to_dict(),
            "phases": {phase: dict(stat) for phase, stat in self.phases.items()},
        }


def _player_score(name: str, grades: List[MoveGrade]) -> PlayerScore:
    counts = {label: 0 for label in GRADE_LABELS}
    for g in grades:
        counts[g.label] = counts.get(g.label, 0) + 1
    n = len(grades)
    accurate = sum(counts.get(label, 0) for label in ACCURATE_LABELS)
    accuracy = 100.0 * accurate / n if n else 0.0
    mean_peer = sum(g.grade_peer for g in grades) / n if n else 0.0
    worst = min(grades, key=lambda g: g.grade_peer) if grades else None
    return PlayerScore(
        name=name,
        n_moves=n,
        label_counts=counts,
        accuracy=accuracy,
        blunders=counts.get("blunder", 0),
        mistakes=counts.get("mistake", 0),
        mean_peer=mean_peer,
        rating=_modal_rating(grades),
        worst=worst,
        phases=phase_breakdown(grades),
    )


def _modal_rating(grades: List[MoveGrade]) -> int:
    """The player's rating for the round: the most common ``mover_elo`` of their moves.

    A player can in principle appear with differing ratings across boards (per-game
    headers), so we take the modal rating; ties break to the highest. ``0`` if empty.
    """
    if not grades:
        return 0
    counts: Dict[int, int] = {}
    for g in grades:
        counts[g.mover_elo] = counts.get(g.mover_elo, 0) + 1
    # max by (frequency, rating) → most common, ties to the higher rating; deterministic.
    return max(counts, key=lambda elo: (counts[elo], elo))


# Leaderboard sort modes (task 0234): which metric is the PRIMARY rank key. Every mode
# keeps the same deterministic tail (the remaining metrics, then name) so ties always
# break identically. 'accuracy' is the historical default and unchanged.
LEADERBOARD_SORTS = ("accuracy", "lead", "blunders")


def _leaderboard_rank_key(s: PlayerScore, sort: str = "accuracy") -> tuple:
    # accuracy desc / lead = mean Δpeer desc / blunders = fewest first; the unused metrics
    # follow as deterministic tie-breaks, always ending in name.
    if sort == "lead":
        return (-s.mean_peer, -s.accuracy, s.blunders, s.name)
    if sort == "blunders":
        return (s.blunders, -s.accuracy, -s.mean_peer, s.name)
    return (-s.accuracy, -s.mean_peer, s.blunders, s.name)


def round_leaderboard(
    games: List[tuple],
    sort: str = "accuracy",
) -> List[PlayerScore]:
    """Rank every player across a round by pooled move quality.

    ``games`` is a list of ``(white_name, black_name, grades)`` tuples — one per board,
    where ``grades`` is that game's :class:`MoveGrade` list. A player's moves are pooled
    across *every* board they appear on (White on one, Black on another all count), keyed
    by player name, then scored with the same per-move grade math the single-game
    :func:`scoreline` uses. Pure reduction — no model calls.

    ``sort`` picks the PRIMARY rank key (one of :data:`LEADERBOARD_SORTS`): ``accuracy``
    (default) ranks by accuracy %; ``lead`` by mean Δpeer desc (rewarding beating peers
    over avoiding mistakes); ``blunders`` fewest-first. The remaining metrics, then name,
    always follow as deterministic tie-breaks.
    """
    by_player: Dict[str, List[MoveGrade]] = {}
    for white_name, black_name, grades in games:
        for g in grades:
            name = white_name if g.mover_white else black_name
            by_player.setdefault(name, []).append(g)
    scores = [_player_score(name, gs) for name, gs in by_player.items()]
    scores.sort(key=lambda s: _leaderboard_rank_key(s, sort))
    return scores


def _worst_cell(worst: Optional[MoveGrade]) -> str:
    """Compact worst-move cell for the leaderboard: ``SAN Δpeer`` (e.g. ``Qd7 -3.2``),
    or ``-`` when the player has no moves. Last column, so longer SAN just overflows
    its pad rather than breaking the numeric columns' alignment."""
    if worst is None:
        return "-"
    return f"{worst.san} {worst.grade_peer:+.1f}"


def render_leaderboard(scores: List[PlayerScore]) -> List[str]:
    """A ranked accuracy table (one row per player), as text lines.

    The trailing ``worst`` column shows each player's biggest single drop as
    ``SAN Δpeer`` (or ``-`` for an empty player); it's display-only and does not change
    the CSV/JSON export schema.
    """
    rows: List[str] = []
    header = (
        f"{'#':>2}  {'player':<14}{'acc%':>6}{'moves':>7}"
        f"{'blun':>6}{'mist':>6}{'meanΔ':>8}  {'worst':<16}"
    )
    rows.append(header)
    rows.append("-" * len(header))
    for i, s in enumerate(scores, start=1):
        rows.append(
            f"{i:>2}  {s.name:<14}{s.accuracy:>6.1f}{s.n_moves:>7}"
            f"{s.blunders:>6}{s.mistakes:>6}{s.mean_peer:>+8.1f}  {_worst_cell(s.worst):<16}"
        )
    return rows


# Machine-readable leaderboard export (task 0214) — feeds broadcast lower-third
# graphics. A stable, flat schema (one row per player) distinct from PlayerScore.to_dict():
# `player`/`avg_delta` are the broadcast-facing names, plus `rating` and a 1-based `rank`.
LEADERBOARD_COLUMNS = ["rank", "player", "rating", "n_moves", "accuracy", "avg_delta"]

# Flat CSV columns for the per-phase breakdown (task 0220): two per phase, appended after
# the base columns so the existing schema is a stable prefix. `{phase}_acc` is that phase's
# accuracy %, `{phase}_moves` its move count (the move counts sum to `n_moves`).
PHASE_CSV_COLUMNS = [f"{phase}_{stat}" for phase in PHASES for stat in ("acc", "moves")]
LEADERBOARD_CSV_COLUMNS = LEADERBOARD_COLUMNS + PHASE_CSV_COLUMNS


def leaderboard_export_rows(scores: List[PlayerScore]) -> List[Dict[str, object]]:
    """Flat, stable per-player rows for JSON/CSV export, ranked in list order.

    ``scores`` is the already-ranked output of :func:`round_leaderboard`; ``rank`` is its
    1-based position. Accuracy is rounded to 1 decimal and ``avg_delta`` (== ``mean_peer``)
    to 2, so the export is stable and lower-third-friendly. ``phases`` is the nested
    per-phase breakdown (task 0220): ``{phase: {n_moves, accuracy, avg_delta}}`` for every
    :data:`PHASES` entry. Pure projection — no model calls.
    """
    return [
        {
            "rank": i,
            "player": s.name,
            "rating": s.rating,
            "n_moves": s.n_moves,
            "accuracy": round(s.accuracy, 1),
            "avg_delta": round(s.mean_peer, 2),
            "phases": {phase: dict(stat) for phase, stat in s.phases.items()},
        }
        for i, s in enumerate(scores, start=1)
    ]


def render_leaderboard_csv(scores: List[PlayerScore]) -> str:
    """The leaderboard export as CSV text (header + one row per player, trailing newline).

    The nested ``phases`` breakdown is flattened into the :data:`PHASE_CSV_COLUMNS`
    (``opening_acc``/``opening_moves``/…) appended after the base columns, so CSV stays a
    flat, machine-readable table while still carrying the per-phase accuracy split.
    """
    import csv
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=LEADERBOARD_CSV_COLUMNS)
    writer.writeheader()
    rows = leaderboard_export_rows(scores)
    for row, score in zip(rows, scores):
        flat = {col: row[col] for col in LEADERBOARD_COLUMNS}
        for phase in PHASES:
            stat = score.phases[phase]
            flat[f"{phase}_acc"] = stat["accuracy"]
            flat[f"{phase}_moves"] = stat["n_moves"]
        writer.writerow(flat)
    return buf.getvalue()


def render_scoreline(line: GameScoreline) -> List[str]:
    """A White-vs-Black grade-label table + mean Δpeer + worst move, as text lines."""
    w, b = line.white, line.black
    rows: List[str] = []
    rows.append(f"{'move quality':<12}{'White':>8}{'Black':>8}")
    rows.append(f"{'-' * 12}{'-' * 8:>8}{'-' * 8:>8}")
    for label in GRADE_LABELS:
        rows.append(f"{label:<12}{w.label_counts[label]:>8}{b.label_counts[label]:>8}")
    rows.append(f"{'moves':<12}{w.n_moves:>8}{b.n_moves:>8}")
    rows.append(f"{'mean Δpeer':<12}{w.mean_peer:>+8.1f}{b.mean_peer:>+8.1f}")
    for side, sl in (("White", w), ("Black", b)):
        if sl.worst is not None:
            g = sl.worst
            rows.append(
                f"worst ({side}): {g.ply:3d}. {g.san} {g.label} (Δpeer {g.grade_peer:+.1f})"
            )
    return rows
