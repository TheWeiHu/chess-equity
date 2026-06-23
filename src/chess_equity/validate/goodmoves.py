"""'Good moves read as good' — the positive-direction validation (task 0117).

The validation gate (:mod:`chess_equity.validate.harness`) proves the *negative*
half of the thesis: a rating-conditioned equity predictor predicts real outcomes
better than the rating-blind centipawn baseline. This module measures the *positive*
half — the central pitch (see :doc:`concept-equity-bar`): **good moves should read as
GOOD, not just less bad.**

The centipawn bar can only ever go down: against perfect play the best a move does is
hold the eval, so a good move reads as ~0 ("less bad"). The equity reframe gives a good
move genuine upside. To test that on real data we look at *moves*, not positions:

For every consecutive evaluated ply-pair ``(before, after)`` in a game, the engine's
own ``cp_eval`` swing is the ground-truth move quality (from the mover's POV), and each
predictor's equity swing is what the *bar* showed the mover. We then ask, per predictor:

- **direction** — on decisive moves, does the bar move the *right way* (Δ>0 on
  engine-approved moves, Δ<0 on blunders)? This is ``sign_accuracy``. Any monotone-in-cp
  predictor scores ~1.0, so it is a sanity floor, not the discriminator.
- **good moves read as good** — the headline. The mean mover-POV equity Δ on
  engine-approved ("good") moves: the bar should read these as a *positive* gain, not a
  saturated ~0. A rating-aware bar leaves headroom a winning-position centipawn→Win%
  map has already spent, so good moves keep visible upside.
- **magnitude** — Pearson correlation of the equity Δ with the cp swing. Reported for
  transparency, but note it is **biased toward the baseline**: the cp swing *is* the
  baseline's only input, so a Win%-of-cp map correlates with it almost by construction.
  It is here to be read honestly, not as the win condition.

Pure functions over rows + :data:`~chess_equity.validate.harness.Predictor` callables —
no I/O, no numpy — so it lives in the light test path beside the metrics module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from chess_equity.data.schema import PositionRow

# A predictor: a row -> predicted White expected-score in [0, 1] (same contract as the
# gate's harness Predictor; re-declared here so this module doesn't import the harness).
Predictor = Callable[[PositionRow], float]

# Centipawn swings are clamped to ±this before anything looks at them: a mate score
# parses to a huge cp value that would otherwise dominate every mean and correlation,
# and past ~10 pawns the practical win-prob is saturated anyway.
CP_CLAMP = 1000.0

# A move counts as engine-"good" when the mover lost at most this many cp (i.e. played
# at/near best); a "blunder" when it dropped at least BLUNDER_DROP. The gap between them
# is intentionally left unclassified so the two buckets are unambiguous.
GOOD_MAX_LOSS = 10.0
BLUNDER_DROP = 100.0

# Sign-accuracy only scores "decisive" moves — those whose cp swing is at least this
# large — so near-neutral noise (where Δ sign is meaningless) doesn't dilute it.
DECISIVE_CP = 25.0

# The robustness sweep (task 0157): the good/blunder cutoffs above are arbitrary
# defaults, so the Δgood > Δblunder direction must be shown not to hinge on them. We
# re-measure across this grid of good × blunder cutoffs and check the direction holds in
# every cell. (sign-acc depends only on DECISIVE_CP — *not* the good/blunder cutoffs —
# so it is invariant across this grid and reported once, not per cell.)
SWEEP_GOOD_CUTOFFS = (5.0, 10.0, 20.0)
SWEEP_BLUNDER_CUTOFFS = (75.0, 100.0, 150.0)


def _clamp(cp: float) -> float:
    return max(-CP_CLAMP, min(CP_CLAMP, cp))


def iter_move_pairs(
    rows: Sequence[PositionRow],
) -> List[Tuple[PositionRow, PositionRow]]:
    """Consecutive evaluated ply-pairs ``(before, after)`` within each game.

    Rows are grouped by ``game_id`` and ordered by ``ply``; a pair is kept only when the
    two plies are adjacent (``after.ply == before.ply + 1``), so a gap from an
    unevaluated ply never fabricates a "move". Rows with no ``game_id`` can't be ordered
    into a game and are skipped (they can't yield a trustworthy move). The move from
    ``before`` to ``after`` was played by **White iff ``after.ply`` is odd**.
    """
    by_game: Dict[str, List[PositionRow]] = defaultdict(list)
    for r in rows:
        if r.game_id is not None:
            by_game[r.game_id].append(r)
    pairs: List[Tuple[PositionRow, PositionRow]] = []
    for game_rows in by_game.values():
        game_rows.sort(key=lambda r: r.ply)
        for before, after in zip(game_rows, game_rows[1:]):
            if after.ply == before.ply + 1:
                pairs.append((before, after))
    return pairs


def _mover_is_white(after: PositionRow) -> bool:
    """The move landing on ``after`` was White's iff its ply index is odd."""
    return after.ply % 2 == 1


def cp_gain_mover(before: PositionRow, after: PositionRow) -> float:
    """The engine's cp swing across the move, from the *mover's* POV (clamped).

    ``cp_eval`` is White-POV, so the White-POV swing is ``after - before``; the mover's
    own gain flips sign when Black moved. ~0 for an engine-best move, strongly negative
    for a blunder (you can't gain cp against perfect play, only lose less).
    """
    white_swing = _clamp(after.cp_eval) - _clamp(before.cp_eval)
    return white_swing if _mover_is_white(after) else -white_swing


def equity_gain_mover(
    before: PositionRow, after: PositionRow, predictor: Predictor
) -> float:
    """The predictor's equity swing across the move, from the mover's POV, in pp.

    ``predictor`` returns a White expected-score in [0, 1]; the mover-POV gain is the
    White-POV swing (flipped for Black) scaled to percentage points so the numbers read
    on the same scale as the bar (0–100).
    """
    white_swing = predictor(after) - predictor(before)
    mover_swing = white_swing if _mover_is_white(after) else -white_swing
    return mover_swing * 100.0


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Pearson correlation, or ``None`` when undefined (n<2 or a constant series)."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sqrt(sum((x - mx) ** 2 for x in xs))
    syy = sqrt(sum((y - my) ** 2 for y in ys))
    if sxx == 0.0 or syy == 0.0:
        return None
    return sxy / (sxx * syy)


@dataclass(frozen=True)
class GoodMovesReport:
    """One predictor's 'good moves read as good' measurement over a set of move-pairs."""

    name: str
    n_moves: int
    n_good: int
    n_blunder: int
    n_decisive: int
    sign_accuracy: Optional[float]  # over decisive moves; None if none qualify
    mean_delta_good: Optional[float]  # mean mover-POV Δequity (pp) on good moves
    mean_delta_blunder: Optional[float]  # mean mover-POV Δequity (pp) on blunders
    correlation: Optional[float]  # Pearson(Δequity, cp gain); baseline-biased


def measure_good_moves(
    rows: Sequence[PositionRow],
    predictors: Dict[str, Predictor],
    *,
    good_max_loss: float = GOOD_MAX_LOSS,
    blunder_drop: float = BLUNDER_DROP,
    decisive_cp: float = DECISIVE_CP,
) -> List[GoodMovesReport]:
    """Score each predictor's move-level Δequity against the engine cp swing.

    Returns one :class:`GoodMovesReport` per predictor (registry order). The move-pairs
    and their cp ground truth are computed once and shared, so every predictor is scored
    over exactly the same moves. Returns an empty list when no adjacent ply-pairs exist
    (e.g. a one-row-per-game dataset) — the caller skips the section rather than emitting
    an empty one.
    """
    pairs = iter_move_pairs(rows)
    if not pairs:
        return []

    cp_gains = [cp_gain_mover(b, a) for b, a in pairs]
    good_idx = [i for i, g in enumerate(cp_gains) if g >= -good_max_loss]
    blunder_idx = [i for i, g in enumerate(cp_gains) if g <= -blunder_drop]
    decisive_idx = [i for i, g in enumerate(cp_gains) if abs(g) >= decisive_cp]

    reports: List[GoodMovesReport] = []
    for name, predictor in predictors.items():
        eq_gains = [equity_gain_mover(b, a, predictor) for b, a in pairs]

        sign_acc: Optional[float] = None
        if decisive_idx:
            correct = sum(
                1 for i in decisive_idx if (eq_gains[i] > 0) == (cp_gains[i] > 0)
            )
            sign_acc = correct / len(decisive_idx)

        mean_good = (
            sum(eq_gains[i] for i in good_idx) / len(good_idx) if good_idx else None
        )
        mean_blunder = (
            sum(eq_gains[i] for i in blunder_idx) / len(blunder_idx)
            if blunder_idx
            else None
        )

        reports.append(
            GoodMovesReport(
                name=name,
                n_moves=len(pairs),
                n_good=len(good_idx),
                n_blunder=len(blunder_idx),
                n_decisive=len(decisive_idx),
                sign_accuracy=sign_acc,
                mean_delta_good=mean_good,
                mean_delta_blunder=mean_blunder,
                correlation=_pearson(eq_gains, cp_gains),
            )
        )
    return reports


def _fmt(x: Optional[float], spec: str = "+.2f") -> str:
    return "—" if x is None else format(x, spec)


def format_good_moves(
    reports: Sequence[GoodMovesReport],
    *,
    baseline: str = "baseline",
    see_also: Optional[str] = None,
) -> str:
    """Render the good-moves measurement as a Markdown section (task 0117).

    One row per predictor: how many moves, the direction sanity floor (sign-accuracy on
    decisive moves), the headline mean Δequity on good vs blunder moves, and the
    (baseline-biased) magnitude correlation. A verdict line states whether the equity
    bar reads good moves *at least as positively* as the baseline — the literal thesis.
    Returns ``""`` when there is nothing to show.

    ``see_also`` (task 0158): when given a path, append a one-line pointer to the fuller
    move-level write-up (``reports/goodmoves_real.md``) so a reader of a single gate
    artifact (``reports/validation_real.md``) can reach the positive-direction half of the
    thesis and its "what this proves / doesn't" caveats.
    """
    if not reports:
        return ""

    out: List[str] = ["## Good moves read as good (move-level Δequity, task 0117)", ""]
    out.append(
        "Per consecutive ply-pair, the engine's cp swing (mover POV, clamped "
        f"±{CP_CLAMP:.0f}) is the ground-truth move quality. **Good** = mover lost "
        f"≤{GOOD_MAX_LOSS:.0f}cp (engine-approved); **blunder** = dropped "
        f"≥{BLUNDER_DROP:.0f}cp. `Δgood`/`Δblunder` are the mean mover-POV equity swing "
        "(pp) the bar showed on each. The thesis: good moves should read as a *positive* "
        "gain, not a saturated ~0."
    )
    out.append("")
    out.append(
        f"`sign-acc` (direction on |cp|≥{DECISIVE_CP:.0f}cp moves) is a sanity floor — "
        "any monotone-in-cp bar scores ~1.0. `corr` is **baseline-biased** (the cp swing "
        "is the baseline's own input), shown for transparency, not as the win condition."
    )
    out.append("")
    out.append("| predictor | moves | good | blunder | sign-acc | Δgood (pp) | Δblunder (pp) | corr |")
    out.append("|---|--:|--:|--:|:--:|--:|--:|--:|")
    for r in reports:
        out.append(
            f"| {r.name} | {r.n_moves} | {r.n_good} | {r.n_blunder} | "
            f"{_fmt(r.sign_accuracy, '.3f')} | {_fmt(r.mean_delta_good)} | "
            f"{_fmt(r.mean_delta_blunder)} | {_fmt(r.correlation, '+.3f')} |"
        )
    out.append("")

    for line in _good_moves_verdict(reports, baseline=baseline):
        out.append(line)
        out.append("")

    if see_also:
        out.append(
            f"See [`{see_also}`]({see_also}) for the fuller move-level write-up — the "
            "reproduce recipe, the rating-signal (blunder-leniency) read, and what this "
            "slice proves and does *not* prove (the rating-conditioned good-move upside "
            "needs Maia's policy — task 0008/0005)."
        )
        out.append("")
    return "\n".join(out)


def reads_good_above_blunder(report: GoodMovesReport) -> bool:
    """Does this bar read engine-approved moves *above* blunders (the core sanity)?

    The minimum the phrase 'good moves read as good, not just less bad' demands: a good
    move's mover-POV equity swing should sit above a blunder's. True for any sane bar;
    a bar that fails this is reading move quality backwards.
    """
    return (
        report.mean_delta_good is not None
        and report.mean_delta_blunder is not None
        and report.mean_delta_good > report.mean_delta_blunder
    )


def _good_moves_verdict(
    reports: Sequence[GoodMovesReport], *, baseline: str
) -> List[str]:
    """The honest two-part read (task 0117).

    1. **Direction** — every bar must read good moves above blunders (sanity floor).
    2. **Rating signal** — with cp-delta as ground truth the rating-*blind* baseline is
       strong almost by construction (cp is its only input), so the rating-conditioned
       edge does NOT show up as bigger good-move upside here — that needs Maia's
       rating-relative move policy (task 0008/0005). What it DOES show robustly is
       *blunder-leniency*: a rating-aware bar reads blunders as less catastrophic
       (the refutation a peer won't find), i.e. ``Δblunder`` nearer zero than baseline.
    """
    by_name = {r.name: r for r in reports}
    base = by_name.get(baseline)
    lines: List[str] = []

    separates = [r.name for r in reports if reads_good_above_blunder(r)]
    if len(separates) == len(reports):
        lines.append(
            "**Direction:** every bar reads engine-approved moves above blunders "
            "(Δgood > Δblunder) — good moves read as good, not as bad. ✅"
        )
    else:
        backwards = [r.name for r in reports if not reads_good_above_blunder(r)]
        lines.append(
            "**Direction:** bars reading move quality backwards (Δgood ≤ Δblunder): "
            + ", ".join(f"`{n}`" for n in backwards)
            + ". ⚠"
        )

    if base is not None and base.mean_delta_blunder is not None:
        base_bl = base.mean_delta_blunder
        lenient = [
            (r.name, r.mean_delta_blunder)
            for r in reports
            if r.name != baseline and r.mean_delta_blunder is not None
        ]
        nicer = [n for n, bl in lenient if bl >= base_bl]
        if lenient and len(nicer) == len(lenient):
            lines.append(
                f"**Rating signal:** every rating-conditioned bar reads blunders as less "
                f"catastrophic than the rating-blind baseline (Δblunder {base_bl:+.2f}pp) "
                "— a refutation a rating-peer won't find is discounted. "
                "(With cp-delta as ground truth the cp-based baseline is strong by "
                "construction; the good-move *upside* needs Maia's rating-relative "
                "policy — task 0008/0005.)"
            )
        elif lenient:
            harsher = ", ".join(
                f"`{n}` ({bl:+.2f} < {base_bl:+.2f})" for n, bl in lenient if bl < base_bl
            )
            lines.append(
                f"**Rating signal:** baseline Δblunder {base_bl:+.2f}pp; harsher-than-"
                f"baseline bars: {harsher}."
            )
    return lines


# --- cutoff-robustness sweep (task 0157) --------------------------------------


@dataclass(frozen=True)
class SweepCell:
    """One predictor's good-moves direction at a single (good, blunder) cutoff pair."""

    good_max_loss: float
    blunder_drop: float
    n_good: int
    n_blunder: int
    mean_delta_good: Optional[float]
    mean_delta_blunder: Optional[float]
    direction_holds: bool  # Δgood > Δblunder in this cell


@dataclass(frozen=True)
class GoodMovesSweep:
    """One predictor swept over a grid of good × blunder cutoffs (task 0157)."""

    name: str
    decisive_cp: float
    sign_accuracy: Optional[float]  # grid-invariant: depends only on decisive_cp
    cells: List[SweepCell]
    all_hold: bool  # the direction holds in every cell


def sweep_good_moves(
    rows: Sequence[PositionRow],
    predictors: Dict[str, Predictor],
    *,
    good_cutoffs: Sequence[float] = SWEEP_GOOD_CUTOFFS,
    blunder_cutoffs: Sequence[float] = SWEEP_BLUNDER_CUTOFFS,
    decisive_cp: float = DECISIVE_CP,
) -> List[GoodMovesSweep]:
    """Re-measure each predictor across a grid of good × blunder cutoffs (task 0157).

    The good/blunder cutoffs in :func:`measure_good_moves` are arbitrary defaults; this
    proves the headline ``Δgood > Δblunder`` direction is not an artifact of them. For
    each predictor we re-run :func:`measure_good_moves` at every ``(good_max_loss,
    blunder_drop)`` pair in the grid and record whether the direction holds in that cell.
    ``sign_accuracy`` is computed once per predictor (it depends only on ``decisive_cp``,
    so it is constant across the grid). Returns one :class:`GoodMovesSweep` per predictor
    in registry order, or an empty list when there are no move-pairs to score.
    """
    if not iter_move_pairs(rows):
        return []

    # sign-acc is grid-invariant, so measure it once per predictor at any cutoff pair.
    base_reports = {
        r.name: r
        for r in measure_good_moves(rows, predictors, decisive_cp=decisive_cp)
    }

    cells_by_name: Dict[str, List[SweepCell]] = {name: [] for name in predictors}
    for g in good_cutoffs:
        for b in blunder_cutoffs:
            for r in measure_good_moves(
                rows,
                predictors,
                good_max_loss=g,
                blunder_drop=b,
                decisive_cp=decisive_cp,
            ):
                cells_by_name[r.name].append(
                    SweepCell(
                        good_max_loss=g,
                        blunder_drop=b,
                        n_good=r.n_good,
                        n_blunder=r.n_blunder,
                        mean_delta_good=r.mean_delta_good,
                        mean_delta_blunder=r.mean_delta_blunder,
                        direction_holds=reads_good_above_blunder(r),
                    )
                )

    sweeps: List[GoodMovesSweep] = []
    for name in predictors:
        cells = cells_by_name[name]
        sweeps.append(
            GoodMovesSweep(
                name=name,
                decisive_cp=decisive_cp,
                sign_accuracy=base_reports[name].sign_accuracy
                if name in base_reports
                else None,
                cells=cells,
                all_hold=all(c.direction_holds for c in cells),
            )
        )
    return sweeps


def format_good_moves_sweep(sweeps: Sequence[GoodMovesSweep]) -> str:
    """Render the cutoff-robustness sweep as a Markdown section (task 0157).

    A compact table per predictor — one row per (good, blunder) cutoff cell with the
    mean Δgood/Δblunder and whether the direction holds — followed by a verdict line
    stating the direction is cutoff-robust, or naming the cells where it breaks. Returns
    ``""`` when there is nothing to show.
    """
    if not sweeps:
        return ""

    out: List[str] = [
        "## Cutoff-robustness sweep (good × blunder grid, task 0157)",
        "",
        "The good/blunder cutoffs above (≤10cp / ≥100cp) are arbitrary defaults, so the "
        "headline `Δgood > Δblunder` direction is re-measured across a grid of good "
        "cutoffs × blunder cutoffs. `holds` is `Δgood > Δblunder` in that cell. `sign-acc` "
        "depends only on the decisive-cp threshold (not the good/blunder cutoffs), so it "
        "is constant across the grid and shown once per predictor.",
        "",
    ]
    for s in sweeps:
        out.append(
            f"**`{s.name}`** — sign-acc {_fmt(s.sign_accuracy, '.3f')} "
            f"(|cp|≥{s.decisive_cp:.0f}cp, grid-invariant)"
        )
        out.append("")
        out.append("| good ≤ | blunder ≥ | good | blunder | Δgood (pp) | Δblunder (pp) | holds |")
        out.append("|--:|--:|--:|--:|--:|--:|:--:|")
        for c in s.cells:
            out.append(
                f"| {c.good_max_loss:.0f}cp | {c.blunder_drop:.0f}cp | {c.n_good} | "
                f"{c.n_blunder} | {_fmt(c.mean_delta_good)} | {_fmt(c.mean_delta_blunder)} "
                f"| {'✅' if c.direction_holds else '⚠'} |"
            )
        out.append("")

    out.extend(_sweep_verdict(sweeps))
    out.append("")
    return "\n".join(out)


def _sweep_verdict(sweeps: Sequence[GoodMovesSweep]) -> List[str]:
    """State whether the direction is cutoff-robust, or name where it breaks."""
    robust = [s for s in sweeps if s.all_hold]
    broken = [s for s in sweeps if not s.all_hold]
    lines: List[str] = []
    if not broken:
        n_cells = len(sweeps[0].cells) if sweeps else 0
        names = ", ".join(f"`{s.name}`" for s in sweeps)
        lines.append(
            f"**Cutoff-robust:** `Δgood > Δblunder` holds in all {n_cells} cells of the "
            f"good × blunder grid for {names} — the direction is not an artifact of the "
            "default cutoffs. ✅"
        )
    else:
        for s in broken:
            bad = ", ".join(
                f"(good≤{c.good_max_loss:.0f}, blunder≥{c.blunder_drop:.0f})"
                for c in s.cells
                if not c.direction_holds
            )
            lines.append(
                f"**Cutoff-sensitive:** `{s.name}` reads move quality backwards "
                f"(Δgood ≤ Δblunder) in: {bad}. ⚠"
            )
        if robust:
            ok = ", ".join(f"`{s.name}`" for s in robust)
            lines.append(f"Direction is cutoff-robust for: {ok}.")
    return lines
