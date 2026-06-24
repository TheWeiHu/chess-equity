"""Single-game equity scorecard — the CLI-first way to interrogate the thesis.

The whole project is one claim: a *rating-conditioned* predictor beats the
*rating-blind* centipawn bar at predicting **real** human outcomes. The statistical
proof of that lives in :mod:`chess_equity.validate` (log-loss/Brier/ECE over thousands
of positions, with a bootstrap gate). This module is the other half — the *fast
iteration loop*: feed it one game and it answers, on the command line, the four
questions you actually ask of a single game:

1. **Here is a game** — the PGN's moves, players, and ratings.
2. **Here is the score** — the objective centipawn eval embedded in the PGN
   (``[%eval …]``), the rating-blind number a normal engine bar would show.
3. **What's the real score?** — the actual result (``1-0``/``0-1``/``1/2-1/2``),
   i.e. the realized White score in ``{1, 0.5, 0}``.
4. **What are we predicting?** — ``equity_white`` = P(White wins) + ½·P(draw) from a
   chosen rating-conditioned model, scored against the realized outcome.

A single game is **illustrative, never proof** — the per-game Brier here is a tracking
number, not a powered statistic. The headline gate stays ``chess-equity validate`` /
``headline`` on a real Lichess dump. This module exists so you can *look at one game*
without opening the browser demo.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence

import chess
import chess.pgn

from chess_equity.adapters import EquityModel
from chess_equity.types import lichess_win_percent

if TYPE_CHECKING:  # avoid importing the grading machinery at module load
    from chess_equity.grading import MoveGrade

# Realized White score for each PGN result token; ``*`` (unfinished) -> unknown.
_RESULT_TO_WHITE_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


@dataclass(frozen=True)
class MoveScore:
    """One ply of the scorecard: the position after ``san`` was played.

    ``cp_white`` is the embedded objective eval from White's POV (None when the PGN
    carries no ``[%eval]`` for this ply). ``equity_white`` is the chosen model's
    rating-conditioned prediction in [0, 100]; ``baseline_white`` is the rating-blind
    Lichess Win% derived from ``cp_white`` (the number to beat), or None when there's
    no embedded eval to derive it from.
    """

    ply: int
    san: str
    fen: str
    cp_white: Optional[float]
    equity_white: float
    baseline_white: Optional[float]


@dataclass(frozen=True)
class Scorecard:
    """A whole game scored move-by-move: predictions vs the realized outcome."""

    white: str
    black: str
    white_elo: int
    black_elo: int
    result: str
    white_score: Optional[float]  # the "real score": 1.0 / 0.5 / 0.0, or None if "*"
    model: str
    moves: List[MoveScore]

    @property
    def equity_brier(self) -> Optional[float]:
        """Mean squared error of the model's equity vs the realized White score.

        Averaged over every ply (the prediction at each position is judged against the
        *final* result — early, genuinely-uncertain positions inflate this, which is why
        a single game's number is illustrative, not proof). None if the game is
        unfinished.
        """
        return self._brier([m.equity_white for m in self.moves])

    @property
    def baseline_brier(self) -> Optional[float]:
        """Same Brier, for the rating-blind centipawn baseline (Lichess Win%).

        Only plies that carry an embedded ``[%eval]`` contribute, so this is comparable
        to :attr:`equity_brier` only over that subset; None if no ply has an eval or the
        game is unfinished.
        """
        return self._brier([m.baseline_white for m in self.moves])

    def _brier(self, preds: List[Optional[float]]) -> Optional[float]:
        if self.white_score is None:
            return None
        errs = [
            (p / 100.0 - self.white_score) ** 2 for p in preds if p is not None
        ]
        if not errs:
            return None
        return sum(errs) / len(errs)


def _white_cp(node: chess.pgn.ChildNode) -> Optional[float]:
    """White-POV centipawns from a node's embedded ``[%eval]`` (mate -> ±10000)."""
    score = node.eval()
    if score is None:
        return None
    return float(score.white().score(mate_score=10000))


def build_scorecard(
    game: chess.pgn.Game,
    model: EquityModel,
    *,
    model_name: str = "model",
    white_elo: Optional[int] = None,
    black_elo: Optional[int] = None,
) -> Scorecard:
    """Walk one game and score every position with ``model`` vs the real result.

    Ratings come from the explicit ``white_elo``/``black_elo`` overrides, else the PGN's
    ``WhiteElo``/``BlackElo`` headers, else 1500 — so ``score --pgn game.pgn`` works with
    zero flags on a real Lichess game.
    """
    headers = game.headers
    we = white_elo if white_elo is not None else _header_elo(headers, "WhiteElo")
    be = black_elo if black_elo is not None else _header_elo(headers, "BlackElo")
    result = headers.get("Result", "*")

    moves: List[MoveScore] = []
    board = game.board()
    for ply, node in enumerate(game.mainline(), start=1):
        san = board.san(node.move)
        board.push(node.move)
        fen = board.fen()
        cp_white = _white_cp(node)
        equity = model.evaluate(fen, we, be)
        baseline = lichess_win_percent(cp_white) if cp_white is not None else None
        moves.append(
            MoveScore(
                ply=ply,
                san=san,
                fen=fen,
                cp_white=cp_white,
                equity_white=equity.equity_white,
                baseline_white=baseline,
            )
        )

    return Scorecard(
        white=headers.get("White", "?"),
        black=headers.get("Black", "?"),
        white_elo=we,
        black_elo=be,
        result=result,
        white_score=_RESULT_TO_WHITE_SCORE.get(result),
        model=model_name,
        moves=moves,
    )


def build_scorecard_from_pgn(
    pgn_text: str,
    model: EquityModel,
    *,
    model_name: str = "model",
    white_elo: Optional[int] = None,
    black_elo: Optional[int] = None,
) -> Scorecard:
    """:func:`build_scorecard` over the first game in a PGN string."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("no game found in PGN")
    return build_scorecard(
        game,
        model,
        model_name=model_name,
        white_elo=white_elo,
        black_elo=black_elo,
    )


def _header_elo(headers: chess.pgn.Headers, key: str) -> int:
    raw = headers.get(key, "")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 1500


# --------------------------------------------------------------------------- #
# Shareable SVG scorecard — the visual sibling of the text scorecard (task 0253)
# --------------------------------------------------------------------------- #

# Dark theme, matching :func:`chess_equity.grading.equity_trajectory_svg` so the
# share card and the trajectory widget read as one family on a stream.
_BG = "#1b1b1b"
_FG = "#f0f0f0"
_DIM = "#9a9a9a"
_WHITE_FILL = "#e8e8e8"  # White's share of an equity bar
_BLACK_FILL = "#3a3a3a"  # Black's share
_ACCENT = "#7fb2e5"


def _xml_escape(text: str) -> str:
    """Escape the five XML special chars so player names never break the SVG."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _last_baseline(card: Scorecard) -> Optional[float]:
    """The most recent ply's rating-blind Lichess Win% (last that carried an eval)."""
    for m in reversed(card.moves):
        if m.baseline_white is not None:
            return m.baseline_white
    return None


def _equity_bar_svg(x: float, y: float, w: float, h: float, equity_white: Optional[float]) -> str:
    """A horizontal White/Black equity bar: ``equity_white``% filled white from the left."""
    if equity_white is None:
        return (
            f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{_BLACK_FILL}" rx="3"/>\n'
            f'  <text x="{x + w / 2}" y="{y + h / 2 + 4}" font-family="sans-serif" '
            f'font-size="11" fill="{_DIM}" text-anchor="middle">no eval</text>\n'
        )
    frac = max(0.0, min(1.0, equity_white / 100.0))
    white_w = round(w * frac, 2)
    return (
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{_BLACK_FILL}" rx="3"/>\n'
        f'  <rect x="{x}" y="{y}" width="{white_w}" height="{h}" fill="{_WHITE_FILL}" rx="3"/>\n'
    )


def _biggest_swing(grades: Sequence["MoveGrade"], card: Scorecard):
    """The ply with the largest |Δequity| (mover POV) and its drama label, or None.

    Reuses :func:`chess_equity.drama.score_event` for the label by replaying each move
    as a minimal ``MoveEvent`` (clock-blind, so the time-scramble signal stays off). The
    swing is the mover's practical change vs the position before the move; the headline
    swing is the biggest such magnitude regardless of whether it crosses a drama bar.
    """
    from chess_equity.drama import MoveEvent
    from chess_equity.drama import score_event
    from chess_equity.grading import white_pov_equity

    if not grades:
        return None
    best = None  # (abs_delta, san, delta, label)
    for i, g in enumerate(grades):
        w_after = white_pov_equity(g)
        w_before = white_pov_equity(grades[i - 1]) if i > 0 else 50.0
        mover_after = w_after if g.mover_white else 100.0 - w_after
        mover_before = w_before if g.mover_white else 100.0 - w_before
        delta = mover_after - mover_before
        if best is None or abs(delta) > best[0]:
            event = MoveEvent(
                game_id="card", ply=g.ply, san=g.san, uci=g.uci, fen="",
                white_to_move=not g.mover_white, white_clock=None, black_clock=None,
                white_elo=card.white_elo, black_elo=card.black_elo,
                equity=w_after, delta_equity=delta, last_move_grade=g.label,
                source="card", compute_ms=0.0,
            )
            drama = score_event(event)
            label = drama.kind.replace("_", " ") if drama is not None else "swing"
            best = (abs(delta), g.san, delta, label)
    if best is None:
        return None
    return best[1], best[2], best[3]  # san, delta, label


def render_scorecard_svg(card: Scorecard, grades: Sequence["MoveGrade"]) -> str:
    """Render a one-game share card as a self-contained, dependency-free SVG (task 0253).

    The visual sibling of :func:`render_scorecard`, sized for Twitter/Discord/stream
    recaps. One fixed dark-theme layout shows: players + Elo, the result, the rating-blind
    objective bar vs the rating-conditioned equity bar (the thesis contrast at the final
    position), the biggest practical Δequity swing with its drama label, per-side accuracy,
    and a mini equity sparkline. Pure stdlib string emit — no external fonts/JS, no torch,
    no network; ``grades`` supplies accuracy/sparkline/drama via :mod:`chess_equity.grading`
    and :mod:`chess_equity.drama`. Theming/PNG export are deferred follow-ups.
    """
    from chess_equity.grading import _accuracy, white_pov_equity

    width, height = 600, 340
    pad = 24

    final_equity = card.moves[-1].equity_white if card.moves else None
    final_objective = _last_baseline(card)
    white_acc = _accuracy([g for g in grades if g.mover_white])
    black_acc = _accuracy([g for g in grades if not g.mover_white])
    swing = _biggest_swing(grades, card)
    result_txt = "unfinished" if card.white_score is None else card.result

    title = (
        f"{_xml_escape(card.white)} ({card.white_elo})  vs  "
        f"{_xml_escape(card.black)} ({card.black_elo})"
    )
    aria = (
        f"Equity scorecard: {title}, result {result_txt}; "
        f"final equity White {0.0 if final_equity is None else final_equity:.0f}%"
    )

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{_xml_escape(aria)}">\n'
    )
    parts.append(f'  <title>{_xml_escape(aria)}</title>\n')
    parts.append(f'  <rect width="{width}" height="{height}" fill="{_BG}"/>\n')

    # Header: players + Elo, result.
    parts.append(
        f'  <text x="{pad}" y="40" font-family="sans-serif" font-size="20" '
        f'font-weight="bold" fill="{_FG}">{title}</text>\n'
    )
    parts.append(
        f'  <text x="{pad}" y="64" font-family="sans-serif" font-size="13" '
        f'fill="{_DIM}">result: {_xml_escape(result_txt)}   ·   model: {_xml_escape(card.model)}</text>\n'
    )

    # The thesis contrast: rating-blind objective bar vs rating-conditioned equity bar.
    # Reserve a gap on the right so the White-POV % label never sits on top of the fill.
    bar_x = pad + 150
    bar_w = width - pad - bar_x - 46
    bar_h = 22
    lbl_x = bar_x + bar_w + 8
    parts.append(
        f'  <text x="{pad}" y="103" font-family="sans-serif" font-size="13" '
        f'fill="{_DIM}">Objective (blind)</text>\n'
    )
    parts.append(_equity_bar_svg(bar_x, 88, bar_w, bar_h, final_objective))
    obj_lbl = "--" if final_objective is None else f"{final_objective:.0f}%"
    parts.append(
        f'  <text x="{lbl_x}" y="103" font-family="sans-serif" font-size="12" '
        f'fill="{_FG}">{obj_lbl}</text>\n'
    )
    parts.append(
        f'  <text x="{pad}" y="139" font-family="sans-serif" font-size="13" '
        f'fill="{_ACCENT}" font-weight="bold">Equity (rating-aware)</text>\n'
    )
    parts.append(_equity_bar_svg(bar_x, 124, bar_w, bar_h, final_equity))
    eq_lbl = "--" if final_equity is None else f"{final_equity:.0f}%"
    parts.append(
        f'  <text x="{lbl_x}" y="139" font-family="sans-serif" font-size="12" '
        f'fill="{_FG}">{eq_lbl}</text>\n'
    )

    # Mini equity sparkline — White-POV equity per ply as rects (font-independent,
    # the same series as grading.equity_sparkline / equity_trajectory_svg).
    spark_y, spark_h = 168, 56
    parts.append(
        f'  <text x="{pad}" y="{spark_y - 6}" font-family="sans-serif" font-size="12" '
        f'fill="{_DIM}">Equity trajectory (White POV)</text>\n'
    )
    if grades:
        spark_w = width - 2 * pad
        n = len(grades)
        cell = spark_w / n
        bw = max(1.0, cell - 1)
        mid_y = spark_y + spark_h / 2
        parts.append(
            f'  <line x1="{pad}" y1="{round(mid_y, 2)}" x2="{pad + spark_w}" '
            f'y2="{round(mid_y, 2)}" stroke="#555" stroke-width="1" stroke-dasharray="4 3"/>\n'
        )
        for i, g in enumerate(grades):
            frac = max(0.0, min(1.0, white_pov_equity(g) / 100.0))
            h = round(spark_h * frac, 2)
            x = round(pad + i * cell, 2)
            yy = round(spark_y + (spark_h - h), 2)
            parts.append(
                f'  <rect x="{x}" y="{yy}" width="{round(bw, 2)}" height="{h}" '
                f'fill="{_ACCENT}" fill-opacity="0.75"/>\n'
            )

    # Biggest practical swing + drama label, and per-side accuracy.
    if swing is not None:
        san, delta, label = swing
        parts.append(
            f'  <text x="{pad}" y="262" font-family="sans-serif" font-size="14" '
            f'fill="{_FG}">Biggest swing: {_xml_escape(san)} ({delta:+.0f} pts) '
            f'<tspan fill="{_ACCENT}">{_xml_escape(label)}</tspan></text>\n'
        )
    parts.append(
        f'  <text x="{pad}" y="290" font-family="sans-serif" font-size="14" '
        f'fill="{_FG}">Accuracy — White {white_acc:.0f}%   Black {black_acc:.0f}%</text>\n'
    )
    parts.append(
        f'  <text x="{pad}" y="320" font-family="sans-serif" font-size="11" '
        f'fill="{_DIM}">chess-equity · one game is illustrative, not proof</text>\n'
    )

    parts.append('</svg>\n')
    return "".join(parts)


def render_scorecard(card: Scorecard) -> List[str]:
    """Render a :class:`Scorecard` as the CLI's human-readable block of lines."""
    lines: List[str] = []
    lines.append(f"# {card.white} ({card.white_elo}) vs {card.black} ({card.black_elo})")
    lines.append(f"# predicting: P(White wins) + 0.5*P(draw), White POV, via {card.model}")
    lines.append("")

    # Per-move table: the score (cp), the rating-blind baseline, and our prediction.
    lines.append(f"{'ply':>4} {'move':<7} {'cp':>7} {'win%':>6} {'equity':>7}")
    lines.append(f"{'':>4} {'':<7} {'(score)':>7} {'(blind)':>6} {'(pred)':>7}")
    for m in card.moves:
        cp = "   --  " if m.cp_white is None else f"{m.cp_white / 100:+6.2f} "
        blind = "   -- " if m.baseline_white is None else f"{m.baseline_white:5.1f} "
        lines.append(
            f"{m.ply:>4} {m.san:<7} {cp:>7} {blind:>6} {m.equity_white:6.1f}%"
        )

    lines.append("")
    real = "unfinished" if card.white_score is None else f"{card.result}  (White scored {card.white_score})"
    lines.append(f"# real score: {real}")

    eb, bb = card.equity_brier, card.baseline_brier
    if eb is not None:
        lines.append(f"# Brier (lower=better, illustrative single-game tracking only):")
        lines.append(f"#   equity   {eb:.4f}")
        if bb is not None:
            verdict = "equity tracks the result better" if eb < bb else (
                "centipawns track the result better" if eb > bb else "tie"
            )
            lines.append(f"#   baseline {bb:.4f}   -> {verdict} on THIS game")
        lines.append(
            "# NB: one game is illustrative, not proof — the powered gate is "
            "`chess-equity validate`/`headline` on a real dump."
        )
    return lines
