"""Personal equity — calibrate the bar to a *specific* player, not just a rating band.

Task 0014 (first slice). A rating band says "a 1600 plays like this on average". But two
1600s differ: one is a tactical shark who melts in endgames, the other a positional grinder.
This module mines one player's annotated game history into a *per-phase quality profile*
(how badly they err in the opening / middlegame / endgame, relative to their own baseline),
then turns that into a **phase-wise rating offset** that feeds the existing equity models.

The mechanism is deliberately small and composable:

- :func:`build_profile` parses a player's games (Lichess PGN with ``[%eval]`` tags) and
  aggregates, per phase, their average centipawn loss and blunder rate.
- :meth:`PlayerProfile.phase_offset` converts "this phase is worse/better than my own
  average" into an Elo delta (negative in weak phases, positive in strong ones), centred on
  zero so the player's *overall* rating is preserved — only its phase distribution shifts.
- :class:`PersonalEquityModel` wraps any :class:`~chess_equity.adapters.EquityModel` and
  applies each player's offset to their effective rating before delegating. Same position,
  two different opponents → two different equities.

Network access (mining a live username) goes through the module-level :func:`_urlopen`
seam — exactly like :mod:`chess_equity.data.download` — so tests inject a fake opener and
the profile-building logic is exercised entirely offline from PGN text.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import IO, Dict, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

import chess
import chess.pgn

from chess_equity.adapters import EquityModel
from chess_equity.data.schema import game_phase, parse_eval
from chess_equity.types import Equity

# The three phases :func:`~chess_equity.data.schema.game_phase` can emit, in board order.
PHASES = ("opening", "middlegame", "endgame")

# A move costing at least this many centipawns (vs the position's eval before it) is a
# "blunder". 300cp ≈ a clean piece; a coarse, engine-free count good enough for a profile.
BLUNDER_CP = 300.0

# How many Elo points one centipawn of *relative* phase weakness is worth. The offset is
# ``ELO_PER_CP_LOSS * (my_overall_acpl - my_phase_acpl)``: a phase 50cp worse than the
# player's own average shifts their effective rating by -100 here. Tunable; intentionally
# modest so a noisy profile can't swing equity wildly.
ELO_PER_CP_LOSS = 2.0

# Clamp the derived offset and the resulting effective rating to sane ranges.
_MAX_OFFSET = 400.0
_MIN_RATING = 100
_MAX_RATING = 4000

# Seam: tests monkeypatch this to avoid the network. Matches urllib.request.urlopen.
_urlopen = urlopen

_LICHESS_GAMES_URL = "https://lichess.org/api/games/user/{user}"


def _non_king_pieces(board: chess.Board) -> int:
    """Pieces on the board excluding the two kings — drives the phase heuristic."""
    return chess.popcount(board.occupied) - 2


def phase_of(board: chess.Board) -> str:
    """The coarse opening/middlegame/endgame label for ``board``.

    Mirrors the dataset's labelling (:func:`chess_equity.data.schema.game_phase`) so a
    profile built from games and an equity computed at inference agree on "what phase".
    """
    return game_phase(board.ply(), _non_king_pieces(board))


@dataclass
class PhaseStats:
    """Aggregated move quality for one player in one game phase."""

    n_moves: int = 0
    sum_cp_loss: float = 0.0
    n_blunders: int = 0

    def add(self, cp_loss: float) -> None:
        self.n_moves += 1
        self.sum_cp_loss += cp_loss
        if cp_loss >= BLUNDER_CP:
            self.n_blunders += 1

    @property
    def avg_cp_loss(self) -> float:
        """Average centipawn loss per move (a.k.a. ACPL) in this phase."""
        return self.sum_cp_loss / self.n_moves if self.n_moves else 0.0

    @property
    def blunder_rate(self) -> float:
        """Fraction of this phase's moves that lost at least :data:`BLUNDER_CP`."""
        return self.n_blunders / self.n_moves if self.n_moves else 0.0


@dataclass
class PlayerProfile:
    """A specific player's phase-by-phase quality, and the Elo offsets it implies."""

    username: str
    n_games: int = 0
    sum_rating: int = 0
    n_rating: int = 0
    phases: Dict[str, PhaseStats] = field(default_factory=dict)

    def _phase(self, phase: str) -> PhaseStats:
        return self.phases.setdefault(phase, PhaseStats())

    @property
    def rating(self) -> Optional[int]:
        """The player's mean rating across the mined games (``None`` if none seen)."""
        return round(self.sum_rating / self.n_rating) if self.n_rating else None

    @property
    def total_moves(self) -> int:
        return sum(s.n_moves for s in self.phases.values())

    @property
    def overall_acpl(self) -> float:
        """Average centipawn loss across *all* phases — the player's own baseline."""
        total = self.total_moves
        if not total:
            return 0.0
        return sum(s.sum_cp_loss for s in self.phases.values()) / total

    def phase_offset(self, phase: str) -> float:
        """Elo offset for ``phase``: negative where the player is weaker than their own
        average, positive where stronger. Zero (no data) leaves the rating untouched."""
        stats = self.phases.get(phase)
        if stats is None or stats.n_moves == 0:
            return 0.0
        delta = self.overall_acpl - stats.avg_cp_loss
        offset = ELO_PER_CP_LOSS * delta
        return max(-_MAX_OFFSET, min(_MAX_OFFSET, offset))

    def effective_rating(self, nominal: int, phase: str) -> int:
        """``nominal`` rating shifted by this phase's offset, clamped to a sane range."""
        adjusted = nominal + self.phase_offset(phase)
        return int(round(max(_MIN_RATING, min(_MAX_RATING, adjusted))))


def _move_cp_loss(eval_before: float, eval_after: float, mover_is_white: bool) -> float:
    """Centipawns the mover gave up, from their POV (clipped at 0 — a move can't *gain*).

    Evals are White-POV centipawns. White wants the eval to rise, Black wants it to fall,
    so White's loss is ``before - after`` and Black's is ``after - before``.
    """
    loss = (eval_before - eval_after) if mover_is_white else (eval_after - eval_before)
    return max(0.0, loss)


def update_profile(profile: PlayerProfile, game: "chess.pgn.Game") -> bool:
    """Fold one game into ``profile`` if the target player is in it. Returns whether it was.

    Walks the mainline; for every move the profiled player made that has an ``[%eval]``
    both before and after, accumulates its centipawn loss into the move's phase. Games not
    featuring the player (by case-insensitive ``White``/``Black`` header) are ignored.
    """
    target = profile.username.strip().lower()
    white = game.headers.get("White", "").strip().lower()
    black = game.headers.get("Black", "").strip().lower()
    if target == white:
        mover_colour = chess.WHITE
        rating_key = "WhiteElo"
    elif target == black:
        mover_colour = chess.BLACK
        rating_key = "BlackElo"
    else:
        return False

    profile.n_games += 1
    try:
        rating = int(game.headers.get(rating_key, ""))
    except (TypeError, ValueError):
        rating = None
    if rating is not None:
        profile.sum_rating += rating
        profile.n_rating += 1

    node = game
    prev_eval = 0.0  # startpos ≈ even; refined as soon as the first [%eval] is seen.
    have_prev = True
    while node.variations:
        node = node.variations[0]
        eval_after = parse_eval(_eval_tag(node.comment))
        mover_is_white = node.parent.board().turn == chess.WHITE
        if eval_after is not None and have_prev and mover_is_white == (mover_colour == chess.WHITE):
            board = node.board()  # position after the move — same convention as the dataset
            loss = _move_cp_loss(prev_eval, eval_after, mover_is_white)
            profile._phase(phase_of(board)).add(loss)
        if eval_after is not None:
            prev_eval = eval_after
            have_prev = True
        else:
            have_prev = False
    return True


def _eval_tag(comment: Optional[str]) -> str:
    """Extract the ``[%eval ...]`` payload from a PGN comment (``""`` if absent)."""
    if not comment:
        return ""
    marker = "[%eval"
    start = comment.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = comment.find("]", start)
    return comment[start:end].strip() if end >= 0 else ""


def build_profile(handle: IO[str], username: str, *, max_games: Optional[int] = None) -> PlayerProfile:
    """Build a :class:`PlayerProfile` for ``username`` from a stream of PGN games.

    ``handle`` is any text stream of concatenated PGN (a file, or the body returned by
    :func:`fetch_user_games`). Only games featuring the player contribute; ``max_games``
    caps how many *of theirs* are folded in.
    """
    profile = PlayerProfile(username=username)
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        update_profile(profile, game)
        if max_games is not None and profile.n_games >= max_games:
            break
    return profile


def fetch_user_games(
    username: str, *, max_games: int = 50, token: Optional[str] = None, rated: bool = True
) -> str:
    """Fetch a player's recent annotated games from the Lichess API as PGN text.

    Read-only and unauthenticated by default (a ``token`` only raises the rate limit). We
    request ``evals=true`` so move quality is computable and ``clocks=false`` to keep the
    payload lean. Goes through :func:`_urlopen` so tests never hit the network.
    """
    url = _LICHESS_GAMES_URL.format(user=quote(username, safe=""))
    params = f"?max={int(max_games)}&evals=true&clocks=false&rated={'true' if rated else 'false'}"
    headers = {"Accept": "application/x-chess-pgn"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url + params, headers=headers)
    with _urlopen(req) as resp:
        body = resp.read()
    return body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else body


def build_profile_for_user(
    username: str, *, max_games: int = 50, token: Optional[str] = None
) -> PlayerProfile:
    """Convenience: fetch ``username``'s games over the network and profile them."""
    import io

    pgn = fetch_user_games(username, max_games=max_games, token=token)
    return build_profile(io.StringIO(pgn), username, max_games=max_games)


def load_profile(
    spec: str, *, max_games: int = 50, token: Optional[str] = None
) -> PlayerProfile:
    """Resolve a profile *spec* (as passed to ``--white-profile`` / ``--black-profile``).

    Two forms, so the same flag covers both the live product path and an offline,
    test-friendly path:

    - ``"<username>"`` — fetch the player's recent annotated games from Lichess and
      profile them (the network seam in :func:`fetch_user_games`).
    - ``"<player>@<file.pgn>"`` — profile ``<player>`` from a local PGN file, no network.
      The name before ``@`` is the ``White``/``Black`` header to match in the file.
    """
    if "@" in spec:
        username, _, path = spec.partition("@")
        if not username or not path:
            raise ValueError(
                f"profile spec {spec!r} must be 'player@file.pgn' (both parts required)"
            )
        with open(path, encoding="utf-8") as fh:
            return build_profile(fh, username, max_games=max_games)
    return build_profile_for_user(spec, max_games=max_games, token=token)


class PersonalEquityModel(EquityModel):
    """Wrap an equity model so each side's rating is shifted by their personal profile.

    ``white_profile`` / ``black_profile`` are optional: a side with no profile is evaluated
    at its nominal rating, so this composes with the band-average bar for unknown opponents.
    """

    def __init__(
        self,
        base: EquityModel,
        *,
        white_profile: Optional[PlayerProfile] = None,
        black_profile: Optional[PlayerProfile] = None,
    ) -> None:
        self.base = base
        self.white_profile = white_profile
        self.black_profile = black_profile

    def evaluate(self, fen: str, white_elo: int, black_elo: int) -> Equity:
        phase = phase_of(chess.Board(fen))
        w = self.white_profile.effective_rating(white_elo, phase) if self.white_profile else white_elo
        b = self.black_profile.effective_rating(black_elo, phase) if self.black_profile else black_elo
        equity = self.base.evaluate(fen, w, b)
        return replace(equity, source=f"{equity.source}+personal")
