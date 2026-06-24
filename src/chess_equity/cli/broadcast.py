"""``chess-equity broadcast`` parser builder."""

from __future__ import annotations

import argparse

from chess_equity.cli._common import add_model_arg, add_profile_args


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    bc = sub.add_parser(
        "broadcast",
        help="stream per-move equity events from a live (or replayed) broadcast",
    )
    src = bc.add_mutually_exclusive_group(required=True)
    src.add_argument("--round", help="Lichess broadcast round id (live feed)")
    src.add_argument("--url", help="arbitrary public PGN URL (generic feed)")
    src.add_argument("--pgn", help="local PGN file, replayed move-by-move as 'live'")
    bc.add_argument("--white-elo", type=int, default=None, help="override White rating")
    bc.add_argument("--black-elo", type=int, default=None, help="override Black rating")
    bc.add_argument("--interval", type=float, default=2.0, help="seconds between polls")
    bc.add_argument("--max-polls", type=int, default=None, help="stop after N polls")
    bc.add_argument(
        "--moves-per-poll", type=int, default=1, help="replay pacing (local --pgn only)"
    )
    bc.add_argument("--token", default=None, help="Lichess API token (optional)")
    bc.add_argument(
        "--board",
        default=None,
        metavar="PLAYER|INDEX|auto[:PLAYER]",
        help="follow one board of a multi-game round: a case-insensitive player-name "
        "substring, or a 0-based board index, or 'auto' to auto-cut the overlay focus to "
        "whichever board has the highest recent drama (task 0256), or 'auto:PLAYER' to "
        "auto-cut but softly bias the focus toward that player's boards (task 0262). "
        "Default: follow every board (task 0182).",
    )
    bc.add_argument(
        "--clock-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="warp the published equity by the side-to-move's time pressure when the feed "
        "carries [%%clk] clocks (task 0097); --no-clock-aware emits the clock-blind bar",
    )
    bc.add_argument(
        "--depth", type=int, default=2,
        help="Stockfish baseline search depth (also the maia-search ply budget)",
    )
    bc.add_argument(
        "--serve-sse",
        type=int,
        default=None,
        metavar="PORT",
        help="stream overlay events as Server-Sent-Events on this port instead of "
        "printing JSON Lines — point an OBS browser source at "
        "http://localhost:PORT/?src=/sse (task 0094)",
    )
    bc.add_argument(
        "--ledger",
        metavar="OUT",
        default=None,
        help="replay a local --pgn and write a flat per-move equity ledger CSV to OUT "
        "(ply, side, san, equity, delta_equity, grade, drama label/score, clocks) for "
        "spreadsheets / post-show graphics — the tabular counterpart to grade "
        "--annotate-pgn (task 0204). Requires --pgn (no live feed).",
    )
    bc.add_argument(
        "--captions",
        action="store_true",
        help="print one human caster sentence per graded move (TTS/chat-ready) instead "
        "of JSON Lines — composed from the move grade, practical swing and mover "
        "rating, with the drama headline appended on a real swing (task 0190)",
    )
    bc.add_argument(
        "--captions-vtt",
        metavar="OUT",
        default=None,
        help="replay a local --pgn and write the per-move caster captions (--captions) "
        "as a timestamped WebVTT subtitle track to OUT — one cue per graded move, keyed "
        "by the game's [%%clk] (cue starts at the elapsed game time the move was made; "
        "clock-less PGNs fall back to even move-index spacing) so the caster line "
        "becomes a real caption/TTS track for the recorded stream (task 0211). Requires "
        "--pgn (no live feed).",
    )
    bc.add_argument(
        "--captions-srt",
        metavar="OUT",
        default=None,
        help="like --captions-vtt but writes the caster captions as an SRT (SubRip) "
        "subtitle track to OUT — same cues, numbered with comma-millisecond timestamps "
        "and raw text, for non-web editors (Premiere/Resolve/CapCut) that can't ingest "
        "WebVTT (task 0229). Requires --pgn (no live feed).",
    )
    bc.add_argument(
        "--divergence-caption-threshold",
        type=float,
        default=None,
        metavar="PTS",
        help="minimum human-vs-engine bar gap (White-POV percentage points, "
        "|equity - cp_implied|) for a graded move's caption to add a spoken DIVERGENCE "
        "callout (task 0273); applies to --captions and the --captions-vtt/-srt tracks. "
        "Default ~15. Needs a model that exposes an objective cp (cp-less feeds emit no "
        "callout).",
    )
    add_profile_args(bc)
    add_model_arg(bc)
    return bc
