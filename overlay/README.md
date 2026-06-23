# chess-equity — streaming overlay (OBS browser source)

A transparent web overlay a streamer drops into OBS/Streamlabs as a **browser
source**. It renders a live, clock-aware **equity bar** — practical win chances
for each player — that updates as a broadcast game progresses, and visibly
diverges from the standard Stockfish centipawn bar viewers are used to.

![equity overlay at a time-scramble divergence](preview.png)

*Above: a captured frame of the bundled replay. The practical equity bar gives
Carlsen 62% while the dashed **centipawn ghost tick** sits at ~48% (engine: dead
even) — Nakamura is down to 3.2s. The clock-aware bar tells the real story.*

This is task **0019** (the streamer-facing deliverable of the live-streaming
wedge). It is a self-contained front-end: it consumes a feed of events and draws
the bar. The live ingestion task (**0018**) produces that feed; the equity model
(**0005**) fills in the numbers. Until those land, a bundled **mock replay**
(`mock-game.json`) drives the overlay so it's demonstrable today.

> **Going live for a real broadcast?** See the end-to-end
> [`docs/STREAMING.md`](../docs/STREAMING.md) runbook — Lichess round URL →
> `chess-equity broadcast --serve-sse` → OBS browser source → troubleshooting.
> This page is the front-end's option reference.

## Quick start (under 2 minutes)

```bash
cd overlay
python3 serve.py            # stdlib only — no pip install
```

Then in OBS: **Sources → + → Browser**, and set the URL to one of:

| URL | Feed |
| --- | --- |
| `http://localhost:8777/`            | bundled replay (`mock-game.json`) |
| `http://localhost:8777/?src=/sse`   | **live** SSE push (same path 0018 will use) |

Set the source size to your scene (e.g. 1920×120 for a bottom strip) and tick
**"Shutdown source when not visible"** off so the feed keeps running. The page
background is transparent, so only the bar composites over your stream.

### Config (query params, all optional)

| Param | Default | Meaning |
| --- | --- | --- |
| `src`    | `./mock-game.json` | SSE endpoint, `ws[s]://` WebSocket, or a `.json` replay file |
| `layout` | `horizontal` | `horizontal` (names flank a wide bar) or `vertical` (classic eval-bar) |
| `theme`  | `dark` | `dark` or `light` label text |
| `cp`     | `1` | show the dashed **centipawn ghost tick** for contrast (`0` to hide) |
| `cpbar`  | `0` | render the centipawn eval as a **full second bar** (greyed, under the equity bar) instead of a tick |
| `caster` | `0` | **caster mode** — flare on big practical swings, highlighted when the engine bar misses them |
| `autofollow` | `0` | **auto-director** — in a multi-game round, auto-switch focus to the board with the biggest live `drama`; a manual pick pins and overrides (task 0188) |
| `focuslock` | `6` | autofollow focus-lock window (plies) — a switched board is held this long so noise can't thrash the bar |
| `speed`  | `1` | replay speed multiplier for `.json` feeds |
| `welo`   | _(feed)_ | override the White rating shown (Maia-2's top band is a coarse `>2000` — pin the real number) |
| `belo`   | _(feed)_ | override the Black rating shown |

Example: `http://localhost:8777/?src=/sse&layout=vertical&cp=0`
Caster setup: `http://localhost:8777/?caster=1&cpbar=1`

### Streamer quickstart — a live Lichess broadcast round → OBS

The quick start above drives the overlay from the bundled replay. To follow a **real
game**, run the [`broadcast`](../src/chess_equity/broadcast.py) ingestor against a
Lichess **broadcast round id** — it polls the round and emits one per-move equity
event per line. All commands are copy-paste; the offline check needs no network.

```bash
# 0. (offline sanity) prove the per-move feed works with no round/network —
#    replays a bundled PGN move-by-move, one JSON event per line:
uv run chess-equity broadcast --pgn data/sample/sample_games.pgn --interval 0 | head

# 1. live: poll a real Lichess broadcast round (Ctrl-C to stop). The round id is
#    the trailing path of a lichess.org/broadcast/…/<ROUND_ID> URL.
uv run chess-equity broadcast --round <ROUND_ID> --white-elo 2700 --black-elo 2700

# 2. serve the overlay (static files + a /sse push endpoint), in another shell:
python3 overlay/serve.py                 # http://localhost:8777/

# 3. in OBS: Sources → + → Browser, and set the URL to:
#    http://localhost:8777/?src=/sse&caster=1
#    (size it to your scene, e.g. 1920×120; transparent background composites over the stream)
```

`--white-elo/--black-elo` pin the ratings the bar conditions on (a broadcast round's
headers are often blank or `?`); `--model maia2` swaps the placeholder baseline for the
real rating-conditioned bar once its weights are installed (see `DEPENDENCIES.md`).

> **One-command live bridge (task 0094).** `chess-equity broadcast --round <id>
> --serve-sse 8777` streams the round straight into the overlay as Server-Sent-Events:
> it serves the overlay's static files **and** an `/sse` endpoint on the one port, so
> you just point an OBS browser source at `http://localhost:8777/?src=/sse` (the
> [config page](#setup-page--no-hand-editing-query-params-task-0021) builds that URL for
> you). No capture-to-file step. Works with `--pgn <file>` (a finished game replayed as
> "live") and `--url <pgn>` too. Each connection gets a fresh replay/stream.
>
> **Tune in early (task 0099).** For a live `--round`/`--url`, you can start the overlay
> *before the round begins*: the bridge keeps polling instead of giving up on the first
> quiet poll, and sends a periodic SSE keep-alive comment so the connection survives the
> wait — the bar populates as soon as the first move lands. A local `--pgn` replay is
> finite and still ends when the game does.
>
> The events are `MoveEvent.to_overlay_event()` / `GameEvent.to_overlay()` — the same
> serialization the JSON-Lines path uses (pinned by
> `tests/test_broadcast_overlay_contract.py`). Without `--serve-sse`, `broadcast` still
> prints those events as **JSON Lines** (the internal shape) for piping/inspection, and
> `python3 overlay/serve.py --game <file>` still replays a saved overlay-event `.json`.

### Setup page — no hand-editing query params (task 0021)

Don't want to assemble that URL by hand? Open **`/config.html`** (served by
`serve.py`): a small form that picks the feed (bundled replay, live `/sse`, or a
custom URL), takes optional rating overrides, toggles layout/theme/caster, and
spits out the ready-to-paste OBS browser-source URL. For a live game, enter the
Lichess broadcast round URL and it shows the `chess-equity broadcast … --serve-sse`
command to run the [0018](../src/chess_equity/broadcast.py) ingestor that feeds the
overlay.

## What it shows

- The **equity bar** (0–100% practical), both names + ratings.
- Both **clocks** (turn red under 10s — the time pressure that drives the wedge).
- The last move's **Δequity grade** pill (task 0008) — flares green when a player
  finds better than their level expects, red on a blunder.
- A dashed **centipawn ghost tick** showing where the classic engine bar would
  sit — so the divergence is visible at a glance (or a **full second bar** with
  `?cpbar=1`).
- In **caster mode** (`?caster=1`), a **drama flare** on big practical equity
  swings, glowing gold when it's a swing the engine bar misses (e.g. a clock
  scramble the centipawn eval calls quiet) — the caster's "look at THIS" cue.

The bundled `mock-game.json` is a bullet time-scramble: around the time scramble
the centipawn eval reads ≈0.00 (or slightly for Black) while the clock-aware
equity bar shows White ~65–75% — then Black blunders under flag pressure and the
two bars converge. That divergence is asserted by `test_overlay.py`.

## Event schema (the contract 0018 must emit)

The feed is a sequence of JSON events. Over SSE/WebSocket, one JSON object per
message. As a replay file, an array (or `{ "events": [...] }`) where each event
may carry `delayMs` = milliseconds to wait before the next one.

All numbers are **White-POV**. `equity` is the practical win chance
`P(win) + 0.5·P(draw)` ∈ `[0,1]`; `cp` is the classic centipawn eval; `clock`
values are **seconds remaining**.

```jsonc
// one-time metadata
{
  "type": "game",
  "format": "bullet",                       // optional label
  "board": 0,                               // optional — 0-based board index in a multi-game round (task 0185)
  "players": {
    "white": { "name": "Carlsen", "rating": 2839 },
    "black": { "name": "Nakamura", "rating": 2802 }
  }
}

// per-move update
{
  "type": "position",
  "ply": 44,
  "board": 0,                               // optional — board index (multi-game round); omitted for a single game
  "move": { "san": "Rxd5??" },              // optional, for display
  "equity": 0.88,                           // REQUIRED — White-POV practical win chance 0..1
  "cp": 60,                                 // optional — classic centipawn eval (White POV)
  "wdl": { "win": 0.80, "draw": 0.16, "loss": 0.04 },  // optional, if the model exposes WDL
  "clock": { "white": 13.2, "black": 1.6 }, // optional — seconds remaining
  "grade": { "label": "blunder", "delta": -0.22 },     // optional — Δequity grade (mover POV)
  "drama": { "kind": "scramble", "magnitude": 0.55,    // optional — caster-mode drama (task 0020)
             "headline": "Time scramble — Black (1.6s) swings the bar -22 pts" }
}

// board roster — only for a multi-game broadcast round (task 0185). Emitted (and
// re-emitted as boards appear) so the overlay can render a live board selector. Pure
// routing metadata: never rendered on the bar. A single-game feed never sends one.
{
  "type": "boards",
  "boards": [
    { "index": 0, "players": { "white": { "name": "Carlsen" }, "black": { "name": "Nakamura" } } },
    { "index": 1, "players": { "white": { "name": "Nepo" }, "black": { "name": "Ding" } } }
  ]
}
```

Only `type` and (for positions) `equity` are required; everything else degrades
gracefully (missing clock hides the clock, missing `cp` hides the ghost tick). In a
multi-game round, events carry a `board` index and the feed sends a `boards` roster so
the overlay shows a selector; the chosen board's events flow to the bar (default: when
only one board exists, no selector appears and every event renders). With `?autofollow=1`
the overlay becomes an **auto-director**: it follows whichever board's latest `drama`
magnitude is highest, holding focus for `?focuslock=` plies after each switch so a real
swing — not noise — wins; the caster can still click the selector to pin a board (which
disables autofollow until reset). The
optional `drama` payload mirrors `chess_equity.drama.DramaEvent` — when present it
supplies the caster-mode flare's headline; otherwise caster mode derives a flare
from the equity swing itself, so it works on any feed.

## Tests

```bash
python3 test_overlay.py        # or: pytest overlay/test_overlay.py
```

Validates the schema and asserts the headline acceptance criterion — that the
equity bar diverges from the centipawn bar by ≥20 points somewhere, and that the
fixture contains a real time-scramble.

## Deferred (follow-ups)

- Wire to the real **0018** broadcast feed (Lichess broadcast WebSocket) and a
  config page that takes a round/game URL (currently config is via query params).
- Emit the server-side `drama` payload from the pipeline (0018 + `chess_equity.drama`)
  so caster headlines use the full classifier (clutch / missed-win / escape / scramble)
  rather than the client-side swing heuristic.
- Precompute-then-live buffering for instant load (0012).
