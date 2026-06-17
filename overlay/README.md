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
| `speed`  | `1` | replay speed multiplier for `.json` feeds |

Example: `http://localhost:8777/?src=/sse&layout=vertical&cp=0`

## What it shows

- The **equity bar** (0–100% practical), both names + ratings.
- Both **clocks** (turn red under 10s — the time pressure that drives the wedge).
- The last move's **Δequity grade** pill (task 0008) — flares green when a player
  finds better than their level expects, red on a blunder.
- A dashed **centipawn ghost tick** showing where the classic engine bar would
  sit — so the divergence is visible at a glance.

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
  "players": {
    "white": { "name": "Carlsen", "rating": 2839 },
    "black": { "name": "Nakamura", "rating": 2802 }
  }
}

// per-move update
{
  "type": "position",
  "ply": 44,
  "move": { "san": "Rxd5??" },              // optional, for display
  "equity": 0.88,                           // REQUIRED — White-POV practical win chance 0..1
  "cp": 60,                                 // optional — classic centipawn eval (White POV)
  "wdl": { "win": 0.80, "draw": 0.16, "loss": 0.04 },  // optional, if the model exposes WDL
  "clock": { "white": 13.2, "black": 1.6 }, // optional — seconds remaining
  "grade": { "label": "blunder", "delta": -0.22 }      // optional — Δequity grade (mover POV)
}
```

Only `type` and (for positions) `equity` are required; everything else degrades
gracefully (missing clock hides the clock, missing `cp` hides the ghost tick).

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
- "Caster mode" **drama/clutch** indicator (0020) on big practical swings.
- Side-by-side classic centipawn bar as a full second bar (currently a ghost tick).
- Precompute-then-live buffering for instant load (0012).
