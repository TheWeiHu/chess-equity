# Go live: streaming the equity overlay to OBS

A single, end-to-end runbook for a streamer or caster: take a real Lichess
broadcast round, compute the **clock-aware equity bar**, and composite it into OBS
as a transparent browser source.

If you only want to *see* the overlay with no network, jump to
[Offline sanity check](#offline-sanity-check-no-network). For the front-end's full
option reference, see [`overlay/README.md`](../overlay/README.md).

Every command below is copy-paste runnable and verified against the actual CLI
(`chess-equity broadcast --help`, `python3 overlay/serve.py --help`).

---

## What you'll end up with

```
Lichess broadcast round  ──►  chess-equity broadcast --serve-sse 8777  ──►  /sse push
                                          (serves overlay + SSE on one port)
                                                        │
                              OBS Browser source: http://localhost:8777/?src=/sse
```

One process polls the round, computes equity, **and** serves the overlay to OBS.
No capture-to-file step.

---

## 0. Prerequisites

- The repo installed so the `chess-equity` CLI runs. This runbook uses `uv run
  chess-equity …`; if you've installed the package into your environment, plain
  `chess-equity …` works the same (entry point: `chess-equity = chess_equity.cli:main`).
- **OBS Studio** (or Streamlabs — anything with a *Browser* source).
- Python 3 on PATH (the overlay's dev server is stdlib-only, no `pip install`).
- A Lichess API token is **optional** (`--token`); public broadcast rounds work
  without one.

Confirm the CLI is reachable:

```bash
uv run chess-equity broadcast --help
```

---

## 1. Get the broadcast round id

A Lichess broadcast URL looks like:

```
https://lichess.org/broadcast/<tournament-slug>/<round-slug>/<ROUND_ID>
```

The **round id** is the trailing path segment (an 8-char id, e.g. `aBcDeFgH`).
Copy that — it's all the CLI needs.

> Tip: open the round in your browser, wait until it shows *live* games, then grab
> the id from the address bar.

---

## 2. Go live (one command)

Start the live bridge. This polls the round, computes the equity bar, and serves
the overlay + an `/sse` push endpoint on the one port:

```bash
uv run chess-equity broadcast --round <ROUND_ID> --serve-sse 8777
```

The process waits for the round to start, then streams events. Leave it running;
`Ctrl-C` stops it.

Useful additions:

| You want… | Add |
| --- | --- |
| Pin real player ratings (Maia-2's top band is a coarse `>2000`) | `--white-elo 2780 --black-elo 2810` |
| A different equity model | `--model maia2` (default is `baseline`; also `wdl-a`, `maia-rollout`, `maia-search`) |
| The plain (clock-blind) bar | `--no-clock-aware` (clock-aware is **on** by default) |
| A slower/faster poll | `--interval 2.0` (seconds between polls; default `2.0`) |
| A different port | `--serve-sse 9000` (then use that port in OBS) |

Other live sources (same `--serve-sse` mechanics):

```bash
# An arbitrary public PGN URL (e.g. a chess.com export):
uv run chess-equity broadcast --url https://example.com/game.pgn --serve-sse 8777

# A local PGN replayed move-by-move as if it were live (great for rehearsal):
uv run chess-equity broadcast --pgn mygame.pgn --serve-sse 8777
```

---

## 3. Point OBS at the feed

1. In OBS: **Sources → + → Browser**.
2. Set the **URL** to:

   ```
   http://localhost:8777/?src=/sse
   ```

   (Use whatever port you passed to `--serve-sse`.)
3. Set the source **size** to your scene — e.g. **1920 × 120** for a bottom strip.
4. Untick **"Shutdown source when not visible"** so the feed keeps running when you
   switch scenes.

The page background is transparent, so only the bar composites over your stream.

### Don't want to hand-edit the URL?

Open the **config page** in a browser:

```
http://localhost:8777/config.html
```

It's a setup form: pick the feed (live `/sse`, replay, or a custom URL), set
rating overrides and layout/theme/feature toggles, and it generates the exact
OBS browser-source URL to paste — and, for a live game, the matching
`chess-equity broadcast … --serve-sse` command to run.

---

## 4. Tune the look (query params)

Append these to the OBS URL (all optional). Full table in
[`overlay/README.md`](../overlay/README.md#config-query-params-all-optional).

| Param | Default | Meaning |
| --- | --- | --- |
| `src` | `./mock-game.json` | feed: `/sse`, a `ws[s]://` WebSocket, or a `.json` replay file |
| `layout` | `horizontal` | `horizontal` (names flank a wide bar) or `vertical` (classic eval-bar) |
| `pov` | `white` | whose POV the bar reads: `white` (classic, never flips), `stm` (adds a side-to-move readout pill, bar unchanged), or `stm-bar` (the whole bar flips to the player on move) — [details](../overlay/README.md#point-of-view-pov) |
| `theme` | `dark` | `dark` or `light` label text |
| `cp` | `1` | show the dashed **centipawn ghost tick** (`0` to hide) |
| `cpbar` | `0` | render centipawn as a full greyed second bar instead of a tick |
| `caster` | `0` | **caster mode** — flares on big practical swings the engine bar misses |
| `welo` / `belo` | _(feed)_ | override the White / Black rating shown |
| `lowclock` | `30` | time-pressure threshold (seconds) for the clock-red cue |
| `speed` | `1` | replay speed multiplier (`.json` feeds only) |

Examples:

```
http://localhost:8777/?src=/sse&layout=vertical&cp=0
http://localhost:8777/?src=/sse&caster=1&cpbar=1
```

> **Model badge.** The bar carries a small badge naming the model behind it (e.g.
> *Maia-2*) so viewers know it's a human win-probability model, not Stockfish. It's not
> a query param — the ingestor stamps it from `--model`, so `broadcast --model maia2`
> shows a *Maia-2* badge. See [the badge note](../overlay/README.md#bar-model-badge).

---

## Offline sanity check (no network)

Before you go live, prove the pipeline works with zero network using the bundled
sample PGN. It replays move-by-move and prints one JSON event per line:

```bash
uv run chess-equity broadcast --pgn data/sample/sample_games.pgn --interval 0 | head
```

You should see a `{"type": "game", …}` header followed by `{"type": "position", …}`
events carrying `equity`, `cp`, `clock`, and `grade`.

To see it **in the overlay** without any CLI, use the stdlib dev server, which
serves the static overlay plus an `/sse` endpoint that replays the bundled
`mock-game.json`:

```bash
python3 overlay/serve.py            # http://localhost:8777/
```

Then in OBS use `http://localhost:8777/` (bundled replay) or
`http://localhost:8777/?src=/sse` (replay pushed over SSE). The dev server takes
`--port` (default `8777`), `--game` (default `mock-game.json`), and `--speed`
(default `1.0`).

> `serve.py` is for **offline/dev**; for a real round use the one-command live
> bridge in [step 2](#2-go-live-one-command).

---

## Troubleshooting

**The bar is frozen / "feed dropped".**
Live broadcasts pause between games and rounds. The bridge keeps polling and
resumes when new moves arrive — give it a few seconds. If it never resumes, the
round may have ended; re-check the round id (step 1) and restart the command.
Also confirm the `chess-equity broadcast --serve-sse` process is still running
(it prints to the terminal it was launched in).

**Wrong board / wrong game.**
A broadcast round can contain several boards. The bridge follows the round's
games; if it's showing the wrong one, double-check you used the correct
`<ROUND_ID>` (the round, not the tournament, slug). For a fixed local game, use
`--pgn <file>` instead so there's no ambiguity.

**Blank / transparent bar in OBS.**
- Make sure the OBS source URL has `?src=/sse` (the live push), not just `/` —
  unless you're intentionally using the bundled replay.
- Confirm the port in the URL matches the `--serve-sse` port (or the `serve.py`
  `--port`).
- A transparent background is **correct** — if the whole source looks empty, add a
  temporary solid color scene behind it to confirm the bar is rendering, then
  remove it.
- Open the same URL in a normal browser tab to see errors; the overlay logs feed
  status to the browser console.

**Ratings look wrong (everyone shows `>2000`).**
The rating-conditioned model reports a coarse band. Pin the real numbers with
`--white-elo`/`--black-elo` on the command, or `welo`/`belo` on the OBS URL
(display-only override).

**Port already in use.**
Pick another port: `--serve-sse 9000`, then point OBS at
`http://localhost:9000/?src=/sse`.

---

## See also

- [`overlay/README.md`](../overlay/README.md) — the overlay front-end: full query
  param reference, layouts, caster mode, and tests.
- `uv run chess-equity broadcast --help` — the authoritative flag list.
