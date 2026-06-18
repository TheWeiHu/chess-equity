# Decision: the web demo's objective bar stays material (Stockfish opt-in)

*Task 0082. Follows PR #74 (task 0074), which added a real-Stockfish cp source to
`web/build_demo.py`.*

## The fork

PR #74 made `--cp-engine stockfish` available in `web/build_demo.py`. That raised a
product question for the **committed, deployed** demo (Légal's Mate on GitHub Pages):

- **(a)** Keep the dependency-free **material count** as the centipawn bar, with
  Stockfish as an opt-in source.
- **(b)** Switch the deployed demo to **real Stockfish** and re-anchor the story on a
  positional game (fortress / long-term compensation / time pressure) that a deep
  engine does not pre-solve.

## Decision — (a)

The committed demo keeps the **material count** as its objective bar. `--cp-engine
stockfish` remains supported for anyone who wants a real engine locally, but the
checked-in `demo-game.json` is generated with material.

### Why

1. **The flagship contradiction only survives a shallow bar.** Légal's Mate is the
   whole point of the demo: White's queen "sacrifice" tanks a *material* count (red)
   while the rating-conditioned equity bar stays winning (green). A deep engine
   *solves* the mate at the queen-grab, so with Stockfish the dramatic green/red split
   collapses into a subtle practical-vs-objective gap. (b) would mean throwing away the
   one position the demo is built around.
2. **Deterministic + dependency-free fits a static page.** GitHub Pages serves
   `demo-game.json` with no backend. A material bar is reproducible regardless of
   Stockfish version, build host, or search depth; a Stockfish bar bakes a specific
   engine build's eval into a checked-in artifact.
3. **The scientific claim already lives elsewhere.** "Equity beats centipawns" is
   *proved* by the validation gate (`chess-equity validate`), which compares equity
   against the **real, rating-blind Stockfish** eval — not by the demo. The demo is a
   teaching foil, not the evidence.

## The framing this pins down

The demo's material bar is a deliberately **shallow** objective proxy, chosen to make
one tactic visible — it is **not** the deep engine, and beating it on Légal's Mate is
not the project's thesis.

> **"Equity beats centipawns"** means a rating-conditioned predictor beats a
> **rating-blind OBJECTIVE eval** (real Stockfish) at predicting **actual human
> outcomes** — i.e. in *practical* terms. It does **not** mean out-tactic-ing a deep
> engine on forced lines. A deep engine is right about the board; it is wrong about
> *this player against that player*.

## If we later want (b) too

Add a *second*, positional demo game alongside Légal's Mate (a fortress or a
long-term-compensation middlegame) built with `--cp-engine stockfish`, and let the page
switch between them. That is additive — it does not require giving up the flagship — so
it is deferred to a follow-up, not part of this decision.
