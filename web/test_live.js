/* Node test for live.js's per-ply eval cache (task 0123).
 *
 * The reported bug: scrubbing the live demo's move slider re-ran every Stockfish + Maia-2
 * eval from scratch instead of reusing the per-ply cache. live.js stamps each node's resp
 * with the ratings it was computed at (resp._we/_be) and hasFresh()/ensureEval() are meant
 * to skip the network when a ply is already fresh. This test loads live.js in a vm with
 * minimal DOM + ChessBoard + a synchronous fetch stub that COUNTS /api/play POSTs, fully
 * charts a small game, then asserts:
 *   - scrubbing across the already-charted game makes ZERO new /api/play calls;
 *   - changing a rating re-evals (cache is correctly rating-aware), then re-scrubbing at
 *     the new ratings is free again.
 * Run: `node web/test_live.js`. No deps.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const HERE = __dirname;
const START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// ---- synchronous promise-ish so postPlay()/postGet() callbacks fire inline ----
function resolved(v) {
  return {
    then(onF) {
      let out;
      try { out = onF ? onF(v) : v; } catch (e) { return resolved(undefined); }
      return out && typeof out.then === "function" ? out : resolved(out);
    },
    catch() { return this; },
  };
}

// Count POSTs to /api/play; everything else (GET /api/games) returns empty.
let playPosts = 0;
function fetchStub(url, opts) {
  const method = (opts && opts.method) || "GET";
  if (url === "/api/play" && method === "POST") {
    playPosts++;
    const body = JSON.parse(opts.body);
    // Echo a minimal, valid eval for the requested FEN (post-move when uci is present we
    // just reuse the same fen string — the test never inspects board legality).
    const resp = {
      fen: body.fen || START, turn: "white", legal: {}, check: false,
      checkmate: false, stalemate: false, game_over: false, cp: 0,
      equity_white: 50, san: body.uci ? "x" : null,
    };
    return resolved({ ok: true, json: () => resolved(resp) });
  }
  // /api/games library: empty so the page doesn't try to load a famous game.
  return resolved({ ok: true, json: () => resolved({ games: [] }) });
}

// ---- minimal DOM + ChessBoard --------------------------------------------------
function fakeEl() {
  return {
    style: {}, dataset: {}, value: "1500", textContent: "", innerHTML: "", hidden: false,
    className: "", max: 0, disabled: false,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    addEventListener() {}, removeEventListener() {},
    appendChild(c) { return c; }, removeChild() {}, remove() {},
    querySelector() { return null; }, querySelectorAll() { return []; },
    setAttribute() {}, getAttribute() { return null; }, scrollIntoView() {},
    getBoundingClientRect() { return { left: 0, top: 0, width: 0, height: 0 }; },
    focus() {}, select() {}, animate() {},
  };
}
const ChessBoard = {
  FILES: "abcdefgh",
  parseFen(fen) { return { grid: [], turn: (fen.split(" ")[1] || "w") }; },
  render() {}, sqName(r, c) { return "abcdefgh"[c] + (8 - r); }, glyph() { return ""; },
};
const sandbox = {
  window: { ChessBoard },
  document: {
    getElementById() { return fakeEl(); },
    createElement() { return fakeEl(); },
    createElementNS() { return fakeEl(); },
    querySelector() { return fakeEl(); },
    querySelectorAll() { return []; },
    addEventListener() {},
  },
  ChessBoard,
  fetch: fetchStub,
  setTimeout(fn) { fn(); },   // run the background-fill steps inline
  JSON, Math, String, parseInt, console,
};
sandbox.window.window = sandbox.window;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(HERE, "live.js"), "utf8"), sandbox);

const live = sandbox.window.ChessEquityLive;
assert.ok(live && typeof live.goPly === "function", "ChessEquityLive must be exposed");

let failures = 0;
function check(name, fn) {
  try { fn(); console.log("PASS", name); }
  catch (e) { failures++; console.log("FAIL", name, "-", e.message); }
}

// A small "game": 6 plies (start + 5 moves). FENs need only be distinct strings; the
// stub doesn't validate them, and live.js keys the cache on node identity + ratings.
function fakeGame() {
  const moves = [{ fen: START, san: "(start)" }];
  for (let i = 1; i <= 5; i++) {
    moves.push({ fen: START + " ply" + i, san: "m" + i, uci: "e2e4" });
  }
  return { name: "Test", white: "W", black: "B", year: 0, moves: moves };
}

check("loading + charting a game evaluates every ply once", () => {
  playPosts = 0;
  live.setGame(fakeGame());                       // goPly(0) + startFill() chart all plies
  assert.strictEqual(live.state.line.length, 6, "line should hold 6 nodes");
  assert.ok(live.state.line.every(live.hasFresh), "every node should be freshly evaluated");
  assert.strictEqual(playPosts, 6, "expected one /api/play per ply, got " + playPosts);
});

check("scrubbing across the charted game makes ZERO new /api/play calls", () => {
  playPosts = 0;
  for (let pass = 0; pass < 2; pass++) {           // sweep right then left, twice
    for (let p = 0; p < live.state.line.length; p++) live.goPly(p);
    for (let p = live.state.line.length - 1; p >= 0; p--) live.goPly(p);
  }
  assert.strictEqual(playPosts, 0, "scrubbing must hit the cache, but made " + playPosts + " call(s)");
});

check("changing a rating re-evals, then re-scrubbing at it is free again", () => {
  playPosts = 0;
  live.setRatings(1500, 2300);                     // new ratings → cache is stale by design
  assert.ok(playPosts > 0, "a rating change must re-score (got 0 calls)");
  assert.ok(live.state.line.every(live.hasFresh), "all plies should be re-charted at the new ratings");
  playPosts = 0;
  for (let p = 0; p < live.state.line.length; p++) live.goPly(p);
  assert.strictEqual(playPosts, 0, "scrubbing at the new ratings must also be cached, got " + playPosts);
});

// --- rating slider: a drag must not re-chart the whole game on every tick (task 0123) ---

check("a rating-drag tick re-scores only the current ply, not the whole game", () => {
  live.setGame(fakeGame());            // freshly charted at 1500/1500, cursor at ply 0
  playPosts = 0;
  live.ratingInput(1500, 1900);        // one intermediate drag tick to a brand-new rating
  assert.strictEqual(playPosts, 1, "a tick must re-score just the current ply, got " + playPosts);
  const fresh = live.state.line.filter(live.hasFresh).length;
  assert.ok(fresh < live.state.line.length,
    "the rest of the game must wait for the drag to settle, but " + fresh + "/" +
    live.state.line.length + " plies were re-charted mid-drag");
});

check("releasing the slider re-charts the rest of the game exactly once", () => {
  playPosts = 0;
  live.ratingCommit();                 // the `change` event when the drag settles
  assert.ok(live.state.line.every(live.hasFresh), "settle must re-chart every ply");
  assert.ok(playPosts >= 1 && playPosts <= live.state.line.length,
    "settle should chart the remaining plies once, got " + playPosts);
});

if (failures) { console.error(failures + " failure(s)"); process.exit(1); }
console.log("ok - live eval cache");
