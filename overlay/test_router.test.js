/* Node test for overlay.js's live board router (task 0185).
 *
 * A multi-game broadcast round feeds every board's events down ONE stream. The overlay
 * must let a caster flip which board the bar follows, routing only the chosen board's
 * events to the DOM. This test loads overlay.js in a vm (no `document`, so it does NOT
 * auto-start — we only want the pure `makeBoardRouter`) and asserts:
 *   - a "boards" roster event populates the selector list (index + players);
 *   - selecting board 2 routes board-2's events and drops the others;
 *   - a single-game feed (events with no `board`) always routes — the default behavior.
 * Run: `node overlay/test_router.test.js`. No deps.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const HERE = __dirname;

// Minimal sandbox: no `document`, so overlay.js's auto-start guard is skipped and we
// load only the pure helpers exposed on `window.EquityOverlay`.
const sandbox = { window: {}, JSON, Math, String, parseInt, parseFloat, console };
sandbox.window.window = sandbox.window;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(HERE, "overlay.js"), "utf8"), sandbox);

const O = sandbox.window.EquityOverlay;
assert.ok(O && typeof O.makeBoardRouter === "function", "EquityOverlay.makeBoardRouter must be exposed");

let failures = 0;
function check(name, fn) {
  try {
    fn();
    console.log("PASS", name);
  } catch (e) {
    failures++;
    console.log("FAIL", name, "-", e.message);
  }
}

// The feed event listing 2 boards (the producer's `boards` roster event, task 0185).
const BOARDS_EVENT = {
  type: "boards",
  boards: [
    { index: 0, players: { white: { name: "Carlsen" }, black: { name: "Nakamura" } } },
    { index: 1, players: { white: { name: "Nepo" }, black: { name: "Ding" } } },
  ],
};
const posBoard0 = { type: "position", board: 0, ply: 10, equity: 0.6, cp: 30, clock: { white: 60, black: 55 } };
const posBoard1 = { type: "position", board: 1, ply: 10, equity: 0.4, cp: -40, clock: { white: 50, black: 48 } };

check("a boards roster event populates the board list (index + players)", () => {
  const r = O.makeBoardRouter();
  r.learn(BOARDS_EVENT);
  const boards = r.boards();
  assert.strictEqual(boards.length, 2, "two boards announced");
  assert.strictEqual(boards[0].index, 0);
  assert.strictEqual(boards[1].index, 1);
  assert.strictEqual(boards[1].players.white.name, "Nepo", "roster carries players for the selector label");
});

check("the first announced board is auto-selected (overlay isn't blank pre-pick)", () => {
  const r = O.makeBoardRouter();
  r.learn(BOARDS_EVENT);
  assert.strictEqual(r.selected(), 0, "first board auto-selected");
  assert.ok(r.accepts(posBoard0), "auto-selected board's events route");
  assert.ok(!r.accepts(posBoard1), "other board's events are dropped");
});

check("selecting board 2 routes its events and drops the others", () => {
  const r = O.makeBoardRouter();
  r.learn(BOARDS_EVENT);
  r.select(1); // "board 2" — the 0-based index 1
  assert.strictEqual(r.selected(), 1);
  assert.ok(r.accepts(posBoard1), "the chosen board's position events route");
  assert.ok(!r.accepts(posBoard0), "a non-selected board's position events are dropped");
});

check("a boards roster event is routing metadata — never rendered", () => {
  const r = O.makeBoardRouter();
  r.learn(BOARDS_EVENT);
  assert.ok(!r.accepts(BOARDS_EVENT), "the boards event itself must not reach the bar");
});

check("the router learns boards from per-game events too (no roster event needed)", () => {
  const r = O.makeBoardRouter();
  r.learn({ type: "game", board: 0, players: { white: { name: "A" }, black: { name: "B" } } });
  r.learn({ type: "game", board: 1, players: { white: { name: "C" }, black: { name: "D" } } });
  assert.strictEqual(r.boards().length, 2, "two boards discovered from game events");
  r.select(1);
  assert.ok(r.accepts(posBoard1) && !r.accepts(posBoard0));
});

check("single-game feed (no board field) always routes — default behavior", () => {
  const r = O.makeBoardRouter();
  const game = { type: "game", players: { white: { name: "A" }, black: { name: "B" } } };
  const pos = { type: "position", ply: 4, equity: 0.5, cp: 0, clock: { white: 60, black: 60 } };
  r.learn(game);
  assert.strictEqual(r.boards().length, 0, "no boards roster for a single game");
  assert.ok(r.accepts(game), "single-game events always route");
  assert.ok(r.accepts(pos), "single-game position events always route");
});

if (failures) {
  console.error(failures + " failure(s)");
  process.exit(1);
}
console.log("ok - board router");
