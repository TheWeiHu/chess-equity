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

// ---- auto-director (task 0188): drama-driven autofollow ----------------------
// Events carry a server drama payload (broadcast.to_overlay_event -> drama.magnitude,
// 0..1). With autofollow on, the router steals focus to the most-dramatic board, held
// by a focus lock so noise can't thrash it; a manual select pins and overrides.
function pos(board, mag) {
  return { type: "position", board: board, ply: 10, equity: 0.5, cp: 0, drama: { magnitude: mag } };
}

check("(a) a higher-drama board event steals focus under autofollow", () => {
  const r = O.makeBoardRouter({ autofollow: true, lockPlies: 3 });
  r.learn(BOARDS_EVENT); // board 0 auto-selected, unlocked
  r.note(pos(0, 0.1)); // a quiet event on the followed board
  assert.strictEqual(r.selected(), 0, "still on board 0 before any bigger swing");
  r.note(pos(1, 0.9)); // board 1 erupts
  assert.strictEqual(r.selected(), 1, "the higher-drama board steals focus");
  assert.ok(r.accepts(pos(1, 0.9)), "the dramatic board's own event now routes");
});

check("(b) the focus lock prevents an immediate re-switch", () => {
  const r = O.makeBoardRouter({ autofollow: true, lockPlies: 3 });
  r.learn(BOARDS_EVENT);
  r.note(pos(1, 0.9)); // steal to board 1, lock = 3
  assert.strictEqual(r.selected(), 1);
  r.note(pos(0, 0.99)); // board 0 is even hotter, but the lock holds...
  assert.strictEqual(r.selected(), 1, "lock blocks the re-switch (tick 1)");
  r.note(pos(0, 0.99));
  assert.strictEqual(r.selected(), 1, "lock blocks the re-switch (tick 2)");
  r.note(pos(0, 0.99));
  assert.strictEqual(r.selected(), 1, "lock blocks the re-switch (tick 3)");
  r.note(pos(0, 0.99)); // lock expired — now the bigger swing wins
  assert.strictEqual(r.selected(), 0, "after the lock expires a real swing takes over");
});

check("(c) a manual select pins the board and disables autofollow", () => {
  const r = O.makeBoardRouter({ autofollow: true, lockPlies: 3 });
  r.learn(BOARDS_EVENT);
  r.select(0); // caster pins board 0
  assert.strictEqual(r.pinned(), true);
  assert.strictEqual(r.autofollow(), false, "autofollow is disabled while pinned");
  r.note(pos(1, 1.0)); // a maximal swing elsewhere must NOT steal focus
  assert.strictEqual(r.selected(), 0, "manual pin overrides the auto-director");
  r.resume(); // reset re-enables autofollow
  assert.strictEqual(r.pinned(), false);
  r.note(pos(1, 1.0));
  assert.strictEqual(r.selected(), 1, "after resume the director follows drama again");
});

check("autofollow is inert without the flag (default routing preserved)", () => {
  const r = O.makeBoardRouter(); // no autofollow
  r.learn(BOARDS_EVENT);
  r.note(pos(1, 1.0));
  assert.strictEqual(r.selected(), 0, "no autofollow → focus stays where learn put it");
  assert.strictEqual(r.autofollow(), false);
});

// ---- auto-advance off a finished board (task 0189) ---------------------------
// The bridge emits a `result` event ({type:"result", board}) when a game ends. The
// router marks that board finished and, if it was the followed board (and not pinned),
// advances focus to the next still-live board so a caster isn't stuck on an ended game.
function result(board) {
  return { type: "result", board: board, game_id: "g" + board, result: "1-0" };
}

check("(0189-a) when the followed board finishes, focus advances to the next live board", () => {
  const r = O.makeBoardRouter();
  r.learn(BOARDS_EVENT); // board 0 auto-selected
  assert.strictEqual(r.selected(), 0);
  r.learn(result(0)); // board 0's game ends
  assert.strictEqual(r.selected(), 1, "focus moved off the finished board to the live one");
  assert.ok(r.accepts(posBoard1), "the now-followed live board's events route");
  assert.ok(!r.accepts(posBoard0), "the finished board's events no longer route");
});

check("(0189-b) a `result` event is routing metadata — never rendered", () => {
  const r = O.makeBoardRouter();
  r.learn(BOARDS_EVENT);
  assert.ok(!r.accepts(result(0)), "the result event itself must not reach the bar");
});

check("(0189-c) a manually-pinned board does NOT auto-advance when it finishes", () => {
  const r = O.makeBoardRouter();
  r.learn(BOARDS_EVENT);
  r.select(0); // caster pins board 0
  assert.strictEqual(r.pinned(), true);
  r.learn(result(0)); // board 0 ends — but it's pinned
  assert.strictEqual(r.selected(), 0, "a pinned board stays put even after it finishes");
});

check("(0189-d) focus stays put when every board has finished", () => {
  const r = O.makeBoardRouter();
  r.learn(BOARDS_EVENT);
  r.learn(result(1)); // the other board finishes first
  assert.strictEqual(r.selected(), 0, "finishing a non-followed board doesn't move focus");
  r.learn(result(0)); // now the followed board finishes too — no live board left
  assert.strictEqual(r.selected(), 0, "with no live board to move to, focus stays put");
});

check("(0189-e) auto-advance composes with autofollow, and the director won't steal back to a finished board", () => {
  const r = O.makeBoardRouter({ autofollow: true, lockPlies: 3 });
  r.learn(BOARDS_EVENT); // board 0 followed
  r.note(pos(1, 0.9)); // board 1 erupts → steal to 1, lock engaged
  assert.strictEqual(r.selected(), 1);
  r.learn(result(1)); // board 1's game ends → advance back to live board 0, lock cleared
  assert.strictEqual(r.selected(), 0, "finished board hands focus to the remaining live one");
  r.note(pos(1, 1.0)); // a late max-drama event on the FINISHED board must not steal focus
  assert.strictEqual(r.selected(), 0, "the auto-director never returns focus to a finished board");
});

if (failures) {
  console.error(failures + " failure(s)");
  process.exit(1);
}
console.log("ok - board router");
