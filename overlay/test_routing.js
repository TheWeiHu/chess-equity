/* Node test for the overlay's live multi-board router (task 0185).
 *
 * A live broadcast round can carry many boards. The feed announces them with a
 * `{type:"boards", boards:[...]}` event; overlay.js's `makeBoardRouter()` decides
 * which board's `game`/`position` events reach the bar. This test loads overlay.js in
 * a vm (no DOM, so its auto-start is skipped) and asserts the routing contract:
 *   - with no roster (single board) every event is accepted — unchanged default;
 *   - once a 2-board roster arrives it defaults to board 0;
 *   - selecting board 2 routes board 2's events and drops board 1's (the headline);
 *   - events without a game_id (legacy feeds) are never silenced.
 * Run: `node overlay/test_routing.js`. No deps.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const HERE = __dirname;
// Load overlay.js with a `window` but NO `document`, so the IIFE attaches
// EquityOverlay to window and its DOMContentLoaded auto-start never fires.
const sandbox = { window: {}, console: { warn() {} } };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(HERE, "overlay.js"), "utf8"), sandbox, {
  filename: "overlay.js",
});
const { makeBoardRouter } = sandbox.window.EquityOverlay;
assert.ok(typeof makeBoardRouter === "function", "overlay.js must expose makeBoardRouter");

const B0 = { index: 0, game_id: "g0", white: "Carlsen", black: "Nakamura" };
const B1 = { index: 1, game_id: "g1", white: "Caruana", black: "Firouzja" };
const posOn = (gid) => ({ type: "position", game_id: gid, ply: 1, equity: 0.5 });

function test_single_board_accepts_everything() {
  const r = makeBoardRouter();
  // No roster announced yet: every event is accepted (default single-board behaviour).
  assert.strictEqual(r.accepts(posOn("g0")), true);
  assert.strictEqual(r.accepts(posOn("anything")), true);
  assert.strictEqual(r.accepts({ type: "position" }), true); // missing game_id
  assert.strictEqual(r.selected(), null);
}

function test_roster_defaults_to_board_zero() {
  const r = makeBoardRouter();
  r.setRoster([B0, B1]);
  assert.strictEqual(r.selected(), "g0", "a fresh roster defaults to the first board");
  assert.strictEqual(r.accepts(posOn("g0")), true, "board 0's events reach the bar");
  assert.strictEqual(r.accepts(posOn("g1")), false, "other boards are filtered out");
}

function test_selecting_board_two_routes_its_events() {
  const r = makeBoardRouter();
  r.setRoster([B0, B1]);
  // The headline: a caster flips to board 2 (g1) and its events now route through,
  // while board 1's (g0) stop.
  r.select("g1");
  assert.strictEqual(r.selected(), "g1");
  assert.strictEqual(r.accepts(posOn("g1")), true, "selected board's events route");
  assert.strictEqual(r.accepts(posOn("g0")), false, "the deselected board is dropped");

  // Concretely: stream a mix and confirm only the selected board's events pass.
  const stream = [posOn("g0"), posOn("g1"), posOn("g0"), posOn("g1")];
  const routed = stream.filter((e) => r.accepts(e));
  assert.strictEqual(routed.length, 2);
  assert.ok(routed.every((e) => e.game_id === "g1"), "only board 2's events survive routing");
}

function test_legacy_events_without_game_id_never_silenced() {
  const r = makeBoardRouter();
  r.setRoster([B0, B1]);
  r.select("g1");
  // An older feed that doesn't tag events with game_id must still render.
  assert.strictEqual(r.accepts({ type: "position", equity: 0.4 }), true);
  assert.strictEqual(r.accepts({ type: "game", players: {} }), true);
}

function test_empty_roster_is_a_noop() {
  const r = makeBoardRouter();
  r.setRoster([]);
  assert.strictEqual(r.selected(), null, "an empty roster selects nothing");
  assert.strictEqual(r.accepts(posOn("g0")), true);
}

const tests = [
  test_single_board_accepts_everything,
  test_roster_defaults_to_board_zero,
  test_selecting_board_two_routes_its_events,
  test_legacy_events_without_game_id_never_silenced,
  test_empty_roster_is_a_noop,
];
let failures = 0;
for (const t of tests) {
  try {
    t();
    console.log("PASS", t.name);
  } catch (e) {
    failures++;
    console.log("FAIL", t.name, "-", e.message);
  }
}
process.exit(failures ? 1 : 0);
