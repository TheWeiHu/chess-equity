/* Node test for the overlay's game-over card (task 0213).
 *
 * When a followed board's game ends, the bar would otherwise just freeze on the final
 * position with no signal it's over. The overlay now derives a clean end card from the
 * terminal `result` event ("1-0" / "0-1" / "1/2-1/2") plus the board's roster players.
 * This test loads overlay.js in a vm (no `document`, so it does NOT auto-start) and
 * asserts the PURE pieces that decide what the card says:
 *   - gameOverCard(result, players) maps each result to result/winner/headline;
 *   - a missing/draw/unknown result has no winner (no false "X wins");
 *   - the board router exposes playersFor(idx) so the renderer can name the winner.
 * Run: `node overlay/test_gameover.test.js`. No deps.
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
assert.ok(O && typeof O.gameOverCard === "function", "EquityOverlay.gameOverCard must be exposed");
assert.ok(typeof O.makeBoardRouter === "function", "EquityOverlay.makeBoardRouter must be exposed");

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

const PLAYERS = { white: { name: "Carlsen" }, black: { name: "Nakamura" } };

check("a White win (1-0) names White as the winner", () => {
  const c = O.gameOverCard("1-0", PLAYERS);
  assert.strictEqual(c.result, "1-0");
  assert.strictEqual(c.winnerSide, "white");
  assert.strictEqual(c.winnerName, "Carlsen");
  assert.strictEqual(c.headline, "Carlsen wins");
});

check("a Black win (0-1) names Black as the winner", () => {
  const c = O.gameOverCard("0-1", PLAYERS);
  assert.strictEqual(c.winnerSide, "black");
  assert.strictEqual(c.winnerName, "Nakamura");
  assert.strictEqual(c.headline, "Nakamura wins");
});

check("a draw (1/2-1/2) has no winner and reads 'Draw'", () => {
  const c = O.gameOverCard("1/2-1/2", PLAYERS);
  assert.strictEqual(c.winnerSide, null);
  assert.strictEqual(c.winnerName, null);
  assert.strictEqual(c.headline, "Draw");
});

check("an unknown/absent result never invents a winner", () => {
  const c = O.gameOverCard(undefined, PLAYERS);
  assert.strictEqual(c.winnerSide, null, "no side claimed on an unknown result");
  assert.strictEqual(c.winnerName, null);
  assert.strictEqual(c.headline, "Game over");
});

check("missing player names fall back to White/Black (no crash)", () => {
  const c = O.gameOverCard("1-0", null);
  assert.strictEqual(c.winnerName, "White", "winner falls back to the side label");
  assert.strictEqual(c.headline, "White wins");
});

// ---- router playersFor: names the winner for the followed board (0213) ----------
const BOARDS_EVENT = {
  type: "boards",
  boards: [
    { index: 0, players: { white: { name: "Carlsen" }, black: { name: "Nakamura" } } },
    { index: 1, players: { white: { name: "Nepo" }, black: { name: "Ding" } } },
  ],
};

check("playersFor(idx) returns the roster players for the card", () => {
  const r = O.makeBoardRouter();
  r.learn(BOARDS_EVENT);
  assert.strictEqual(r.playersFor(1).white.name, "Nepo", "board 1's white player");
  assert.strictEqual(r.playersFor(0).black.name, "Nakamura", "board 0's black player");
});

check("playersFor(unknown) is null so the card falls back to on-screen names", () => {
  const r = O.makeBoardRouter();
  r.learn(BOARDS_EVENT);
  assert.strictEqual(r.playersFor(9), null, "no roster entry → null");
});

check("end-to-end: the followed board's result cards its actual winner", () => {
  // board 0 followed; it ends 0-1 → the card should name board 0's Black (Nakamura),
  // sourced from the roster the same way the renderer does.
  const r = O.makeBoardRouter();
  r.learn(BOARDS_EVENT);
  assert.strictEqual(r.selected(), 0, "board 0 auto-followed");
  const players = r.playersFor(r.selected());
  const card = O.gameOverCard("0-1", players);
  assert.strictEqual(card.headline, "Nakamura wins");
});

if (failures) {
  console.error(failures + " failure(s)");
  process.exit(1);
}
console.log("ok - game-over card");
