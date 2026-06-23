/* Node test for overlay.js's equity POV mapping (task 0206).
 *
 * The overlay can show equity from two points of view, selected by ?pov=:
 *   - white (default): the classic always-White bar — the value is White's win
 *     equity regardless of whose turn it is.
 *   - stm: side-to-move — the value flips to (1 - eq) when Black is on the move, so
 *     the readout always measures the player about to move.
 *
 * Both mappings are the pure `EquityOverlay.orient(equityWhite, pov, whiteToMove)`
 * helper, plus `whiteToMove(evt)` which resolves whose turn it is (explicit field, or
 * ply parity). This loads the REAL overlay.js in a vm sandbox and exercises them so a
 * regression can't pass a hand-mirrored re-implementation.
 *
 * Run: `node overlay/test_pov.test.js`. No deps.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const HERE = __dirname;

// overlay.js is an IIFE ending in (typeof window !== "undefined" ? window : this).
// With `window` undefined in the sandbox, `this` (the sandbox global) receives
// EquityOverlay. The pure helpers under test touch no browser globals, but overlay.js
// references `document`/`URLSearchParams` at module scope guarded by typeof checks, so a
// bare sandbox is enough — there's no top-level DOM access.
const sandbox = { JSON, Math, console, URLSearchParams };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(HERE, "overlay.js"), "utf8"), sandbox);

const O = sandbox.EquityOverlay;
assert.ok(O && typeof O.orient === "function", "EquityOverlay.orient must be exposed");
assert.ok(typeof O.whiteToMove === "function", "EquityOverlay.whiteToMove must be exposed");

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

// --- white-POV: the classic bar — always White, side-to-move is ignored ----------
check("white-POV returns White equity regardless of side to move", () => {
  assert.strictEqual(O.orient(0.7, "white", true), 0.7);
  assert.strictEqual(O.orient(0.7, "white", false), 0.7);
  assert.strictEqual(O.orient(0.3, "white", false), 0.3);
});

// --- stm-POV: flips sign relative to white-POV when Black is to move --------------
check("stm-POV with White to move equals the white-POV value", () => {
  assert.strictEqual(O.orient(0.7, "stm", true), 0.7);
  assert.strictEqual(O.orient(0.42, "stm", true), 0.42);
});

check("stm-POV with Black to move flips the white-POV value (1 - eq)", () => {
  assert.ok(Math.abs(O.orient(0.7, "stm", false) - 0.3) < 1e-12);
  assert.ok(Math.abs(O.orient(0.3, "stm", false) - 0.7) < 1e-12);
  // The defining property: a stm value is the white-POV value, sign-flipped on Black's move.
  const eq = 0.62;
  assert.ok(Math.abs(O.orient(eq, "stm", false) - (1 - O.orient(eq, "white", false))) < 1e-12);
});

check("orient clamps out-of-range equity before mapping", () => {
  assert.strictEqual(O.orient(1.5, "white", true), 1);
  assert.strictEqual(O.orient(-0.2, "white", true), 0);
  assert.strictEqual(O.orient(1.5, "stm", false), 0); // 1 - clamp(1.5)=1 -> 0
});

// --- whiteToMove: explicit field wins; else ply parity (White moves on odd plies) -
check("whiteToMove honors an explicit white_to_move field", () => {
  assert.strictEqual(O.whiteToMove({ white_to_move: false }), false);
  assert.strictEqual(O.whiteToMove({ white_to_move: true }), true);
});

check("whiteToMove honors an explicit stm string field", () => {
  assert.strictEqual(O.whiteToMove({ stm: "black" }), false);
  assert.strictEqual(O.whiteToMove({ stm: "white" }), true);
});

check("whiteToMove derives from ply parity (after an even ply, White is to move)", () => {
  assert.strictEqual(O.whiteToMove({ ply: 30 }), true);   // even -> White to move
  assert.strictEqual(O.whiteToMove({ ply: 31 }), false);  // odd  -> Black to move
});

check("whiteToMove defaults to White when nothing indicates the side", () => {
  assert.strictEqual(O.whiteToMove({}), true);
  assert.strictEqual(O.whiteToMove(null), true);
});

// --- the two mappings genuinely differ for the same position (Black to move) ------
check("the white and stm mappings differ when Black is to move", () => {
  const eq = 0.74; // White clearly ahead
  const black = O.whiteToMove({ ply: 41 }); // odd -> Black to move
  assert.strictEqual(black, false);
  assert.notStrictEqual(O.orient(eq, "white", black), O.orient(eq, "stm", black));
});

if (failures) {
  console.log("\n" + failures + " test(s) FAILED");
  process.exit(1);
}
console.log("\nall POV tests passed");
