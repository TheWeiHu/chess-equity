/* Node test for overlay.js's momentum cue (task 0208).
 *
 * The momentum arrow shows the DIRECTION and SIZE of the last equity swing so a viewer
 * can see who just gained at a glance, then fades over the next few moves. The pure brain
 * is two helpers on `window.EquityOverlay`:
 *   - `momentum(prevEq, eq, opts)` — the swing's side/delta/magnitude from the event delta
 *     (no new feed field), or null when negligible / not enough history.
 *   - `makeMomentumTracker(opts)` — a timer-free, tick-driven decay state machine (mirrors
 *     feed.js makeStaleTracker): a fresh swing sets the arrow full-strength; quiet moves
 *     decay it by 1/decayTicks until it clears.
 *
 * This loads the REAL overlay.js in a vm sandbox and exercises them so a regression can't
 * pass a hand-mirrored re-implementation.
 *
 * Run: `node overlay/test_momentum.test.js`. No deps.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const HERE = __dirname;

// overlay.js is an IIFE ending in (typeof window !== "undefined" ? window : this). With
// `window` undefined in the sandbox, `this` (the sandbox global) receives EquityOverlay.
// The helpers under test touch no browser globals.
const sandbox = { JSON, Math, console, URLSearchParams };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(HERE, "overlay.js"), "utf8"), sandbox);

const O = sandbox.EquityOverlay;
assert.ok(O && typeof O.momentum === "function", "EquityOverlay.momentum must be exposed");
assert.ok(typeof O.makeMomentumTracker === "function", "EquityOverlay.makeMomentumTracker must be exposed");

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

// --- momentum(): direction + magnitude from the event delta ------------------------
check("momentum points toward White when equity rises", () => {
  const m = O.momentum(0.50, 0.62);
  assert.ok(m, "a 12pt swing is not negligible");
  assert.strictEqual(m.side, "white");
  assert.ok(Math.abs(m.delta - 0.12) < 1e-9);
});

check("momentum points toward Black when equity falls", () => {
  const m = O.momentum(0.62, 0.50);
  assert.ok(m);
  assert.strictEqual(m.side, "black");
  assert.ok(m.delta < 0);
});

check("momentum is null when the swing is negligible (< minDelta)", () => {
  assert.strictEqual(O.momentum(0.50, 0.51), null, "1pt < 2pt default is negligible");
  // ...and the threshold is tunable.
  assert.ok(O.momentum(0.50, 0.51, { minDelta: 0.005 }), "lower minDelta makes 1pt show");
});

check("momentum is null without two real numbers (first move / bad input)", () => {
  assert.strictEqual(O.momentum(null, 0.6), null);
  assert.strictEqual(O.momentum(0.6, NaN), null);
});

check("momentum magnitude is normalized to 0..1 and clamps at a big swing", () => {
  const small = O.momentum(0.50, 0.60); // 0.10 / 0.40 = 0.25
  assert.ok(Math.abs(small.magnitude - 0.25) < 1e-9);
  const huge = O.momentum(0.10, 0.95); // way past 0.40 -> clamps to 1
  assert.strictEqual(huge.magnitude, 1);
});

check("momentum clamps out-of-range equity before differencing", () => {
  // prev clamps to 1, eq clamps to 0 -> delta -1, side black
  const m = O.momentum(1.5, -0.2);
  assert.strictEqual(m.side, "black");
  assert.ok(Math.abs(m.delta + 1) < 1e-9);
});

// --- makeMomentumTracker(): a fresh swing, then decay over quiet moves --------------
check("a fresh swing sets the arrow to full strength in its direction", () => {
  const t = O.makeMomentumTracker({ decayTicks: 4 });
  assert.strictEqual(t.note(null, 0.50), null, "first move has no prior -> nothing to show");
  const s = t.note(0.50, 0.64);
  assert.ok(s, "the swing produces a render state");
  assert.strictEqual(s.side, "white");
  assert.strictEqual(s.strength, 1);
});

check("a quiet move decays the standing arrow by 1/decayTicks and clears after N", () => {
  const t = O.makeMomentumTracker({ decayTicks: 4 });
  t.note(null, 0.50);
  t.note(0.50, 0.64); // strength 1, side white
  let s = t.note(0.64, 0.645); // quiet (< 2pt) -> decay one step
  assert.ok(s, "still visible after one quiet move");
  assert.ok(Math.abs(s.strength - 0.75) < 1e-9);
  assert.strictEqual(s.side, "white", "the standing direction is held while it fades");
  t.note(0.645, 0.646); // 0.50
  t.note(0.646, 0.647); // 0.25
  assert.strictEqual(t.note(0.647, 0.648), null, "cleared after decayTicks quiet moves");
});

check("a new swing in the other direction resets to full strength and flips side", () => {
  const t = O.makeMomentumTracker({ decayTicks: 4 });
  t.note(null, 0.50);
  t.note(0.50, 0.70); // white, strength 1
  t.note(0.70, 0.701); // decay to 0.75
  const s = t.note(0.701, 0.40); // big drop toward black
  assert.strictEqual(s.side, "black");
  assert.strictEqual(s.strength, 1, "a fresh swing re-arms full strength");
});

check("peek() reports the current state without advancing the decay", () => {
  const t = O.makeMomentumTracker({ decayTicks: 4 });
  t.note(null, 0.50);
  t.note(0.50, 0.70);
  const a = t.peek();
  const b = t.peek();
  assert.deepStrictEqual(a, b, "peek is idempotent");
  assert.strictEqual(a.strength, 1);
});

if (failures) {
  console.log("\n" + failures + " test(s) FAILED");
  process.exit(1);
}
console.log("\nall momentum tests passed");
