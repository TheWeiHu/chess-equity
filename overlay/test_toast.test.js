/* Node test for overlay.js's drama toast (task 0241).
 *
 * A dramatic move (the server's chess_equity.drama.score_event classification, carried
 * on the per-move feed as `evt.drama`) should flash a transient labelled toast over the
 * bar ('CLUTCH +12%', 'ESCAPE', 'MISSED WIN', 'SCRAMBLE') that then auto-hides as quiet
 * moves arrive. The pure brain is two helpers on `window.EquityOverlay`:
 *   - `dramaToast(drama)` — maps a server `drama` payload to a render spec
 *     {kind, label, side, magnitude}, or null when the move carries no drama kind.
 *   - `makeToastTracker(opts)` — a timer-free, tick-driven decay machine (mirrors
 *     makeMomentumTracker): a drama event sets the toast full-strength; each subsequent
 *     quiet move (drama undefined) decays it by 1/holdTicks until it clears.
 *
 * This loads the REAL overlay.js in a vm sandbox (no `document`, so it does NOT
 * auto-start) and exercises the helpers so a regression can't pass a hand-mirrored
 * re-implementation.
 *
 * Run: `node overlay/test_toast.test.js`. No deps.
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
assert.ok(O && typeof O.dramaToast === "function", "EquityOverlay.dramaToast must be exposed");
assert.ok(typeof O.makeToastTracker === "function", "EquityOverlay.makeToastTracker must be exposed");

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

// --- dramaToast(): map the server drama payload to a labelled render spec ------------
check("a clutch event labels CLUTCH with the signed swing", () => {
  const t = O.dramaToast({ kind: "clutch", delta_equity: 12, magnitude: 0.3, mover_white: true });
  assert.ok(t, "a classified drama event produces a toast");
  assert.strictEqual(t.kind, "clutch");
  assert.strictEqual(t.label, "CLUTCH +12%");
  assert.strictEqual(t.side, "white");
});

check("a missed_win event labels MISSED WIN (keyword only, the situation is the story)", () => {
  const t = O.dramaToast({ kind: "missed_win", delta_equity: -18, mover_white: false });
  assert.strictEqual(t.label, "MISSED WIN");
  assert.strictEqual(t.side, "black", "mover_white:false -> black");
});

check("an escape event labels ESCAPE", () => {
  const t = O.dramaToast({ kind: "escape", delta_equity: 30, mover_white: true });
  assert.strictEqual(t.label, "ESCAPE");
});

check("a scramble event labels SCRAMBLE with the signed swing", () => {
  const t = O.dramaToast({ kind: "scramble", delta_equity: -22, mover_white: false });
  assert.strictEqual(t.label, "SCRAMBLE -22%");
});

check("a quiet move (no drama / no kind / unknown kind) produces no toast", () => {
  assert.strictEqual(O.dramaToast(undefined), null);
  assert.strictEqual(O.dramaToast(null), null);
  assert.strictEqual(O.dramaToast({}), null, "no kind -> null");
  assert.strictEqual(O.dramaToast({ kind: "whatever" }), null, "unknown kind -> null, not garbage");
});

check("magnitude is taken from the payload, else normalized from the delta, clamped 0..1", () => {
  assert.ok(Math.abs(O.dramaToast({ kind: "clutch", magnitude: 0.42, delta_equity: 5 }).magnitude - 0.42) < 1e-9);
  // no magnitude field -> |delta| / 40
  assert.ok(Math.abs(O.dramaToast({ kind: "clutch", delta_equity: 20 }).magnitude - 0.5) < 1e-9);
  // out-of-range magnitude clamps
  assert.strictEqual(O.dramaToast({ kind: "escape", magnitude: 5, delta_equity: 50 }).magnitude, 1);
});

// --- makeToastTracker(): appears on a drama move, then clears on quiet moves ----------
check("a drama event sets the toast to full strength with its label", () => {
  const t = O.makeToastTracker({ holdTicks: 3 });
  const s = t.note({ kind: "clutch", delta_equity: 14, mover_white: true });
  assert.ok(s, "the drama event produces a render state");
  assert.strictEqual(s.label, "CLUTCH +14%");
  assert.strictEqual(s.strength, 1);
});

check("the toast holds (decaying) across quiet moves, then auto-hides after holdTicks", () => {
  const t = O.makeToastTracker({ holdTicks: 3 });
  t.note({ kind: "escape", delta_equity: 25, mover_white: false }); // strength 1
  let s = t.note(undefined); // quiet -> decay one step
  assert.ok(s, "still visible one quiet move after the drama");
  assert.ok(Math.abs(s.strength - (2 / 3)) < 1e-9);
  assert.strictEqual(s.label, "ESCAPE", "the standing label is held while it fades");
  t.note(null); // 1/3
  assert.strictEqual(t.note(undefined), null, "cleared after holdTicks quiet moves");
});

check("a quiet move right after clear stays cleared (no flicker back on)", () => {
  const t = O.makeToastTracker({ holdTicks: 2 });
  t.note({ kind: "clutch", delta_equity: 11, mover_white: true });
  t.note(undefined); // 0.5
  assert.strictEqual(t.note(undefined), null, "decayed to 0");
  assert.strictEqual(t.note(undefined), null, "stays null");
});

check("a fresh drama event re-arms the toast to full strength and swaps the label", () => {
  const t = O.makeToastTracker({ holdTicks: 4 });
  t.note({ kind: "clutch", delta_equity: 11, mover_white: true });
  t.note(undefined); // decaying
  const s = t.note({ kind: "missed_win", delta_equity: -20, mover_white: false });
  assert.strictEqual(s.strength, 1, "a fresh event re-arms full strength");
  assert.strictEqual(s.label, "MISSED WIN");
  assert.strictEqual(s.side, "black");
});

check("peek() reports the current state without advancing the decay", () => {
  const t = O.makeToastTracker({ holdTicks: 4 });
  t.note({ kind: "scramble", delta_equity: -15, mover_white: true });
  const a = t.peek();
  const b = t.peek();
  assert.deepStrictEqual(a, b, "peek is idempotent");
  assert.strictEqual(a.strength, 1);
});

if (failures) {
  console.log("\n" + failures + " test(s) FAILED");
  process.exit(1);
}
console.log("\nall drama-toast tests passed");
