/* Node test for overlay.js's focus-cut lower-third banner (task 0264).
 *
 * The server-side auto-director (`broadcast --board auto`, tasks 0256/0260) emits a
 * `focus` routing event the moment it cuts to a livelier board, carrying a caption-ready
 * `reason` cue ("cut to Bd3: +0.9 swing vs +0.4"). The overlay should turn that into a
 * brief lower-third banner so the auto-director's payoff is visible on the OBS source.
 *
 * The pure brain is one helper on `window.EquityOverlay`:
 *   - `focusBanner(evt)` — maps a `focus` event to a render spec {board, text}, or null
 *     when there's nothing to flash (not a focus event, or a routing-only focus with no
 *     `reason` — e.g. a caster pin cut or the silent opening adopt).
 *
 * This loads the REAL overlay.js in a vm sandbox (no `document`, so it does NOT
 * auto-start) and exercises the helper so a regression can't pass a hand-mirrored
 * re-implementation.
 *
 * Run: `node overlay/test_focus_banner.test.js`. No deps.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const HERE = __dirname;

// overlay.js is an IIFE ending in (typeof window !== "undefined" ? window : this). With
// `window` undefined in the sandbox, `this` (the sandbox global) receives EquityOverlay.
// The helper under test touches no browser globals.
const sandbox = { JSON, Math, console, URLSearchParams };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(HERE, "overlay.js"), "utf8"), sandbox);

const O = sandbox.EquityOverlay;
assert.ok(O && typeof O.focusBanner === "function", "EquityOverlay.focusBanner must be exposed");

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

// --- focusBanner(): a drama cut carries a reason → banner ------------------------------
check("a drama cut event yields a banner with the director's reason verbatim", () => {
  const b = O.focusBanner({
    type: "focus",
    board: 2,
    game_id: "g3",
    reason: "cut to Bd3: +0.9 swing vs +0.4",
  });
  assert.ok(b, "a focus event with a reason produces a banner");
  assert.strictEqual(b.board, 2);
  assert.strictEqual(b.text, "cut to Bd3: +0.9 swing vs +0.4", "reason is shown verbatim (caption-ready)");
});

check("a caster-pin focus reason also banners (it explains the cut too)", () => {
  const b = O.focusBanner({ type: "focus", board: 0, reason: "caster pin: hold Bd1" });
  assert.ok(b);
  assert.strictEqual(b.text, "caster pin: hold Bd1");
});

// --- routing-only / non-focus events produce no banner --------------------------------
check("a routing-only focus event (no reason) produces no banner", () => {
  // The pin INPUT channel emits `{type:focus, board}` with NO reason — pure re-route.
  assert.strictEqual(O.focusBanner({ type: "focus", board: 1 }), null);
});

check("a focus event with an empty/blank reason produces no banner", () => {
  assert.strictEqual(O.focusBanner({ type: "focus", board: 1, reason: "" }), null);
  assert.strictEqual(O.focusBanner({ type: "focus", board: 1, reason: "   " }), null);
});

check("a non-string reason produces no banner (defensive)", () => {
  assert.strictEqual(O.focusBanner({ type: "focus", board: 1, reason: 42 }), null);
  assert.strictEqual(O.focusBanner({ type: "focus", board: 1, reason: null }), null);
});

check("non-focus events and junk produce no banner", () => {
  assert.strictEqual(O.focusBanner({ type: "position", reason: "x" }), null);
  assert.strictEqual(O.focusBanner({ type: "result", reason: "x" }), null);
  assert.strictEqual(O.focusBanner(undefined), null);
  assert.strictEqual(O.focusBanner(null), null);
  assert.strictEqual(O.focusBanner({}), null);
});

check("a focus reason with a non-numeric board still banners with board=null", () => {
  const b = O.focusBanner({ type: "focus", reason: "cut to Bd2: +0.7 swing vs +0.2" });
  assert.ok(b, "reason is the gate, not the board field");
  assert.strictEqual(b.board, null);
  assert.strictEqual(b.text, "cut to Bd2: +0.7 swing vs +0.2");
});

check("reason is trimmed of surrounding whitespace", () => {
  const b = O.focusBanner({ type: "focus", board: 1, reason: "  cut to Bd2: +0.5 swing vs +0.1  " });
  assert.strictEqual(b.text, "cut to Bd2: +0.5 swing vs +0.1");
});

if (failures) {
  console.log("\n" + failures + " test(s) FAILED");
  process.exit(1);
}
console.log("\nall focus-banner tests passed");
