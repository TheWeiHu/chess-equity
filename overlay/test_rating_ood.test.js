/* Node test for overlay.js's out-of-distribution high-rating read (task 0255).
 *
 * The pipeline emits a plain `rating_ood` boolean (chess_equity.broadcast.is_rating_ood:
 * both ratings over the coarse `>2000` Maia-2 bucket). The overlay marks the bar
 * lower-confidence iff that boolean is strictly true — the threshold decision lives on the
 * server, so the overlay's read is the pure `EquityOverlay.ratingOod(flag)` boolean,
 * exercised here against the REAL overlay.js so a regression can't pass a hand-mirrored
 * re-implementation.
 *
 * Run: `node overlay/test_rating_ood.test.js`. No deps.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const HERE = __dirname;
const sandbox = { JSON, Math, console, URLSearchParams };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(HERE, "overlay.js"), "utf8"), sandbox);

const O = sandbox.EquityOverlay;
assert.ok(
  O && typeof O.ratingOod === "function",
  "EquityOverlay.ratingOod must be exposed"
);

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

check("rating_ood:true marks the bar", () => {
  assert.strictEqual(O.ratingOod(true), true);
});

check("rating_ood:false (in distribution) leaves the bar clean", () => {
  assert.strictEqual(O.ratingOod(false), false);
});

check("missing field (older feed) degrades to no marker", () => {
  assert.strictEqual(O.ratingOod(undefined), false);
  assert.strictEqual(O.ratingOod(null), false);
});

check("a truthy-but-non-true value is not honored (strict boolean)", () => {
  assert.strictEqual(O.ratingOod(1), false);
  assert.strictEqual(O.ratingOod("yes"), false);
});

if (failures) {
  console.log("\n" + failures + " test(s) FAILED");
  process.exit(1);
}
console.log("\nall rating-ood tests passed");
