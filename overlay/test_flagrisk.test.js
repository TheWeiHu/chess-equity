/* Node test for overlay.js's flag-risk alert read (task 0243).
 *
 * The pipeline emits a per-side `flag_risk: {white:{risk,alert}, black:{risk,alert}}`
 * block (chess_equity.clock.flag_risk + is_flag_risk_alert); the overlay lights a 🚩
 * time-trouble badge for a side iff that side's `.alert` is true. The decision lives on
 * the server (it applies the threshold), so the overlay's read is the pure
 * `EquityOverlay.flagRiskAlert(side)` boolean — exercised here against the REAL overlay.js
 * so a regression can't pass a hand-mirrored re-implementation.
 *
 * Run: `node overlay/test_flagrisk.test.js`. No deps.
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
  O && typeof O.flagRiskAlert === "function",
  "EquityOverlay.flagRiskAlert must be exposed"
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

check("alert:true lights the badge", () => {
  assert.strictEqual(O.flagRiskAlert({ risk: 0.51, alert: true }), true);
});

check("alert:false (comfortable clock) stays dark", () => {
  assert.strictEqual(O.flagRiskAlert({ risk: 0.0, alert: false }), false);
});

check("clock-blind / missing side never alerts", () => {
  // null, undefined, and an empty object all degrade to no badge.
  assert.strictEqual(O.flagRiskAlert(null), false);
  assert.strictEqual(O.flagRiskAlert(undefined), false);
  assert.strictEqual(O.flagRiskAlert({}), false);
});

check("a truthy-but-non-true alert is not honored (strict boolean)", () => {
  // The server emits a real boolean; anything else is treated as no alert.
  assert.strictEqual(O.flagRiskAlert({ alert: 1 }), false);
  assert.strictEqual(O.flagRiskAlert({ alert: "yes" }), false);
});

if (failures) {
  console.log("\n" + failures + " test(s) FAILED");
  process.exit(1);
}
console.log("\nall flag-risk tests passed");
