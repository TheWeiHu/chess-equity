/* Node test for overlay.js's colorblind-safe palette toggle (task 0254).
 *
 * The OBS overlay's default red(--bad)/green(--accent) signalling is the classic
 * deuteranopia/protanopia failure case. `?palette=cvd` (or the config.html toggle) must
 * swap it to a CVD-safe blue/orange set by adding the `palette-cvd` class to #overlay.
 * This test loads overlay.js in a vm (no `document`, so it does NOT auto-start — we only
 * want the pure `paletteClass` mapper) and asserts the param maps to the alternate class:
 *   - "cvd"      -> "palette-cvd"  (the alternate CSS class is applied);
 *   - default/"" -> ""            (no class — default palette untouched, additive).
 * Run: `node overlay/test_palette.test.js`. No deps.
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
assert.ok(O && typeof O.paletteClass === "function", "EquityOverlay.paletteClass must be exposed");

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

check("palette=cvd maps to the alternate CSS class", () => {
  assert.strictEqual(O.paletteClass("cvd"), "palette-cvd", "the CVD-safe class is applied");
});

check("the default palette adds no class (additive toggle)", () => {
  assert.strictEqual(O.paletteClass("default"), "", "default => no class");
  assert.strictEqual(O.paletteClass(""), "", "empty => no class");
  assert.strictEqual(O.paletteClass(undefined), "", "missing => no class");
  assert.strictEqual(O.paletteClass("bogus"), "", "unknown value falls back to default");
});

// The alternate palette must actually exist in overlay.css, scoped to the class the
// helper emits — otherwise the toggle would add a no-op class.
check("overlay.css defines the .palette-cvd overrides for --accent and --bad", () => {
  const css = fs.readFileSync(path.join(HERE, "overlay.css"), "utf8");
  assert.ok(/\.palette-cvd\b/.test(css), "overlay.css must define .palette-cvd");
  const block = css.slice(css.indexOf(".overlay.palette-cvd"));
  assert.ok(/--accent\s*:/.test(block), ".palette-cvd must override --accent (green->blue)");
  assert.ok(/--bad\s*:/.test(block), ".palette-cvd must override --bad (red->orange)");
});

if (failures) {
  console.log("\n" + failures + " FAILED");
  process.exit(1);
}
console.log("\nall palette tests passed");
