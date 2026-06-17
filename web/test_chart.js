/* Node test for the across-the-game chart geometry (task 0038).
 *
 * app.js is a browser IIFE; we load it in a vm with minimal DOM/fetch stubs (so the
 * load-time `fetch(...)` no-ops) and exercise the pure `chartGeometry` helper it
 * exposes on window.ChessEquityDemo. Run: `node web/test_chart.js`. No deps.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const HERE = __dirname;
const game = JSON.parse(fs.readFileSync(path.join(HERE, "demo-game.json"), "utf8"));

// Minimal stubs: app.js only touches document/DOM inside functions called after the
// fetch resolves, so a never-resolving fetch keeps init() from running at load.
const noopChain = { then() { return noopChain; }, catch() { return noopChain; } };
const sandbox = {
  window: { location: { search: "" } },
  document: { createElementNS() { return {}; } },
  fetch() { return noopChain; },
  URLSearchParams,
  Math,
  console,
};
sandbox.window.window = sandbox.window;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(HERE, "app.js"), "utf8"), sandbox);

const api = sandbox.window.ChessEquityDemo;
assert.ok(api && typeof api.chartGeometry === "function", "chartGeometry must be exposed");

const bands = game.rating_bands;
const opts = { width: 480, height: 160, pad: 24 };

let failures = 0;
function check(name, fn) {
  try { fn(); console.log("PASS", name); }
  catch (e) { failures++; console.log("FAIL", name, "-", e.message); }
}

check("one point per ply", () => {
  const g = api.chartGeometry(game.moves, bands, 1500, 1500, opts);
  assert.strictEqual(g.points.length, game.moves.length);
});

check("points stay within the padded box", () => {
  const g = api.chartGeometry(game.moves, bands, 1500, 1500, opts);
  for (const p of g.points) {
    assert.ok(p.x >= g.pad - 1e-6 && p.x <= g.width - g.pad + 1e-6, "x in range");
    for (const y of [p.eqY, p.cpY]) {
      assert.ok(y >= g.pad - 1e-6 && y <= g.height - g.pad + 1e-6, "y in range");
    }
  }
});

check("x spans the full inner width end to end", () => {
  const g = api.chartGeometry(game.moves, bands, 1500, 1500, opts);
  assert.ok(Math.abs(g.points[0].x - g.pad) < 1e-6, "first ply at left pad");
  const last = g.points[g.points.length - 1];
  assert.ok(Math.abs(last.x - (g.width - g.pad)) < 1e-6, "last ply at right edge");
});

check("equity line is rating-conditioned (moves with the slider)", () => {
  const lo = api.chartGeometry(game.moves, bands, 1100, 2300, opts);
  const hi = api.chartGeometry(game.moves, bands, 2300, 1100, opts);
  const moved = lo.points.some((p, i) => Math.abs(p.eqY - hi.points[i].eqY) > 1e-6);
  assert.ok(moved, "equity Y should differ across rating grids");
});

check("centipawn line is rating-blind (does NOT move with the slider)", () => {
  const a = api.chartGeometry(game.moves, bands, 1100, 2300, opts);
  const b = api.chartGeometry(game.moves, bands, 2300, 1100, opts);
  assert.ok(a.points.every((p, i) => Math.abs(p.cpY - b.points[i].cpY) < 1e-9));
});

check("higher y means lower win% (axis is flipped)", () => {
  const g = api.chartGeometry(game.moves, bands, 1500, 1500, opts);
  assert.ok(g.yFor(100) < g.yFor(0), "100% should sit above 0%");
  assert.ok(Math.abs(g.yFor(50) - g.y50) < 1e-9);
});

check("polyline point strings match the computed points", () => {
  const g = api.chartGeometry(game.moves, bands, 1500, 1500, opts);
  assert.strictEqual(g.eqPoints.split(" ").length, g.points.length);
  assert.strictEqual(g.cpPoints.split(" ").length, g.points.length);
  assert.strictEqual(g.eqPoints.split(" ")[0], g.points[0].x + "," + g.points[0].eqY);
});

check("hover label carries san + both readings", () => {
  const g = api.chartGeometry(game.moves, bands, 1500, 1500, opts);
  assert.ok(g.points[0].label.includes("start"));
  assert.ok(/equity \d+% · cp-bar \d+%/.test(g.points[1].label));
});

check("single-ply game centers the point (no divide-by-zero)", () => {
  const one = [game.moves[0]];
  const g = api.chartGeometry(one, bands, 1500, 1500, opts);
  assert.strictEqual(g.points.length, 1);
  assert.ok(Number.isFinite(g.points[0].x));
});

if (failures) { console.error(failures + " failure(s)"); process.exit(1); }
console.log("ok - chart geometry");
