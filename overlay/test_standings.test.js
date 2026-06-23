/* Node test for the lower-third standings panel (task 0231).
 *
 * The panel turns a `grade --round --json` leaderboard export into a top-N OBS lower
 * third. The contract is that rows render in strict RANK order (a sub-floor cameo or a
 * file-order shuffle can never jump the board), sliced to the requested N. This loads
 * the REAL standings.js in a vm sandbox and exercises the pure helpers `orderTopN`,
 * `rowsHTML`, and `extract` so a regression can't pass a hand-mirrored re-implementation.
 *
 * Run: `node overlay/test_standings.test.js`. No deps.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const HERE = __dirname;

// standings.js is an IIFE ending in (typeof window !== "undefined" ? window : this).
// With `window`/`document` undefined in the sandbox the DOM `mount()` is a no-op at load,
// so a bare sandbox loads it cleanly and exposes EquityStandings on `this`.
const sandbox = { JSON, Math, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(HERE, "standings.js"), "utf8"), sandbox);

const S = sandbox.EquityStandings;
assert.ok(S && typeof S.orderTopN === "function", "EquityStandings.orderTopN must be exposed");
assert.ok(typeof S.rowsHTML === "function", "EquityStandings.rowsHTML must be exposed");
assert.ok(typeof S.extract === "function", "EquityStandings.extract must be exposed");

// The committed illustrative fixture — exercise the panel against the same JSON OBS reads.
const fixture = JSON.parse(fs.readFileSync(path.join(HERE, "mock-leaderboard.json"), "utf8"));

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

// --- extract: accept both the bare export array and the {leaderboard:[…]} wrapper -----
check("extract pulls rows from the {leaderboard:[…]} fixture wrapper", () => {
  const rows = S.extract(fixture);
  assert.ok(Array.isArray(rows) && rows.length === 8, "fixture has 8 players");
  assert.strictEqual(rows[0].player, "Carlsen");
});
check("extract passes a bare export array straight through", () => {
  const bare = [{ rank: 1, player: "a" }, { rank: 2, player: "b" }];
  // strictEqual: a bare array must be returned by reference, not copied.
  assert.strictEqual(S.extract(bare), bare);
});
check("extract yields an empty array for junk payloads", () => {
  // The arrays cross the vm-realm boundary, so compare by shape (Array + length),
  // not deepStrictEqual against a test-realm [] (whose prototype differs).
  for (const junk of [null, undefined, {}, 42, "x"]) {
    const out = S.extract(junk);
    assert.ok(Array.isArray(out) && out.length === 0, "junk -> empty array");
  }
});

// --- orderTopN: SHUFFLED input must come back in strict rank order, sliced to N --------
check("orderTopN sorts a shuffled leaderboard back into rank order", () => {
  const rows = S.extract(fixture);
  // Reverse so file order is the OPPOSITE of rank order — the worst possible input.
  const shuffled = rows.slice().reverse();
  assert.strictEqual(shuffled[0].rank, 8, "precondition: input starts at rank 8");
  const ordered = S.orderTopN(shuffled, 99);
  const ranks = ordered.map((r) => r.rank);
  assert.deepStrictEqual(ranks, [1, 2, 3, 4, 5, 6, 7, 8], "rows come back in rank order");
});
check("orderTopN slices to the requested top N", () => {
  const ordered = S.orderTopN(S.extract(fixture), 3);
  assert.strictEqual(ordered.length, 3);
  assert.deepStrictEqual(ordered.map((r) => r.player), ["Carlsen", "Nakamura", "Caruana"]);
});
check("orderTopN clamps a bad/zero/negative top to the default of 5", () => {
  assert.strictEqual(S.orderTopN(S.extract(fixture), 0).length, 5);
  assert.strictEqual(S.orderTopN(S.extract(fixture), -3).length, 5);
  assert.strictEqual(S.orderTopN(S.extract(fixture), "xyz").length, 5);
});
check("orderTopN tie-breaks equal ranks by player name, deterministically", () => {
  const tied = [
    { rank: 1, player: "Zoe" },
    { rank: 1, player: "Ann" },
    { rank: 1, player: "Mia" },
  ];
  const names = S.orderTopN(tied, 99).map((r) => r.player);
  assert.deepStrictEqual(names, ["Ann", "Mia", "Zoe"]);
});

// --- rowsHTML: the rendered markup carries players in rank order ----------------------
check("rowsHTML emits one row per player, in rank order", () => {
  const html = S.rowsHTML(S.extract(fixture).slice().reverse(), 5);
  const names = (html.match(/st-name">([^<]+)</g) || []).map((m) => m.replace(/.*">|</g, ""));
  assert.deepStrictEqual(names, ["Carlsen", "Nakamura", "Caruana", "Nepomniachtchi", "Ding"]);
  // and the displayed rank cells are 1..5 in order
  const rankCells = (html.match(/st-rank">([^<]*)</g) || []).map((m) => m.replace(/.*">|</g, ""));
  assert.deepStrictEqual(rankCells, ["1", "2", "3", "4", "5"]);
});
check("rowsHTML shows the row's own rank field, not its array index", () => {
  // An export with a gap (no rank 2) must still print the true rank numbers.
  const gapped = [{ rank: 1, player: "a", accuracy: 90, avg_delta: 1 }, { rank: 3, player: "c", accuracy: 80, avg_delta: 2 }];
  const html = S.rowsHTML(gapped, 5);
  const rankCells = (html.match(/st-rank">([^<]*)</g) || []).map((m) => m.replace(/.*">|</g, ""));
  assert.deepStrictEqual(rankCells, ["1", "3"]);
});
check("rowsHTML formats accuracy as a percent and Δpeer as a signed number", () => {
  const html = S.rowsHTML([{ rank: 1, player: "x", accuracy: 96.3, avg_delta: 3.12 }], 1);
  assert.ok(html.indexOf("96.3%") !== -1, "accuracy shown as 96.3%");
  assert.ok(html.indexOf("+3.1") !== -1, "positive delta gets a + sign");
  const neg = S.rowsHTML([{ rank: 1, player: "x", accuracy: 50, avg_delta: -4.4 }], 1);
  assert.ok(neg.indexOf("-4.4") !== -1, "negative delta keeps its sign");
});
check("rowsHTML escapes a hostile player name so it can't break the markup", () => {
  const html = S.rowsHTML([{ rank: 1, player: "<b>x</b>", accuracy: 50, avg_delta: 0 }], 1);
  assert.ok(html.indexOf("<b>x</b>") === -1, "raw tag must not appear");
  assert.ok(html.indexOf("&lt;b&gt;x&lt;/b&gt;") !== -1, "name is HTML-escaped");
});

if (failures) {
  console.log("\n" + failures + " standings test(s) FAILED");
  process.exit(1);
}
console.log("\nall standings tests passed");
