/* Node test for feed.js's live/stale feed state machine (task 0202).
 *
 * The stale-feed UI (task 0178) dims the bar and shows a "⚠ reconnecting…" marker when
 * a LIVE feed drops, so an on-air frozen bar can't be mistaken for a live one. The pure
 * brain of that UI is `EquityFeed.makeStaleTracker` in feed.js: a timer-free state
 * machine (the caller injects `now` in ms) that returns an edge-only transition string
 * ("stale"/"recovered"), else null, so the overlay fires its DOM side-effect exactly once
 * per transition.
 *
 * Until now the only coverage was overlay/test_overlay.py's `class StaleTracker` — a
 * PYTHON RE-IMPLEMENTATION that mirrors the JS by hand and can silently drift from the
 * real code. This test loads the ACTUAL feed.js in a vm sandbox and exercises the shipped
 * helper directly, so a regression in feed.js can't pass a green Python mirror.
 *
 * Run: `node overlay/test_stale.test.js`. No deps.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const HERE = __dirname;

// feed.js is an IIFE: (function (global) {...})(typeof window !== "undefined" ? window : this).
// In this sandbox `window` is undefined, so `this` (the sandbox global) receives
// `EquityFeed`. makeStaleTracker only touches `Date` (and only when `now` is omitted,
// which we never do), so no browser globals are needed to load it.
const sandbox = { JSON, Math, Date, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(HERE, "feed.js"), "utf8"), sandbox);

const Feed = sandbox.EquityFeed;
assert.ok(Feed && typeof Feed.makeStaleTracker === "function", "EquityFeed.makeStaleTracker must be exposed");

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

check("enters STALE on silence — only when no event for >= staleMs, and only on the edge", () => {
  const t = Feed.makeStaleTracker(10000);
  assert.strictEqual(t.event(0), null, "first event reports no transition");
  assert.strictEqual(t.isStale(), false);
  // Polling before the threshold keeps it live...
  assert.strictEqual(t.poll(5000), null, "below threshold stays live");
  assert.strictEqual(t.isStale(), false);
  // ...crossing the threshold flips it exactly once...
  assert.strictEqual(t.poll(10000), "stale", "crossing staleMs goes stale");
  assert.strictEqual(t.isStale(), true);
  // ...and further polls while already stale report no edge.
  assert.strictEqual(t.poll(20000), null, "stale transition fires only on the edge");
});

check("enters STALE on transport error immediately, before any timeout", () => {
  const t = Feed.makeStaleTracker(10000);
  t.event(0);
  assert.strictEqual(t.fail(), "stale", "a dropped connection forces stale at once");
  assert.strictEqual(t.isStale(), true);
  assert.strictEqual(t.fail(), null, "an already-stale fail must not re-fire");
});

check("recovers on the next event, exactly once", () => {
  const t = Feed.makeStaleTracker(10000);
  t.event(0);
  assert.strictEqual(t.poll(10000), "stale");
  assert.strictEqual(t.isStale(), true);
  // The next event clears the stale state and reports "recovered"...
  assert.strictEqual(t.event(12000), "recovered", "next event recovers");
  assert.strictEqual(t.isStale(), false);
  // ...and a normal event while already live reports no transition.
  assert.strictEqual(t.event(13000), null, "a live-to-live event is not a transition");
});

check("no-op before the first event ever arrives (no frozen-from-birth feed)", () => {
  const t = Feed.makeStaleTracker(10000);
  assert.strictEqual(t.poll(999999), null, "polling before any event never declares stale");
  assert.strictEqual(t.isStale(), false);
});

check("defaults to a 10s window when staleMs is omitted/zero", () => {
  const t = Feed.makeStaleTracker();
  t.event(0);
  assert.strictEqual(t.poll(9999), null, "just under the default 10s stays live");
  assert.strictEqual(t.poll(10000), "stale", "the default window is 10000ms");
});

check("recover then re-stale: the cycle is repeatable and stays edge-only", () => {
  const t = Feed.makeStaleTracker(10000);
  t.event(0);
  assert.strictEqual(t.poll(10000), "stale");
  assert.strictEqual(t.event(11000), "recovered");
  // A second silence stretch after recovery goes stale again on its own edge.
  assert.strictEqual(t.poll(20000), null, "below the new threshold (11000+10000) stays live");
  assert.strictEqual(t.poll(21000), "stale", "a second silence stretch goes stale again");
  assert.strictEqual(t.poll(30000), null, "still edge-only the second time around");
});

if (failures) {
  console.error(failures + " failure(s)");
  process.exit(1);
}
console.log("\nAll stale-tracker tests passed.");
