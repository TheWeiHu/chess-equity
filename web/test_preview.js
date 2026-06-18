/* Node test for the hover board preview (task 0078).
 *
 * app.js is a browser IIFE. test_chart.js exercises the pure geometry with near-noop
 * DOM stubs; here we need real DOM behaviour, so we build a *compact* fake DOM (no
 * deps), run app.js end to end with a synchronous fetch of demo-game.json, then
 * dispatch a `mouseenter` on a chart dot and assert the floating preview populates.
 * Run: `node web/test_preview.js`.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const HERE = __dirname;
const game = JSON.parse(fs.readFileSync(path.join(HERE, "demo-game.json"), "utf8"));

// ---- compact fake DOM ------------------------------------------------------

function makeEl(tag) {
  const el = {
    tagName: tag,
    children: [],
    attrs: {},
    style: {},
    _handlers: {},
    _text: "",
    hidden: false,
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      contains(c) { return this._set.has(c); },
    },
    setAttribute(k, v) { this.attrs[k] = v; },
    getAttribute(k) { return this.attrs[k]; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(type, fn) { (this._handlers[type] = this._handlers[type] || []).push(fn); },
    dispatchEvent(ev) { (this._handlers[ev.type] || []).forEach((fn) => fn(ev)); },
  };
  Object.defineProperty(el, "innerHTML", {
    get() { return ""; },
    set(v) { if (v === "") this.children = []; },
  });
  Object.defineProperty(el, "textContent", {
    get() { return this._text; },
    set(v) { this._text = v; },
  });
  return el;
}

const byId = {};
const document = {
  title: "",
  createElement: makeEl,
  createElementNS(_ns, tag) { return makeEl(tag); },
  getElementById(id) { return (byId[id] = byId[id] || makeEl("div")); },
  addEventListener() {},
};

// Synchronous promise-ish so fetch(...).then(r => r.json()).then(init) runs at load.
function resolved(val) {
  return { then(f) { return resolved(f(val)); }, catch() { return this; } };
}

const sandbox = {
  window: { location: { search: "" }, ChessEquityDemo: null },
  document,
  // app.js fetches games.json (the catalog) at load before fetching the game itself.
  // Return a one-entry manifest for it so setupGamePicker populates the dropdown
  // instead of hitting its "no catalog" hide-path; any other URL resolves to the game.
  fetch(url) {
    if (url === "games.json") {
      return resolved({ ok: true, json: () => ({ games: [{ file: "demo-game.json", name: "Demo" }] }) });
    }
    return resolved({ json: () => game });
  },
  URLSearchParams,
  Math,
  parseInt,
  console,
};
sandbox.window.window = sandbox.window;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(HERE, "app.js"), "utf8"), sandbox);

// ---- assertions ------------------------------------------------------------

let failures = 0;
function check(name, fn) {
  try { fn(); console.log("PASS", name); }
  catch (e) { failures++; console.log("FAIL", name, "-", e.message); }
}

const chart = document.getElementById("chart");
const dots = chart.children.filter((c) => c.tagName === "circle");
const preview = document.getElementById("board-preview");
const previewBoard = document.getElementById("preview-board");

check("chart rendered one dot per ply", () => {
  assert.strictEqual(dots.length, game.moves.length);
});

check("preview markup is hidden by default in index.html", () => {
  // The initial hidden state comes from the HTML attribute, not JS, so assert the
  // markup rather than the fake-DOM default.
  const html = fs.readFileSync(path.join(HERE, "index.html"), "utf8");
  assert.ok(/id="board-preview"[^>]*\shidden/.test(html), "board-preview must start hidden");
});

check("hovering a dot populates the preview board and shows it", () => {
  const target = dots[5]; // a mid-game ply (Nxe5 region)
  target.dispatchEvent({ type: "mouseenter" });
  assert.strictEqual(preview.hidden, false, "preview should become visible");
  assert.strictEqual(previewBoard.children.length, 64, "preview board should have 64 squares");
  assert.ok(document.getElementById("preview-caption")._text.length > 0, "caption should name the ply");
});

check("the preview shows the hovered ply, not the current board ply", () => {
  // app.js starts at ply 0; hovering ply 5 must render ply 5's position, leaving the
  // main board untouched (click-scrub, tested below, is what moves the main board).
  const expectedPieces = game.moves[5].fen.split(" ")[0].replace(/[^a-zA-Z]/g, "").length;
  const shownPieces = previewBoard.children.filter(
    (sq) => sq.children.length > 0
  ).length;
  assert.strictEqual(shownPieces, expectedPieces, "preview piece count matches ply 5 FEN");
});

check("mousemove repositions the floating preview", () => {
  dots[5].dispatchEvent({ type: "mousemove", clientX: 100, clientY: 80 });
  assert.strictEqual(preview.style.left, "116px");
  assert.strictEqual(preview.style.top, "96px");
});

check("preview flips to the other side of the cursor near the right/bottom edge", () => {
  // 200x200 viewport, 160x160 preview, 16px offset. At cursor (190,190) the default
  // cursor+offset would put it at 206 — off-screen — so it must flip to the left/above:
  // 190 - 16 - 160 = 14 (>= the 16px min? no -> clamped to 16).
  const pos = sandbox.window.ChessEquityDemo.clampPreviewPos(190, 190, 160, 160, 200, 200, 16);
  assert.ok(pos.left + 160 <= 200, "preview right edge must stay within the viewport");
  assert.ok(pos.top + 160 <= 200, "preview bottom edge must stay within the viewport");
  assert.ok(pos.left >= 16 && pos.top >= 16, "preview must not spill off the top/left");
});

check("preview keeps the plain cursor+offset placement when there is room", () => {
  // With room (or no viewport measurement), it stays at cursor + 16 — the old behaviour.
  // (Compare fields, not the object: it crosses the vm realm so its prototype differs.)
  const roomy = sandbox.window.ChessEquityDemo.clampPreviewPos(100, 80, 160, 160, 1000, 800, 16);
  assert.strictEqual(roomy.left, 116);
  assert.strictEqual(roomy.top, 96);
  const unmeasured = sandbox.window.ChessEquityDemo.clampPreviewPos(100, 80, 0, 0, 0, 0, 16);
  assert.strictEqual(unmeasured.left, 116);
  assert.strictEqual(unmeasured.top, 96);
});

check("mouseleave hides the preview again", () => {
  dots[5].dispatchEvent({ type: "mouseleave" });
  assert.strictEqual(preview.hidden, true);
});

check("click-scrub still works (main board ply changes, unchanged behaviour)", () => {
  const mainBoard = document.getElementById("board");
  dots[3].dispatchEvent({ type: "click" });
  // After scrubbing to ply 3 the main board re-renders with that position's pieces.
  const expected = game.moves[3].fen.split(" ")[0].replace(/[^a-zA-Z]/g, "").length;
  const shown = mainBoard.children.filter((sq) => sq.children.length > 0).length;
  assert.strictEqual(shown, expected, "main board reflects clicked ply");
});

// ---- touch + keyboard parity (task 0101) -----------------------------------

check("tapping a dot pops the preview at the tap point (touch path)", () => {
  hidePreview_via_mouseleave();
  // A touch-generated click carries clientX/clientY; the preview should show + position.
  dots[6].dispatchEvent({ type: "click", clientX: 200, clientY: 120 });
  assert.strictEqual(preview.hidden, false, "tap should reveal the preview");
  assert.strictEqual(previewBoard.children.length, 64, "preview board renders on tap");
  assert.strictEqual(preview.style.left, "216px", "preview positioned at the tap x+16");
  assert.strictEqual(preview.style.top, "136px", "preview positioned at the tap y+16");
});

check("chart dots are keyboard-focusable (tabindex=0)", () => {
  assert.strictEqual(dots[6].getAttribute("tabindex"), "0");
});

check("focusing a dot shows that ply's preview (keyboard path)", () => {
  hidePreview_via_mouseleave();
  dots[4].dispatchEvent({ type: "focus" });
  assert.strictEqual(preview.hidden, false, "focus should reveal the preview");
  const expectedPieces = game.moves[4].fen.split(" ")[0].replace(/[^a-zA-Z]/g, "").length;
  const shownPieces = previewBoard.children.filter((sq) => sq.children.length > 0).length;
  assert.strictEqual(shownPieces, expectedPieces, "focused dot previews its own ply");
});

check("blur hides the preview (keyboard dismiss)", () => {
  dots[4].dispatchEvent({ type: "focus" });
  dots[4].dispatchEvent({ type: "blur" });
  assert.strictEqual(preview.hidden, true, "blur should dismiss the preview");
});

// A tiny helper: reset the preview to hidden between cases via the public mouseleave path
// (the fake DOM has no global listeners, so we reset through a dot we already have).
function hidePreview_via_mouseleave() {
  dots[6].dispatchEvent({ type: "mouseleave" });
}

if (failures) { console.error(failures + " failure(s)"); process.exit(1); }
console.log("ok - hover board preview");
