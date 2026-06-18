/* board.js — one chess-board renderer shared by both demos (guided app.js + live.js).
 *
 * Before this, each demo had its own board code with different piece glyphs and square
 * styling, so the two boards looked different. This is the single source of truth:
 * SOLID glyphs (U+265A–F) for both colours (CSS colours white vs black), and the same
 * square/coord markup. Exposes window.ChessBoard = { parseFen, render, sqName, FILES }.
 */
(function () {
  "use strict";

  // Clean Unicode chess glyphs: the SOLID set (U+265A-F) for BOTH colours so White and
  // Black pieces are the exact same glyph (and thus the same size — the outline U+2654-9
  // set has different metrics and rendered a size apart). CSS (.piece.white/.black in
  // board.css) fills White light with a soft dark edge and Black solid dark.
  var PIECES = { K: "♚", Q: "♛", R: "♜", B: "♝", N: "♞", P: "♟",
                 k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
  // The glyph for any type — used by the promotion picker (rendered dark).
  function glyph(type) { return PIECES[type.toLowerCase()]; }
  var FILES = "abcdefgh";

  // Parse a FEN's placement field into an 8x8 grid (row 0 == rank 8, col 0 == file a).
  function parseFen(fen) {
    var rows = fen.split(" ")[0].split("/"), grid = [];
    for (var r = 0; r < 8; r++) {
      var line = [], chars = rows[r].split("");
      for (var c = 0; c < chars.length; c++) {
        var ch = chars[c];
        if (/\d/.test(ch)) { for (var k = 0; k < +ch; k++) line.push(null); }
        else line.push(ch);
      }
      grid.push(line);
    }
    return { grid: grid, turn: fen.split(" ")[1] || "w" };
  }

  function sqName(r, c) { return FILES[c] + (8 - r); }

  function coord(kind, text) {
    var s = document.createElement("span");
    s.className = "coord " + kind;
    s.textContent = text;
    return s;
  }

  // Render `fen` into board element `el`. opts (all optional):
  //   flipped    bool — view from Black's side
  //   coords     bool — file/rank labels in the edges
  //   highlight  {from,to} square names (last move) | [names]
  //   selected   square name (a picked-up piece)
  //   dests      [square names] — legal destinations (dot / capture ring)
  //   check      square name — king in check
  //   onSquare   fn(name, ev) — per-square click
  //   draggable  fn(name) -> bool — make that piece draggable
  //   onDragStart / onDrop  fn(name, ev)
  function render(el, fen, opts) {
    opts = opts || {};
    var grid = parseFen(fen).grid;

    var hl = {};
    if (opts.highlight) {
      if (opts.highlight.from) { hl[opts.highlight.from] = 1; hl[opts.highlight.to] = 1; }
      else if (opts.highlight.forEach) opts.highlight.forEach(function (n) { hl[n] = 1; });
    }
    var dests = {};
    (opts.dests || []).forEach(function (n) { dests[n] = 1; });

    el.innerHTML = "";
    for (var d = 0; d < 8; d++) {
      for (var e = 0; e < 8; e++) {
        var rr = opts.flipped ? 7 - d : d, cc = opts.flipped ? 7 - e : e;
        var name = sqName(rr, cc), piece = grid[rr][cc];
        var sq = document.createElement("div");
        sq.className = "sq " + ((rr + cc) % 2 === 0 ? "light" : "dark");
        if (hl[name]) sq.classList.add("hl");
        if (name === opts.selected) sq.classList.add("sel");
        if (name === opts.check) sq.classList.add("check");
        if (dests[name]) { sq.classList.add("dest"); if (piece) sq.classList.add("capture"); }
        if (piece) {
          var span = document.createElement("span");
          // t-<type> lets CSS normalise per-glyph size (the font draws ♚/♞ a size off)
          span.className = "piece t-" + piece.toLowerCase() +
            " " + (piece === piece.toUpperCase() ? "white" : "black");
          span.textContent = PIECES[piece];
          if (opts.draggable && opts.draggable(name)) {
            span.draggable = true;
            if (opts.onDragStart) span.addEventListener("dragstart", bind(opts.onDragStart, name));
          }
          sq.appendChild(span);
        }
        if (opts.coords) {
          if (d === 7) sq.appendChild(coord("file", FILES[cc]));
          if (e === 0) sq.appendChild(coord("rank", String(8 - rr)));
        }
        sq.dataset.name = name;
        if (opts.onSquare) sq.addEventListener("click", bind(opts.onSquare, name));
        if (opts.onDrop) {
          sq.addEventListener("dragover", function (ev) { ev.preventDefault(); });
          sq.addEventListener("drop", bindPrevent(opts.onDrop, name));
        }
        el.appendChild(sq);
      }
    }
  }

  function bind(fn, name) { return function (ev) { fn(name, ev); }; }
  function bindPrevent(fn, name) { return function (ev) { ev.preventDefault(); fn(name, ev); }; }

  window.ChessBoard = { parseFen: parseFen, render: render, sqName: sqName, FILES: FILES, glyph: glyph };
})();
