/* board.js — one chess-board renderer shared by both demos (guided app.js + live.js).
 *
 * Before this, each demo had its own board code with different piece glyphs and square
 * styling, so the two boards looked different. This is the single source of truth:
 * SOLID glyphs (U+265A–F) for both colours (CSS colours white vs black), and the same
 * square/coord markup. Exposes window.ChessBoard = { parseFen, render, sqName, FILES }.
 */
(function () {
  "use strict";

  // Minimal flat piece set — deliberate inline-SVG silhouettes (not the system chess
  // font), drawn once on a 45x45 grid. Fill + edge come from CSS (.piece.white/.black),
  // so the same paths render as light-with-edge or solid-dark. Simple high-contrast
  // shapes that stay sharp at small sizes and on the green squares.
  var SHAPES = {
    p: '<circle cx="22.5" cy="15" r="5"/>' +
       '<path d="M15.5 34c0-8 3-12 7-12s7 4 7 12z"/>',
    r: '<path d="M13.5 19V12H17v3h3.5v-3h4v3H28v-3h3.5V19Z"/>' +
       '<path d="M15 19h15v15H15z"/>' +
       '<path d="M12.5 34h20v4h-20z"/>',
    n: '<path d="M14 38h18c0-8-.4-14-2-18-1.6-4-5-6.6-9-7l1.6-3.6c.3-.8-.6-1.6-1.6-1C19 12 17.5 13 16 14c-3 2-5 6-5 10.5 0 2 1.6 3 3 2l2.6-3.6 2 1L19 31c-1.6 2.6-2.6 4-2.6 7Z"/>',
    b: '<circle cx="22.5" cy="8.5" r="2"/>' +
       '<path d="M22.5 10c4 4 5.5 9 5.5 13 0 5-2.6 8.6-5.5 9.5-3-.9-5.5-4.5-5.5-9.5 0-4 1.5-9 5.5-13Z"/>' +
       '<path d="M14 34c0-3 4-4 8.5-4s8.5 1 8.5 4l1 4H13Z"/>',
    q: '<path d="M11 22l2.5 12h18L34 22l-5 5-2.6-9-2.9 9-3-9-3 9-2.6-9Z"/>' +
       '<circle cx="11" cy="20" r="2"/><circle cx="22.5" cy="15" r="2"/><circle cx="34" cy="20" r="2"/>' +
       '<path d="M12.5 34h20v4h-20z"/>',
    k: '<path d="M21 6h3v3h3v3h-3v4h-3v-4h-3V9h3z"/>' +
       '<path d="M22.5 16c-5.5 0-9.5 4.5-9.5 10 0 4 3.5 7 9.5 7s9.5-3 9.5-7c0-5.5-4-10-9.5-10Z"/>' +
       '<path d="M12.5 33h20v5h-20z"/>'
  };
  function svgFor(type) {
    return '<svg viewBox="0 0 45 45" aria-hidden="true" focusable="false">' +
      SHAPES[type.toLowerCase()] + '</svg>';
  }
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
          span.className = "piece " + (piece === piece.toUpperCase() ? "white" : "black");
          span.innerHTML = svgFor(piece);
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

  window.ChessBoard = { parseFen: parseFen, render: render, sqName: sqName, FILES: FILES, pieceSvg: svgFor };
})();
