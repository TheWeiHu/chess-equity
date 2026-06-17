/* live.js — interactive board backed by web/server.py.
 *
 * The browser holds no chess rules: it POSTs the current FEN (+ an optional move and
 * the two ratings) to /api/play and renders whatever the server returns — legal moves,
 * game state, the Stockfish centipawn bar and the Maia-2 equity bar. Click a piece to
 * see its legal destinations, click one to play it.
 */
(function () {
  "use strict";

  var PIECES = { K: "♚", Q: "♛", R: "♜", B: "♝", N: "♞", P: "♟",
                 k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
  var FILES = "abcdefgh";
  var MATE_CP = 10000;

  var state = {
    fen: null, legal: {}, turn: "white", sel: null, last: null,
    we: 1500, be: 1500, flipped: false, busy: false, history: [],
  };

  function $(id) { return document.getElementById(id); }

  function parseGrid(fen) {
    return fen.split(" ")[0].split("/").map(function (row) {
      var line = [];
      row.split("").forEach(function (ch) {
        if (/\d/.test(ch)) { for (var k = 0; k < +ch; k++) line.push(null); }
        else line.push(ch);
      });
      return line;
    });
  }

  function sqName(r, c) { return FILES[c] + (8 - r); }

  // ---- rendering -----------------------------------------------------------

  function render() {
    if (!state.fen) return;   // nothing to draw until the first /api/play resolves
    var grid = parseGrid(state.fen);
    var dests = state.sel ? (state.legal[state.sel] || []) : [];
    var checkRC = null;
    if (state._check) {
      var king = state.turn === "white" ? "K" : "k";
      for (var r = 0; r < 8 && !checkRC; r++)
        for (var c = 0; c < 8; c++)
          if (grid[r][c] === king) checkRC = sqName(r, c);
    }

    var board = $("board");
    board.innerHTML = "";
    for (var d = 0; d < 8; d++) {
      for (var e = 0; e < 8; e++) {
        var rr = state.flipped ? 7 - d : d;
        var cc = state.flipped ? 7 - e : e;
        var name = sqName(rr, cc);
        var piece = grid[rr][cc];
        var sq = document.createElement("div");
        sq.className = "sq " + ((rr + cc) % 2 === 0 ? "light" : "dark");
        if (state.last && (name === state.last.from || name === state.last.to)) sq.classList.add("hl");
        if (name === state.sel) sq.classList.add("sel");
        if (checkRC === name) sq.classList.add("check");
        if (dests.indexOf(name) >= 0) { sq.classList.add("dest"); if (piece) sq.classList.add("capture"); }
        if (piece) {
          var span = document.createElement("span");
          span.className = "piece " + (piece === piece.toUpperCase() ? "white" : "black");
          span.textContent = PIECES[piece];
          if (state.legal[name]) {            // only pieces with legal moves are draggable
            span.draggable = true;
            span.addEventListener("dragstart", onDragStart);
          }
          sq.appendChild(span);
        }
        if (d === 7) sq.appendChild(coord("file", FILES[cc]));
        if (e === 0) sq.appendChild(coord("rank", String(8 - rr)));
        sq.dataset.name = name;
        sq.addEventListener("click", onSquare);
        sq.addEventListener("dragover", function (ev) { ev.preventDefault(); });
        sq.addEventListener("drop", onDrop);
        board.appendChild(sq);
      }
    }
    board.classList.toggle("busy", state.busy);
  }

  function coord(kind, text) {
    var s = document.createElement("span");
    s.className = "coord " + kind;
    s.textContent = text;
    return s;
  }

  function renderBars(resp) {
    var eq = resp.equity_white;
    var cpWhite = cpToWhite(resp.cp) * 100;
    $("equity-fill").style.width = eq + "%";
    $("equity-readout").textContent = Math.round(eq) + "% White";
    $("cp-fill").style.width = cpWhite + "%";
    $("cp-readout").textContent = Math.abs(resp.cp) >= MATE_CP
      ? (resp.cp > 0 ? "#" : "-#") : (resp.cp >= 0 ? "+" : "") + (resp.cp / 100).toFixed(1);

    var best = $("best");
    best.textContent = resp.best_san ? "Stockfish prefers " + resp.best_san : "";

    var gap = eq - cpWhite;
    var div = $("divergence");
    if (!resp.game_over && Math.abs(gap) >= 15) {
      div.hidden = false;
      div.textContent = "Equity favours " + (gap > 0 ? "White" : "Black") + " by " +
        Math.round(Math.abs(gap)) + " pts over the centipawn bar — at these ratings the " +
        "realistic result differs from perfect play.";
    } else { div.hidden = true; }
  }

  function cpToWhite(cp) { return 1 / (1 + Math.exp(-0.00368208 * cp)); }

  function renderStatus(resp) {
    var s = $("status");
    if (resp.checkmate) s.innerHTML = "<span class='san'>Checkmate</span> — " +
      (resp.turn === "white" ? "Black" : "White") + " wins.";
    else if (resp.stalemate) s.innerHTML = "<span class='san'>Draw</span> (no result).";
    else s.innerHTML = (resp.turn === "white" ? "White" : "Black") + " to move" +
      (resp.check ? " — <span class='san'>check</span>" : "");
  }

  function renderMoves() {
    var ol = $("moves");
    ol.innerHTML = "";
    state.history.forEach(function (h, i) {
      if (!h.san) return;
      var li = document.createElement("li");
      li.textContent = (i % 2 === 0 ? (i / 2 + 1) + ". " : "") + h.san;
      ol.appendChild(li);
    });
    var ply = state.history.filter(function (h) { return h.san; }).length;
    if (ply) {
      var box = ol.lastChild;
      if (box && box.scrollIntoView) box.scrollIntoView({ block: "nearest" });
    }
  }

  function applyResp(resp, san, lastMove) {
    state.fen = resp.fen;
    state.legal = resp.legal || {};
    state.turn = resp.turn;
    state._check = resp.check;
    state.sel = null;
    state.last = lastMove || null;
    $("fen").value = resp.fen;
    render();
    renderBars(resp);
    renderStatus(resp);
    renderMoves();
  }

  // ---- networking ----------------------------------------------------------

  function play(opts) {
    // opts: { uci?, fen?, record? } — record pushes onto the undo stack.
    if (state.busy) return;
    state.busy = true;
    render();
    var body = {
      fen: opts.fen != null ? opts.fen : state.fen,
      white_elo: state.we, black_elo: state.be,
    };
    if (opts.uci) body.uci = opts.uci;
    fetch("/api/play", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (res) {
        state.busy = false;
        if (!res.ok) { showFenErr(res.j.error || "error"); render(); return; }
        hideFenErr();
        var last = opts.uci ? { from: opts.uci.slice(0, 2), to: opts.uci.slice(2, 4) } : state.last;
        if (opts.record) state.history.push({ fen: res.j.fen, san: res.j.san, last: last, resp: res.j });
        applyResp(res.j, res.j.san, last);
      })
      .catch(function (err) { state.busy = false; showFenErr(String(err)); render(); });
  }

  // ---- interaction ---------------------------------------------------------

  // Play from -> to, detecting a pawn promotion (which pops the picker first).
  function attemptMove(from, to) {
    var grid = parseGrid(state.fen);
    var piece = grid[8 - parseInt(from[1], 10)][FILES.indexOf(from[0])];
    var toRank = parseInt(to[1], 10);
    var uci = from + to;
    if ((piece === "P" && toRank === 8) || (piece === "p" && toRank === 1)) { promote(uci); return; }
    play({ uci: uci, record: true });
  }

  function onSquare(ev) {
    if (state.busy) return;
    var name = ev.currentTarget.dataset.name;
    // Completing a move (click-to-move)?
    if (state.sel && (state.legal[state.sel] || []).indexOf(name) >= 0) { attemptMove(state.sel, name); return; }
    // Otherwise select a piece that has legal moves.
    state.sel = state.legal[name] ? name : null;
    render();
  }

  function onDragStart(ev) {
    if (state.busy) { ev.preventDefault(); return; }
    var from = ev.target.parentElement.dataset.name;
    state.sel = from;
    paintDrag(from);
    if (ev.dataTransfer) { ev.dataTransfer.effectAllowed = "move"; ev.dataTransfer.setData("text/plain", from); }
    ev.target.addEventListener("dragend", onDragEnd, { once: true });
  }

  function onDrop(ev) {
    ev.preventDefault();
    var to = ev.currentTarget.dataset.name;
    if (state.sel && (state.legal[state.sel] || []).indexOf(to) >= 0) attemptMove(state.sel, to);
    else render();   // illegal target — clear the drag marks
  }

  function onDragEnd() {
    // Dropped outside a legal square (no move started): repaint from state.
    if (!state.busy) render();
  }

  function squareEl(name) { return document.querySelector('#board .sq[data-name="' + name + '"]'); }

  // Mark a piece + its legal destinations without a full re-render (a re-render mid
  // drag would remove the element being dragged and abort the drag).
  function paintDrag(from) {
    document.querySelectorAll("#board .sq").forEach(function (sq) {
      sq.classList.remove("dest", "capture", "sel");
    });
    var src = squareEl(from);
    if (src) src.classList.add("sel");
    (state.legal[from] || []).forEach(function (to) {
      var sq = squareEl(to);
      if (!sq) return;
      sq.classList.add("dest");
      if (sq.querySelector(".piece")) sq.classList.add("capture");
    });
  }

  function promote(baseUci) {
    var frame = document.querySelector(".board-frame");
    var old = frame.querySelector(".promo");
    if (old) old.remove();
    var menu = document.createElement("div");
    menu.className = "promo";
    [["q", "♛"], ["r", "♜"], ["b", "♝"], ["n", "♞"]].forEach(function (p) {
      var b = document.createElement("button");
      b.textContent = p[1];
      b.addEventListener("click", function () { menu.remove(); play({ uci: baseUci + p[0], record: true }); });
      menu.appendChild(b);
    });
    frame.appendChild(menu);
  }

  function showFenErr(msg) { var e = $("fen-err"); e.hidden = false; e.textContent = msg; }
  function hideFenErr() { $("fen-err").hidden = true; }

  // ---- controls ------------------------------------------------------------

  function undo() {
    if (state.busy || !state.history.length) return;
    state.history.pop();
    var prev = state.history[state.history.length - 1];
    if (prev) applyResp(prev.resp, prev.san, prev.last);
    else { state.last = null; play({ fen: START, record: false }); }
  }

  var START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  function wire() {
    $("new").addEventListener("click", function () {
      state.history = []; state.last = null; state.sel = null; play({ fen: START, record: false });
    });
    $("undo").addEventListener("click", undo);
    $("flip").addEventListener("click", function () { state.flipped = !state.flipped; render(); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "f" || e.key === "F") { state.flipped = !state.flipped; render(); }
    });
    function slider(id, outId, key) {
      var el = $(id);
      el.addEventListener("input", function () {
        state[key] = parseInt(el.value, 10);
        $(outId).textContent = el.value;
        play({ fen: state.fen, record: false });   // re-eval current position
      });
    }
    slider("white-elo", "white-elo-out", "we");
    slider("black-elo", "black-elo-out", "be");
    $("load-fen").addEventListener("click", function () {
      state.history = []; state.last = null; state.sel = null;
      play({ fen: $("fen").value.trim(), record: false });
    });
  }

  wire();
  play({ fen: START, record: false });
})();
