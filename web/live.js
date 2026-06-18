/* live.js — the interactive board. One position at a time, scored live by the backend.
 *
 * Model: state.line is a list of nodes [{fen, san, last, resp}] with a cursor state.ply.
 * node[0] is the start position. Step through the line (first/prev/next/last/scrub/click/
 * move-list), load a famous game (/api/games + /api/game), or play a move from the current
 * position — that truncates any future and continues, so taking over mid-game branches the
 * line. The browser holds no chess rules: legality, SAN and game-over come from the server.
 */
(function () {
  "use strict";

  var FILES = "abcdefgh";
  var MATE_CP = 10000;
  var START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  // The first two full moves are opening book — Maia-2 equity there is noise that swings
  // wildly for no real reason, so we don't show it (the cp bar is accurate, so it stays).
  // Plies: 0 = start, 1..4 = White/Black move 1 and 2. Equity shows from move 3 (ply 5).
  var BOOK_PLIES = 4;
  function inBook(p) { return p <= BOOK_PLIES; }

  var state = {
    line: [{ fen: START, san: "(start)", last: null, resp: null }],
    ply: 0, sel: null, we: 1500, be: 1500, flipped: false, busy: false,
    meta: null,        // {name, white, black, year} of a loaded famous game, else null
    branched: false,   // true once you've played your own move off a loaded game
  };

  function $(id) { return document.getElementById(id); }
  function node() { return state.line[state.ply]; }
  function legalMap() { var r = node().resp; return (r && r.legal) || {}; }
  function cpToWhite(cp) { return 1 / (1 + Math.exp(-0.00368208 * cp)); }
  function setThinking(on) { var b = document.querySelector(".bars"); if (b) b.classList.toggle("thinking", on); }

  // ---- rendering -----------------------------------------------------------

  function render() {
    var n = node();
    var legal = legalMap();
    var check = null;
    if (n.resp && n.resp.check) {
      var grid = ChessBoard.parseFen(n.fen).grid;
      var king = n.resp.turn === "white" ? "K" : "k";
      for (var r = 0; r < 8 && !check; r++)
        for (var c = 0; c < 8; c++)
          if (grid[r][c] === king) check = ChessBoard.sqName(r, c);
    }
    ChessBoard.render($("board"), n.fen, {
      flipped: state.flipped,
      coords: true,
      highlight: n.last,
      selected: state.sel,
      dests: state.sel ? (legal[state.sel] || []) : [],
      check: check,
      onSquare: onSquare,
      draggable: function (name) { return !!legal[name]; },
      onDragStart: onDragStart,
      onDrop: onDrop,
    });
    renderControls();
    renderBars();
    renderStatus();
    renderMoves();
    renderChart();
  }

  // ---- over-the-game bar chart (White win%: Maia equity vs Stockfish centipawn) ----

  var SVG_NS = "http://www.w3.org/2000/svg";
  function svgEl(name, attrs) {
    var el = document.createElementNS(SVG_NS, name);
    for (var k in attrs) if (attrs[k] != null) el.setAttribute(k, attrs[k]);
    return el;
  }
  // an eval is "fresh" only if it was computed at the current pair of ratings
  function hasFresh(n) { return n.resp && n.resp._we === state.we && n.resp._be === state.be; }

  function renderChart() {
    var svg = $("chart"); if (!svg) return;
    var n = state.line.length, W = 480, H = 150, padX = 10, padY = 10;
    var innerW = W - 2 * padX, innerH = H - 2 * padY;
    var base = H - padY;                               // bars grow up from the bottom
    var mid = padY + innerH / 2;                       // the 50% reference line
    function xFor(i) { return n <= 1 ? padX + innerW / 2 : padX + (i / (n - 1)) * innerW; }
    var slot = n <= 1 ? innerW : innerW / (n - 1);
    // each MOVE is a tight pair: equity (green) just left of centre, centipawn (grey)
    // just right, with a clear gap to the next move's pair.
    var bw = Math.max(1.2, Math.min(6, slot * 0.30)), gap = Math.max(0.5, bw * 0.22);
    svg.innerHTML = "";
    // a soft highlighted box behind the CURRENT move's pair — the clean "you are here"
    var hx = xFor(state.ply), boxW = Math.max(bw * 2 + gap + 5, slot * 0.74);
    svg.appendChild(svgEl("rect", { x: hx - boxW / 2, y: padY - 2, width: boxW, height: innerH + 4, rx: 3, class: "chart-now" }));
    svg.appendChild(svgEl("line", { x1: padX, y1: mid, x2: W - padX, y2: mid, class: "chart-mid" }));
    function bar(x, v, cls) {                          // v in 0..100 (White win%)
      var h = Math.max(1.5, (v / 100) * innerH);
      svg.appendChild(svgEl("rect", { x: x, y: base - h, width: bw, height: h, rx: Math.min(1.4, bw / 2), class: cls }));
    }
    state.line.forEach(function (nd, i) {
      if (!hasFresh(nd)) return;
      var cx = xFor(i);
      if (!inBook(i)) bar(cx - bw - gap / 2, nd.resp.equity_white, "bar-eq");  // Maia equity
      bar(cx + gap / 2, cpToWhite(nd.resp.cp) * 100, "bar-cp");                // Stockfish
    });
    // generous transparent click target + tooltip per move
    state.line.forEach(function (nd, i) {
      var hit = svgEl("rect", { x: xFor(i) - slot / 2, y: padY, width: Math.max(slot, 2), height: innerH, fill: "transparent", style: "cursor:pointer" });
      var t = svgEl("title", {});
      var cpv = hasFresh(nd) ? Math.round(cpToWhite(nd.resp.cp) * 100) : "…";
      var eqv = hasFresh(nd) && !inBook(i) ? Math.round(nd.resp.equity_white) + "%" : (inBook(i) ? "book" : "…");
      t.textContent = (i === 0 ? "start" : nd.san) + " — Maia " + eqv + " · Stockfish " + cpv + "%";
      hit.appendChild(t);
      hit.addEventListener("click", function () { goPly(i); });
      svg.appendChild(hit);
    });
  }

  // Progressively evaluate every ply in the background so the chart fills in over the
  // whole game. A generation token cancels the run when the game or ratings change.
  var fillGen = 0;
  function startFill() {
    var gen = ++fillGen, i = 0;
    function step() {
      if (gen !== fillGen) return;
      while (i < state.line.length && hasFresh(state.line[i])) i++;
      updateProgress();
      if (i >= state.line.length) return;
      var p = i, nd = state.line[p];
      postPlay({ fen: nd.fen }, function (j, ok) {
        if (gen !== fillGen) return;
        if (ok) { j._we = state.we; j._be = state.be; nd.resp = j; renderChart(); if (state.ply === p) { renderBars(); renderStatus(); } }
        i = p + 1; setTimeout(step, 0);
      });
    }
    step();
  }
  function updateProgress() {
    var el = $("chart-progress"); if (!el) return;
    var done = state.line.filter(hasFresh).length, total = state.line.length;
    el.textContent = done < total ? "charting " + done + "/" + total + "…" : "";
  }

  function renderControls() {
    $("scrub").max = state.line.length - 1;
    $("scrub").value = state.ply;
  }

  // The grouped-scores card heading: which position the two bars are reading.
  function headText() {
    if (state.ply === 0) return "Starting position";
    var s = node().san;
    return s ? "After " + s : "Move " + state.ply;
  }
  // Fill the "why the bars differ" card. It is always populated (never an empty box):
  // the green "differ" state when the two reads disagree, a muted state otherwise.
  function setDiff(kind, text) {
    var div = $("divergence");
    div.className = "divergence" + (kind === "differ" ? "" : " agree");
    div.textContent = text;
  }

  function renderBars() {
    var sh = $("scores-head"); if (sh) sh.textContent = headText();
    var r = node().resp;
    if (!r) {
      $("equity-readout").textContent = "—"; $("cp-readout").textContent = "—";
      setDiff("agree", "Evaluating this position…");
      return;
    }
    var eq = r.equity_white, cpW = cpToWhite(r.cp) * 100;
    var book = inBook(state.ply);
    // Opening book: equity is meaningless this early, so park the bar at even and say "Book".
    $("equity-fill").style.width = (book ? 50 : eq) + "%";
    $("equity-readout").textContent = book ? "Book" : Math.round(eq) + "% White";
    $("equity-readout").classList.toggle("book", book);
    $("cp-fill").style.width = cpW + "%";
    $("cp-readout").textContent = Math.abs(r.cp) >= MATE_CP
      ? (r.cp > 0 ? "#" : "-#") : (r.cp >= 0 ? "+" : "") + (r.cp / 100).toFixed(1);
    var gap = eq - cpW;
    if (book) {
      setDiff("agree", "Opening book — equity stays parked until the position leaves theory.");
    } else if (r.game_over) {
      setDiff("agree", "The game is decided here.");
    } else if (Math.abs(gap) >= 15) {
      setDiff("differ", "Equity favours " + (gap > 0 ? "White" : "Black") + " by " +
        Math.round(Math.abs(gap)) + " pts over the centipawn bar — at these ratings the " +
        "realistic result differs from perfect play.");
    } else {
      setDiff("agree", "Equity and the engine agree here — the rating-conditioned read " +
        "tracks near-perfect play.");
    }
  }

  function renderStatus() {
    var r = node().resp, s = $("status");
    // The loaded game (or "Your line" once you've branched) is shown here — no separate card.
    var ctx = state.meta ? (state.branched ? "Your line" : state.meta.name) + " · " : "";
    if (!r) { s.textContent = ctx + "evaluating…"; return; }
    if (r.checkmate) { s.innerHTML = ctx + "<span class='san'>Checkmate</span> — " + (r.turn === "white" ? "Black" : "White") + " wins."; return; }
    if (r.stalemate) { s.innerHTML = ctx + "<span class='san'>Draw</span> (no result)."; return; }
    s.innerHTML = ctx + (r.turn === "white" ? "White" : "Black") + " to move" + (r.check ? " — <span class='san'>check</span>" : "");
  }

  function renderMoves() {
    var box = $("moves");
    box.innerHTML = "";
    // one row per full move: number, White ply, Black ply — a vertical scroller.
    for (var i = 1; i < state.line.length; i += 2) {
      var row = document.createElement("div");
      row.className = "mv-row";
      var num = document.createElement("span");
      num.className = "mv-num";
      num.textContent = Math.ceil(i / 2) + ".";
      row.appendChild(num);
      [i, i + 1].forEach(function (p) {
        var cell = document.createElement("span");
        if (p < state.line.length) {
          cell.className = "mv-ply" + (p === state.ply ? " current" : "");
          cell.textContent = state.line[p].san;
          cell.addEventListener("click", function () { goPly(p); });
        } else {
          cell.className = "mv-ply empty";
        }
        row.appendChild(cell);
      });
      box.appendChild(row);
    }
    var cur = box.querySelector(".mv-ply.current");
    if (cur && cur.scrollIntoView) cur.scrollIntoView({ block: "nearest" });
  }

  // ---- evaluation + navigation ---------------------------------------------

  // Ensure the node at ply p has a fresh eval (for the current ratings), then refresh.
  function ensureEval(p) {
    var n = state.line[p];
    if (n.resp && n.resp._we === state.we && n.resp._be === state.be) return;
    if (state.busy) return;
    state.busy = true; setThinking(true);
    postPlay({ fen: n.fen }, function (j, ok) {
      state.busy = false; setThinking(false);
      if (!ok) { showErr(j.error || "error"); return; }
      j._we = state.we; j._be = state.be;
      n.resp = j;
      if (state.ply === p) render();
    });
  }

  function goPly(p) {
    state.ply = Math.max(0, Math.min(state.line.length - 1, p));
    state.sel = null;
    render();
    ensureEval(state.ply);
  }

  function postPlay(extra, cb) {
    var body = { white_elo: state.we, black_elo: state.be };
    for (var k in extra) body[k] = extra[k];
    fetch("/api/play", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then(function (r) { return r.json().then(function (j) { return [r.ok, j]; }); })
      .then(function (rj) { cb(rj[1], rj[0]); })
      .catch(function (err) { cb({ error: String(err) }, false); });
  }

  function showErr(msg) { $("status").textContent = msg; }

  // ---- interaction ---------------------------------------------------------

  function onSquare(name) {
    if (state.busy) return;
    var legal = legalMap();
    if (state.sel && (legal[state.sel] || []).indexOf(name) >= 0) { attemptMove(state.sel, name); return; }
    state.sel = legal[name] ? name : null;
    render();
  }

  function attemptMove(from, to) {
    var grid = ChessBoard.parseFen(node().fen).grid;
    var piece = grid[8 - parseInt(from[1], 10)][FILES.indexOf(from[0])];
    var uci = from + to, toRank = parseInt(to[1], 10);
    if ((piece === "P" && toRank === 8) || (piece === "p" && toRank === 1)) { promote(uci); return; }
    animateMove(from, to);
    doMove(uci);
  }

  // Play `uci` from the current position: truncate any future line and continue.
  function doMove(uci) {
    if (state.busy) return;
    state.busy = true; setThinking(true);
    var base = node().fen;
    postPlay({ fen: base, uci: uci }, function (j, ok) {
      state.busy = false; setThinking(false);
      if (!ok) { showErr(j.error || "error"); render(); return; }
      // Playing a move drops any future line — if a famous game was loaded, you've now
      // branched into your own line (the status caption then says "Your line").
      var deviated = state.ply < state.line.length - 1 || (state.meta && !state.branched);
      j._we = state.we; j._be = state.be;
      state.line = state.line.slice(0, state.ply + 1);
      state.line.push({ fen: j.fen, san: j.san, last: { from: uci.slice(0, 2), to: uci.slice(2, 4) }, resp: j });
      state.ply++; state.sel = null;
      if (state.meta && deviated) state.branched = true;
      render();
    });
  }

  // Optimistic, animated piece move (FLIP slide); the server response then syncs.
  function animateMove(from, to) {
    var board = $("board");
    var fromSq = board.querySelector('.sq[data-name="' + from + '"]');
    var toSq = board.querySelector('.sq[data-name="' + to + '"]');
    if (!fromSq || !toSq) return;
    var piece = fromSq.querySelector(".piece");
    if (!piece) return;
    board.querySelectorAll(".sq.sel, .sq.dest, .sq.capture").forEach(function (s) { s.classList.remove("sel", "dest", "capture"); });
    var fr = fromSq.getBoundingClientRect(), tr = toSq.getBoundingClientRect();
    var dx = tr.left - fr.left, dy = tr.top - fr.top;
    var captured = toSq.querySelector(".piece");
    if (captured) captured.remove();
    toSq.appendChild(piece);
    if (piece.animate) {
      piece.animate([{ transform: "translate(" + (-dx) + "px," + (-dy) + "px)" }, { transform: "none" }],
        { duration: 150, easing: "cubic-bezier(.4,0,.2,1)" });
    }
  }

  function onDragStart(name, ev) {
    if (state.busy) { ev.preventDefault(); return; }
    state.sel = name;
    paintDrag(name);
    if (ev.dataTransfer) { ev.dataTransfer.effectAllowed = "move"; ev.dataTransfer.setData("text/plain", name); }
    ev.target.addEventListener("dragend", function () { if (!state.busy) render(); }, { once: true });
  }
  function onDrop(name) {
    var legal = legalMap();
    if (state.sel && (legal[state.sel] || []).indexOf(name) >= 0) attemptMove(state.sel, name);
    else render();
  }
  function squareEl(name) { return document.querySelector('#board .sq[data-name="' + name + '"]'); }
  function paintDrag(from) {
    document.querySelectorAll("#board .sq").forEach(function (sq) { sq.classList.remove("dest", "capture", "sel"); });
    var src = squareEl(from); if (src) src.classList.add("sel");
    (legalMap()[from] || []).forEach(function (to) {
      var sq = squareEl(to); if (!sq) return;
      sq.classList.add("dest"); if (sq.querySelector(".piece")) sq.classList.add("capture");
    });
  }

  function promote(baseUci) {
    var frame = document.querySelector(".board-frame");
    var old = frame.querySelector(".promo"); if (old) old.remove();
    var menu = document.createElement("div");
    menu.className = "promo";
    ["q", "r", "b", "n"].forEach(function (p) {
      var b = document.createElement("button");
      // same clean glyph set as the board, so the picker matches the experience
      b.innerHTML = '<span class="piece black">' + window.ChessBoard.glyph(p) + '</span>';
      b.setAttribute("aria-label", { q: "queen", r: "rook", b: "bishop", n: "knight" }[p]);
      b.addEventListener("click", function () { menu.remove(); animateMove(baseUci.slice(0, 2), baseUci.slice(2, 4)); doMove(baseUci + p); });
      menu.appendChild(b);
    });
    frame.appendChild(menu);
  }

  // ---- game library + controls ---------------------------------------------

  function newGame() {
    state.line = [{ fen: START, san: "(start)", last: null, resp: null }];
    state.ply = 0; state.sel = null; state.meta = null; state.branched = false;
    $("game-select").value = "";
    goPly(0); startFill();
  }

  function loadGame(id) {
    postGet("/api/game?id=" + encodeURIComponent(id), function (g, ok) {
      if (!ok) { showErr(g.error || "could not load game"); return; }
      state.line = g.moves.map(function (m) {
        return { fen: m.fen, san: m.san, resp: null,
          last: m.uci ? { from: m.uci.slice(0, 2), to: m.uci.slice(2, 4) } : null };
      });
      state.branched = false;
      state.ply = 0; state.sel = null;
      state.meta = { name: g.name, white: g.white, black: g.black, year: g.year };
      goPly(0); startFill();
    });
  }

  function postGet(url, cb) {
    fetch(url).then(function (r) { return r.json().then(function (j) { return [r.ok, j]; }); })
      .then(function (rj) { cb(rj[1], rj[0]); })
      .catch(function (err) { cb({ error: String(err) }, false); });
  }

  function wire() {
    $("first").addEventListener("click", function () { goPly(0); });
    $("prev").addEventListener("click", function () { goPly(state.ply - 1); });
    $("next").addEventListener("click", function () { goPly(state.ply + 1); });
    $("last").addEventListener("click", function () { goPly(state.line.length - 1); });
    $("flip").addEventListener("click", function () { state.flipped = !state.flipped; render(); });
    $("scrub").addEventListener("input", function (e) { goPly(parseInt(e.target.value, 10)); });
    $("new").addEventListener("click", newGame);
    document.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") goPly(state.ply - 1);
      else if (e.key === "ArrowRight") goPly(state.ply + 1);
      else if (e.key === "Home") goPly(0);
      else if (e.key === "End") goPly(state.line.length - 1);
      else if (e.key === "f" || e.key === "F") { state.flipped = !state.flipped; render(); }
    });
    function slider(id, outId, key) {
      var el = $(id);
      el.addEventListener("input", function () {
        state[key] = parseInt(el.value, 10);
        $(outId).textContent = el.value;
        ensureEval(state.ply);   // re-score the current position now…
        startFill();             // …and re-chart the whole game at the new ratings
      });
    }
    slider("white-elo", "white-elo-out", "we");
    slider("black-elo", "black-elo-out", "be");
  }

  function loadLibrary() {
    postGet("/api/games", function (data, ok) {
      var sel = $("game-select");
      var blank = document.createElement("option");
      blank.value = ""; blank.textContent = ok && data.games && data.games.length ? "— pick a game —" : "(start the server for the library)";
      sel.appendChild(blank);
      if (ok && data.games) {
        data.games.forEach(function (g) {
          var o = document.createElement("option");
          o.value = g.id;
          o.textContent = g.name + (g.year ? " (" + g.year + ")" : "") + " · " + g.plies + " plies";
          sel.appendChild(o);
        });
      }
      sel.addEventListener("change", function () { if (sel.value) loadGame(sel.value); });
    });
  }

  wire();
  loadLibrary();
  goPly(0);   // start position, evaluated live
  startFill();
})();
