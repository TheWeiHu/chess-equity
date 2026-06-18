/* live.js — the interactive board. One position at a time, scored live by the backend.
 *
 * Model: state.line is a list of nodes [{fen, san, last, resp}] with a cursor state.ply.
 * node[0] is the start position. You can:
 *   • step through the line (first/prev/next/last/scrub/click) — each position is
 *     evaluated lazily by /api/play when you land on it,
 *   • load a famous game (/api/games + /api/game) into the line,
 *   • play a move from the current position — that truncates any future and continues,
 *     so taking over mid-game just branches the line.
 * The browser holds no chess rules: legality, SAN, game-over and the game library all
 * come from python-chess on the server.
 */
(function () {
  "use strict";

  var FILES = "abcdefgh";
  var MATE_CP = 10000;
  var START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  var state = {
    line: [{ fen: START, san: "(start)", last: null, resp: null }],
    ply: 0, sel: null, we: 1500, be: 1500, flipped: false, busy: false,
    meta: null,   // {name, white, black} of a loaded famous game, else null
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
  }

  function renderControls() {
    $("scrub").max = state.line.length - 1;
    $("scrub").value = state.ply;
  }

  function renderBars() {
    var r = node().resp;
    var eqFill = $("equity-fill"), cpFill = $("cp-fill");
    if (!r) {
      $("equity-readout").textContent = "—"; $("cp-readout").textContent = "—";
      $("best").textContent = ""; $("divergence").hidden = true;
      return;
    }
    var eq = r.equity_white, cpW = cpToWhite(r.cp) * 100;
    eqFill.style.width = eq + "%";
    $("equity-readout").textContent = Math.round(eq) + "% White";
    cpFill.style.width = cpW + "%";
    $("cp-readout").textContent = Math.abs(r.cp) >= MATE_CP
      ? (r.cp > 0 ? "#" : "-#") : (r.cp >= 0 ? "+" : "") + (r.cp / 100).toFixed(1);
    $("best").textContent = r.best_san ? "Stockfish prefers " + r.best_san : "";
    var gap = eq - cpW, div = $("divergence");
    if (!r.game_over && Math.abs(gap) >= 15) {
      div.hidden = false;
      div.textContent = "Equity favours " + (gap > 0 ? "White" : "Black") + " by " +
        Math.round(Math.abs(gap)) + " pts over the centipawn bar — at these ratings the " +
        "realistic result differs from perfect play.";
    } else { div.hidden = true; }
  }

  function renderStatus() {
    var r = node().resp, s = $("status");
    if (!r) { s.textContent = "evaluating…"; return; }
    if (r.checkmate) { s.innerHTML = "<span class='san'>Checkmate</span> — " + (r.turn === "white" ? "Black" : "White") + " wins."; return; }
    if (r.stalemate) { s.innerHTML = "<span class='san'>Draw</span> (no result)."; return; }
    s.innerHTML = (r.turn === "white" ? "White" : "Black") + " to move" + (r.check ? " — <span class='san'>check</span>" : "");
  }

  function renderMoves() {
    var ol = $("moves");
    ol.innerHTML = "";
    state.line.forEach(function (nd, i) {
      if (i === 0) return;
      var li = document.createElement("li");
      var label = document.createElement("span");
      label.textContent = (i % 2 === 1 ? Math.ceil(i / 2) + ". " : "") + nd.san;
      li.appendChild(label);
      if (i === state.ply) li.classList.add("current");
      li.addEventListener("click", function () { goPly(i); });
      ol.appendChild(li);
    });
    var cur = ol.querySelector("li.current");
    if (cur && cur.scrollIntoView) cur.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  function renderPlayers() {
    var m = state.meta, html;
    if (m) {
      var blurb = state.branched
        ? "You took over from " + m.name + " — now playing your own line. Pick the game again to reset."
        : m.name + (m.year ? " · " + m.year : "") + " — step through it, or play a move from any point to take over.";
      html = "<div class='side'><span class='nm'>⬜ " + (m.white || "White") + "</span></div>" +
        "<div class='vs'>vs</div><div class='side'><span class='nm'>⬛ " + (m.black || "Black") + "</span></div>" +
        "<div class='blurb'>" + blurb + "</div>";
    } else {
      html = "<div class='side'><span class='nm'>Free play</span></div><div class='vs'></div>" +
        "<div class='side muted'>move either side</div>";
    }
    $("players").innerHTML = html;
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
      if (!ok) { showFenErr(j.error || "error"); return; }
      hideFenErr();
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
      if (!ok) { showFenErr(j.error || "error"); render(); return; }
      hideFenErr();
      j._we = state.we; j._be = state.be;
      // Playing a move drops any future line — if a famous game was loaded, you've now
      // branched into your own line. Flag it so the players card says so.
      var deviated = state.ply < state.line.length - 1 || (state.meta && !state.branched);
      state.line = state.line.slice(0, state.ply + 1);
      state.line.push({ fen: j.fen, san: j.san, last: { from: uci.slice(0, 2), to: uci.slice(2, 4) }, resp: j });
      state.ply++; state.sel = null;
      if (state.meta && deviated) { state.branched = true; renderPlayers(); }
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
    [["q", "♛"], ["r", "♜"], ["b", "♝"], ["n", "♞"]].forEach(function (p) {
      var b = document.createElement("button");
      b.textContent = p[1];
      b.addEventListener("click", function () { menu.remove(); animateMove(baseUci.slice(0, 2), baseUci.slice(2, 4)); doMove(baseUci + p[0]); });
      menu.appendChild(b);
    });
    frame.appendChild(menu);
  }

  function showFenErr(msg) { var e = $("fen-err"); e.hidden = false; e.textContent = msg; }
  function hideFenErr() { $("fen-err").hidden = true; }

  // ---- game library + controls ---------------------------------------------

  function newGame() {
    state.line = [{ fen: START, san: "(start)", last: null, resp: null }];
    state.ply = 0; state.sel = null; state.meta = null; state.branched = false;
    $("game-select").value = "";
    renderPlayers(); goPly(0);
  }

  function loadGame(id) {
    postGet("/api/game?id=" + encodeURIComponent(id), function (g, ok) {
      if (!ok) { showFenErr(g.error || "could not load game"); return; }
      state.line = g.moves.map(function (m) {
        return { fen: m.fen, san: m.san, resp: null,
          last: m.uci ? { from: m.uci.slice(0, 2), to: m.uci.slice(2, 4) } : null };
      });
      state.branched = false;
      state.ply = 0; state.sel = null;
      state.meta = { name: g.name, white: g.white, black: g.black, year: g.year };
      renderPlayers(); goPly(0);
    });
  }

  function loadFen(fen) {
    state.line = [{ fen: fen, san: "(start)", last: null, resp: null }];
    state.ply = 0; state.sel = null; state.meta = null; state.branched = false;
    $("game-select").value = "";
    renderPlayers(); goPly(0);
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
    $("load-fen").addEventListener("click", function () { loadFen($("fen").value.trim()); });
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
        ensureEval(state.ply);   // equity is rating-conditioned → re-score this position
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
  renderPlayers();
  goPly(0);   // start position, evaluated live
})();
