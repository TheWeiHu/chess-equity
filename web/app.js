/* app.js — the chess-equity web demo (task 0010).
 *
 * Loads demo-game.json and renders a board + two bars: the classic centipawn eval
 * and the rating-conditioned equity. Rating sliders re-read equity from the JSON's
 * rating grid; the centipawn bar is rating-blind, so moving a slider moves only the
 * equity bar. No build step, no deps — vanilla DOM.
 */
(function () {
  "use strict";

  var PIECES = {
    K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
    k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
  };

  // ---- pure helpers --------------------------------------------------------

  // Classic Lichess logistic: centipawns (White POV) -> White win fraction [0,1].
  function cpToWhite(cp) {
    return 1 / (1 + Math.exp(-0.00368208 * cp));
  }

  // Snap an arbitrary rating to the nearest available band in the JSON grid.
  function nearestBand(bands, value) {
    return bands.reduce(function (best, b) {
      return Math.abs(b - value) < Math.abs(best - value) ? b : best;
    }, bands[0]);
  }

  function equityAt(move, bands, whiteElo, blackElo) {
    var key = nearestBand(bands, whiteElo) + "-" + nearestBand(bands, blackElo);
    return move.equity[key];
  }

  // ---- state ---------------------------------------------------------------

  var state = { data: null, ply: 0, whiteElo: 1500, blackElo: 1500 };

  function $(id) { return document.getElementById(id); }

  // ---- board ---------------------------------------------------------------

  function renderBoard(fen) {
    var rows = fen.split(" ")[0].split("/");
    var board = $("board");
    board.innerHTML = "";
    for (var r = 0; r < 8; r++) {
      var file = 0;
      var chars = rows[r].split("");
      for (var c = 0; c < chars.length; c++) {
        var ch = chars[c];
        if (/\d/.test(ch)) {
          for (var k = 0; k < parseInt(ch, 10); k++) addSquare(board, r, file++, null);
        } else {
          addSquare(board, r, file++, ch);
        }
      }
    }
  }

  function addSquare(board, rank, file, piece) {
    var sq = document.createElement("div");
    sq.className = "sq " + ((rank + file) % 2 === 0 ? "light" : "dark");
    if (piece) {
      var span = document.createElement("span");
      span.className = "piece " + (piece === piece.toUpperCase() ? "white" : "black");
      span.textContent = PIECES[piece];
      sq.appendChild(span);
    }
    board.appendChild(sq);
  }

  // ---- bars ----------------------------------------------------------------

  function renderBars() {
    var move = state.data.moves[state.ply];
    var eqWhite = equityAt(move, state.data.rating_bands, state.whiteElo, state.blackElo);
    var cpWhite = cpToWhite(move.cp) * 100;

    $("equity-fill").style.width = eqWhite + "%";
    $("equity-readout").textContent = Math.round(eqWhite) + "% White";
    $("cp-fill").style.width = cpWhite + "%";
    $("cp-readout").textContent = (move.cp >= 0 ? "+" : "") + (move.cp / 100).toFixed(1);

    // Call out the flagship divergence between the two bars.
    var gap = eqWhite - cpWhite;
    var div = $("divergence");
    if (Math.abs(gap) >= 15) {
      div.hidden = false;
      var side = gap > 0 ? "White" : "Black";
      div.textContent =
        "Equity favours " + side + " by " + Math.round(Math.abs(gap)) +
        " pts over the centipawn bar — the material count misreads this position.";
    } else {
      div.hidden = true;
    }
  }

  // ---- move list -----------------------------------------------------------

  function renderMoves() {
    var ol = $("moves");
    ol.innerHTML = "";
    state.data.moves.forEach(function (move, i) {
      var li = document.createElement("li");
      var num = i === 0 ? "" : Math.ceil(i / 2) + (i % 2 === 1 ? "." : "…") + " ";
      li.textContent = num + move.san;
      if (move.grade) {
        if (move.grade.delta > 0) li.classList.add("good");
        else if (move.grade.delta < 0) li.classList.add("bad");
        var g = document.createElement("span");
        g.className = "grade";
        g.textContent = move.grade.label;
        li.appendChild(g);
      }
      if (i === state.ply) li.classList.add("current");
      li.addEventListener("click", function () { goto(i); });
      ol.appendChild(li);
    });
  }

  function renderCaption() {
    var move = state.data.moves[state.ply];
    var cap = $("move-caption");
    if (state.ply === 0) {
      cap.innerHTML = "<span class='san'>Starting position</span>";
      return;
    }
    var g = move.grade ? " — " + move.grade.label + " (" +
      (move.grade.delta > 0 ? "+" : "") + move.grade.delta + ")" : "";
    cap.innerHTML = "Move " + Math.ceil(state.ply / 2) + ": <span class='san'>" +
      move.san + "</span>" + g;
  }

  // ---- wiring --------------------------------------------------------------

  function render() {
    renderBoard(state.data.moves[state.ply].fen);
    renderBars();
    renderMoves();
    renderCaption();
    $("scrub").value = state.ply;
  }

  function goto(ply) {
    state.ply = Math.max(0, Math.min(state.data.moves.length - 1, ply));
    render();
  }

  function setupRatingSlider(id, outId, key, defaultElo) {
    var bands = state.data.rating_bands;
    var el = $(id);
    el.min = 0;
    el.max = bands.length - 1;
    // Default to the game's real rating (nearest band) so imports auto-fill; fall
    // back to 1500, then the lowest band.
    var want = nearestBand(bands, defaultElo || 1500);
    el.value = bands.indexOf(want) >= 0 ? bands.indexOf(want) : 0;
    function apply() {
      var elo = bands[parseInt(el.value, 10)];
      state[key] = elo;
      $(outId).textContent = elo;
      renderBars();
    }
    el.addEventListener("input", apply);
    apply();
  }

  function init(data) {
    state.data = data;
    $("scrub").max = data.moves.length - 1;
    var g = data.game || {};
    if (g.name) document.title = g.name + " — chess-equity";
    setupRatingSlider("white-elo", "white-elo-out", "whiteElo", g.white_elo_default);
    setupRatingSlider("black-elo", "black-elo-out", "blackElo", g.black_elo_default);
    $("prev").addEventListener("click", function () { goto(state.ply - 1); });
    $("next").addEventListener("click", function () { goto(state.ply + 1); });
    $("scrub").addEventListener("input", function (e) { goto(parseInt(e.target.value, 10)); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") goto(state.ply - 1);
      if (e.key === "ArrowRight") goto(state.ply + 1);
    });
    render();
  }

  // Which game file to load: ?game=<name.json>, defaulting to the bundled demo.
  // Restrict to a bare JSON filename in this folder (no scheme, path, or traversal)
  // so the param can't point the page at an arbitrary URL.
  function gameFile() {
    var raw = new URLSearchParams(window.location.search).get("game");
    if (raw && /^[A-Za-z0-9._-]+\.json$/.test(raw) && raw.indexOf("..") === -1) {
      return raw;
    }
    return "demo-game.json";
  }

  var file = gameFile();
  fetch(file)
    .then(function (r) { return r.json(); })
    .then(init)
    .catch(function (err) {
      document.body.insertAdjacentHTML(
        "beforeend",
        "<p style='color:#e0585b;padding:1.5rem'>Could not load " + file + " (" +
          err + "). Serve this folder over HTTP, e.g. <code>python3 -m http.server -d web</code>.</p>"
      );
    });

  // Expose pure helpers for testing.
  window.ChessEquityDemo = { cpToWhite: cpToWhite, nearestBand: nearestBand, equityAt: equityAt };
})();
