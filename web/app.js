/* app.js — the chess-equity web demo (task 0010).
 *
 * Loads demo-game.json and renders a board + two bars: the classic centipawn eval
 * and the rating-conditioned equity. Rating sliders re-read equity from the JSON's
 * rating grid; the centipawn bar is rating-blind, so moving a slider moves only the
 * equity bar. No build step, no deps — vanilla DOM.
 */
(function () {
  "use strict";

  // Board rendering (piece glyphs + squares) is shared with the live demo via board.js.

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

  // Place the floating board-preview near the cursor without spilling off-screen.
  // Default is cursor + `off`; if that would overflow the right/bottom viewport edge,
  // flip the box to the other side of the cursor, then clamp so it never leaves the
  // top/left. `vw`/`vh` of 0 (no measurement available) means "don't clamp" — the
  // plain cursor+off placement, the historical behaviour. Pure → unit-testable.
  function clampPreviewPos(cx, cy, w, h, vw, vh, off) {
    var left = cx + off;
    if (vw && left + w + off > vw) left = cx - off - w;
    if (vw && left < off) left = off;
    var top = cy + off;
    if (vh && top + h + off > vh) top = cy - off - h;
    if (vh && top < off) top = off;
    return { left: left, top: top };
  }

  // Geometry for the across-the-game chart: equity (rating-conditioned) vs the classic
  // centipawn bar, both as White win% per ply. Pure (no DOM) so it is unit-testable;
  // renderChart() turns it into SVG. Y is flipped (0% at the bottom).
  function chartGeometry(moves, bands, whiteElo, blackElo, opts) {
    opts = opts || {};
    var W = opts.width || 480, H = opts.height || 160, pad = opts.pad || 24;
    var innerW = W - 2 * pad, innerH = H - 2 * pad;
    var n = moves.length;
    function xFor(i) { return n <= 1 ? pad + innerW / 2 : pad + (i / (n - 1)) * innerW; }
    function yFor(pct) { return pad + (1 - pct / 100) * innerH; }
    // Drama (clutch / missed_win / escape) is precomputed per band in build_demo.py;
    // look up the events for the band the slider currently selects (sparse — most
    // bands have none) and mark those plies on the equity line.
    var dramaByPly = {};
    if (opts.drama) {
      var dKey = nearestBand(bands, whiteElo) + "-" + nearestBand(bands, blackElo);
      (opts.drama[dKey] || []).forEach(function (d) { dramaByPly[d.ply] = d; });
    }
    var points = moves.map(function (m, i) {
      var eq = equityAt(m, bands, whiteElo, blackElo);
      var cp = cpToWhite(m.cp) * 100;
      var grade = m.grade ? " · " + m.grade.label + " " + (m.grade.delta > 0 ? "+" : "") + m.grade.delta : "";
      var drama = dramaByPly[i] || null;
      return {
        ply: i, x: xFor(i), eqY: yFor(eq), cpY: yFor(cp), eqVal: eq, cpVal: cp, drama: drama,
        label: (i === 0 ? "start" : m.san) + ": equity " + Math.round(eq) +
          "% · cp-bar " + Math.round(cp) + "%" + grade +
          (drama ? " · ⚡ " + drama.kind + ": " + drama.headline : ""),
      };
    });
    return {
      width: W, height: H, pad: pad, y50: yFor(50), xFor: xFor, yFor: yFor, points: points,
      eqPoints: points.map(function (p) { return p.x + "," + p.eqY; }).join(" "),
      cpPoints: points.map(function (p) { return p.x + "," + p.cpY; }).join(" "),
    };
  }

  // ---- state ---------------------------------------------------------------

  var state = { data: null, ply: 0, whiteElo: 1500, blackElo: 1500 };

  function $(id) { return document.getElementById(id); }

  // ---- board ---------------------------------------------------------------
  // Review board (non-interactive) via the shared renderer. The main board gets edge
  // coordinates; the floating mini-preview (boardEl passed) stays clean.

  function renderBoard(fen, boardEl) {
    ChessBoard.render(boardEl || $("board"), fen, { coords: !boardEl });
  }

  // ---- hover board preview -------------------------------------------------
  // A caster scanning the equity curve wants to *see* a ply's position without
  // clicking (which would move the main board and lose their place). Hovering a
  // chart dot renders that ply into a small floating board, reusing renderBoard.

  function showPreview(ply) {
    var preview = $("board-preview");
    if (!preview) return;
    var move = state.data.moves[ply];
    renderBoard(move.fen, $("preview-board"));
    var cap = $("preview-caption");
    if (cap) {
      cap.textContent = ply === 0
        ? "Starting position"
        : Math.ceil(ply / 2) + (ply % 2 === 1 ? ". " : "… ") + move.san;
    }
    preview.hidden = false;
  }

  function movePreview(ev) {
    var preview = $("board-preview");
    if (!preview || preview.hidden) return;
    var rect = preview.getBoundingClientRect ? preview.getBoundingClientRect() : null;
    var pos = clampPreviewPos(
      ev.clientX, ev.clientY,
      rect ? rect.width : 0, rect ? rect.height : 0,
      window.innerWidth || 0, window.innerHeight || 0,
      16
    );
    preview.style.left = pos.left + "px";
    preview.style.top = pos.top + "px";
  }

  function hidePreview() {
    var preview = $("board-preview");
    if (preview) preview.hidden = true;
  }

  // Position the (already-shown) preview for a tap or a keyboard focus — touch/keyboard
  // users have no moving pointer, so place it from the tap's coordinates when present,
  // else next to the focused dot. Mouse hover keeps using movePreview, untouched.
  function placePreview(ev, el) {
    var preview = $("board-preview");
    if (!preview || preview.hidden) return;
    var x, y;
    if (ev && ev.clientX != null) {
      x = ev.clientX; y = ev.clientY;
    } else if (el && el.getBoundingClientRect) {
      var r = el.getBoundingClientRect();
      x = r.left; y = r.bottom;
    } else {
      return;
    }
    preview.style.left = x + 16 + "px";
    preview.style.top = y + 16 + "px";
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
      var barLabel =
        state.data.cp_engine === "stockfish"
          ? "the engine's objective eval"
          : "the material count";
      div.textContent =
        "Equity favours " + side + " by " + Math.round(Math.abs(gap)) +
        " pts over the centipawn bar — " + barLabel + " misreads this position.";
    } else {
      div.hidden = true;
    }
  }

  // ---- across-the-game chart -----------------------------------------------

  var SVG_NS = "http://www.w3.org/2000/svg";

  function svgEl(name, attrs) {
    var el = document.createElementNS(SVG_NS, name);
    for (var k in attrs) { if (attrs[k] != null) el.setAttribute(k, attrs[k]); }
    return el;
  }

  function renderChart() {
    var svg = $("chart");
    if (!svg) return;
    var g = chartGeometry(
      state.data.moves, state.data.rating_bands, state.whiteElo, state.blackElo,
      { width: 480, height: 160, pad: 24, drama: state.data.drama }
    );
    svg.setAttribute("viewBox", "0 0 " + g.width + " " + g.height);
    svg.innerHTML = "";
    // 50% reference line.
    svg.appendChild(svgEl("line", {
      x1: g.pad, y1: g.y50, x2: g.width - g.pad, y2: g.y50, class: "chart-mid",
    }));
    // Current-ply cursor.
    var cx = g.xFor(state.ply);
    svg.appendChild(svgEl("line", {
      x1: cx, y1: g.pad, x2: cx, y2: g.height - g.pad, class: "chart-cursor",
    }));
    // The two lines: centipawn bar (rating-blind) under equity (rating-conditioned).
    svg.appendChild(svgEl("polyline", { points: g.cpPoints, class: "chart-cp" }));
    svg.appendChild(svgEl("polyline", { points: g.eqPoints, class: "chart-eq" }));
    // One dot per ply on the equity line: native <title> tooltip on hover, click
    // scrubs, and hover pops a floating board preview of that ply (no click needed).
    g.points.forEach(function (p) {
      var dot = svgEl("circle", {
        cx: p.x, cy: p.eqY, r: p.ply === state.ply ? 4 : (p.drama ? 3.5 : 2.5),
        class: "chart-dot" + (p.ply === state.ply ? " current" : "") +
          (p.drama ? " drama drama-" + p.drama.kind : ""),
      });
      var title = svgEl("title", {});
      title.textContent = p.label;
      dot.appendChild(title);
      dot.addEventListener("click", function () { goto(p.ply); });
      dot.addEventListener("mouseenter", function () { showPreview(p.ply); });
      dot.addEventListener("mousemove", movePreview);
      dot.addEventListener("mouseleave", hidePreview);
      // Touch + keyboard parity (task 0101): tap pops the preview at the tap point, and
      // making the dot focusable lets tab-through reveal it; dismissal is wired globally
      // (outside tap / Escape) plus blur here. Mouse hover above is unchanged.
      dot.setAttribute("tabindex", "0");
      dot.addEventListener("click", function (ev) { showPreview(p.ply); placePreview(ev, dot); });
      dot.addEventListener("focus", function () { showPreview(p.ply); placePreview(null, dot); });
      dot.addEventListener("blur", hidePreview);
      svg.appendChild(dot);
    });
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
    renderChart();
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
      renderChart();  // equity line is rating-conditioned, so it moves with the slider
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
      if (e.key === "Escape") hidePreview();  // dismiss the tap/focus preview (task 0101)
    });
    // Dismiss the preview on a tap/click outside any chart dot. A dot's own click fires
    // first (showing the preview) and this bubbles after; ignore taps on a chart dot so
    // they don't immediately re-hide it.
    document.addEventListener("click", function (e) {
      var t = e.target;
      var onDot = t && t.classList && t.classList.contains && t.classList.contains("chart-dot");
      if (!onDot) hidePreview();
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

  // Populate the catalog selector from games.json so you can scroll through the
  // bundled games. Switching navigates to ?game=<file>; a fresh load keeps the URL,
  // board, sliders and chart all in sync without bespoke teardown. Best-effort: if
  // games.json is missing (e.g. an imported single game), hide the picker.
  function setupGamePicker(current) {
    var sel = $("game-select");
    if (!sel) return;
    fetch("games.json")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (manifest) {
        var games = manifest && manifest.games;
        if (!games || !games.length) { sel.parentNode.hidden = true; return; }
        games.forEach(function (g) {
          var opt = document.createElement("option");
          opt.value = g.file;
          opt.textContent = g.name;
          if (g.file === current) opt.selected = true;
          sel.appendChild(opt);
        });
        sel.addEventListener("change", function () {
          window.location.search = "?game=" + encodeURIComponent(sel.value);
        });
      })
      .catch(function () { sel.parentNode.hidden = true; });
  }

  var file = gameFile();
  setupGamePicker(file);
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
  window.ChessEquityDemo = {
    cpToWhite: cpToWhite, nearestBand: nearestBand, equityAt: equityAt,
    chartGeometry: chartGeometry, clampPreviewPos: clampPreviewPos,
  };
})();
