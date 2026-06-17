/* overlay.js — render the equity bar from a feed of events.
 *
 * Reads config from query params, opens the feed (feed.js), and updates the
 * DOM on each event. Pure formatting helpers are exposed on window.EquityOverlay
 * for testing.
 */
(function (global) {
  "use strict";

  // ---- pure helpers (unit-testable) -------------------------------------

  function clamp01(x) {
    if (typeof x !== "number" || isNaN(x)) return 0.5;
    return x < 0 ? 0 : x > 1 ? 1 : x;
  }

  // White-POV equity (0..1) -> percent string for each side.
  function pct(equityWhite, side) {
    const w = clamp01(equityWhite);
    const v = side === "white" ? w : 1 - w;
    return Math.round(v * 100) + "%";
  }

  // Centipawns (White POV) -> a 0..1 position for the ghost tick, via the
  // standard logistic Lichess uses. Independent of the practical equity, so
  // the two visibly diverge under time pressure — the whole point.
  function cpToWhitePos(cp) {
    if (typeof cp !== "number" || isNaN(cp)) return null;
    return 1 / (1 + Math.exp(-0.00368208 * cp));
  }

  // Seconds -> M:SS (or H:MM:SS). Sub-10s shows tenths for the scramble feel.
  function formatClock(secs) {
    if (typeof secs !== "number" || isNaN(secs) || secs < 0) return "";
    if (secs < 10) return secs.toFixed(1);
    const s = Math.floor(secs);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = String(s % 60).padStart(2, "0");
    return h > 0 ? h + ":" + String(m).padStart(2, "0") + ":" + ss : m + ":" + ss;
  }

  // ---- DOM wiring --------------------------------------------------------

  function params() {
    const p = new URLSearchParams(global.location ? global.location.search : "");
    return {
      src: p.get("src") || "./mock-game.json",
      layout: p.get("layout") || "horizontal",
      theme: p.get("theme") || "dark",
      cp: p.get("cp") !== "0",
      speed: parseFloat(p.get("speed")) || 1,
    };
  }

  function q(sel) {
    return document.querySelector(sel);
  }

  function setText(sel, val) {
    const el = q(sel);
    if (el) el.textContent = val == null ? "" : String(val);
  }

  let gradeTimer = null;

  function applyGame(evt) {
    const pl = evt.players || {};
    if (pl.white) {
      setText("[data-white-name]", pl.white.name || "White");
      setText("[data-white-rating]", pl.white.rating != null ? pl.white.rating : "");
    }
    if (pl.black) {
      setText("[data-black-name]", pl.black.name || "Black");
      setText("[data-black-rating]", pl.black.rating != null ? pl.black.rating : "");
    }
  }

  function applyPosition(evt, cfg) {
    const eq = clamp01(evt.equity);

    // Bar widths.
    const whiteEl = q("[data-bar-white]");
    if (whiteEl) {
      const dim = cfg.layout === "vertical" ? "height" : "width";
      whiteEl.style[dim] = (eq * 100).toFixed(1) + "%";
    }
    setText("[data-white-pct]", pct(eq, "white"));
    setText("[data-black-pct]", pct(eq, "black"));

    // Centipawn ghost tick (the divergence cue).
    const ghost = q("[data-cp-ghost]");
    if (ghost) {
      const pos = cfg.cp ? cpToWhitePos(evt.cp) : null;
      if (pos == null) {
        ghost.classList.remove("show");
      } else {
        ghost.classList.add("show");
        ghost.style.left = (pos * 100).toFixed(1) + "%";
      }
    }

    // Clocks (rolling players carry over from the game event).
    const clk = evt.clock || {};
    updateClock("[data-white-clock]", clk.white);
    updateClock("[data-black-clock]", clk.black);

    // Per-move Δequity grade pill.
    if (evt.grade && evt.grade.label) showGrade(evt.grade);

    // Late-arriving player metadata in a position event.
    if (evt.players) applyGame(evt);
  }

  function updateClock(sel, secs) {
    const el = q(sel);
    if (!el) return;
    if (secs == null) {
      el.textContent = "";
      return;
    }
    el.textContent = formatClock(secs);
    el.classList.toggle("low", secs < 10);
  }

  function showGrade(grade) {
    const el = q("[data-grade]");
    if (!el) return;
    el.textContent = grade.label + (grade.delta != null ? " " + fmtDelta(grade.delta) : "");
    el.classList.toggle("bad", (grade.delta || 0) < 0);
    el.hidden = false;
    if (gradeTimer) clearTimeout(gradeTimer);
    gradeTimer = setTimeout(function () {
      el.hidden = true;
    }, 3500);
  }

  function fmtDelta(d) {
    const sign = d >= 0 ? "+" : "";
    return sign + Math.round(d * 100) + "%";
  }

  function dispatch(evt, cfg) {
    if (!evt || !evt.type) return;
    if (evt.type === "game") applyGame(evt);
    else if (evt.type === "position") applyPosition(evt, cfg);
  }

  function start() {
    const cfg = params();
    const root = q("#overlay");
    if (root) {
      root.classList.remove("layout-horizontal", "layout-vertical");
      root.classList.add("layout-" + cfg.layout);
    }
    document.body.className = "theme-" + cfg.theme;

    if (global.EquityFeed) {
      global.EquityFeed.connect(cfg.src, {
        speed: cfg.speed,
        onEvent: function (evt) {
          dispatch(evt, cfg);
        },
        onError: function (e) {
          // Stay silent on-screen (it's an overlay); log for setup debugging.
          if (global.console) console.warn("[equity-overlay] feed error", e);
        },
      });
    }
  }

  // Expose pure helpers for tests; auto-start in a browser.
  global.EquityOverlay = {
    clamp01: clamp01,
    pct: pct,
    cpToWhitePos: cpToWhitePos,
    formatClock: formatClock,
    fmtDelta: fmtDelta,
  };

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", start);
    } else {
      start();
    }
  }
})(typeof window !== "undefined" ? window : this);
