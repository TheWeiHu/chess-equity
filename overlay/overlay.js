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

  // Caster mode (task 0022): a big PRACTICAL equity swing — and a flag for the
  // swings the engine misses (practical bar moves far more than the centipawn bar).
  // All inputs White-POV; returns null when the swing is too small to flare on.
  // MAX_SWING (0.40) mirrors chess_equity.drama.MAX_SWING so magnitude is comparable.
  function dramaSwing(prevEquity, equity, prevCp, cp, opts) {
    opts = opts || {};
    var minSwing = opts.minSwing == null ? 0.1 : opts.minSwing; // 10pts to flare
    var blindRatio = opts.blindRatio == null ? 2.0 : opts.blindRatio;
    if (typeof prevEquity !== "number" || typeof equity !== "number") return null;
    if (isNaN(prevEquity) || isNaN(equity)) return null;
    var swing = equity - prevEquity;
    if (Math.abs(swing) < minSwing) return null;
    var cpA = cpToWhitePos(prevCp);
    var cpB = cpToWhitePos(cp);
    var engineBlind;
    if (cpA == null || cpB == null) {
      engineBlind = true; // no engine eval to compare against
    } else {
      engineBlind = Math.abs(swing) >= blindRatio * Math.abs(cpB - cpA);
    }
    return {
      side: swing > 0 ? "white" : "black",
      swing: swing,
      magnitude: Math.min(1, Math.abs(swing) / 0.4),
      engineBlind: engineBlind,
    };
  }

  // Practical-vs-engine DIVERGENCE — the wedge made visible (task 0048). Unlike
  // dramaSwing (a swing INTO a move), this is a LEVEL comparison at one position:
  // when the practical equity bar and the classic centipawn bar disagree on who is
  // winning by more than `threshold` win-prob points, there is a "human edge" — e.g.
  // the engine says lost only via an inhuman refutation the rated player won't find.
  // Returns null when the two roughly agree. `side` is who the practical bar favors
  // relative to the engine.
  function humanEdge(equityWhite, cp, opts) {
    opts = opts || {};
    var threshold = opts.threshold == null ? 0.15 : opts.threshold; // 15 pts
    if (typeof equityWhite !== "number" || isNaN(equityWhite)) return null;
    var cpPos = cpToWhitePos(cp);
    if (cpPos == null) return null;
    var gap = clamp01(equityWhite) - cpPos;
    if (Math.abs(gap) < threshold) return null;
    return {
      side: gap > 0 ? "white" : "black",
      gap: gap,
      magnitude: Math.min(1, Math.abs(gap) / 0.5),
    };
  }

  // Caster-facing label for a human-edge divergence: which side the practical bar
  // favors, and by how many points beyond what the engine sees.
  function edgeLabel(e) {
    var pts = Math.round(Math.abs(e.gap) * 100);
    var who = e.side === "white" ? "White" : "Black";
    return "human edge · " + who + " +" + pts + " vs engine";
  }

  // Streamer rating override (task 0021): when the setup page sets ?welo=/?belo=,
  // that rating wins over whatever the feed reports. Useful because Maia-2's top
  // band is a coarse ">2000" — a caster can pin the real ratings for context.
  function overrideRating(feedRating, override) {
    if (override != null && override !== "") return override;
    return feedRating == null ? "" : feedRating;
  }

  // Is this side in time trouble? True when a real remaining-seconds reading is at or
  // under the threshold (default 30s, tunable via ?lowclock=). Drives the visual cue
  // (task 0104) so a viewer can see a clock-driven equity shift, not just a positional
  // one. ``null``/missing/negative clocks are never "pressure".
  function timePressure(secs, threshold) {
    return typeof secs === "number" && !isNaN(secs) && secs >= 0 && secs <= threshold;
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
      cpbar: p.get("cpbar") === "1",
      caster: p.get("caster") === "1",
      speed: parseFloat(p.get("speed")) || 1,
      welo: p.get("welo"),
      belo: p.get("belo"),
      lowclock: parseFloat(p.get("lowclock")) || 30,
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
  let dramaTimer = null;
  // Previous position, so caster mode can measure the swing into THIS move.
  let prevEquity = null;
  let prevCp = null;

  function applyGame(evt, cfg) {
    cfg = cfg || {};
    const pl = evt.players || {};
    if (pl.white) {
      setText("[data-white-name]", pl.white.name || "White");
      setText("[data-white-rating]", overrideRating(pl.white.rating, cfg.welo));
    }
    if (pl.black) {
      setText("[data-black-name]", pl.black.name || "Black");
      setText("[data-black-rating]", overrideRating(pl.black.rating, cfg.belo));
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

    const cpPos = cpToWhitePos(evt.cp);

    // Classic centipawn eval — either as a full second bar (?cpbar=1) or a ghost
    // tick on the equity bar (?cp, the default). The full bar supersedes the tick.
    const cpBar = q("[data-cp-bar]");
    if (cpBar) cpBar.hidden = !(cfg.cpbar && cpPos != null);
    if (cfg.cpbar && cpPos != null) {
      const cpWhite = q("[data-cp-bar-white]");
      if (cpWhite) {
        const dim = cfg.layout === "vertical" ? "height" : "width";
        cpWhite.style[dim] = (cpPos * 100).toFixed(1) + "%";
      }
    }
    const ghost = q("[data-cp-ghost]");
    if (ghost) {
      const showTick = cfg.cp && !cfg.cpbar && cpPos != null;
      if (!showTick) {
        ghost.classList.remove("show");
      } else {
        ghost.classList.add("show");
        ghost.style.left = (cpPos * 100).toFixed(1) + "%";
      }
    }

    // Human-edge badge (task 0048): persistent while the practical bar and the
    // engine bar disagree on the position — shown/hidden per move like the ghost tick.
    const edgeEl = q("[data-edge]");
    if (edgeEl) {
      const he = humanEdge(eq, evt.cp);
      if (he) {
        edgeEl.textContent = edgeLabel(he);
        edgeEl.classList.toggle("white", he.side === "white");
        edgeEl.classList.toggle("black", he.side === "black");
        edgeEl.hidden = false;
      } else {
        edgeEl.hidden = true;
      }
    }

    // Caster mode: flare on a big practical swing the engine bar misses (task 0022).
    if (cfg.caster) {
      const d = dramaSwing(prevEquity, eq, prevCp, evt.cp);
      if (d) showDrama(d, evt.drama);
    }
    prevEquity = eq;
    prevCp = evt.cp;

    // Clocks (rolling players carry over from the game event).
    const clk = evt.clock || {};
    updateClock("[data-white-clock]", clk.white);
    updateClock("[data-black-clock]", clk.black);

    // Time-pressure cue (task 0104): tint the nameplate of whichever side is low on
    // the clock so a clock-driven equity shift reads as such, not as a positional one.
    const lc = cfg.lowclock != null ? cfg.lowclock : 30;
    setPressure(".player-white", clk.white, lc);
    setPressure(".player-black", clk.black, lc);

    // Per-move Δequity grade pill.
    if (evt.grade && evt.grade.label) showGrade(evt.grade);

    // Late-arriving player metadata in a position event.
    if (evt.players) applyGame(evt, cfg);
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

  // Toggle the time-pressure class on a player's nameplate from its remaining clock.
  function setPressure(playerSel, secs, threshold) {
    const el = q(playerSel);
    if (el) el.classList.toggle("time-pressure", timePressure(secs, threshold));
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

  // Caster-facing one-liner for a swing. Prefers a server-provided drama headline
  // (chess_equity.drama, once 0018/0020 emit it); otherwise builds one from the swing.
  function dramaHeadline(d, serverDrama) {
    if (serverDrama && serverDrama.headline) return serverDrama.headline;
    const pts = Math.round(d.swing * 100);
    const arrow = d.swing > 0 ? "▲" : "▼";
    const who = d.side === "white" ? "White" : "Black";
    const tail = d.engineBlind ? " · engine bar misses it" : "";
    return arrow + " " + who + " " + (pts > 0 ? "+" : "") + pts + " pts" + tail;
  }

  function showDrama(d, serverDrama) {
    const el = q("[data-drama]");
    if (!el) return;
    el.textContent = dramaHeadline(d, serverDrama);
    el.classList.toggle("white", d.side === "white");
    el.classList.toggle("black", d.side === "black");
    el.classList.toggle("engine-blind", !!d.engineBlind);
    el.hidden = false;
    // Restart the flare animation on each fire.
    el.style.animation = "none";
    void el.offsetWidth;
    el.style.animation = "";
    if (dramaTimer) clearTimeout(dramaTimer);
    dramaTimer = setTimeout(function () {
      el.hidden = true;
    }, 4000);
  }

  function dispatch(evt, cfg) {
    if (!evt || !evt.type) return;
    if (evt.type === "game") applyGame(evt, cfg);
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

    // Show any rating overrides immediately, before the first game event arrives.
    if (cfg.welo != null && cfg.welo !== "") setText("[data-white-rating]", cfg.welo);
    if (cfg.belo != null && cfg.belo !== "") setText("[data-black-rating]", cfg.belo);

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
    timePressure: timePressure,
    overrideRating: overrideRating,
    fmtDelta: fmtDelta,
    dramaSwing: dramaSwing,
    dramaHeadline: dramaHeadline,
    humanEdge: humanEdge,
    edgeLabel: edgeLabel,
  };

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", start);
    } else {
      start();
    }
  }
})(typeof window !== "undefined" ? window : this);
