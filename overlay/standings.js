/* OBS lower-third standings panel (task 0231).
 *
 * Renders a clean top-N accuracy standings lower third for a streamed round, fed
 * by the `grade --round --json` leaderboard export (grading.py `leaderboard_export_rows`).
 * That export is a JSON array, already rank-sorted, one object per player:
 *   { rank, player, rating, n_moves, accuracy (0-100), avg_delta, phases:{…} }
 * The lower third shows rank / player / accuracy% / Δpeer for the top N — the four
 * fields a broadcast graphic needs; `rating`/`phases` are carried but optional.
 *
 * Like overlay.js/feed.js this is an IIFE exposing a single global (`EquityStandings`)
 * with PURE helpers (orderTopN / rowsHTML) plus a thin DOM `mount()`. The pure helpers
 * touch no browser globals, so a test can load this file in a bare `vm` sandbox and
 * exercise rank ordering directly (see test_standings.test.js).
 *
 * Query params (standings.html):
 *   ?src=URL   leaderboard JSON (a bare export array, or {leaderboard:[…]} like the
 *              committed mock fixture). Default ./mock-leaderboard.json.
 *   ?top=N     how many qualified players to show (default 5, clamped 1..20).
 *   ?title=…   panel title (default "STANDINGS").
 *   ?theme=dark|light   label theme (default dark).
 */
(function (global) {
  "use strict";

  var DEFAULT_TOP = 5;

  // A leaderboard payload may be the raw export array, or an object wrapping it under
  // `leaderboard` (the committed mock fixture, which also carries a `_comment`). Accept both.
  function extract(payload) {
    if (Array.isArray(payload)) return payload;
    if (payload && Array.isArray(payload.leaderboard)) return payload.leaderboard;
    return [];
  }

  function clampTop(n) {
    n = Math.floor(Number(n));
    if (!isFinite(n) || n < 1) return DEFAULT_TOP;
    return Math.min(n, 20);
  }

  // Pure: return the top-N rows in strict rank order. The export is already sorted, but
  // we re-sort defensively (by `rank`, then `player` name) so a shuffled or hand-edited
  // feed still renders deterministically and a cameo can't jump the board by file order.
  function orderTopN(rows, top) {
    var n = clampTop(top);
    var sorted = (rows || []).slice().sort(function (a, b) {
      var ra = Number(a && a.rank);
      var rb = Number(b && b.rank);
      if (isFinite(ra) && isFinite(rb) && ra !== rb) return ra - rb;
      var na = String((a && a.player) || "");
      var nb = String((b && b.player) || "");
      return na < nb ? -1 : na > nb ? 1 : 0;
    });
    return sorted.slice(0, n);
  }

  // Minimal HTML escape so a player name can't break the row markup.
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function fmtAccuracy(v) {
    var n = Number(v);
    return (isFinite(n) ? n.toFixed(1) : "0.0") + "%";
  }

  // Δpeer (mean peer-relative loss, centipawns) — signed, 1 dp, like render_leaderboard.
  function fmtDelta(v) {
    var n = Number(v);
    if (!isFinite(n)) n = 0;
    return (n >= 0 ? "+" : "") + n.toFixed(1);
  }

  // Pure: one `<div class="st-row">` per row, in rank order. The displayed `#` is the
  // row's own `rank` field (not its array index), so an export with gaps still reads true.
  function rowsHTML(rows, top) {
    return orderTopN(rows, top)
      .map(function (r) {
        var rank = Number(r && r.rank);
        var hash = isFinite(rank) ? rank : "";
        return (
          '<div class="st-row" data-rank="' + esc(hash) + '">' +
          '<span class="st-rank">' + esc(hash) + "</span>" +
          '<span class="st-name">' + esc((r && r.player) || "?") + "</span>" +
          '<span class="st-acc">' + esc(fmtAccuracy(r && r.accuracy)) + "</span>" +
          '<span class="st-delta">' + esc(fmtDelta(r && r.avg_delta)) + "</span>" +
          "</div>"
        );
      })
      .join("");
  }

  function panelHTML(rows, top, title) {
    return (
      '<div class="st-title">' + esc(title || "STANDINGS") + "</div>" +
      '<div class="st-head">' +
      '<span class="st-rank">#</span>' +
      '<span class="st-name">player</span>' +
      '<span class="st-acc">acc</span>' +
      '<span class="st-delta">&Delta;peer</span>' +
      "</div>" +
      '<div class="st-rows">' + rowsHTML(rows, top) + "</div>"
    );
  }

  // DOM path — only runs in a browser. Reads query params, fetches the leaderboard JSON,
  // and paints the panel. Guarded so module load is side-effect-free under a vm sandbox.
  function mount() {
    if (typeof document === "undefined") return;
    var panel = document.querySelector("[data-standings]");
    if (!panel) return;

    var params =
      typeof URLSearchParams !== "undefined" && typeof location !== "undefined"
        ? new URLSearchParams(location.search)
        : new URLSearchParams("");
    var src = params.get("src") || "./mock-leaderboard.json";
    var top = clampTop(params.get("top"));
    var title = params.get("title") || "STANDINGS";
    var theme = params.get("theme") === "light" ? "theme-light" : "theme-dark";
    if (typeof document.body !== "undefined" && document.body) {
      document.body.className = theme;
    }

    function paint(rows) {
      panel.innerHTML = panelHTML(rows, top, title);
    }

    if (typeof fetch === "function") {
      fetch(src)
        .then(function (res) {
          return res.json();
        })
        .then(function (payload) {
          paint(extract(payload));
        })
        .catch(function () {
          panel.innerHTML = '<div class="st-title">STANDINGS</div>' +
            '<div class="st-empty">no leaderboard feed</div>';
        });
    }
  }

  global.EquityStandings = {
    extract: extract,
    orderTopN: orderTopN,
    rowsHTML: rowsHTML,
    panelHTML: panelHTML,
    fmtAccuracy: fmtAccuracy,
    fmtDelta: fmtDelta,
    mount: mount,
  };

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", mount);
    } else {
      mount();
    }
  }
})(typeof window !== "undefined" ? window : this);
