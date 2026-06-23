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

  // Who is to move in the position an event describes? Prefer the authoritative
  // white_to_move flag the broadcast bridge now emits on every position event
  // (the post-move FEN's turn); honor an stm string next. Only when neither is
  // present (a bare replay feed) do we derive from ply parity: White moves on ODD
  // plies, so the position AFTER an even ply is White-to-move. Defaults to White
  // (matching the white-POV default) when nothing says otherwise.
  function whiteToMove(evt) {
    if (!evt) return true;
    if (typeof evt.white_to_move === "boolean") return evt.white_to_move;
    if (evt.stm === "white") return true;
    if (evt.stm === "black") return false;
    if (typeof evt.ply === "number" && !isNaN(evt.ply)) return evt.ply % 2 === 0;
    return true;
  }

  // Orient a White-POV equity (0..1) for the chosen point of view.
  //   pov="white" (default, classic bar): always White's win equity — side ignored.
  //   pov="stm" (side-to-move): the value flips to 1 - eq when Black is to move, so
  //     the readout always measures the player on the move. This is the one place the
  //     two POVs diverge — a stm value is the white-POV value with its sign flipped
  //     whenever it's Black's turn.
  function orient(equityWhite, pov, isWhiteToMove) {
    const w = clamp01(equityWhite);
    if (pov === "stm" && isWhiteToMove === false) return 1 - w;
    return w;
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

  // Game-over card content (task 0213): when a followed board's game ends the bar
  // would otherwise just freeze on the final position with no signal it's over. Map a
  // terminal `result` ("1-0" / "0-1" / "1/2-1/2") + the board's roster `players` to a
  // clean end-card payload: the result token, the winner's name, and a headline. Draw
  // (or any unknown result) has no winner. Pure (no DOM) so it's unit-testable; the
  // final equity is added by the renderer, which knows the bar's last value.
  function gameOverCard(result, players) {
    var pl = players || {};
    var white = (pl.white && pl.white.name) || "White";
    var black = (pl.black && pl.black.name) || "Black";
    var winnerSide = result === "1-0" ? "white" : result === "0-1" ? "black" : null;
    var winnerName = winnerSide === "white" ? white : winnerSide === "black" ? black : null;
    var headline = winnerName
      ? winnerName + " wins"
      : result === "1/2-1/2"
      ? "Draw"
      : "Game over";
    return { result: result, winnerSide: winnerSide, winnerName: winnerName, headline: headline };
  }

  // Momentum cue (task 0208): the DIRECTION and SIZE of the last equity swing, so a
  // viewer can see at a glance who just gained — distinct from dramaSwing (which only
  // FLARES on big engine-blind swings in caster mode) and humanEdge (a LEVEL gap vs the
  // engine). All inputs White-POV; derived purely from the event delta (eq - prevEquity),
  // no new feed field. Returns null on the first move or when the swing is negligible.
  // `magnitude` is the swing normalized to MAX_SWING (0.4) like dramaSwing, so the two
  // read on the same scale.
  function momentum(prevEquity, equity, opts) {
    opts = opts || {};
    var minDelta = opts.minDelta == null ? 0.02 : opts.minDelta; // < 2pts is "negligible"
    if (typeof prevEquity !== "number" || typeof equity !== "number") return null;
    if (isNaN(prevEquity) || isNaN(equity)) return null;
    var delta = clamp01(equity) - clamp01(prevEquity);
    if (Math.abs(delta) < minDelta) return null;
    return {
      side: delta > 0 ? "white" : "black", // who the swing moved equity TOWARD
      delta: delta,
      magnitude: Math.min(1, Math.abs(delta) / 0.4),
    };
  }

  // Decaying momentum tracker (task 0208): a timer-free, tick-driven state machine (mirrors
  // feed.js makeStaleTracker — the caller drives it, no clocks) so the arrow lingers and
  // fades after a swing instead of blinking off the next move. `note(prevEq, eq)` feeds one
  // position: a fresh swing (>= minDelta) sets the arrow to FULL strength in its direction;
  // a quiet move decays the standing arrow by 1/decayTicks until it disappears after
  // `decayTicks` quiet moves. Returns the render state {side, magnitude, strength} where
  // strength is 1..0 (drives opacity), or null when there's nothing to show.
  function makeMomentumTracker(opts) {
    opts = opts || {};
    var decayTicks = opts.decayTicks == null ? 4 : opts.decayTicks; // quiet moves until it clears
    var step = decayTicks > 0 ? 1 / decayTicks : 1;
    var side = null;       // direction of the standing swing
    var magnitude = 0;     // normalized size of the swing that set the arrow (held while visible)
    var strength = 0;      // 1 on a fresh swing, decays toward 0 each quiet tick
    return {
      note: function (prevEquity, equity) {
        var m = momentum(prevEquity, equity, opts);
        if (m) {
          side = m.side;
          magnitude = m.magnitude;
          strength = 1;
        } else {
          strength = Math.max(0, strength - step);
        }
        if (strength <= 0 || side === null) return null;
        return { side: side, magnitude: magnitude, strength: strength };
      },
      // Current render state without advancing (for tests/inspection).
      peek: function () {
        if (strength <= 0 || side === null) return null;
        return { side: side, magnitude: magnitude, strength: strength };
      },
    };
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

  // Live board router (task 0185): a multi-game broadcast round feeds every board's
  // events down one stream. The router learns the roster (from a "boards" roster event
  // or per-game `board` indices) and decides which events reach the overlay so a caster
  // can flip boards live. Pure + DOM-free so it's unit-testable: `learn(evt)` updates the
  // roster, `select(idx)` chooses the followed board, `accepts(evt)` gates rendering.
  //
  // Default = single-board behavior: a feed whose events carry no `board` (a single game)
  // always `accepts`, so nothing changes for the common case. The first board seen in a
  // multi-game round is auto-selected so the bar shows something before the caster picks.
  //
  // Auto-director (task 0188): with `opts.autofollow`, `note(evt)` reads each event's
  // server `drama.magnitude` and steals focus to whichever board has the biggest live
  // swing — but only after a focus lock of `opts.lockPlies` plies has elapsed since the
  // last switch, so transient noise can't thrash the bar (a real swing wins; a blip waits
  // out the lock). A manual `select(idx)` PINS the board and disables autofollow until
  // `resume()`, so a caster can always override the director.
  //
  // Anti-flap stickiness (task 0203): the focus lock above is a min-dwell AFTER a cut, but
  // pre-cut the director used to steal on a bare `mag > leader` — so on a busy round two
  // boards leapfrog by a hair every tick and the bar flip-flops, which is unwatchable on
  // air. Now a challenger must out-drama the leader by `opts.dramaMargin` for
  // `opts.challengePlies` CONSECUTIVE challenge ticks before it can cut. A rival that falls
  // back under the margin loses its streak; the leader going quiet lets a real rival
  // accumulate its streak across interleaved quiet-board ticks. Defaults are sticky (margin
  // 0.1, K 2) so anti-flap is on by default; set `challengePlies:1, dramaMargin:0` for the
  // old hair-trigger behavior.
  function makeBoardRouter(opts) {
    opts = opts || {};
    var autofollow = !!opts.autofollow;
    var lockPlies = opts.lockPlies == null ? 6 : opts.lockPlies; // post-cut min-dwell window
    var dramaMargin = opts.dramaMargin == null ? 0.1 : opts.dramaMargin; // lead a rival must show
    var challengePlies = opts.challengePlies == null ? 2 : opts.challengePlies; // sustained ticks (K)
    var boards = []; // [{index, players}], in board order
    var selected = null; // followed board index; null = none chosen yet
    var pinned = false; // a manual select pins the board, disabling autofollow + auto-advance
    var lockRemaining = 0; // plies left before autofollow may switch again
    var challenger = null; // board currently building a sustained-lead challenge (or null)
    var challengeStreak = 0; // consecutive challenge ticks the challenger has held the margin
    var lastDrama = {}; // board index -> latest drama magnitude seen
    var finished = {}; // board index -> true once its game has a terminal result (0189)

    function has(idx) {
      for (var i = 0; i < boards.length; i++) if (boards[i].index === idx) return true;
      return false;
    }

    function dramaMag(evt) {
      var d = evt && evt.drama;
      return d && typeof d.magnitude === "number" && !isNaN(d.magnitude) ? d.magnitude : 0;
    }

    // The next still-live board in round order, or null if every board has finished.
    function nextLiveBoard() {
      for (var i = 0; i < boards.length; i++) {
        if (!finished[boards[i].index]) return boards[i].index;
      }
      return null;
    }

    // If the followed board has finished and the caster hasn't pinned it, advance focus
    // to the next live board (task 0189). A no-op when nothing is selected, the board is
    // still live, or every board has ended (stay put on the final position).
    function autoAdvance() {
      if (pinned || selected === null || !finished[selected]) return;
      var next = nextLiveBoard();
      if (next !== null) selected = next;
    }

    return {
      boards: function () {
        return boards;
      },
      selected: function () {
        return selected;
      },
      // Roster players for a board index (or null) — used by the game-over card (0213)
      // to name the winner. A single-game feed has no roster, so this returns null and
      // the card falls back to the on-screen names.
      playersFor: function (idx) {
        for (var i = 0; i < boards.length; i++) {
          if (boards[i].index === idx) return boards[i].players || null;
        }
        return null;
      },
      autofollow: function () {
        return autofollow && !pinned;
      },
      // Whether the caster has manually pinned a board (autofollow + auto-advance disabled).
      pinned: function () {
        return pinned;
      },
      // A manual pick (caster clicks the selector): follow this board AND pin it, so neither
      // the auto-director (0188) nor a finished-board auto-advance (0189) can yank focus away.
      select: function (idx) {
        selected = idx;
        pinned = true;
        lockRemaining = 0;
        challenger = null; // a fresh pick abandons any half-built challenge (0203)
        challengeStreak = 0;
      },
      // Re-enable autofollow after a manual pin (the "reset" the caster reaches for).
      resume: function () {
        pinned = false;
        lockRemaining = 0;
        challenger = null; // unpinning starts the director clean — no stale streak (0203)
        challengeStreak = 0;
      },
      // Auto-director step (task 0188): given an incoming event, decide whether the most
      // dramatic board should steal focus. No-op unless autofollow is on, the feed is
      // multi-board (numeric `board`), and the board isn't pinned. Honors the focus lock.
      note: function (evt) {
        if (!autofollow || pinned) return;
        if (!evt || typeof evt.board !== "number") return;
        var mag = dramaMag(evt);
        if (selected === null) {
          // Nothing followed yet — adopt the first board we see, unlocked so a bigger
          // swing elsewhere can immediately take over.
          selected = evt.board;
          lastDrama[evt.board] = mag;
          return;
        }
        lastDrama[evt.board] = mag;
        if (evt.board === selected) {
          if (lockRemaining > 0) lockRemaining--; // a ply on the focused board still ticks
          return; // already focused — nothing to steal
        }
        if (lockRemaining > 0) {
          lockRemaining--; // locked: a real swing has to wait out the focus window
          return;
        }
        // Anti-flap (0203): a rival cuts only after it leads the leader by `dramaMargin`
        // for `challengePlies` consecutive challenge ticks. Interleaved quiet boards (which
        // never clear the margin) don't reset the standing challenger; only the challenger
        // itself falling back under the margin does.
        if (mag - (lastDrama[selected] || 0) >= dramaMargin) {
          if (challenger === evt.board) {
            challengeStreak++;
          } else {
            challenger = evt.board; // a new rival takes over the challenge slot
            challengeStreak = 1;
          }
          if (challengeStreak >= challengePlies) {
            selected = evt.board; // a sustained, margin-clearing swing steals focus...
            lockRemaining = lockPlies; // ...and the lock guards it from an immediate flip
            challenger = null;
            challengeStreak = 0;
          }
        } else if (challenger === evt.board) {
          challenger = null; // the standing challenger cooled off — drop its streak
          challengeStreak = 0;
        }
      },
      // Update the roster from a routing event. A "boards" event carries the full
      // roster; a "game" event with a numeric `board` adds one board; a "result" event
      // marks a board's game as ended. Auto-selects the first board so a fresh overlay
      // isn't blank before the caster chooses, and auto-advances off a finished board.
      learn: function (evt) {
        if (!evt) return;
        if (evt.type === "boards" && Array.isArray(evt.boards)) {
          boards = evt.boards.slice();
        } else if (evt.type === "game" && typeof evt.board === "number") {
          if (!has(evt.board)) boards.push({ index: evt.board, players: evt.players });
        } else if (evt.type === "result" && typeof evt.board === "number") {
          finished[evt.board] = true;
        }
        if (selected === null && boards.length) selected = boards[0].index;
        // Advance after every learn: a result for the followed board moves us now; a
        // later live board appearing while we're stranded on a finished one moves us then.
        autoAdvance();
      },
      // Should this event be rendered, given the current selection? "boards" and
      // "result" events are routing metadata (never rendered). Events with no `board`
      // (single-game feed) always pass. When a board is selected, only its events pass.
      accepts: function (evt) {
        if (!evt || evt.type === "boards" || evt.type === "result") return false;
        if (typeof evt.board !== "number") return true;
        if (selected === null) return true;
        return evt.board === selected;
      },
    };
  }

  // ---- DOM wiring --------------------------------------------------------

  function params() {
    const p = new URLSearchParams(global.location ? global.location.search : "");
    return {
      src: p.get("src") || "./mock-game.json",
      layout: p.get("layout") || "horizontal",
      theme: p.get("theme") || "dark",
      pov: p.get("pov") === "stm" ? "stm" : "white",   // bar POV: classic White vs side-to-move
      cp: p.get("cp") !== "0",
      cpbar: p.get("cpbar") === "1",
      caster: p.get("caster") === "1",
      legend: p.get("legend") === "1",
      autofollow: p.get("autofollow") === "1",
      focuslock: parseInt(p.get("focuslock"), 10),
      dramamargin: parseFloat(p.get("dramamargin")),
      challenge: parseInt(p.get("challenge"), 10),
      speed: parseFloat(p.get("speed")) || 1,
      welo: p.get("welo"),
      belo: p.get("belo"),
      lowclock: parseFloat(p.get("lowclock")) || 30,
      stale: parseFloat(p.get("stale")) || 10,
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
  // Decaying momentum arrow (task 0208): lingers a few moves after a swing, then fades.
  let momentumTracker = makeMomentumTracker();

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
    lastBarEquity = eq; // remembered for the game-over card (task 0213)

    // Bar widths.
    const whiteEl = q("[data-bar-white]");
    if (whiteEl) {
      const dim = cfg.layout === "vertical" ? "height" : "width";
      whiteEl.style[dim] = (eq * 100).toFixed(1) + "%";
    }
    setText("[data-white-pct]", pct(eq, "white"));
    setText("[data-black-pct]", pct(eq, "black"));

    // Side-to-move readout (?pov=stm). The bar stays player-honest (white-fill is
    // always White's equity — it never jumps mid-game), so the POV toggle reframes
    // only the numeric readout to whoever is on the move, via orient().
    const stmEl = q("[data-stm-pct]");
    if (stmEl) {
      if (cfg.pov === "stm") {
        const wtm = whiteToMove(evt);
        const mover = wtm ? "White" : "Black";
        stmEl.textContent = mover + " to move · " + Math.round(orient(eq, "stm", wtm) * 100) + "%";
        stmEl.hidden = false;
      } else {
        stmEl.hidden = true;
      }
    }

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

    // Momentum arrow (task 0208): always-on glance cue for the direction + size of the
    // last swing. Fed every move (even quiet ones) so the standing arrow decays and clears.
    showMomentum(momentumTracker.note(prevEquity, eq));

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

  // Momentum arrow (task 0208): render the decaying last-swing cue. `m` is the tracker
  // state {side, magnitude, strength} or null. Hidden when null (negligible / decayed
  // out). The arrow points UP when equity moved toward White, DOWN toward Black; its
  // opacity follows `strength` (fades over the decay window) and its size hints at the
  // swing `magnitude`, so it reads at a glance without text.
  function showMomentum(m) {
    const el = q("[data-momentum]");
    if (!el) return;
    if (!m) {
      el.hidden = true;
      return;
    }
    const up = m.side === "white";
    el.textContent = up ? "▲" : "▼";
    el.classList.toggle("white", up);
    el.classList.toggle("black", !up);
    // Opacity carries the decay; font-size nudges up with the swing magnitude.
    el.style.opacity = (0.25 + 0.75 * m.strength).toFixed(2);
    el.style.fontSize = (15 + Math.round(9 * m.magnitude)) + "px";
    el.hidden = false;
  }

  // Live board selector for a multi-game round (task 0185). Hidden for a single board
  // (<= 1 known), revealed and populated otherwise; changing it tells the router which
  // board to follow, and subsequent events for that board flow to the bar.
  let boardRouter = null;
  let boardSelectWired = false;
  // The last White-POV equity drawn on the bar (task 0213) — the value the bar would
  // freeze on at game end, shown on the game-over card. Defaults to even.
  let lastBarEquity = 0.5;

  function boardLabel(b) {
    const pl = b.players || {};
    const w = (pl.white && pl.white.name) || "White";
    const bl = (pl.black && pl.black.name) || "Black";
    return "Board " + (b.index + 1) + ": " + w + " – " + bl;
  }

  function renderBoardSelector(router) {
    const sel = q("[data-board-select]");
    if (!sel) return;
    const boards = router.boards();
    if (boards.length <= 1) {
      sel.hidden = true; // single board → no selector (default behavior preserved)
      return;
    }
    if (sel.options.length !== boards.length) {
      sel.innerHTML = "";
      boards.forEach(function (b) {
        const opt = document.createElement("option");
        opt.value = String(b.index);
        opt.textContent = boardLabel(b);
        sel.appendChild(opt);
      });
    }
    sel.hidden = false;
    if (router.selected() != null) sel.value = String(router.selected());
    if (!boardSelectWired) {
      boardSelectWired = true;
      sel.addEventListener("change", function () {
        router.select(parseInt(sel.value, 10));
      });
    }
  }

  function dispatch(evt, cfg) {
    if (!evt || !evt.type) return;
    if (boardRouter) {
      // A `result` for the FOLLOWED board (task 0213): capture the focus BEFORE learn(),
      // which auto-advances off the finished board, so we can tell the end-card belongs
      // to the board we were watching. Non-followed boards just route (no card).
      var followedBefore = boardRouter.selected();
      if (evt.type === "result") {
        var forFollowed =
          typeof evt.board !== "number" || evt.board === followedBefore;
        if (forFollowed) {
          var players = boardRouter.playersFor(
            typeof evt.board === "number" ? evt.board : followedBefore
          );
          renderGameOverCard(gameOverCard(evt.result, players));
        }
      }
      boardRouter.learn(evt);
      boardRouter.note(evt); // auto-director may steal focus to the most-dramatic board
      renderBoardSelector(boardRouter);
      // Drop events for boards we aren't following (and "boards"/"result" metadata).
      if (!boardRouter.accepts(evt)) return;
    } else if (evt.type === "result") {
      // No router (degenerate single-board wiring): still card the end.
      renderGameOverCard(gameOverCard(evt.result, null));
      return;
    }
    // Any live position clears a stale game-over card (focus moved to a live board).
    if (evt.type === "position") clearGameOverCard();
    if (evt.type === "game") applyGame(evt, cfg);
    else if (evt.type === "position") applyPosition(evt, cfg);
  }

  // Game-over card (task 0213): overlay the bar with the final result + winner + the
  // bar's last equity, so a finished game reads as finished instead of a frozen bar.
  // `card` is the pure gameOverCard() payload; the equity comes from what's on the bar.
  function renderGameOverCard(card) {
    var el = q("[data-gameover]");
    if (!el || !card) return;
    setText("[data-gameover-headline]", card.headline);
    setText("[data-gameover-result]", card.result || "");
    var eqEl = q("[data-gameover-equity]");
    if (eqEl) {
      // The followed board's last shown White-equity (the value the bar froze on).
      eqEl.textContent = "final equity " + pct(lastBarEquity, "white") + " White";
    }
    el.hidden = false;
  }

  function clearGameOverCard() {
    var el = q("[data-gameover]");
    if (el) el.hidden = true;
  }

  // Stale-feed UI (task 0178): when the live feed drops, the bar would otherwise
  // freeze silently and mislead viewers. Dim the overlay and reveal a small
  // "reconnecting" marker; clear both on the next event.
  function setStale(on) {
    const root = q("#overlay");
    if (root) root.classList.toggle("feed-stale", !!on);
    const marker = q("[data-stale]");
    if (marker) marker.hidden = !on;
  }

  function start() {
    const cfg = params();
    momentumTracker = makeMomentumTracker(); // fresh decay state per (re)start
    boardRouter = makeBoardRouter({
      autofollow: cfg.autofollow,
      lockPlies: isNaN(cfg.focuslock) ? undefined : cfg.focuslock,
      dramaMargin: isNaN(cfg.dramamargin) ? undefined : cfg.dramamargin,
      challengePlies: isNaN(cfg.challenge) ? undefined : cfg.challenge,
    });
    const root = q("#overlay");
    if (root) {
      root.classList.remove("layout-horizontal", "layout-vertical");
      root.classList.add("layout-" + cfg.layout);
      root.classList.toggle("pov-stm", cfg.pov === "stm");
    }
    document.body.className = "theme-" + cfg.theme;

    // Legend key (task 0201): off by default; revealed only when the streamer opts in
    // via ?legend=1 (the config.html toggle). Static content, so just unhide it.
    const legendEl = q("[data-legend]");
    if (legendEl) legendEl.hidden = !cfg.legend;

    // Show any rating overrides immediately, before the first game event arrives.
    if (cfg.welo != null && cfg.welo !== "") setText("[data-white-rating]", cfg.welo);
    if (cfg.belo != null && cfg.belo !== "") setText("[data-black-rating]", cfg.belo);

    if (global.EquityFeed) {
      global.EquityFeed.connect(cfg.src, {
        speed: cfg.speed,
        staleMs: cfg.stale * 1000,
        onEvent: function (evt) {
          dispatch(evt, cfg);
        },
        onError: function (e) {
          // Stay silent on-screen (it's an overlay); log for setup debugging.
          if (global.console) console.warn("[equity-overlay] feed error", e);
        },
        onStale: function () {
          setStale(true);
        },
        onFresh: function () {
          setStale(false);
        },
      });
    }
  }

  // Expose pure helpers for tests; auto-start in a browser.
  global.EquityOverlay = {
    clamp01: clamp01,
    pct: pct,
    orient: orient,
    whiteToMove: whiteToMove,
    cpToWhitePos: cpToWhitePos,
    formatClock: formatClock,
    timePressure: timePressure,
    overrideRating: overrideRating,
    fmtDelta: fmtDelta,
    dramaSwing: dramaSwing,
    dramaHeadline: dramaHeadline,
    humanEdge: humanEdge,
    edgeLabel: edgeLabel,
    momentum: momentum,
    makeMomentumTracker: makeMomentumTracker,
    makeBoardRouter: makeBoardRouter,
    gameOverCard: gameOverCard,
  };

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", start);
    } else {
      start();
    }
  }
})(typeof window !== "undefined" ? window : this);
