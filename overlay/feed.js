/* feed.js — transport abstraction for the equity overlay.
 *
 * One job: turn a `src` (SSE endpoint, WebSocket URL, or a .json replay file)
 * into a stream of overlay events delivered to an onEvent(evt) callback.
 *
 * Event schema (the contract the live ingestion task, 0018, must emit) is
 * documented in README.md. Two event types:
 *   - "game"     : one-time metadata (players, format).
 *   - "position" : per-move update (equity, clocks, cp, grade).
 *
 * Kept dependency-free so the overlay loads instantly in OBS.
 */
(function (global) {
  "use strict";

  function isWebSocket(src) {
    return /^wss?:\/\//i.test(src);
  }
  function isJsonReplay(src) {
    return /\.json($|\?)/i.test(src);
  }

  function nowMs() {
    return typeof Date !== "undefined" && Date.now ? Date.now() : 0;
  }

  // Pure stale-state machine (no real timers — the caller supplies `now` in ms),
  // so enter-stale / recover-from-stale are unit-testable without a browser.
  // A live feed is STALE when no event has arrived for `staleMs`, or when the
  // transport errors; it recovers on the next event. Each method returns a
  // transition string ("stale" / "recovered") only on the edge, else null, so
  // callers fire their UI side-effect exactly once per transition.
  function makeStaleTracker(staleMs) {
    staleMs = staleMs || 10000;
    var lastEventAt = null;
    var stale = false;
    return {
      // Record an event at time `now`. Returns "recovered" if it cleared a
      // stale state, else null.
      event: function (now) {
        lastEventAt = now == null ? nowMs() : now;
        if (stale) {
          stale = false;
          return "recovered";
        }
        return null;
      },
      // Force stale (a transport error / dropped connection). Returns "stale"
      // on the transition into stale, else null.
      fail: function () {
        if (!stale) {
          stale = true;
          return "stale";
        }
        return null;
      },
      // Poll at time `now`: go stale if no event for >= staleMs. Returns "stale"
      // on the transition, else null. A no-op until the first event arrives.
      poll: function (now) {
        if (stale || lastEventAt == null) return null;
        if ((now == null ? nowMs() : now) - lastEventAt >= staleMs) {
          stale = true;
          return "stale";
        }
        return null;
      },
      isStale: function () {
        return stale;
      },
    };
  }

  // Live SSE: server pushes `data: {json}\n\n` frames.
  function connectSSE(src, onEvent, onError) {
    const es = new EventSource(src);
    es.onmessage = function (m) {
      try {
        onEvent(JSON.parse(m.data));
      } catch (e) {
        onError && onError(e);
      }
    };
    es.onerror = function (e) {
      onError && onError(e);
    };
    return function close() {
      es.close();
    };
  }

  // Live WebSocket: server pushes JSON text frames.
  function connectWS(src, onEvent, onError) {
    const ws = new WebSocket(src);
    ws.onmessage = function (m) {
      try {
        onEvent(JSON.parse(m.data));
      } catch (e) {
        onError && onError(e);
      }
    };
    ws.onerror = function (e) {
      onError && onError(e);
    };
    return function close() {
      ws.close();
    };
  }

  // Replay a static .json file. The file is either an array of events or
  // { events: [...] }. Each position event may carry `delayMs` (time until
  // the next event); `speed` scales those delays so casters can fast-forward.
  function connectReplay(src, onEvent, onError, speed) {
    let stopped = false;
    let timer = null;
    fetch(src)
      .then(function (r) {
        if (!r.ok) throw new Error("feed " + r.status + " " + src);
        return r.json();
      })
      .then(function (data) {
        const events = Array.isArray(data) ? data : data.events || [];
        let i = 0;
        function step() {
          if (stopped || i >= events.length) return;
          const evt = events[i++];
          onEvent(evt);
          const wait = Math.max(0, (evt.delayMs || 0) / (speed || 1));
          timer = setTimeout(step, wait);
        }
        step();
      })
      .catch(function (e) {
        onError && onError(e);
      });
    return function close() {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }

  // Dispatch on src shape. Returns a close() function.
  //
  // Stale detection (opts.onStale / opts.onFresh): for LIVE feeds (SSE/WS) the
  // overlay must not silently freeze when the connection drops. We wrap onEvent
  // and onError through a stale tracker and poll for silence; onStale fires once
  // when the feed goes quiet for `staleMs` (default 10s) or the transport errors,
  // onFresh fires once when events resume. .json replays end legitimately, so
  // they get error-driven stale (a failed fetch) but no silence poller.
  function connect(src, opts) {
    opts = opts || {};
    const onEvent = opts.onEvent || function () {};
    const onError = opts.onError || function () {};
    const onStale = opts.onStale || function () {};
    const onFresh = opts.onFresh || function () {};
    const staleMs = opts.staleMs == null ? 10000 : opts.staleMs;
    const tracker = makeStaleTracker(staleMs);

    function handleEvent(evt) {
      if (tracker.event(nowMs()) === "recovered") onFresh();
      onEvent(evt);
    }
    function handleError(e) {
      if (tracker.fail() === "stale") onStale();
      onError(e);
    }

    const live = !isJsonReplay(src);
    let close;
    if (isWebSocket(src)) close = connectWS(src, handleEvent, handleError);
    else if (isJsonReplay(src)) close = connectReplay(src, handleEvent, handleError, opts.speed);
    else close = connectSSE(src, handleEvent, handleError);

    // Silence poller — only for live transports, and only when stale detection
    // is enabled (staleMs > 0) and timers exist.
    let poller = null;
    if (live && staleMs > 0 && typeof setInterval === "function") {
      poller = setInterval(function () {
        if (tracker.poll(nowMs()) === "stale") onStale();
      }, Math.max(250, Math.floor(staleMs / 4)));
    }
    return function closeAll() {
      if (poller && typeof clearInterval === "function") clearInterval(poller);
      close();
    };
  }

  global.EquityFeed = { connect: connect, makeStaleTracker: makeStaleTracker };
})(typeof window !== "undefined" ? window : this);
