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
  function connect(src, opts) {
    opts = opts || {};
    const onEvent = opts.onEvent || function () {};
    const onError = opts.onError || function () {};
    if (isWebSocket(src)) return connectWS(src, onEvent, onError);
    if (isJsonReplay(src)) return connectReplay(src, onEvent, onError, opts.speed);
    return connectSSE(src, onEvent, onError);
  }

  global.EquityFeed = { connect: connect };
})(typeof window !== "undefined" ? window : this);
