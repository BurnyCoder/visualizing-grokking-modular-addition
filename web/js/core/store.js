// store.js — the two primitives of the site's update pipeline: a batched observable store and a latest-wins runner.
//
// Local explanation: `createStore(initial)` holds a state object; `set(patch)` merges the patch at once but notifies
// listeners only once per scheduled flush (by default the next animation frame), passing the union of the keys that
// changed since the last flush. A store created with `{parent}` is a scope: it owns the keys of `initial` (e.g. its
// own step while an animation plays) and forwards every other key to the parent, whose changes it relays to its own
// listeners within the same flush. `latestWins(fn)` wraps an async function so that at most one call runs at a time;
// calls arriving meanwhile are merged into a single follow-up call (the latest state always wins, never a stale one).
// Global context: core/state.js builds the page's global store on this; figures/common.js `watch` uses latestWins so a
// slow figure never draws frames out of order or piles up work while a run is played.
// Pure module (no DOM at import time) so tests/js/unit/store.test.mjs can test it with Node.
// Sources: requestAnimationFrame batching (https://developer.mozilla.org/docs/Web/API/Window/requestAnimationFrame);
// queueMicrotask fallback (https://developer.mozilla.org/docs/Web/API/queueMicrotask).

/** Default flush scheduler: the next animation frame in a browser, a microtask elsewhere (Node tests). */
const nextFrame = (fn) => (globalThis.requestAnimationFrame ? globalThis.requestAnimationFrame(() => fn()) : queueMicrotask(fn));

/**
 * Create an observable store.
 * @param {object} initial  initial values (for a scope: the keys it owns)
 * @param {{parent?: object, schedule?: (fn: Function) => void}} o  parent store of a scope; flush scheduler
 */
export function createStore(initial, { parent = null, schedule = nextFrame } = {}) {
  const own = parent ? new Set(Object.keys(initial)) : null; // keys a scope keeps for itself (null: owns everything)
  const state = parent ? { ...parent.state, ...initial } : { ...initial }; // mutated in place: importers keep one reference
  const listeners = []; // {keys: Set|null, fn}
  let pending = new Set(); // keys changed since the last flush
  let scheduled = false; // a flush is already queued

  /** Call every listener interested in one of `changed` (snapshot: listeners may unsubscribe while running). */
  function notify(changed) {
    for (const entry of listeners.slice()) {
      if (entry.keys && !changed.some((k) => entry.keys.has(k))) continue; // not interested
      try { entry.fn(state, changed); } catch (e) { console.error(e); } // one failing listener must not stop the rest
    }
  }

  /** Deliver the batched changes now. */
  function flush() {
    scheduled = false;
    if (!pending.size) return;
    const changed = [...pending];
    pending = new Set();
    notify(changed);
  }

  /** Merge `patch` (own keys here, other keys forwarded to the parent) and schedule one notification. */
  function set(patch) {
    const forward = {}; // keys that belong to the parent store
    for (const [k, v] of Object.entries(patch)) {
      if (own && !own.has(k)) { forward[k] = v; continue; }
      if (state[k] === v) continue; // unchanged: no notification
      state[k] = v;
      pending.add(k);
    }
    if (parent && Object.keys(forward).length) parent.set(forward);
    if (pending.size && !scheduled) { scheduled = true; schedule(flush); }
  }

  /** Call fn(state, changedKeys) now (unless immediate=false) and after every flush touching `keys` (null = all). */
  function subscribe(keys, fn, { immediate = true } = {}) {
    const entry = { keys: keys ? new Set(keys) : null, fn };
    listeners.push(entry);
    if (immediate) fn(state, []);
    return () => { const i = listeners.indexOf(entry); if (i >= 0) listeners.splice(i, 1); };
  }

  if (parent) { // relay the parent's changes of keys this scope does not own, in the parent's flush (no extra frame)
    parent.subscribe(null, (ps, changed) => {
      const relayed = changed.filter((k) => !own.has(k));
      for (const k of relayed) state[k] = ps[k];
      if (relayed.length) notify(relayed);
    }, { immediate: false });
  }
  return { state, set, subscribe, flush, parent };
}

/**
 * Wrap an async `fn` so calls never overlap: the first call runs at once; calls made while it runs are merged with
 * `merge(previousArgs, newArgs)` into one pending call, run when the current one finishes. Errors are logged, not thrown.
 * Returns a function that starts (or joins) the run and resolves when no call is left.
 */
export function latestWins(fn, merge = (_, latest) => latest) {
  let running = null; // promise of the current pump, null when idle
  let queued = null; // merged arguments of the next call, null when none
  const pump = async () => {
    while (queued) {
      const args = queued;
      queued = null;
      try { await fn(args); } catch (e) { console.error(e); }
    }
    running = null;
  };
  return (args) => {
    queued = queued ? merge(queued, args) : args;
    running ||= pump();
    return running;
  };
}
