// state.js — one shared, observable state for every figure on a page (run, step, inputs, head, neuron, frequency).
//
// Local explanation: `state` holds the current selection; `set(patch)` merges a change and notifies the
// subscribers whose keys changed; the selection is mirrored into the URL hash (#run=paper&step=9401&a=5...)
// so any view can be shared as a link and survives reloads.
// Global context: the dashboard's control bar, click-to-select on heatmaps/plots, the guided tour and the
// chapter figures all read and write this one object, which keeps every panel in sync.
// Source: URLSearchParams (https://developer.mozilla.org/docs/Web/API/URLSearchParams).

const DEFAULTS = { run: "paper", step: 40001, a: 5, b: 71, ap: 40, head: 0, neuron: 0, freq: 14 };
const NUMERIC = new Set(["step", "a", "b", "ap", "head", "neuron", "freq"]); // keys parsed as integers from the URL

export const state = { ...DEFAULTS, ...readHash() }; // current selection
const listeners = []; // {keys:Set|null, fn}

/** Parse the URL hash into a partial state object. */
function readHash() {
  const out = {};
  const params = new URLSearchParams(location.hash.slice(1)); // "#a=1&b=2" → a=1, b=2
  for (const [k, v] of params) if (k in DEFAULTS) out[k] = NUMERIC.has(k) ? Number(v) : v;
  return out;
}

let hashTimer = null; // debounce URL updates while scrubbing
/** Write the non-default part of the state into the URL hash (without adding history entries). */
function writeHash() {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    const params = new URLSearchParams();
    for (const k of Object.keys(DEFAULTS)) if (state[k] !== DEFAULTS[k]) params.set(k, state[k]);
    const h = params.toString();
    history.replaceState(null, "", h ? `#${h}` : location.pathname + location.search);
  }, 150);
}

/** Merge `patch` into the state and notify subscribers interested in any changed key. */
export function set(patch) {
  const changed = Object.keys(patch).filter((k) => state[k] !== patch[k]);
  if (!changed.length) return;
  Object.assign(state, patch);
  writeHash();
  for (const { keys, fn } of listeners) if (!keys || changed.some((k) => keys.has(k))) fn(state, changed);
}

/** Call fn(state) now and whenever any of `keys` (array; null = any key) changes. Returns an unsubscribe function. */
export function subscribe(keys, fn, { immediate = true } = {}) {
  const entry = { keys: keys ? new Set(keys) : null, fn };
  listeners.push(entry);
  if (immediate) fn(state, []);
  return () => listeners.splice(listeners.indexOf(entry), 1);
}
