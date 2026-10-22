// state.js — one shared, observable state for every figure on a page (run, step, inputs, head, neuron, frequency),
// plus per-animation scopes.
//
// Local explanation: the global store (core/store.js) holds the current selection; `set(patch)` merges a change at
// once and notifies subscribers once per animation frame; the selection is mirrored into the URL hash
// (#run=paper&step=9401&a=5...) so any view can be shared as a link and survives reloads (not while a run is playing,
// to avoid a history write per frame). `storeOf(el)` returns the store a figure should use: the global one, or the
// scope of its nearest [data-scope] ancestor, which owns its own step / stepf / playing so an animated figure can
// play without moving the rest of the page.
// Global context: the dashboard's control bar, click-to-select on heatmaps/plots, the guided tour and the chapter
// figures all read and write this state, which keeps every panel in sync.
// Source: URLSearchParams (https://developer.mozilla.org/docs/Web/API/URLSearchParams).
import { createStore } from "./store.js";

const DEFAULTS = { run: "paper", step: 40001, a: 100, b: 50, ap: 40, head: 0, neuron: 0, freq: 14 }; // URL-backed keys
const NUMERIC = new Set(["step", "a", "b", "ap", "head", "neuron", "freq"]); // keys parsed as integers from the URL
const TRANSIENT = { stepf: null, playing: false }; // not in the URL: fractional play position, playback flag

/** Parse the URL hash into a partial state object. */
function readHash() {
  const out = {};
  const params = new URLSearchParams(location.hash.slice(1)); // "#a=1&b=2" → a=1, b=2
  for (const [k, v] of params) if (k in DEFAULTS) out[k] = NUMERIC.has(k) ? Number(v) : v;
  return out;
}

const store = createStore({ ...DEFAULTS, ...TRANSIENT, stage: null, ...readHash() }); // the page's global store; `stage`:
// forward-flow's highlighted stage, set by its clicks and chapter 5's scenes (global, so animation scopes forward it)
export const state = store.state; // current selection (one object, updated in place)
export const set = store.set;
export const subscribe = store.subscribe;

let hashTimer = null; // debounce URL updates while scrubbing
/** Write the non-default part of the state into the URL hash (without adding history entries). */
function writeHash() {
  clearTimeout(hashTimer);
  if (state.playing) return; // written once playback stops
  hashTimer = setTimeout(() => {
    const params = new URLSearchParams();
    for (const k of Object.keys(DEFAULTS)) if (state[k] !== DEFAULTS[k]) params.set(k, state[k]);
    const h = params.toString();
    history.replaceState(null, "", h ? `#${h}` : location.pathname + location.search);
  }, 150);
}
subscribe([...Object.keys(DEFAULTS), "playing"], writeHash, { immediate: false });

const scopes = new WeakMap(); // [data-scope] element → its scoped store
/** The store for element `el`: its nearest [data-scope] ancestor's scope (created on first use), else the global store. */
export function storeOf(el) {
  const host = el?.closest?.("[data-scope]");
  if (!host) return store;
  if (!scopes.has(host)) scopes.set(host, createStore({ step: Number(host.dataset.step) || state.step, ...TRANSIENT }, { parent: store }));
  return scopes.get(host);
}
