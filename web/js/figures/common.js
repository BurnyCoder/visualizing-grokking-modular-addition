// common.js — small helpers shared by the figure modules.
//
// Local explanation: which run a figure shows (fixed via data-run, else the global selection), subscription to
// the global state, a theme-change redraw hook, simple DOM builders and frequency/basis labels.
// Global context: every figure module (curves, weights, activations, geometry, output, gradients, interventions,
// seeds, teaching) uses these so behaviour is uniform across the dashboard and the chapters.
import { state, subscribe, set } from "../core/state.js";
import { run } from "../core/data.js";

export { state, set };

/** The run id a figure should display. */
export const runOf = (ds) => ds.run || state.run;

/** Subscribe `fn(state)` to keys, re-running also when the theme changes (charts read theme colours). */
export function watch(keys, fn) {
  const unsub = subscribe(keys, fn);
  const theme = () => fn(state, ["theme"]);
  window.addEventListener("grokviz-theme", theme);
  return () => { unsub(); window.removeEventListener("grokviz-theme", theme); };
}

/** Create an element with class and optional HTML, append it to parent, return it. */
export function el(parent, tag, cls = "", html = "") {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html) e.innerHTML = html;
  parent.append(e);
  return e;
}

/** Human label of basis index i: 0 → "Const", 2k−1 → "cos k", 2k → "sin k" (paper convention). */
export const basisName = (i) => (i === 0 ? "Const" : `${i % 2 ? "cos" : "sin"} ${Math.ceil(i / 2)}`);

/** Format a step number compactly (40001 → "40.0k"). */
export const kstep = (s) => (s >= 1000 ? `${(s / 1000).toFixed(1)}k` : String(s));

/** Run manifest entry for the figure's run. */
export const runInfo = (ds) => run(runOf(ds));

/** A note line under a figure (e.g. which checkpoint is displayed). */
export function note(parent) {
  return el(parent, "p", "fig-note");
}
