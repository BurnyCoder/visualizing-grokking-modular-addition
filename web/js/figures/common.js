// common.js — small helpers shared by the figure modules.
//
// Local explanation: which run a figure shows (fixed via data-run, else the global selection); `watch`, the one way a
// figure subscribes to the state (gated, coalesced, deduplicated rendering); `follow` for cheap overlay updates; a
// dedupe key per stored frame; simple DOM builders and frequency/basis labels.
// Global context: every figure module (curves, weights, activations, geometry, output, gradients, interventions,
// seeds) uses these so behaviour is uniform across the dashboard and the chapters.
// Render pipeline: store flush (core/store.js) → visibility gate (core/visibility.js) → latest-wins runner → dedupe
// → the figure's draw function; offscreen figures only remember that they are dirty and draw when scrolled back.
import { state, set, storeOf } from "../core/state.js";
import { latestWins } from "../core/store.js";
import { frameAt, lerpInto } from "../core/anim.js";
import { onVisibility } from "../core/visibility.js";
import { countRender, track } from "../core/hooks.js";
import { nearestIndex, run } from "../core/data.js";

export { state, set, storeOf };

/** The run id a figure should display. */
export const runOf = (ds) => ds.run || state.run;

const union = (a, b) => [...new Set([...a, ...b])]; // merge changed-key lists

/**
 * Draw `fn(state, changedKeys)` now and whenever one of `keys` changes (or the theme does), for the figure whose body
 * is `body`: one draw at a time with the latest state (no out-of-order frames), none while the figure is offscreen
 * (it draws once when it comes back), and none when `dedupe(state)` (optional, may be async) returns the same key as
 * for the last draw — e.g. the same stored checkpoint while the step moves between two snapshots. `theme: false` for
 * figures whose colours do not depend on the theme (canvas heatmaps), so a theme switch does not recompute them.
 * `minGap` (ms) is for expensive figures (thousands of SVG points, many grids): while a run plays they redraw at most
 * that often, always ending with the latest state, so they never hold up the frames of the rest of the page.
 */
export function watch(body, keys, fn, { dedupe = null, theme = true, minGap = 0 } = {}) {
  const st = storeOf(body); // global store or the figure's animation scope
  const id = body.closest?.("[data-fig]")?.dataset.fig || body.dataset?.widget || "?"; // for render statistics
  let lastKey; // dedupe key of the last draw
  let dirty = null; // changed keys collected while offscreen
  const draw = latestWins(async (changed) => {
    const s = st.state;
    if (dedupe) {
      const k = await dedupe(s);
      if (k === lastKey && !changed.includes("theme")) return; // same frame already on screen
      lastKey = k;
    }
    if (body.hasAttribute("aria-live")) body.setAttribute("aria-live", s.playing ? "off" : "polite"); // no announcements per played frame
    const t0 = performance.now();
    await fn(s, changed);
    countRender(id, performance.now() - t0);
  }, union);
  const gate = onVisibility(body, () => { if (dirty) { const c = dirty; dirty = null; track(draw(c)); } });
  let lastStart = -Infinity, held = null, timer = null; // minGap throttling while playing
  const trigger = (changed) => {
    if (!gate.visible()) { dirty = union(dirty || [], changed); return; }
    const wait = minGap && st.state.playing ? lastStart + minGap - performance.now() : 0;
    if (wait > 0) { // too soon: remember the change and draw once the gap has passed (the latest state wins)
      held = union(held || [], changed);
      timer ||= setTimeout(() => { timer = null; const c = held; held = null; trigger(c); }, wait);
      return;
    }
    lastStart = performance.now();
    track(draw(changed));
  };
  const unsub = st.subscribe(keys, (_, changed) => trigger(changed));
  const onTheme = () => trigger(["theme"]);
  if (theme) window.addEventListener("grokviz-theme", onTheme);
  return () => { unsub(); window.removeEventListener("grokviz-theme", onTheme); };
}

/** Cheap synchronous updates (overlay marks, labels) of the figure's store on `keys`; no gating needed. */
export const follow = (body, keys, fn) => storeOf(body).subscribe(keys, fn);

/**
 * Dedupe key for `watch`: the run, the stored frame of `field` (ckpt_steps, weight_steps, geometry_steps, spectra_steps)
 * nearest the current step, and the values of extra state keys.
 */
export const frameKey = (ds, field, ...keys) => async (s) => {
  const id = runOf(ds), steps = (await run(id))[field];
  return [id, steps ? steps[nearestIndex(steps, s.step)] : "-", ...keys.map((k) => s[k])].join("|");
};

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

/** A note line under a figure (e.g. which checkpoint is displayed). */
export function note(parent) {
  return el(parent, "p", "fig-note");
}

/**
 * Neuron indices sorted by frequency cluster: neurons whose best single-frequency FVE exceeds 0.85 come first, ordered
 * by that frequency; the rest (unclustered / always-firing) last. `freq[i]`, `fve[i]` as in metrics.neuron_frequencies.
 */
export const clusterOrder = (freq, fve, n = freq.length) =>
  Array.from({ length: n }, (_, i) => i).sort((x, y) => ((fve[x] > 0.85 ? freq[x] : 99) - (fve[y] > 0.85 ? freq[y] : 99)) || x - y);

/**
 * Fill `out` with frame data at the store's position `s`: each stored frame of `a` (one per entry of `steps`) holds
 * `out.length` numbers at `offset`. Paused: the nearest stored frame. Playing: `lerp` of the two stored frames around
 * the fractional step (core/anim.js). Returns a label saying which, so a blended picture is always disclosed.
 */
export function fill(out, a, steps, s, { offset = 0, lerp = lerpInto } = {}) {
  const size = a.data.length / steps.length, frame = (f) => a.data.subarray(f * size + offset, f * size + offset + out.length);
  const fmt = (x) => x.toLocaleString("en-US");
  if (s.playing && s.stepf != null) {
    const { i, j, t } = frameAt(steps, s.stepf);
    lerp(out, frame(i), frame(j), t);
    return t > 0 ? `step ≈ ${fmt(Math.round(s.stepf))}, blended between the stored frames at ${fmt(steps[i])} and ${fmt(steps[j])}` : `step ${fmt(steps[i])}`;
  }
  const f = nearestIndex(steps, s.step);
  out.set(frame(f));
  return `step ${fmt(steps[f])} (stored frame)`;
}
