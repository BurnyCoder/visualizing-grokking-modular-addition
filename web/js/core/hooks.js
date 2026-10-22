// hooks.js — test and capture hooks: render counts per figure and a "settled" signal, as window.grokviz.anim.
//
// Local explanation: figures/common.js reports every render (`countRender`) and every render promise (`track`);
// `stats()` returns {renders: {figureId: n}, inflight}; `settled()` resolves once two animation frames have passed
// with no render in flight (state flushes happen on animation frames, so one frame is not enough).
// Global context: tests/test_site_browser.py checks that offscreen figures do not render while the step changes, and
// screenshot/clip capture waits for `settled()` instead of fixed sleeps.
// Source: requestAnimationFrame (https://developer.mozilla.org/docs/Web/API/Window/requestAnimationFrame).

const renders = {}; // figure id → number of renders
const active = new Set(); // render promises still running

/** Count one render of figure `id`. */
export const countRender = (id) => { renders[id] = (renders[id] || 0) + 1; };

/** Remember a render promise until it settles. */
export function track(p) {
  active.add(p);
  p.finally(() => active.delete(p));
  return p;
}

const frame = () => new Promise((r) => requestAnimationFrame(r));
/** Resolve when no render is running or about to run. */
async function settled() {
  for (;;) {
    await frame();
    await frame();
    if (!active.size) return;
    await Promise.allSettled([...active]);
  }
}

window.grokviz = { ...(window.grokviz || {}), anim: { stats: () => ({ renders: { ...renders }, inflight: active.size }), settled } };
