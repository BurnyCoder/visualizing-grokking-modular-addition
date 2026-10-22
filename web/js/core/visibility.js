// visibility.js — one shared IntersectionObserver telling figures whether they are on (or near) the screen.
//
// Local explanation: `onVisibility(el, onShow)` starts observing `el` (once, however many callers) and returns
// {visible()}; every `onShow()` registered for `el` runs each time it comes back within the margin. Elements count as
// visible until the observer's first report, because figures are only created when they approach the viewport
// (app.js) and must draw their first frame right away.
// Global context: figures/common.js `watch` skips renders of offscreen figures and marks them dirty, so playing a run
// on the dashboard only redraws the panels the reader can see; a figure may call `watch` several times (one per tile).
// Source: IntersectionObserver (https://developer.mozilla.org/docs/Web/API/Intersection_Observer_API).

const MARGIN = "200px 0px"; // render slightly before a figure scrolls in (creation in app.js uses 400px)
const entries = new WeakMap(); // element → {visible, shows: callbacks}
let io = null; // created on first use

/** Observe `el`; returns {visible()} and calls onShow() whenever it becomes visible again. */
export function onVisibility(el, onShow) {
  io ||= new IntersectionObserver((records) => {
    for (const r of records) {
      const e = entries.get(r.target);
      if (!e) continue;
      const was = e.visible;
      e.visible = r.isIntersecting;
      if (e.visible && !was) e.shows.forEach((f) => f());
    }
  }, { rootMargin: MARGIN });
  let e = entries.get(el);
  if (!e) {
    e = { visible: true, shows: [] };
    entries.set(el, e);
    io.observe(el);
  }
  e.shows.push(onShow);
  return { visible: () => e.visible };
}
