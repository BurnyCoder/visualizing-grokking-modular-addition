// scrollplay.js — play animations while they are on screen, and apply scrollytelling scenes.
//
// Local explanation: `scrollPlay(el, player)` registers an animated figure: it starts playing when at least 60 % of
// it (or of the screen, for figures taller than the screen) is visible and pauses below 20 %; only one animation
// plays at a time; once the reader touches its controls it no longer starts by itself; nothing autoplays under
// reduced motion, and a hidden tab pauses everything. `initScenes()` keeps one {{scene:id}} marker active: the last
// one above two thirds of the screen height (rechecked once per frame while scrolling, so jumps via links count
// too). When the active marker changes, its `state` is set and its `play` range ({from, to, speed} in training
// steps) is played on the page's Player (under reduced motion it jumps to the end of the range instead).
// Global context: chapter animations ({{anim:id}}) and scrollytelling chapters; the dashboard never autoplays.
// Sources: IntersectionObserver thresholds (https://developer.mozilla.org/docs/Web/API/Intersection_Observer_API); the
// "step enters" pattern of scrollytelling libraries (https://github.com/russellsamora/scrollama);
// Page Visibility API (https://developer.mozilla.org/docs/Web/API/Page_Visibility_API).
import { reducedMotion } from "../core/motion.js";
import { set } from "../core/state.js";
import { playSteps } from "./page-player.js";

const registered = new Map(); // figure element → {player, manual}
const THRESHOLDS = Array.from({ length: 11 }, (_, i) => i / 10); // report every 10 % of visibility
let io = null;

/** Share of `r`'s element on screen, relative to the element or to the screen when the element is taller. */
const shown = (r) => r.intersectionRect.height / Math.max(1, Math.min(r.boundingClientRect.height, r.rootBounds?.height ?? innerHeight));

function onVisibility(records) {
  for (const r of records) {
    const e = registered.get(r.target);
    if (!e) continue;
    const v = r.isIntersecting ? shown(r) : 0;
    if (v < 0.2) e.player.pause();
    else if (v >= 0.6 && !e.manual && !reducedMotion() && !e.player.playing) {
      for (const o of registered.values()) if (o !== e) o.player.pause(); // one animation at a time
      e.player.play();
    }
  }
}

/** Register an animated figure element with its Player. */
export function scrollPlay(el, player) {
  io ||= new IntersectionObserver(onVisibility, { threshold: THRESHOLDS });
  const entry = { player, manual: false };
  registered.set(el, entry);
  el.querySelector(".anim-bar")?.addEventListener("pointerdown", () => { entry.manual = true; }); // the reader took over
  io.observe(el);
}

document.addEventListener("visibilitychange", () => { if (document.hidden) for (const e of registered.values()) e.player.pause(); });

/** Apply one scene: set its state, then play (or, under reduced motion, show the end of) its step range. */
function applyScene(scene) {
  if (scene.state) set(scene.state);
  if (scene.play) playSteps(scene.play);
}

/** Watch every {{scene:id}} marker on the page; apply the active one whenever it changes. */
export function initScenes() {
  const markers = [...document.querySelectorAll(".scene[data-scene]")];
  if (!markers.length) return;
  let current = null, queued = false;
  const update = () => {
    queued = false;
    const active = markers.filter((m) => m.getBoundingClientRect().top < (2 * innerHeight) / 3).at(-1) ?? null;
    if (active && active !== current) applyScene(JSON.parse(active.dataset.scene));
    current = active;
  };
  addEventListener("scroll", () => { if (!queued) { queued = true; requestAnimationFrame(update); } }, { passive: true });
  update();
}
