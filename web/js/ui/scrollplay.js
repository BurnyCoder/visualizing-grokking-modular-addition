// scrollplay.js — play animations while they are on screen, and apply scrollytelling scenes.
//
// Local explanation: `scrollPlay(el, player)` registers an animated figure: it starts playing when at least 60 % of
// it (or of the screen, for figures taller than the screen) is visible and pauses below 20 %; only one animation
// plays at a time; once the reader touches its controls it no longer starts by itself; nothing autoplays under
// reduced motion, and a hidden tab pauses everything. `initScenes()` watches the {{scene:id}} markers: when one
// crosses the middle of the screen, its `state` is set and its `play` range ({from, to, speed} in training steps)
// is played on the page's Player (under reduced motion it jumps to the end of the range instead).
// Global context: chapter animations ({{anim:id}}) and scrollytelling chapters; the dashboard never autoplays.
// Sources: IntersectionObserver thresholds and rootMargin (https://developer.mozilla.org/docs/Web/API/Intersection_Observer_API);
// Page Visibility API (https://developer.mozilla.org/docs/Web/API/Page_Visibility_API).
import { reducedMotion } from "../core/motion.js";
import { nearest } from "../core/anim.js";
import { set } from "../core/state.js";
import { pagePlayer } from "./page-player.js";

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
async function applyScene(scene) {
  if (scene.state) set(scene.state);
  if (!scene.play) return;
  const player = pagePlayer();
  await player.ready;
  const lo = nearest(player.steps, scene.play.from), hi = nearest(player.steps, scene.play.to);
  player.setRange([lo, hi]);
  if (scene.play.speed) player.setSpeed(scene.play.speed);
  if (reducedMotion()) { player.seek(hi); return; }
  player.seek(lo);
  player.play();
}

/** Watch every {{scene:id}} marker on the page. */
export function initScenes() {
  const markers = document.querySelectorAll(".scene[data-scene]");
  if (!markers.length) return;
  const sio = new IntersectionObserver((records) => {
    for (const r of records) if (r.isIntersecting) applyScene(JSON.parse(r.target.dataset.scene));
  }, { rootMargin: "-50% 0px -50% 0px" }); // a zero-height band across the middle of the screen
  markers.forEach((m) => sio.observe(m));
}
