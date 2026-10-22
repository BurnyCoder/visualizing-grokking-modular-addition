// page-player.js — the one Player of a page, playing the selected run on the page's global state.
//
// Local explanation: created on first use; follows the run selection (loads the run's checkpoint steps and phases
// from the manifest, ignoring answers for a run that is no longer selected), uses the reader's reduced-motion
// preference, and while playing warms the next weight snapshots (core/data.js prefetchModels) for model-based panels.
// `playSteps` plays a range of training steps on it (tour steps and scrollytelling scenes).
// Global context: the dashboard's player bar, the chapter scrubber widget, scrollytelling scenes and the guided tour
// all control this same Player, so they never disagree about the step.
import { Player } from "../core/player.js";
import { nearest } from "../core/anim.js";
import { storeOf } from "../core/state.js";
import { reducedMotion } from "../core/motion.js";
import { prefetchModels, run } from "../core/data.js";

let player = null;

/** The page's Player (global store), created on first call; `player.ready` resolves once a run's steps are loaded. */
export function pagePlayer() {
  if (player) return player;
  const store = storeOf(null); // no [data-scope] ancestor → the global store
  player = new Player(store, { reduced: reducedMotion });
  let loaded;
  player.ready = new Promise((r) => { loaded = r; });
  store.subscribe(["run"], async (s) => {
    const id = s.run, r = await run(id);
    if (store.state.run !== id) return; // a newer selection wins
    player.setSteps(r.ckpt_steps, r.phases);
    loaded();
  });
  store.subscribe(["step"], (s) => { if (s.playing) prefetchModels(s.run, s.step); }, { immediate: false });
  return player;
}

/** Play training steps `from` … `to` on the page's Player (tour steps, scenes); under reduced motion show `to` instead. */
export async function playSteps({ from, to, speed = 1, loop = "once" }) {
  const p = pagePlayer();
  await p.ready;
  const lo = nearest(p.steps, from), hi = nearest(p.steps, to);
  p.setRange([lo, hi]);
  p.setSpeed(speed);
  p.setLoop(loop);
  if (reducedMotion()) { p.seek(hi); return; }
  p.seek(lo);
  p.play();
}
