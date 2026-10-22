// anim-figure.js — turn a figure placed with {{anim:id}} into a self-playing animation.
//
// Local explanation: the figure element carries data-scope (its own step, core/state.js storeOf), data-run (the run
// it plays) and data-anim = {run, range, speed, loop, poster} from its `anim:` block in content/figures.yaml.
// `mountAnim` creates a Player on the figure's scope, loads the run's checkpoints and phases, selects the range (a
// preset name such as "grokking", or [from, to] in training steps), shows the poster frame (the `poster` step, else
// the end of the range — the most informative still frame, which reduced-motion readers keep), adds a mini player
// bar and registers the figure for scroll-play (ui/scrollplay.js).
// Global context: chapters animate any dashboard figure this way without changing the figure's code.
import { Player } from "../core/player.js";
import { nearest } from "../core/anim.js";
import { storeOf } from "../core/state.js";
import { reducedMotion } from "../core/motion.js";
import { run } from "../core/data.js";
import { playerBar } from "./player-bar.js";
import { scrollPlay } from "./scrollplay.js";

/** Mount the player, its bar and scroll-play on an {{anim:id}} figure element. */
export async function mountAnim(fig) {
  const cfg = JSON.parse(fig.dataset.anim);
  const player = new Player(storeOf(fig), { reduced: reducedMotion });
  const r = await run(cfg.run);
  player.setSteps(r.ckpt_steps, r.phases, Array.isArray(cfg.range) ? cfg.range.map((s) => nearest(r.ckpt_steps, s)) : cfg.range);
  player.setSpeed(cfg.speed);
  player.setLoop(cfg.loop);
  player.seek(cfg.poster != null ? nearest(r.ckpt_steps, cfg.poster) : player.range[1]);
  playerBar(fig.querySelector(".anim-bar"), player, { variant: "mini" });
  scrollPlay(fig, player);
  return player;
}
