// player-bar.js — playback controls for a Player, in three sizes.
//
// Local explanation: `playerBar(el, player, {variant, keys})` builds the controls inside `el` and keeps them in sync
// with the Player:
//   full    (dashboard)     ⏮ ◀ ▶/❚❚ ▶ ⏭, scrubber over the checkpoints with the phase bands and the play range
//                           marked, "step · phase" label, speed, loop mode, range and phase chips, keyboard help;
//   compact (chapter bar)   ▶/❚❚, scrubber, label, speed;
//   mini    (inside an animated figure) ▶/❚❚, scrubber, label.
// With `keys: true` the page keyboard drives it: Space play/pause, ←/→ one checkpoint (Shift: ten), [ / ] slower /
// faster, Home/End range start/end, L loop mode, ? key help. The scrubber works in checkpoint indices, so every
// position is a stored checkpoint and the phase bands sit where those checkpoints are.
// Global context: replaces the separate play loops of the dashboard and the chapter scrubber; one Player per store.
// Sources: HTML range input (https://developer.mozilla.org/docs/Web/HTML/Element/input/range); keyboard events
// (https://developer.mozilla.org/docs/Web/API/KeyboardEvent/key).
import { SPEEDS, nearest, phaseOf } from "../core/anim.js";
import { cssVar } from "../viz/colors.js";

const LOOPS = { loop: "⟲ loop", pingpong: "⇄ back and forth", once: "→ once" }; // loop modes in cycling order
const CHIPS = { grokking: "Grokking", whole: "Whole run", late: "Late training", memorisation: "Memorisation", circuit: "Circuit formation", cleanup: "Cleanup" };
const KEYS = [["Space", "play / pause"], ["← →", "one checkpoint (Shift: ten)"], ["[ ]", "slower / faster"], ["Home End", "start / end of the range"], ["L", "loop mode"], ["?", "this help"]];
const speedName = (x) => ({ 0.25: "¼×", 0.5: "½×" })[x] || `${x}×`;

/** Build the controls for `player` in `el`; returns {destroy}. */
export function playerBar(el, player, { variant = "full", keys = false } = {}) {
  const full = variant === "full";
  el.classList.add("player-bar", variant);
  el.innerHTML = `
    <div class="pb-main">
      ${full ? '<button type="button" class="btn pb-icon" data-act="start" title="Start of the range (Home)" aria-label="Start of the range">⏮</button><button type="button" class="btn pb-icon" data-act="back" title="Previous checkpoint (←)" aria-label="Previous checkpoint">◀</button>' : ""}
      <button type="button" class="btn pb-play" data-act="play" aria-label="Play">▶</button>
      ${full ? '<button type="button" class="btn pb-icon" data-act="fwd" title="Next checkpoint (→)" aria-label="Next checkpoint">▶|</button><button type="button" class="btn pb-icon" data-act="end" title="End of the range (End)" aria-label="End of the range">⏭</button>' : ""}
      <div class="scrubber-wrap"><input type="range" min="0" max="0" step="1" value="0" aria-label="Training step"><div class="phase-strip" aria-hidden="true"></div><div class="pb-range" aria-hidden="true"></div></div>
      <output class="pb-label mono"></output>
      ${variant !== "mini" ? `<select class="pb-speed" aria-label="Playback speed" title="Speed ([ and ])">${SPEEDS.map((s) => `<option value="${s}">${speedName(s)}</option>`).join("")}</select>` : ""}
      ${full ? '<button type="button" class="btn pb-loop" data-act="loop" title="Loop mode (L)"></button><button type="button" class="btn pb-icon" data-act="help" title="Keyboard shortcuts (?)" aria-label="Keyboard shortcuts">?</button>' : ""}
    </div>
    ${full ? `<div class="pb-chips" role="group" aria-label="Play range">${Object.entries(CHIPS).map(([k, v]) => `<button type="button" class="chip" data-range="${k}">${v}</button>`).join("")}</div>
      <div class="pb-help" hidden>${KEYS.map(([k, v]) => `<kbd>${k}</kbd> ${v}`).join(" · ")}</div>` : ""}`;
  const $ = (s) => el.querySelector(s);
  const slider = $("input[type=range]"), label = $(".pb-label"), play = $(".pb-play"), speed = $(".pb-speed");
  let shownSteps = null; // steps list the strip was drawn for

  const acts = {
    play: () => player.toggle(),
    start: () => player.seek(player.range[0], { tween: true }),
    end: () => player.seek(player.range[1], { tween: true }),
    back: () => player.stepBy(-1),
    fwd: () => player.stepBy(1),
    loop: () => { const m = Object.keys(LOOPS); player.setLoop(m[(m.indexOf(player.loop) + 1) % m.length]); },
    help: () => { const h = $(".pb-help"); if (h) h.hidden = !h.hidden; },
  };
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.act) acts[b.dataset.act]();
    if (b.dataset.range) player.jumpTo(b.dataset.range);
  });
  slider.addEventListener("input", () => player.seek(Number(slider.value)));
  speed?.addEventListener("change", () => player.setSpeed(Number(speed.value)));

  /** Phase bands and the play range under the scrubber (positions in checkpoint-index space). */
  function drawStrip() {
    const strip = $(".phase-strip"), steps = player.steps, n = steps.length - 1 || 1, ph = player.phases;
    strip.replaceChildren();
    if (ph?.t1) {
      for (const [a, b, c] of [[0, ph.t1, "--phase1"], [ph.t1, ph.t2, "--phase2"], [ph.t2, ph.t3, "--phase3"]]) {
        if (a == null || b == null) continue;
        const i = nearest(steps, a), j = nearest(steps, b); // checkpoint indices of the band's ends
        const span = document.createElement("span");
        span.style.left = `${(100 * i) / n}%`;
        span.style.width = `${(100 * (j - i)) / n}%`;
        span.style.background = cssVar(c).replace(/0\.\d+\)$/, "0.45)"); // the theme's band colour, more opaque
        strip.append(span);
      }
    }
    shownSteps = steps;
  }

  function update() {
    if (!player.steps) return;
    if (shownSteps !== player.steps) { slider.max = String(player.steps.length - 1); drawStrip(); }
    const i = Math.round(player.pos), step = player.steps[i], n = player.steps.length - 1 || 1;
    slider.value = String(i);
    const phase = phaseOf(step, player.phases);
    label.textContent = `step ${step.toLocaleString("en-US")}${phase && variant !== "mini" ? ` · ${phase}` : ""}`;
    play.textContent = player.playing ? "❚❚" : "▶";
    play.setAttribute("aria-label", player.playing ? "Pause" : "Play");
    if (speed) speed.value = String(player.speed);
    const loop = $(".pb-loop");
    if (loop) loop.textContent = LOOPS[player.loop];
    const [lo, hi] = player.range, mark = $(".pb-range");
    Object.assign(mark.style, { left: `${(100 * lo) / n}%`, width: `${(100 * (hi - lo)) / n}%` });
    el.querySelectorAll(".chip").forEach((c) => {
      c.classList.toggle("active", c.dataset.range === player.rangeName);
      c.hidden = !player.presets[c.dataset.range]; // runs without detected phases offer only "whole run"
    });
  }
  const off = player.onChange(update);
  update();

  /** Page keyboard shortcuts (ignored while typing in a field or when a button has focus: Space would click it). */
  function onKey(e) {
    if (e.target.closest?.("input, select, textarea, button, [contenteditable]") || e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key, i = SPEEDS.indexOf(player.speed);
    const map = {
      " ": acts.play, ArrowLeft: () => player.stepBy(e.shiftKey ? -10 : -1), ArrowRight: () => player.stepBy(e.shiftKey ? 10 : 1),
      "[": () => player.setSpeed(SPEEDS[Math.max(i - 1, 0)]), "]": () => player.setSpeed(SPEEDS[Math.min(i + 1, SPEEDS.length - 1)]),
      Home: acts.start, End: acts.end, l: acts.loop, L: acts.loop, "?": acts.help,
    };
    if (!map[k]) return;
    e.preventDefault(); // no page scroll on Space / Home / End
    map[k]();
  }
  if (keys) document.addEventListener("keydown", onKey);
  return { destroy: () => { off(); document.removeEventListener("keydown", onKey); } };
}
