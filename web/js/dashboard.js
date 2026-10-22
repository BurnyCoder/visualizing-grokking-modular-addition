// dashboard.js — controller of the Big Dashboard's control bar and guided tour.
//
// Local explanation: fills the run selector from the manifest, mounts the full player bar (ui/player-bar.js: play,
// scrubber with phase bands, speed, loop, range chips, keyboard) on the page's Player, binds the a/b/a′/head/neuron/
// frequency inputs two-way to the shared state; the tour steps through content/tour.yaml, setting the state and
// highlighting one panel at a time.
// Global context: the panels themselves are ordinary figures that subscribe to the same state.
// Source: HTML number input (https://developer.mozilla.org/docs/Web/HTML/Element/input/number).
import { manifest, run } from "./core/data.js";
import { set, subscribe } from "./core/state.js";
import { reducedMotion } from "./core/motion.js";
import { pagePlayer } from "./ui/page-player.js";
import { playerBar } from "./ui/player-bar.js";

const $ = (id) => document.getElementById(id);

export async function initDashboard() {
  const m = await manifest();
  const sel = $("ctl-run");
  for (const r of m.runs) sel.append(new Option(`${r.id} — ${r.title}`, r.id));
  sel.onchange = () => set({ run: sel.value });
  playerBar($("ctl-player"), pagePlayer(), { variant: "full", keys: true });
  subscribe(["run"], async (s) => {
    sel.value = s.run;
    const r = await run(s.run);
    setStatus(`run ${r.id}: key frequencies ${r.key_freqs.join(", ")}; phases ${fmtPhases(r.phases)}`);
  });
  for (const [id, key, max] of [["ctl-a", "a", 112], ["ctl-b", "b", 112], ["ctl-ap", "ap", 112], ["ctl-neuron", "neuron", 511], ["ctl-freq", "freq", 56]]) {
    const input = $(id);
    subscribe([key], (s) => { input.value = String(s[key]); });
    input.onchange = () => set({ [key]: Math.max(0, Math.min(max, Number(input.value) || 0)) });
  }
  const head = $("ctl-head");
  subscribe(["head"], (s) => { head.value = String(s.head); });
  head.onchange = () => set({ head: Number(head.value) });
  initTour();
}

function fmtPhases(ph) {
  if (!ph || !ph.t1) return "not detected";
  return `memorisation ends ≈${ph.t1}, circuit formation ends ≈${ph.t2 ?? "—"}, cleanup ends ≈${ph.t3 ?? "—"}`;
}

function setStatus(t) { $("ctl-status").textContent = t; }

/** Guided tour through content/tour.yaml steps. */
function initTour() {
  const steps = (window.GROKVIZ && window.GROKVIZ.tour) || [];
  const box = $("tour");
  let i = 0;
  const show = () => {
    const t = steps[i];
    document.querySelectorAll(".fig.tour-focus").forEach((f) => f.classList.remove("tour-focus"));
    set({ ...(t.state || {}) });
    const fig = document.getElementById(`fig-${t.panel}`);
    if (fig) { fig.classList.add("tour-focus"); fig.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "center" }); }
    box.querySelector(".tour-step").textContent = `${i + 1} / ${steps.length}`;
    box.querySelector(".tour-title").textContent = t.title;
    box.querySelector(".tour-text").innerHTML = t.text;
    window.grokvizRenderMath?.(box);
  };
  $("ctl-tour").onclick = () => { if (!steps.length) return; box.hidden = false; i = 0; show(); };
  box.querySelector(".tour-next").onclick = () => { i = Math.min(i + 1, steps.length - 1); show(); };
  box.querySelector(".tour-prev").onclick = () => { i = Math.max(i - 1, 0); show(); };
  box.querySelector(".tour-close").onclick = () => { box.hidden = true; document.querySelectorAll(".fig.tour-focus").forEach((f) => f.classList.remove("tour-focus")); };
}
