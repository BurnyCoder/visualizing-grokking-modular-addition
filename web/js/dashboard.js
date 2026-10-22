// dashboard.js — controller of the Big Dashboard's control bar and guided tour.
//
// Local explanation: fills the run selector from the manifest, binds the step slider (snapping to stored
// checkpoints of the selected run, with the phase bands drawn underneath), the play button, the a/b/a′/head/
// neuron/frequency inputs and keyboard ←/→, all two-way bound to the shared state; the tour steps through
// content/tour.yaml, setting the state and highlighting one panel at a time.
// Global context: the panels themselves are ordinary figures that subscribe to the same state.
// Source: HTML range input (https://developer.mozilla.org/docs/Web/HTML/Element/input/range).
import { manifest, nearestIndex as nearest, run } from "./core/data.js";
import { set, state, subscribe } from "./core/state.js";
import { reducedMotion } from "./core/motion.js";
import { cssVar } from "./viz/colors.js";

const $ = (id) => document.getElementById(id);

export async function initDashboard() {
  const m = await manifest();
  const sel = $("ctl-run");
  for (const r of m.runs) sel.append(new Option(`${r.id} — ${r.title}`, r.id));
  sel.onchange = () => set({ run: sel.value });
  const slider = $("ctl-step"), label = $("ctl-step-label");
  let steps = [];
  let showStep = () => {};
  subscribe(["run"], async (s) => {
    sel.value = s.run;
    const r = await run(s.run);
    steps = r.ckpt_steps;
    slider.max = String(steps.length - 1);
    drawPhases(r);
    $("ctl-freq").value = String(s.freq);
    setStatus(`run ${r.id}: key frequencies ${r.key_freqs.join(", ")}; phases ${fmtPhases(r.phases)}`);
    set({ step: steps[nearest(steps, s.step)] });
    showStep(s); // also when the step value itself did not change
  });
  showStep = (s) => { // move the slider/label to the stored checkpoint nearest to s.step
    if (!steps.length) return;
    const i = nearest(steps, s.step);
    slider.value = String(i);
    label.textContent = `step ${steps[i].toLocaleString()}`;
  };
  subscribe(["step"], showStep);
  slider.oninput = () => set({ step: steps[Number(slider.value)] });
  for (const [id, key, max] of [["ctl-a", "a", 112], ["ctl-b", "b", 112], ["ctl-ap", "ap", 112], ["ctl-neuron", "neuron", 511], ["ctl-freq", "freq", 56]]) {
    const input = $(id);
    subscribe([key], (s) => { input.value = String(s[key]); });
    input.onchange = () => set({ [key]: Math.max(0, Math.min(max, Number(input.value) || 0)) });
  }
  const head = $("ctl-head");
  subscribe(["head"], (s) => { head.value = String(s.head); });
  head.onchange = () => set({ head: Number(head.value) });
  let timer = null; // play/pause through checkpoints
  $("ctl-play").onclick = (e) => {
    if (timer) { clearInterval(timer); timer = null; e.target.textContent = "▶"; return; }
    e.target.textContent = "❚❚";
    timer = setInterval(() => {
      const i = nearest(steps, state.step);
      if (i >= steps.length - 1) { clearInterval(timer); timer = null; e.target.textContent = "▶"; return; }
      set({ step: steps[i + 1] });
    }, 220);
  };
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea")) return;
    const i = nearest(steps, state.step);
    if (e.key === "ArrowRight") set({ step: steps[Math.min(i + 1, steps.length - 1)] });
    if (e.key === "ArrowLeft") set({ step: steps[Math.max(i - 1, 0)] });
  });
  initTour();
}

function fmtPhases(ph) {
  if (!ph || !ph.t1) return "not detected";
  return `memorisation ends ≈${ph.t1}, circuit formation ends ≈${ph.t2 ?? "—"}, cleanup ends ≈${ph.t3 ?? "—"}`;
}

function setStatus(t) { $("ctl-status").textContent = t; }

/** Colour bands under the slider for the three phases of the selected run. */
function drawPhases(r) {
  const strip = $("ctl-phases");
  strip.replaceChildren();
  const ph = r.phases, end = r.ckpt_steps.at(-1);
  if (!ph || !ph.t1) return;
  for (const [a, b, c] of [[0, ph.t1, "--phase1"], [ph.t1, ph.t2, "--phase2"], [ph.t2, ph.t3, "--phase3"]]) {
    if (a == null || b == null) continue;
    const span = document.createElement("span");
    span.style.left = `${(100 * a) / end}%`;
    span.style.width = `${(100 * (b - a)) / end}%`;
    span.style.background = cssVar(c).replace(/0\.\d+\)$/, "0.45)");
    strip.append(span);
  }
}

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
