// model.js — widgets that drive the real network: the in-chapter training scrubber and the step-by-step forward
// pass of chapter 5.
//
// Local explanation: `scrubber` is a compact copy of the dashboard's run selector + step slider (+ play), bound to
// the shared state, so chapter figures can be moved through training; `forward-walkthrough` runs GrokModel's
// detailed forward pass on the selected (a, b) at the selected step and prints every intermediate quantity
// (embeddings, attention scores and pattern per head, residuals, a few neurons, top logits, probabilities).
// Global context: lets a reader check every equation of chapter 5 against the actual numbers of the trained model.
// Source: paper App. A equations (https://arxiv.org/abs/2301.05217).
import { manifest, model, nearestIndex as nearest, run } from "../core/data.js";
import { set, state, subscribe } from "../core/state.js";
import { softmax } from "../model/grokmodel.js";
import { frameKey, watch } from "../figures/common.js";

const fmt = (xs, n = 4) => `(${Array.from(xs).slice(0, n).map((x) => x.toFixed(3)).join(", ")}, …)`;

export const widgets = {
  scrubber: {
    async render(el) {
      el.classList.add("chapter-scrubber");
      el.innerHTML = `<div class="cb-row"><label>Run <select></select></label><button class="btn" type="button">▶</button>
        <input type="range" min="0" max="0" value="0" style="flex:1 1 14rem"><output class="mono"></output></div>`;
      const sel = el.querySelector("select"), slider = el.querySelector("input"), label = el.querySelector("output"), play = el.querySelector("button");
      const m = await manifest();
      for (const r of m.runs) sel.append(new Option(r.id, r.id));
      let steps = [];
      const showStep = (s) => { if (!steps.length) return; const i = nearest(steps, s.step); slider.value = String(i); label.textContent = `step ${steps[i].toLocaleString()}`; };
      subscribe(["run"], async (s) => { sel.value = s.run; steps = (await run(s.run)).ckpt_steps; slider.max = String(steps.length - 1); set({ step: steps[nearest(steps, s.step)] }); showStep(s); });
      subscribe(["step"], showStep);
      sel.onchange = () => set({ run: sel.value });
      slider.oninput = () => set({ step: steps[Number(slider.value)] });
      let timer = null;
      play.onclick = () => {
        if (timer) { clearInterval(timer); timer = null; play.textContent = "▶"; return; }
        play.textContent = "❚❚";
        timer = setInterval(() => { const i = nearest(steps, state.step); if (i >= steps.length - 1) { clearInterval(timer); timer = null; play.textContent = "▶"; } else set({ step: steps[i + 1] }); }, 200);
      };
    },
  },

  "forward-walkthrough": {
    async render(el) {
      el.innerHTML = `<p class="widget-title">One forward pass of the paper's trained network</p>
        <div class="row"><label>a <input type="number" min="0" max="112" class="wa"></label><label>b <input type="number" min="0" max="112" class="wb"></label>
        <span class="fig-note">step <output class="ws mono"></output> (change it with the dashboard or chapter 18's slider)</span></div><div class="workings"></div>`;
      const wa = el.querySelector(".wa"), wb = el.querySelector(".wb"), out = el.querySelector(".workings"), ws = el.querySelector(".ws");
      wa.onchange = () => set({ a: Math.max(0, Math.min(112, Number(wa.value))) });
      wb.onchange = () => set({ b: Math.max(0, Math.min(112, Number(wb.value))) });
      watch(el, ["run", "step", "a", "b"], async (s) => { // gated + skipped while the step stays on the same weight snapshot
        wa.value = s.a; wb.value = s.b;
        const net = await model(s.run, s.step);
        ws.textContent = net.step;
        const f = net.forwardDetailed(s.a, s.b);
        const p = net.p, L = Array.from(f.logits).slice(0, p), prob = softmax(L);
        const order = L.map((v, c) => [v, c]).sort((x, y) => y[0] - x[0]).slice(0, 5);
        const lines = [
          `tokens t = (${s.a}, ${s.b}, ${p})  — "${s.a} ${s.b} ="`,
          `Step 1  x0_0 = W_E[:, ${s.a}] + W_pos[0] = ${fmt(f.resid0[0])}   (128 numbers)`,
          `        x0_1 = W_E[:, ${s.b}] + W_pos[1] = ${fmt(f.resid0[1])}`,
          `        x0_2 = W_E[:, =] + W_pos[2]   = ${fmt(f.resid0[2])}`,
          `Step 2  attention from '=' to (a, b, =), per head:`,
          ...f.patterns.map((rows, h) => `        head ${h}: ${rows[2].map((x) => x.toFixed(4)).join(", ")}`),
          `        x1 = x0_2 + Σ heads = ${fmt(f.residMid[2])}`,
          `Step 3  neuron pre-activations (first 6): ${fmt(f.pre, 6)}`,
          `        after ReLU: ${fmt(f.post, 6)}   — ${f.post.filter((v) => v > 0).length} of 512 neurons are active`,
          `        x2 = x1 + W_out·MLP + b_out = ${fmt(f.residPost)}`,
          `Step 4  top logits: ${order.map(([v, c]) => `c=${c}: ${v.toFixed(2)}`).join(", ")}`,
          `        softmax probability of the correct answer ${(s.a + s.b) % p}: ${prob[(s.a + s.b) % p].toFixed(6)}`,
          `        prediction = argmax = ${order[0][1]}   ${order[0][1] === (s.a + s.b) % p ? "✓ correct" : "✗ wrong"}`,
        ];
        out.textContent = lines.join("\n");
      }, { dedupe: frameKey({}, "weight_steps", "a", "b"), theme: false });
    },
  },
};
