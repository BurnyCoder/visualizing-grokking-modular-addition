// gradients.js — dashboard group F: what the training signal looks like at every checkpoint.
//
// Local explanation: exact full-batch gradients recomputed at each checkpoint (src/grokviz/gradients.py):
// per-parameter gradient norms, the angle between the gradient and the weights (weight decay pulls along −θ),
// the angle between the training-loss and test-loss gradients, the gradient-to-decay ratio, and which Fourier
// frequencies the embedding gradient is pushing (step × frequency heatmap).
// Global context: links the progress measures to their cause — the paper attributes circuit formation and
// cleanup to weight decay (Sec. 5.2–5.3); these plots show the forces directly.
// Sources: paper Sec. 5.2–5.3 (https://arxiv.org/abs/2301.05217); AdamW (https://arxiv.org/abs/1711.05101).
import { array, column, run, row } from "../core/data.js";
import { Plot, phaseShapes } from "../viz/plot.js";
import { Heatmap } from "../viz/heatmap.js";
import { cssVar } from "../viz/colors.js";
import { note, runOf, set, watch } from "./common.js";

/** Metric-over-training plot for gradient arrays (same pattern as curves.js). */
function overTraining(traces, layout) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 260, stepAxis: true });
      const info = note(body);
      let shown = null;
      watch(["run", "step"], async (s, changed) => {
        const id = runOf(ds), r = await run(id);
        // hide (not destroy) the plot when a run has no gradients, so switching back to one that has them redraws it
        plot.el.hidden = !r.arrays.grad_scalars;
        info.textContent = r.arrays.grad_scalars ? "" : "No gradient data for this run.";
        if (!r.arrays.grad_scalars) { shown = null; return; }
        if (shown !== id || changed.includes("theme")) {
          shown = id;
          await plot.draw(await traces(id, r), { ...layout, shapes: phaseShapes(r.phases, r.ckpt_steps.at(-1)), xaxis: { title: { text: "training step" } } });
        }
        plot.cursor(s.step);
      });
    },
  };
}

const ln = (x, y, name, color, extra = {}) => ({ x: Array.from(x), y: Array.from(y), name, type: "scatter", mode: "lines", line: { color, width: 1.8 }, ...extra });

export const figures = {
  "grad-norms": overTraining(async (id, r) => {
    const g = await array(id, "grad_norm");
    const T = g.shape[0], P = g.shape[1];
    return g.entry.columns.map((name, j) => ln(r.ckpt_steps, Array.from({ length: T }, (_, t) => g.data[t * P + j]), name.replace("blocks.0.", ""), undefined, { line: { width: 1.3 } }));
  }, { yaxis: { type: "log", title: { text: "‖gradient of train loss‖" }, exponentformat: "power" } }),

  "grad-alignment": overTraining(async (id, r) => [
    ln(r.ckpt_steps, await column(id, "grad_scalars", "cos_grad_weight"), "cos(gradient, weights)", cssVar("--accent")),
    ln(r.ckpt_steps, await column(id, "grad_scalars", "cos_train_test"), "cos(train-loss gradient, test-loss gradient)", cssVar("--restricted")),
  ], { yaxis: { title: { text: "cosine similarity" }, range: [-1.05, 1.05] } }),

  "grad-vs-decay": overTraining(async (id, r) => [
    ln(r.ckpt_steps, await column(id, "grad_scalars", "grad_over_decay"), "‖∇L‖ ÷ (λ‖θ‖)", cssVar("--excluded")),
  ], { yaxis: { type: "log", title: { text: "loss pull ÷ weight-decay pull" }, exponentformat: "power" } }),

  "grad-we-fourier-heatmap": {
    async render(body, ds) {
      body.innerHTML = "";
      let hm = null, shown = null;
      const info = note(body);
      watch(["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id);
        if (!r.arrays.grad_we_fourier) return;
        if (shown !== id) {
          shown = id;
          const a = await array(id, "grad_we_fourier");
          const T = a.shape[0], K = 56, grid = new Float32Array(K * T);
          for (let t = 0; t < T; t++) {
            const v = row(a, t);
            let tot = 0;
            for (let k = 1; k <= K; k++) tot += v[2 * k - 1] ** 2 + v[2 * k] ** 2;
            for (let k = 1; k <= K; k++) grid[(K - k) * T + t] = (v[2 * k - 1] ** 2 + v[2 * k] ** 2) / (tot || 1); // share of gradient energy
          }
          body.querySelector(".heatmap-wrap")?.remove();
          hm = new Heatmap(body, { rows: K, cols: T, size: 640, height: 240, diverging: false, format: (v) => `${(100 * v).toFixed(1)} %`,
            label: (rr, cc) => `k = ${K - rr}, step ${r.ckpt_steps[cc]}`, onClick: (rr, cc) => set({ step: r.ckpt_steps[cc], freq: K - rr }) });
          body.append(info);
          hm.draw(grid);
        }
        info.textContent = `Share of the embedding gradient's energy in each frequency (rows k = 1 bottom … 56 top) at every checkpoint (columns). Current step ${s.step}.`;
      });
    },
  },
};
