// gradients.js — dashboard group F: what the training signal looks like at every checkpoint.
//
// Local explanation: exact full-batch gradients recomputed at each checkpoint (src/grokviz/gradients.py):
// per-parameter gradient norms, the angle between the gradient and the weights (weight decay pulls along −θ),
// the angle between the training-loss and test-loss gradients, and which Fourier frequencies the embedding gradient
// is pushing (step × frequency heatmap); for our runs also Adam's actual step direction (from the saved moments)
// compared with the gradient, the weights and the weight-decay step, and each parameter's logged update size.
// Global context: links the progress measures to their cause — the paper attributes circuit formation and
// cleanup to weight decay (Sec. 5.2–5.3); these plots show the forces directly.
// Sources: paper Sec. 5.2–5.3 (https://arxiv.org/abs/2301.05217); AdamW (https://arxiv.org/abs/1711.05101).
import { array, column, run } from "../core/data.js";
import { Plot, phaseShapes } from "../viz/plot.js";
import { cssVar } from "../viz/colors.js";
import { stepByFreq } from "./weights.js";
import { note, runOf, watch } from "./common.js";

/** Metric-over-training plot for gradient arrays (same pattern as curves.js); `needs` = the array it plots.
 * Charts with one line per parameter (11 legend entries) pass a taller `height` so the plot keeps its room. */
function overTraining(traces, layout, needs = "grad_scalars", missing = "No gradient data for this run.", height = 260) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height, stepAxis: true });
      const info = note(body);
      let shown = null;
      watch(body, ["run", "step"], async (s, changed) => {
        const id = runOf(ds), r = await run(id);
        // hide (not destroy) the plot when a run has no gradients, so switching back to one that has them redraws it
        plot.el.hidden = !r.arrays[needs];
        info.textContent = r.arrays[needs] ? "" : missing;
        if (!r.arrays[needs]) { shown = null; return; }
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
  }, { yaxis: { type: "log", title: { text: "‖gradient of train loss‖" }, exponentformat: "power" } }, undefined, undefined, 400),

  "grad-alignment": overTraining(async (id, r) => [
    ln(r.ckpt_steps, await column(id, "grad_scalars", "cos_grad_weight"), "cos(gradient, weights)", cssVar("--accent")),
    ln(r.ckpt_steps, await column(id, "grad_scalars", "cos_train_test"), "cos(train-loss gradient, test-loss gradient)", cssVar("--restricted")),
  ], { yaxis: { title: { text: "cosine similarity" }, range: [-1.05, 1.05] } }),

  "adam-alignment": overTraining(async (id) => {
    const x = (await array(id, "adam_scalars")).entry.steps, mk = { mode: "lines+markers", marker: { size: 4 } };
    return [ln(x, await column(id, "adam_scalars", "cos_adam_grad"), "cos(Adam direction, gradient)", cssVar("--accent"), mk),
      ln(x, await column(id, "adam_scalars", "cos_adam_weight"), "cos(Adam direction, weights)", cssVar("--restricted"), mk)];
  }, { yaxis: { title: { text: "cosine similarity" }, range: [-1.05, 1.05] } }, "adam_scalars", "Adam's internal state was not saved for this run (only for our retrained runs)."),

  "adam-vs-decay": overTraining(async (id) => {
    const x = (await array(id, "adam_scalars")).entry.steps;
    return [ln(x, await column(id, "adam_scalars", "adam_over_decay"), "loss step ÷ weight-decay step", cssVar("--excluded"), { mode: "lines+markers", marker: { size: 4 } })];
  }, { yaxis: { type: "log", title: { text: "|Adam direction| ÷ (λ|θ|)" }, exponentformat: "power" } }, "adam_scalars",
  "Adam's internal state was not saved for this run (only for our retrained runs); without weight decay the ratio is undefined."),

  "update-norms": overTraining(async (id) => {
    const u = await array(id, "update_norm"), T = u.shape[0], P = u.shape[1], every = u.entry.step_every;
    const x = Array.from({ length: T }, (_, t) => t * every);
    return u.entry.columns.map((name, j) => ln(x, Array.from({ length: T }, (_, t) => u.data[t * P + j]), name.replace("blocks.0.", ""), undefined, { line: { width: 1.3 } }));
  }, { yaxis: { type: "log", title: { text: "‖update‖ (log)" }, exponentformat: "power" } }, "update_norm",
  "Update sizes were logged only for our retrained runs.", 400),

  "grad-we-fourier-heatmap": stepByFreq("grad_we_fourier", {
    perFreq: (v) => { // share of the gradient's energy (cos² + sin², constant excluded) in each frequency
      const e = Array.from({ length: 56 }, (_, i) => v[2 * i + 1] ** 2 + v[2 * i + 2] ** 2), tot = e.reduce((x, y) => x + y, 0) || 1;
      return e.map((x) => x / tot);
    },
    format: (v) => `${(100 * v).toFixed(1)} %`, what: "Share of the embedding gradient's energy in each frequency at every checkpoint.",
  }),
};
