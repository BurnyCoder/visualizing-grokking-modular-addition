// curves.js — dashboard group A: metrics over the whole training run, with phase bands and a step cursor.
//
// Local explanation: each figure loads per-step or per-checkpoint arrays exported by src/grokviz/export.py and
// draws them once per run; moving the global step only moves the dotted cursor (cheap). Clicking a curve sets
// the global step, so every other panel jumps to that moment of training.
// Global context: these are the "as it is trained" curves: grokking itself (loss/accuracy), the paper's progress
// measures (restricted/excluded loss, Gini, sum of squared weights — Fig. 7) and ARENA's trig fractions.
// Sources: paper Sec. 5 and Fig. 2, 6, 7, 14, 15 (https://arxiv.org/abs/2301.05217); ARENA 1.5.2 section 3
// (https://learn.arena.education/chapter1_transformer_interp/32_grokking/3-analysis-during-training/).
import { array, column, run } from "../core/data.js";
import { Plot, phaseShapes } from "../viz/plot.js";
import { cssVar, freqColor } from "../viz/colors.js";
import { el, follow, frameKey, note, runOf, storeOf, watch } from "./common.js";
import { stepHeatmap } from "./weights.js";

const stepsEvery = (n, every, step0 = 0) => Array.from({ length: n }, (_, i) => step0 + i * every); // x values for per-10-step curves

/**
 * Generic "metric over training" figure: `traces(runId, r)` returns Plotly traces; redraws when the run
 * changes; moves the cursor when the step changes.
 */
function overTraining(traces, layout = {}, height = 270) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height, stepAxis: true });
      let shown = null;
      watch(body, ["run", "step"], async (s, changed) => {
        const id = runOf(ds);
        if (id !== shown || changed.includes("theme")) {
          shown = id;
          const r = await run(id);
          const t = await traces(id, r);
          const xmax = r.ckpt_steps[r.ckpt_steps.length - 1];
          await plot.draw(t, { ...layout, shapes: phaseShapes(r.phases, xmax), xaxis: { title: { text: "training step" }, ...(layout.xaxis || {}) } });
        }
        plot.cursor(s.step);
      });
    },
  };
}

const line = (x, y, name, color, extra = {}) => ({ x: Array.from(x), y: Array.from(y), name, type: "scatter", mode: "lines", line: { color, width: 1.8 }, ...extra });

/** Train and test loss traces at every 10th step (also used by the dashboard hero, figures/hero.js). */
export async function lossTraces(id) {
  const tr = await array(id, "curve_train_loss"), te = await array(id, "curve_test_loss");
  return [line(stepsEvery(tr.data.length, 10), tr.data, "train loss", cssVar("--train")),
    line(stepsEvery(te.data.length, 10), te.data, "test loss", cssVar("--test"))];
}

/** The two step × frequency views of alpha-growth: each frequency's cos(w_k(a+b−c)) coefficient in the logits, and the loss without it. */
const VIEWS = {
  alpha: { array: "cos_coeffs", perRow: (v) => Array.from(v), format: (v) => v.toFixed(1), limits: { min: 0 }, label: "coefficient of cos(w_k(a+b−c))", log: false,
    what: "Coefficient of the unit-normalised cos(w_k(a+b−c)) in the logits at every checkpoint (colour from 0; the few small negative values show as 0)." },
  excluded: { array: "excl_per_freq", perRow: (v) => Array.from(v, (x) => Math.log10(x)), format: (v) => `10^${v.toFixed(1)}`, label: "training loss without k", log: true,
    what: "Training loss after removing frequency k from the logits (colour: log10 of the loss) at every checkpoint." },
};

export const figures = {
  "alpha-growth": {
    async render(body, ds) {
      body.innerHTML = "";
      const sel = el(body, "label", "hero-ctl", 'show <select><option value="alpha">how strongly cos(w_k(a+b−c)) appears in the logits</option><option value="excluded">the training loss without frequency k</option></select>').querySelector("select");
      const heats = {}; // one heatmap per view, the other hidden (and therefore not redrawn: figures/common.js gate)
      for (const [k, v] of Object.entries(VIEWS)) { heats[k] = el(body, "div"); heats[k].hidden = k !== sel.value; stepHeatmap(v.array, v).render(heats[k], ds); }
      const plot = new Plot(body, { height: 210, stepAxis: true }); // the selected frequency, drawn up to the current step
      const info = note(body);
      const draw = async (s) => {
        const id = runOf(ds), r = await run(id), v = VIEWS[sel.value], a = await array(id, v.array), K = a.shape[1];
        const y = Array.from({ length: a.shape[0] }, (_, t) => a.data[t * K + s.freq - 1]);
        const upto = r.ckpt_steps.findIndex((x) => x > s.step), n = upto < 0 ? y.length : upto;
        await plot.draw([line(r.ckpt_steps, y, `k = ${s.freq}, whole run`, cssVar("--ink-soft"), { opacity: 0.35 }),
          line(r.ckpt_steps.slice(0, n), y.slice(0, n), `k = ${s.freq} up to the current step`, freqColor(s.freq), { line: { color: freqColor(s.freq), width: 2.6 } })],
        { yaxis: { title: { text: v.label }, ...(v.log ? { type: "log", exponentformat: "power" } : { rangemode: "tozero" }) }, xaxis: { title: { text: "training step" } },
          shapes: phaseShapes(r.phases, r.ckpt_steps.at(-1)) });
        plot.cursor(s.stepf ?? s.step);
        info.textContent = `Line: frequency k = ${s.freq} (${r.key_freqs.includes(s.freq) ? "a key frequency of this run" : "not a key frequency of this run"}); pick another by clicking a heatmap row or in the control bar.`;
      };
      sel.onchange = () => { for (const [k, d] of Object.entries(heats)) d.hidden = k !== sel.value; draw(storeOf(body).state); };
      watch(body, ["run", "step", "freq"], draw, { dedupe: async (s) => `${await frameKey(ds, "ckpt_steps", "freq")(s)}|${sel.value}` });
      follow(body, ["stepf"], (s) => plot.cursor(s.stepf ?? s.step));
    },
  },

  "loss-curves": overTraining(lossTraces, { yaxis: { type: "log", title: { text: "loss (log scale)" }, exponentformat: "power" } }),
  "loss-curves-linear": overTraining(lossTraces, { yaxis: { title: { text: "loss (linear scale)" }, rangemode: "tozero" } }),

  "acc-curves": overTraining(async (id, r) => [
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "acc_train"), "train accuracy", cssVar("--train")),
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "acc_test"), "test accuracy", cssVar("--test")),
  ], { yaxis: { title: { text: "accuracy" }, range: [0, 1.02] } }),

  "progress-measures": overTraining(async (id, r) => [
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "loss_train"), "train loss", cssVar("--train")),
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "loss_test"), "test loss", cssVar("--test")),
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "restricted_loss"), "restricted loss", cssVar("--restricted"), { line: { color: cssVar("--restricted"), width: 2.4 } }),
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "excluded_loss"), "excluded loss", cssVar("--excluded"), { line: { color: cssVar("--excluded"), width: 2.4 } }),
  ], { yaxis: { type: "log", title: { text: "loss (log scale)" }, exponentformat: "power" } }, 300),

  "excluded-per-freq": overTraining(async (id, r) => {
    const ex = await array(id, "excl_per_freq");
    const T = ex.shape[0], K = ex.shape[1];
    const traces = r.key_freqs.map((k) => line(r.ckpt_steps, Array.from({ length: T }, (_, t) => ex.data[t * K + k - 1]), `exclude k=${k}`, freqColor(k)));
    traces.push(line(r.ckpt_steps, await column(id, "ckpt_scalars", "loss_train"), "train loss", cssVar("--train"), { line: { color: cssVar("--train"), dash: "dot" } }));
    traces.push(line(r.ckpt_steps, await column(id, "ckpt_scalars", "loss_test"), "test loss", cssVar("--test"), { line: { color: cssVar("--test"), dash: "dot" } }));
    return traces;
  }, { yaxis: { type: "log", title: { text: "loss (log scale)" }, exponentformat: "power" } }, 300),

  "gini-curves": overTraining(async (id, r) => [
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "gini_we"), "Gini of W_E Fourier norms", "#d1495b"),
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "gini_wl"), "Gini of W_L Fourier norms", "#2e86ab"),
  ], { yaxis: { title: { text: "Gini (0 = spread, 1 = sparse)" }, range: [0, 1] } }),

  "ssw-curves": overTraining(async (id, r) => {
    const ssw = await array(id, "ssw");
    const T = ssw.shape[0], P = ssw.shape[1];
    const total = line(r.ckpt_steps, await column(id, "ckpt_scalars", "ssw_total"), "total", cssVar("--ink"), { line: { color: cssVar("--ink"), width: 2.6 } });
    const parts = ssw.entry.columns.map((name, j) => line(r.ckpt_steps, Array.from({ length: T }, (_, t) => ssw.data[t * P + j]), name.replace("blocks.0.", ""), undefined, { line: { width: 1.2 }, visible: "legendonly" }));
    return [total, ...parts];
  }, { yaxis: { title: { text: "sum of squared weights" } } }),

  "trig-loss-ratio": overTraining(async (id, r) => {
    const te = await column(id, "ckpt_scalars", "loss_test");
    const rl = await column(id, "ckpt_scalars", "restricted_loss");
    return [line(r.ckpt_steps, te.map((v, i) => v / rl[i]), "test loss ÷ restricted loss", cssVar("--accent"))];
  }, { yaxis: { type: "log", title: { text: "ratio" }, exponentformat: "power" } }),

  "trig-fraction": overTraining(async (id, r) => [
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "trig_frac_pre"), "neuron pre-activations", "#edae49"),
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "trig_frac_post"), "neuron activations", "#00798c"),
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "trig_frac_logit"), "logits", "#d1495b"),
  ], { yaxis: { title: { text: "variance in cos/sin(w_k(a+b)) directions" }, range: [0, 1] } }),

  "cos-coeffs": overTraining(async (id, r) => {
    const cc = await array(id, "cos_coeffs");
    const T = cc.shape[0], K = cc.shape[1];
    return r.key_freqs.map((k) => line(r.ckpt_steps, Array.from({ length: T }, (_, t) => cc.data[t * K + k - 1]), `k=${k}`, freqColor(k)));
  }, { yaxis: { title: { text: "coefficient of cos(w_k(a+b−c))" } } }),

  "restricted-split-curves": overTraining(async (id, r) => [
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "loss_all"), "loss on all pairs", cssVar("--ink")),
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "restricted_loss_train"), "restricted loss, training pairs (wide)", cssVar("--train"), { line: { color: cssVar("--train"), width: 5 }, opacity: 0.5 }), // wide underneath: the test curve lies on top of it
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "restricted_loss_test"), "restricted loss, test pairs", cssVar("--test")),
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "excluded_loss_max"), "excluded loss, worst single frequency", cssVar("--excluded"), { line: { color: cssVar("--excluded"), dash: "dot", width: 2 } }),
  ], { yaxis: { type: "log", title: { text: "loss (log scale)" }, exponentformat: "power" } }, 300),

  "fve-curves": overTraining(async (id, r) => [
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "logit_fve"), "logits explained by Σ α_k cos(w_k(a+b−c))", "#d1495b"),
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "frac_neurons_085"), "fraction of neurons with one dominant frequency", "#00798c"),
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "restricted_acc_test"), "test accuracy using only the key frequencies", "#edae49"),
    line(r.ckpt_steps, await column(id, "ckpt_scalars", "wl_resid_sq_frac"), "share of W_L outside the key frequencies", "#8d6a9f", { line: { color: "#8d6a9f", dash: "dot" } }),
  ], { yaxis: { title: { text: "fraction" }, range: [0, 1] } }),
};
