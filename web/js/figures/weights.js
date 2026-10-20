// weights.js — dashboard group B: the weights themselves, mostly seen through the Fourier basis.
//
// Local explanation: bar charts of the Fourier-component norms of W_E (embedding) and W_L = W_U W_out
// (neuron→logit map) at the current step; a step×component heatmap of W_E's spectrum over all of training; the
// singular values of W_E; each head's attention lookup C_j and OV output in the Fourier basis; W_logit in the
// Fourier basis with neurons sorted by frequency cluster; and the positional embeddings.
// Global context: the paper's "first line of evidence" (periodicity / Fourier sparsity of the weights, Fig. 3,
// 10, 11) and ARENA's SVD and W_logit views, all scrubbable over training.
// Sources: paper Sec. 4.1–4.2, App. C.1.3 (https://arxiv.org/abs/2301.05217); ARENA 1.5.2 "Understanding the
// embedding" and "W_logit in Fourier basis" (https://learn.arena.education/chapter1_transformer_interp/32_grokking/2-circuit-and-feature-analysis/).
import { array, model, nearestIndex, row, run, manifest } from "../core/data.js";
import { Plot } from "../viz/plot.js";
import { Heatmap } from "../viz/heatmap.js";
import { cssVar } from "../viz/colors.js";
import { el, kstep, note, runOf, watch, set } from "./common.js";

const K = 56; // frequencies 1..56 for p = 113

/** Split a 113-long Fourier vector into Const, cos k and sin k series for grouped bars. */
function cosSin(v) {
  return { cos: Array.from({ length: K }, (_, i) => v[2 * i + 1]), sin: Array.from({ length: K }, (_, i) => v[2 * i + 2]), c: v[0] };
}

/** Grouped cos/sin bar chart of a [T, 113] Fourier-norm array at the current step. */
function spectrumBars(arrayName, ytitle) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 250 });
      const info = note(body);
      watch(["run", "step"], async (s) => {
        const id = runOf(ds);
        const r = await run(id), a = await array(id, arrayName);
        const t = nearestIndex(r.ckpt_steps, s.step);
        const { cos, sin, c } = cosSin(row(a, t));
        const ks = Array.from({ length: K }, (_, i) => i + 1);
        await plot.draw([
          { x: ks, y: cos, type: "bar", name: "cos k", marker: { color: "#2e86ab" } },
          { x: ks, y: sin, type: "bar", name: "sin k", marker: { color: "#d1495b" } },
        ], { barmode: "group", bargap: 0.15, xaxis: { title: { text: "frequency k" }, dtick: 4 }, yaxis: { title: { text: ytitle } }, hovermode: "closest" });
        info.textContent = `step ${r.ckpt_steps[t]} · constant component ${c.toFixed(3)} · key frequencies of this run: ${r.key_freqs.join(", ")}`;
      });
      plot.el.on?.("plotly_click", (ev) => set({ freq: ev.points[0].x }));
    },
  };
}

/** Step × frequency heatmap of a [T, 113] spectrum (combined √(cos²+sin²) per frequency), with a cursor row. */
function spectrumOverTraining(arrayName) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      let hm = null, shown = null;
      const info = note(body);
      watch(["run", "step"], async (s) => {
        const id = runOf(ds);
        const r = await run(id), a = await array(id, arrayName);
        const T = a.shape[0];
        if (id !== shown) {
          shown = id;
          body.querySelector(".heatmap-wrap")?.remove();
          hm = new Heatmap(body, { rows: K, cols: T, size: 520, diverging: false, format: (v) => v.toFixed(3),
            label: (rr, cc) => `k = ${K - rr}, step ${r.ckpt_steps[cc]}`, onClick: (rr, cc) => set({ step: r.ckpt_steps[cc], freq: K - rr }) });
          body.append(info);
          const grid = new Float32Array(K * T);
          for (let t = 0; t < T; t++) {
            const v = row(a, t);
            for (let k = 1; k <= K; k++) grid[(K - k) * T + t] = Math.hypot(v[2 * k - 1], v[2 * k]); // high k at top
          }
          hm.draw(grid);
        }
        info.textContent = `rows: frequency k (1 at bottom, 56 at top) · columns: checkpoints from step 0 to ${kstep(r.ckpt_steps[T - 1])} · current step ${s.step} (click to jump)`;
      });
    },
  };
}

export const figures = {
  "we-fourier-bars": spectrumBars("we_fourier", "norm of component in W_E"),
  "wl-fourier-bars": spectrumBars("wl_fourier", "norm of component in W_L"),
  "we-fourier-heatmap": spectrumOverTraining("we_fourier"),
  "wl-fourier-heatmap": spectrumOverTraining("wl_fourier"),

  "we-svals": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 240 });
      watch(["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id), a = await array(id, "we_svals");
        const t = nearestIndex(r.ckpt_steps, s.step);
        const sv = Array.from(row(a, t));
        await plot.draw([{ x: sv.map((_, i) => i + 1), y: sv, type: "scatter", mode: "lines+markers", marker: { size: 4, color: cssVar("--accent") }, line: { color: cssVar("--accent") }, name: `step ${r.ckpt_steps[t]}` }],
          { xaxis: { title: { text: "singular value index" } }, yaxis: { title: { text: "singular value of W_E" }, rangemode: "tozero" }, hovermode: "closest" });
      });
    },
  },

  "attn-C-fourier": headSpectra("attn_C_fourier", "coefficient of C_j", true),
  "ov-fourier": headSpectra("ov_fourier", "norm in W_O W_V W_E", false),

  "wlogit-sorted": {
    async render(body, ds) {
      body.innerHTML = "";
      const m = await manifest();
      let hm = null;
      const info = note(body);
      watch(["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id);
        const net = await model(id, s.step);
        const freqs = await array(id, "neuron_freq");
        const fve = await array(id, "neuron_fve");
        const t = nearestIndex(r.ckpt_steps, net.step);
        const nf = row(freqs, t), nv = row(fve, t);
        const p = net.p, n = net.n;
        const F = fourierBasis(p);
        const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => ((nv[x] > 0.85 ? nf[x] : 99) - (nv[y] > 0.85 ? nf[y] : 99)) || x - y);
        const grid = new Float32Array(n * p);
        order.forEach((neuron, rIdx) => {
          for (let i = 0; i < p; i++) { // component i of column `neuron` of W_L: Σ_c F[i][c] W_L[c][neuron]
            let s2 = 0;
            for (let c = 0; c < p; c++) s2 += F[i * p + c] * net.WL[c * n + neuron];
            grid[rIdx * p + i] = s2;
          }
        });
        if (!hm) hm = new Heatmap(body, { rows: n, cols: p, size: 420, diverging: true, format: (v) => v.toFixed(3),
          label: (rr, cc) => `neuron ${order[rr]} (k=${nv[order[rr]] > 0.85 ? nf[order[rr]] : "none"}), component ${m.basis_names[cc]}`,
          onClick: (rr) => set({ neuron: order[rr] }) });
        body.append(info);
        hm.draw(grid);
        info.textContent = `step ${net.step}: rows = 512 neurons sorted by their frequency cluster (unclustered last); columns = Fourier components of the output c (Const, cos 1, sin 1, …). Click a row to select that neuron.`;
      });
    },
  },

  "pos-embed": {
    async render(body, ds) {
      body.innerHTML = "";
      const hm = new Heatmap(body, { rows: 3, cols: 128, size: 520, diverging: true, rowLabel: "position", colLabel: "dimension", format: (v) => v.toFixed(3) });
      watch(["run", "step"], async (s) => {
        const net = await model(runOf(ds), s.step);
        hm.draw(net.Wpos.data);
      });
    },
  },
};

/** Per-head Fourier spectra ([T, 4, 113]) at the current step as small bar charts. */
function headSpectra(arrayName, ytitle, signed) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 280 });
      watch(["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id), a = await array(id, arrayName);
        const t = nearestIndex(r.ckpt_steps, s.step);
        const v = row(a, t), H = a.shape[1], P = a.shape[2];
        const traces = [];
        for (let h = 0; h < H; h++) {
          const comb = Array.from({ length: K }, (_, i) => (signed ? Math.hypot(v[h * P + 2 * i + 1], v[h * P + 2 * i + 2]) : Math.hypot(v[h * P + 2 * i + 1], v[h * P + 2 * i + 2])));
          traces.push({ x: comb.map((_, i) => i + 1), y: comb, type: "bar", name: `head ${h}` });
        }
        await plot.draw(traces, { barmode: "group", xaxis: { title: { text: "frequency k" }, dtick: 4 }, yaxis: { title: { text: `${ytitle} (√(cos²+sin²))` } }, hovermode: "closest" });
      });
    },
  };
}

const basisCache = new Map();
/** Real orthonormal Fourier basis (paper order) as a flat p×p Float32Array, cached. */
export function fourierBasis(p) {
  if (basisCache.has(p)) return basisCache.get(p);
  const F = new Float32Array(p * p);
  for (let x = 0; x < p; x++) F[x] = 1 / Math.sqrt(p); // constant row
  for (let k = 1; k <= (p - 1) / 2; k++) {
    for (let x = 0; x < p; x++) {
      F[(2 * k - 1) * p + x] = Math.cos((2 * Math.PI * k * x) / p) * Math.sqrt(2 / p);
      F[2 * k * p + x] = Math.sin((2 * Math.PI * k * x) / p) * Math.sqrt(2 / p);
    }
  }
  basisCache.set(p, F);
  return F;
}
