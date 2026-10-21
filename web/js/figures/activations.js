// activations.js — dashboard group C: what the network computes on every input (a, b), at the current step.
//
// Local explanation: attention and neuron grids are computed in the browser by GrokModel from the checkpoint's
// weights (113×113 = 12,769 inputs); the 2D Fourier transform of a neuron grid shows which waves it contains;
// per-neuron frequency / fraction-of-variance / quadratic share come from the exported per-checkpoint arrays.
// Global context: the paper's periodicity evidence (Fig. 4, 9, 12), single-frequency neurons and clusters
// (Sec. 4.3, Fig. 5) and ARENA's neuron-cluster scatter plots, at any moment of training.
// Sources: paper Sec. 4.1, 4.3 (https://arxiv.org/abs/2301.05217); ARENA 1.5.2 "Neurons cluster by frequency"
// (https://learn.arena.education/chapter1_transformer_interp/32_grokking/2-circuit-and-feature-analysis/).
import { array, model, nearestIndex, row, run } from "../core/data.js";
import { Heatmap, miniHeatmap } from "../viz/heatmap.js";
import { Plot } from "../viz/plot.js";
import { freqColor } from "../viz/colors.js";
import { fourierBasis } from "./weights.js";
import { basisName, clusterOrder, el, note, runOf, set, state, watch } from "./common.js";

/** 2D Fourier transform of a p×p grid (rows a, cols b): C = F G Fᵀ (orthonormal basis). */
export function fft2(grid, p) {
  const F = fourierBasis(p);
  const tmp = new Float32Array(p * p), out = new Float32Array(p * p);
  for (let i = 0; i < p; i++) for (let b = 0; b < p; b++) { // tmp = F · G (transform along a)
    let s = 0;
    for (let a = 0; a < p; a++) s += F[i * p + a] * grid[a * p + b];
    tmp[i * p + b] = s;
  }
  for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) { // out = tmp · Fᵀ (transform along b)
    let s = 0;
    for (let b = 0; b < p; b++) s += tmp[i * p + b] * F[j * p + b];
    out[i * p + j] = s;
  }
  return out;
}

const centre = (g) => { const m = g.reduce((x, y) => x + y, 0) / g.length; return g.map((v) => v - m); }; // remove the constant

/**
 * A p×p 2D-Fourier map from a precomputed array [S, p, p] (log10 of each component's size relative to the frame's
 * largest; src/grokviz/geometry.py `spectra`), shown at the spectra frame nearest the selected step. Rows: component in
 * a, columns: component in b (Const, cos 1, sin 1, …). Runs other than the two showcase runs have only the final frame.
 */
export function spectrumMap(arrayName, what) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      const hm = new Heatmap(body, { rows: 113, cols: 113, size: 300, diverging: false, format: (v) => `10^${v.toFixed(1)} of max`,
        label: (r, c) => `${basisName(r)} (in a) × ${basisName(c)} (in b)` });
      const info = note(body);
      watch(["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id);
        if (!r.spectra_steps) { info.textContent = "no 2D Fourier maps for this run"; return; }
        const f = nearestIndex(r.spectra_steps, s.step);
        hm.draw(row(await array(id, arrayName), f), { min: -4, max: 0 });
        const only = r.spectra_steps.length === 1 ? " (this run has this map only at its final checkpoint)" : "";
        info.textContent = `step ${r.spectra_steps[f]}${only}: ${what}. Colour: log10 of each component's size relative to the largest (darkest = 10⁻⁴ or less).`;
      });
    },
  };
}

export const figures = {
  "neuron-b2-map": spectrumMap("spec_neuron_b2", "mean over all 512 neurons of the squared 2D Fourier coefficients of their activations"),
  "neuron-freq-map": {
    async render(body, ds) {
      body.innerHTML = "";
      let hm = null;
      const info = note(body);
      watch(["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id);
        if (!r.spectra_steps) { info.textContent = "no neuron–frequency map for this run"; return; }
        const f = nearestIndex(r.spectra_steps, s.step);
        const a = await array(id, "spec_neuron_freq"), v = row(a, f), [n, K] = a.shape.slice(1);
        const best = Array.from({ length: n }, (_, i) => { let k = 0; for (let j = 1; j < K; j++) if (v[i * K + j] > v[i * K + k]) k = j; return k + 1; });
        const fve = Array.from({ length: n }, (_, i) => v[i * K + best[i] - 1]);
        const order = clusterOrder(best, fve, n), grid = new Float32Array(n * K);
        order.forEach((neuron, rIdx) => { for (let j = 0; j < K; j++) grid[rIdx * K + j] = v[neuron * K + j]; });
        if (!hm) hm = new Heatmap(body, { rows: n, cols: K, size: 420, height: 420, diverging: false, format: (x) => `${(100 * x).toFixed(1)} %`,
          label: (rr, cc) => `neuron ${order[rr]}, frequency k = ${cc + 1}`, onClick: (rr, cc) => set({ neuron: order[rr], freq: cc + 1 }) });
        body.append(info);
        hm.draw(grid, { min: 0, max: 1 });
        const only = r.spectra_steps.length === 1 ? " (final checkpoint only for this run)" : "";
        info.textContent = `step ${r.spectra_steps[f]}${only}: share of each neuron's variance (rows, sorted by cluster) explained by each frequency's 2D Fourier terms (columns k = 1…56). A bright single cell per row = a single-frequency neuron. Click to select a neuron and frequency.`;
      });
    },
  },

  "attn-grid": {
    async render(body, ds) {
      body.innerHTML = "";
      const hm = new Heatmap(body, { rows: 113, cols: 113, size: 300, diverging: false, format: (v) => v.toFixed(3), onClick: (a, b) => set({ a, b }) });
      const info = note(body);
      watch(["run", "step", "head"], async (s) => {
        const net = await model(runOf(ds), s.step);
        hm.draw(net.attentionGrid(s.head, 0), { min: 0, max: 1 });
        info.textContent = `head ${s.head}, step ${net.step}: attention from '=' to the first number a, for every input (a = row, b = column). Click a cell to select that input.`;
      });
    },
  },

  "attn-grids-all": {
    async render(body, ds) {
      body.innerHTML = "";
      const grid = el(body, "div", "small-multiples");
      const info = note(body);
      watch(["run", "step"], async (s) => {
        const net = await model(runOf(ds), s.step);
        grid.replaceChildren();
        for (let h = 0; h < net.H; h++) {
          const f = el(grid, "figure");
          f.append(miniHeatmap(net.attentionGrid(h, 0), 113, 113, false));
          el(f, "figcaption", "", `head ${h}`);
          f.style.cursor = "pointer";
          f.onclick = () => set({ head: h });
        }
        info.textContent = `step ${net.step}: attention from '=' to a for all four heads (rows a, columns b, yellow = 1, purple = 0).`;
      });
    },
  },

  "avg-attn": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 230 });
      watch(["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id), a = await array(id, "avg_attn");
        const t = nearestIndex(r.ckpt_steps, s.step), v = row(a, t);
        const heads = [0, 1, 2, 3];
        await plot.draw(["a (position 0)", "b (position 1)", "= (position 2)"].map((name, j) => ({ x: heads.map((h) => `head ${h}`), y: heads.map((h) => v[h * 3 + j]), name, type: "bar" })),
          { barmode: "stack", yaxis: { title: { text: "mean attention from '='" }, range: [0, 1] }, hovermode: "closest" });
      });
    },
  },

  "neuron-grid": {
    async render(body, ds) {
      body.innerHTML = "";
      const row1 = el(body, "div", "heatmap-wrap");
      const left = el(row1, "div"), right = el(row1, "div");
      el(left, "p", "fig-note", "activation over all inputs (rows a, columns b)");
      el(right, "p", "fig-note", "same grid in the 2D Fourier basis (rows: component in a, columns: component in b)");
      const hm = new Heatmap(left, { rows: 113, cols: 113, size: 300, diverging: false, format: (v) => v.toFixed(3), onClick: (a, b) => set({ a, b }) });
      const hf = new Heatmap(right, { rows: 113, cols: 113, size: 300, diverging: true, format: (v) => v.toFixed(3),
        label: (i, j) => `${basisLabel(i)} (in a) × ${basisLabel(j)} (in b)`, onClick: (i) => { if (i > 0) set({ freq: Math.ceil(i / 2) }); } });
      const info = note(body);
      watch(["run", "step", "neuron"], async (s) => {
        const id = runOf(ds), r = await run(id), net = await model(id, s.step);
        const g = net.neuronGrid(s.neuron, true);
        hm.draw(g);
        hf.draw(fft2(centre(g), net.p));
        const t = nearestIndex(r.ckpt_steps, net.step);
        const f = row(await array(id, "neuron_freq"), t)[s.neuron], v = row(await array(id, "neuron_fve"), t)[s.neuron];
        info.textContent = `neuron ${s.neuron}, step ${net.step}: best frequency k = ${f}, which explains ${(100 * v).toFixed(1)} % of its variance (constant removed before the transform).`;
      });
    },
  },

  "neuron-small-multiples": {
    async render(body, ds) {
      body.innerHTML = "";
      const grid = el(body, "div", "small-multiples");
      const info = note(body);
      watch(["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id), net = await model(id, s.step);
        const t = nearestIndex(r.ckpt_steps, net.step);
        const freqs = row(await array(id, "neuron_freq"), t), fve = row(await array(id, "neuron_fve"), t);
        const picks = [];
        for (const k of r.key_freqs) { // 3 example neurons per key frequency cluster of the final model
          let n = 0;
          for (let i = 0; i < freqs.length && n < 3; i++) if (freqs[i] === k && fve[i] > 0.85) { picks.push(i); n++; }
        }
        grid.replaceChildren();
        for (const i of picks.length ? picks : [0, 1, 2, 3, 4, 5, 6, 7]) {
          const f = el(grid, "figure");
          f.append(miniHeatmap(net.neuronGrid(i, true), 113, 113, false));
          el(f, "figcaption", "", `neuron ${i} · k=${freqs[i]} · ${(100 * fve[i]).toFixed(0)} %`);
          f.style.cursor = "pointer";
          f.onclick = () => set({ neuron: i });
        }
        info.textContent = `step ${net.step}: example neurons from each frequency cluster (chosen at this step). Click one to inspect it.`;
      });
    },
  },

  "neuron-freq-scatter": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 300 });
      watch(["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id);
        const t = nearestIndex(r.ckpt_steps, s.step);
        const f = Array.from(row(await array(id, "neuron_freq"), t)), v = Array.from(row(await array(id, "neuron_fve"), t));
        const act = Array.from(row(await array(id, "frac_active"), t));
        await plot.draw([{ x: f, y: v, mode: "markers", type: "scatter", text: f.map((_, i) => `neuron ${i}`),
          marker: { size: 6, color: act, colorscale: "Viridis", cmin: 0, cmax: 1, colorbar: { title: { text: "fraction of inputs<br>where active", side: "right" }, thickness: 10 } } }],
        { xaxis: { title: { text: "best frequency k" }, range: [0, 57] }, yaxis: { title: { text: "fraction of variance explained" }, range: [0, 1.02] }, hovermode: "closest",
          shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.85, y1: 0.85, line: { dash: "dot", width: 1 } }] });
        plot.el.on?.("plotly_click", (ev) => set({ neuron: ev.points[0].pointIndex }));
      });
    },
  },

  "quad-scatter": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 300 });
      watch(["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id);
        const t = nearestIndex(r.ckpt_steps, s.step);
        const pre = Array.from(row(await array(id, "neuron_quad_pre"), t)), post = Array.from(row(await array(id, "neuron_quad_post"), t));
        const f = Array.from(row(await array(id, "neuron_freq"), t));
        await plot.draw([{ x: pre, y: post, mode: "markers", type: "scatter", text: f.map((k, i) => `neuron ${i}, k=${k}`), marker: { size: 6, color: f.map((k) => freqColor(k)) } }],
          { xaxis: { title: { text: "quadratic share before ReLU" }, range: [0, 1] }, yaxis: { title: { text: "quadratic share after ReLU" }, range: [0, 1] }, hovermode: "closest",
            shapes: [{ type: "line", x0: 0, y0: 0, x1: 1, y1: 1, line: { dash: "dot", width: 1 } }] });
      });
    },
  },

  "cluster-sizes": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 260, stepAxis: true });
      let shown = null;
      watch(["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id);
        if (shown !== id) {
          shown = id;
          const f = await array(id, "neuron_freq"), v = await array(id, "neuron_fve");
          const T = f.shape[0], n = f.shape[1];
          const traces = r.key_freqs.map((k) => ({ x: r.ckpt_steps, y: Array.from({ length: T }, (_, t) => { let c = 0; for (let i = 0; i < n; i++) if (f.data[t * n + i] === k && v.data[t * n + i] > 0.85) c++; return c; }),
            name: `k=${k}`, type: "scatter", mode: "lines", stackgroup: "one", line: { color: freqColor(k), width: 0.5 } }));
          await plot.draw(traces, { yaxis: { title: { text: "neurons with FVE > 0.85" } }, xaxis: { title: { text: "training step" } } });
        }
        plot.cursor(s.step);
      });
    },
  },
};

/** Basis index → name (0 Const, 2k−1 cos k, 2k sin k). */
export const basisLabel = (i) => (i === 0 ? "Const" : `${i % 2 ? "cos" : "sin"} ${Math.ceil(i / 2)}`);
void state;
