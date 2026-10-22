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
import { clusterOrder, follow, frameKey, kstep, note, runOf, set, storeOf, watch } from "./common.js";

const K = 56; // frequencies 1..56 for p = 113

/** Split a 113-long Fourier vector into Const, cos k and sin k series for grouped bars. */
function cosSin(v) {
  return { cos: Array.from({ length: K }, (_, i) => v[2 * i + 1]), sin: Array.from({ length: K }, (_, i) => v[2 * i + 2]), c: v[0] };
}

/** Grouped cos/sin bar chart of a [T, 113] Fourier-norm array at the current step; the selected frequency is shaded. */
function spectrumBars(arrayName, ytitle) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 250 }).on("plotly_click", (ev) => set({ freq: ev.points[0].x })); // click a bar → select k
      const info = note(body);
      watch(body, ["run", "step", "freq"], async (s) => {
        const id = runOf(ds);
        const r = await run(id), a = await array(id, arrayName);
        const t = nearestIndex(r.ckpt_steps, s.step);
        const { cos, sin, c } = cosSin(row(a, t));
        const ks = Array.from({ length: K }, (_, i) => i + 1);
        await plot.draw([
          { x: ks, y: cos, type: "bar", name: "cos k", marker: { color: "#2e86ab" } },
          { x: ks, y: sin, type: "bar", name: "sin k", marker: { color: "#d1495b" } },
        ], { barmode: "group", bargap: 0.15, xaxis: { title: { text: "frequency k" }, dtick: 4 }, yaxis: { title: { text: ytitle } }, hovermode: "closest",
          shapes: [{ type: "rect", xref: "x", yref: "paper", x0: s.freq - 0.5, x1: s.freq + 0.5, y0: 0, y1: 1, fillcolor: cssVar("--accent"), opacity: 0.15, line: { width: 0 }, layer: "below" }] });
        info.textContent = `step ${r.ckpt_steps[t]} · constant component ${c.toFixed(3)} · key frequencies of this run: ${r.key_freqs.join(", ")} · shaded: selected k = ${s.freq} (click a bar to select)`;
      }, { dedupe: frameKey(ds, "ckpt_steps", "freq") });
    },
  };
}

/**
 * Step × frequency heatmap of a per-checkpoint array [T, 113] (Fourier components): `perFreq(row)` turns one row into 56
 * values (k = 1…56, drawn high k at the top). The current checkpoint's column and the selected frequency's row are
 * outlined (overlay marks: moving them redraws nothing); clicking a cell selects that step and frequency.
 */
export function stepByFreq(arrayName, { perFreq, format, what }) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      let hm = null, steps = null;
      const info = note(body);
      const mark = (s) => { if (!hm) return; hm.marks.set("step", { col: nearestIndex(steps, s.step) }); hm.marks.set("freq", { row: K - s.freq }); };
      watch(body, ["run"], async (s) => {
        const id = runOf(ds), r = await run(id);
        body.querySelector(".heatmap-wrap")?.remove();
        hm = null;
        if (!r.arrays[arrayName]) { info.textContent = "No data for this run."; return; }
        const a = await array(id, arrayName), T = a.shape[0], grid = new Float32Array(K * T);
        for (let t = 0; t < T; t++) perFreq(row(a, t)).forEach((v, j) => { grid[(K - 1 - j) * T + t] = v; }); // row K−k holds frequency k
        steps = r.ckpt_steps;
        hm = new Heatmap(body, { rows: K, cols: T, size: 640, height: 240, diverging: false, format,
          label: (rr, cc) => `k = ${K - rr}, step ${steps[cc]}`, onClick: (rr, cc) => storeOf(body).set({ step: steps[cc], freq: K - rr }) });
        body.append(info);
        hm.draw(grid);
        info.textContent = `${what} Rows: frequency k (1 at the bottom, 56 at the top); columns: checkpoints from step 0 to ${kstep(steps[T - 1])}. Highlighted: the current step's column and the selected frequency's row (click a cell to select both).`;
        mark(s);
      }, { theme: false });
      follow(body, ["step", "freq"], mark);
    },
  };
}

/** √(cos² + sin²) of each frequency k = 1…56 in a 113-long Fourier vector (paper order Const, cos 1, sin 1, …). */
export const freqNorms = (v) => Array.from({ length: K }, (_, i) => Math.hypot(v[2 * i + 1], v[2 * i + 2]));

export const figures = {
  "we-fourier-bars": spectrumBars("we_fourier", "norm of component in W_E"),
  "wl-fourier-bars": spectrumBars("wl_fourier", "norm of component in W_L"),
  "we-fourier-heatmap": stepByFreq("we_fourier", { perFreq: freqNorms, format: (v) => v.toFixed(3), what: "Norm of each frequency in W_E at every checkpoint." }),
  "wl-fourier-heatmap": stepByFreq("wl_fourier", { perFreq: freqNorms, format: (v) => v.toFixed(3), what: "Norm of each frequency in W_L at every checkpoint." }),

  "we-svals": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 240 });
      watch(body, ["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id), a = await array(id, "we_svals");
        const t = nearestIndex(r.ckpt_steps, s.step);
        const sv = Array.from(row(a, t));
        await plot.draw([{ x: sv.map((_, i) => i + 1), y: sv, type: "scatter", mode: "lines+markers", marker: { size: 4, color: cssVar("--accent") }, line: { color: cssVar("--accent") }, name: `step ${r.ckpt_steps[t]}` }],
          { xaxis: { title: { text: "singular value index" } }, yaxis: { title: { text: "singular value of W_E" }, rangemode: "tozero" }, hovermode: "closest" });
      }, { dedupe: frameKey(ds, "ckpt_steps") });
    },
  },

  "attn-C-fourier": headSpectra("attn_C_fourier", "norm of C_j's component"),
  "ov-fourier": headSpectra("ov_fourier", "norm in W_O W_V W_E"),

  "wlogit-sorted": {
    async render(body, ds) {
      body.innerHTML = "";
      const m = await manifest();
      let hm = null, order = [], nf = [], nv = []; // current neuron order and clusters (read by the hover label)
      const info = note(body);
      const mark = (s) => hm?.marks.set("neuron", { row: order.indexOf(s.neuron) }); // outline the selected neuron's row
      watch(body, ["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id);
        const net = await model(id, s.step);
        const t = nearestIndex(r.ckpt_steps, net.step);
        nf = row(await array(id, "neuron_freq"), t); nv = row(await array(id, "neuron_fve"), t);
        const p = net.p, n = net.n;
        const F = fourierBasis(p);
        order = clusterOrder(nf, nv, n);
        const grid = new Float32Array(n * p);
        order.forEach((neuron, rIdx) => {
          for (let i = 0; i < p; i++) { // component i of column `neuron` of W_L: Σ_c F[i][c] W_L[c][neuron]
            let s2 = 0;
            for (let c = 0; c < p; c++) s2 += F[i * p + c] * net.WL[c * n + neuron];
            grid[rIdx * p + i] = s2;
          }
        });
        if (!hm) hm = new Heatmap(body, { rows: n, cols: p, size: 420, height: 420, diverging: true, format: (v) => v.toFixed(3),
          label: (rr, cc) => `neuron ${order[rr]} (k=${nv[order[rr]] > 0.85 ? nf[order[rr]] : "none"}), component ${m.basis_names[cc]}`,
          onClick: (rr) => set({ neuron: order[rr] }) });
        body.append(info);
        hm.draw(grid);
        mark(s);
        info.textContent = `step ${net.step}: rows = 512 neurons sorted by their frequency cluster (unclustered last); columns = Fourier components of the output c (Const, cos 1, sin 1, …). Click a row to select that neuron (outlined).`;
      }, { dedupe: frameKey(ds, "weight_steps"), theme: false });
      follow(body, ["neuron"], mark);
    },
  },

  "wneur-fourier": {
    async render(body, ds) {
      body.innerHTML = "";
      const m = await manifest();
      let hm = null, order = [], nf = [], nv = []; // current neuron order and clusters (read by the hover label)
      const info = note(body);
      const mark = (s) => hm?.marks.set("neuron", { row: order.indexOf(s.neuron) });
      watch(body, ["run", "step", "head"], async (s) => {
        const id = runOf(ds), r = await run(id);
        const net = await model(id, s.step);
        const t = nearestIndex(r.ckpt_steps, net.step);
        nf = row(await array(id, "neuron_freq"), t); nv = row(await array(id, "neuron_fve"), t);
        const p = net.p, n = net.n, h = Number(s.head) || 0;
        const F = fourierBasis(p);
        order = clusterOrder(nf, nv, n);
        const grid = new Float32Array(n * p);
        order.forEach((neuron, rIdx) => {
          for (let i = 0; i < p; i++) { // component i over input tokens a of W_neur[h] = W_in·OV_h(a) (net.PA, grokmodel.js)
            let acc = 0;
            for (let a = 0; a < p; a++) acc += F[i * p + a] * net.PA[h][a][neuron];
            grid[rIdx * p + i] = acc;
          }
        });
        if (!hm) hm = new Heatmap(body, { rows: n, cols: p, size: 420, height: 420, diverging: true, format: (v) => v.toFixed(3),
          label: (rr, cc) => `neuron ${order[rr]} (k=${nv[order[rr]] > 0.85 ? nf[order[rr]] : "none"}), component ${m.basis_names[cc]} of a`,
          onClick: (rr) => set({ neuron: order[rr] }) });
        body.append(info);
        hm.draw(grid);
        mark(s);
        info.textContent = `step ${net.step}, head ${h}: rows = 512 neurons sorted by frequency cluster; columns = Fourier components (Const, cos 1, sin 1, …) over the token a of W_neur = W_in·W_O·W_V·W_E — what head ${h} adds to each neuron's input when it attends fully to a. Other heads: pick one in the dashboard's control bar. Click a row to select a neuron (outlined).`;
      }, { dedupe: frameKey(ds, "weight_steps", "head"), theme: false });
      follow(body, ["neuron"], mark);
    },
  },

  "pos-embed": {
    async render(body, ds) {
      body.innerHTML = "";
      const hm = new Heatmap(body, { rows: 3, cols: 128, size: 520, diverging: true, rowLabel: "position", colLabel: "dimension", format: (v) => v.toFixed(3) });
      watch(body, ["run", "step"], async (s) => {
        const net = await model(runOf(ds), s.step);
        hm.draw(net.Wpos.data);
      }, { dedupe: frameKey(ds, "weight_steps"), theme: false });
    },
  },
};

/** Per-head Fourier spectra ([T, 4, 113]) at the current step as grouped bars of √(cos² + sin²) per frequency. */
function headSpectra(arrayName, ytitle) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 280 });
      watch(body, ["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id), a = await array(id, arrayName);
        const t = nearestIndex(r.ckpt_steps, s.step);
        const v = row(a, t), H = a.shape[1], P = a.shape[2];
        const traces = [];
        for (let h = 0; h < H; h++) {
          const comb = freqNorms(v.subarray(h * P, (h + 1) * P));
          traces.push({ x: comb.map((_, i) => i + 1), y: comb, type: "bar", name: `head ${h}` });
        }
        await plot.draw(traces, { barmode: "group", xaxis: { title: { text: "frequency k" }, dtick: 4 }, yaxis: { title: { text: `${ytitle} (√(cos²+sin²))` } }, hovermode: "closest" });
      }, { dedupe: frameKey(ds, "ckpt_steps") });
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
