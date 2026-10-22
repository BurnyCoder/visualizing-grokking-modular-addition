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
import { cssVar, freqColor } from "../viz/colors.js";
import { basisName, clusterOrder, el, fill, follow, frameKey, kstep, note, runOf, set, storeOf, watch } from "./common.js";
import { CanvasPlot } from "../viz/canvas.js";
import { fourierBasis, freqCoeffs } from "../model/fourier.js";

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
 * Step × row heatmap of a per-checkpoint array: `perRow(row, state)` turns one checkpoint's row into `rows` values, row 0
 * drawn at the bottom, coloured with fixed `limits` {min, max} when given; `keys` are extra state keys it depends on
 * (e.g. the selected head). By default the rows are frequencies k = 1…56: `rowName(i)` labels row i, `select(i)` is the
 * state a click on it selects and `selected(state)` the row outlined (null: none). The current checkpoint's column is
 * outlined too (overlay marks: moving them redraws nothing).
 */
export function stepHeatmap(arrayName, { perRow, format, what, limits = null, keys = [], rows = K, rowName = (i) => `k = ${i + 1}`,
  select = (i) => ({ freq: i + 1 }), selected = (s) => s.freq - 1, rowsText = "frequency k (1 at the bottom, 56 at the top)" }) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      let hm = null, steps = null;
      const info = note(body);
      const mark = (s) => {
        if (!hm) return;
        const i = selected(s);
        hm.marks.set("step", { col: nearestIndex(steps, s.step) });
        hm.marks.set("row", i == null ? null : { row: rows - 1 - i });
      };
      watch(body, ["run", ...keys], async (s) => {
        const id = runOf(ds), r = await run(id);
        body.querySelector(".heatmap-wrap")?.remove();
        hm = null;
        if (!r.arrays[arrayName]) { info.textContent = "No data for this run."; return; }
        const a = await array(id, arrayName), T = a.shape[0], grid = new Float32Array(rows * T);
        for (let t = 0; t < T; t++) perRow(row(a, t), s).forEach((v, i) => { grid[(rows - 1 - i) * T + t] = v; }); // row 0 at the bottom
        steps = r.ckpt_steps;
        hm = new Heatmap(body, { rows, cols: T, size: 640, height: 240, diverging: false, format,
          label: (rr, cc) => `${rowName(rows - 1 - rr)}, step ${steps[cc]}`, onClick: (rr, cc) => storeOf(body).set({ step: steps[cc], ...select(rows - 1 - rr) }) });
        body.append(info);
        hm.draw(grid, limits);
        info.textContent = `${what} Rows: ${rowsText}; columns: checkpoints from step 0 to ${kstep(steps[T - 1])}. Highlighted: the current step's column${selected(s) == null ? "" : " and the selected row"} (click a cell to select).`;
        mark(s);
      }, { theme: false });
      follow(body, ["step", "freq"], mark);
    },
  };
}

/** √(cos² + sin²) of each frequency k = 1…56 in a 113-long Fourier vector (paper order Const, cos 1, sin 1, …). */
export const freqNorms = (v) => Array.from({ length: K }, (_, i) => Math.hypot(v[2 * i + 1], v[2 * i + 2]));

export const figures = {
  "freq-norm-race": {
    async render(body, ds) {
      body.innerHTML = "";
      const sel = el(body, "label", "hero-ctl", 'matrix <select><option value="we_fourier">W_E (embedding)</option><option value="wl_fourier">W_L (neuron → logit map)</option></select>').querySelector("select");
      const cp = new CanvasPlot(body, { height: 330, onClick: (k) => set({ freq: k }) });
      const info = note(body);
      const TOP = 12, buf = new Float32Array(2 * K + 1), shownY = new Map(), scale = new Map(); // frequency → displayed row
      const draw = async (s) => {
        const id = runOf(ds), r = await run(id), a = await array(id, sel.value);
        if (!scale.has(`${id}|${sel.value}`)) { // one x scale for the whole run: bars grow and shrink visibly
          let m = 0;
          for (let t = 0; t < a.shape[0]; t++) m = Math.max(m, ...freqNorms(a.data.subarray(t * 113, (t + 1) * 113)));
          scale.set(`${id}|${sel.value}`, m || 1);
        }
        const max = scale.get(`${id}|${sel.value}`), label = fill(buf, a, r.ckpt_steps, s);
        const norms = freqNorms(buf), init = freqNorms(a.data.subarray(0, 113)); // now, and at initialisation (step 0)
        const top = norms.map((v, i) => [v, i + 1]).sort((x, y) => y[0] - x[0]).slice(0, TOP);
        const rects = [], labels = [];
        top.forEach(([v, k], rank) => {
          const target = TOP - 1 - rank, y0 = shownY.get(k) ?? -1; // a frequency entering the top 12 rises from below
          const y = s.playing ? y0 + (target - y0) * 0.3 : target; // ease the reordering while playing; exact when paused
          shownY.set(k, y);
          rects.push({ x0: 0, x1: init[k - 1], y0: y - 0.36, y1: y + 0.36, stroke: cssVar("--ink-soft") }, // ghost: the value at initialisation
            { x0: 0, x1: v, y0: y - 0.3, y1: y + 0.3, fill: freqColor(k), stroke: k === s.freq ? cssVar("--ink") : null, id: k,
              label: `<span class="tt-term">k = ${k}</span>norm ${v.toFixed(2)} (at initialisation ${init[k - 1].toFixed(2)}); click to select` });
          labels.push({ x: -0.015 * max, y, text: `k = ${k}`, align: "right", color: cssVar(r.key_freqs.includes(k) ? "--ink" : "--ink-soft") },
            { x: v + 0.015 * max, y, text: v.toFixed(1), align: "left", color: cssVar("--ink-soft") });
        });
        for (const k of [...shownY.keys()]) if (!top.some(([, kk]) => kk === k)) shownY.delete(k); // dropped out of the top 12
        cp.draw({ domain: [-0.16 * max, 1.12 * max, -0.7, TOP - 0.3], rects, labels });
        info.textContent = `${label}. The 12 largest frequency norms of ${sel.value === "we_fourier" ? "W_E" : "W_L"}, longest at the top; outlines show each frequency's norm at initialisation, dark labels the run's key frequencies. Click a bar to select its frequency everywhere.`;
      };
      sel.onchange = () => draw(storeOf(body).state);
      watch(body, ["run", "step", "stepf", "playing", "freq"], draw);
    },
  },

  "we-fourier-bars": spectrumBars("we_fourier", "norm of component in W_E"),
  "wl-fourier-bars": spectrumBars("wl_fourier", "norm of component in W_L"),
  "we-fourier-heatmap": stepHeatmap("we_fourier", { perRow: freqNorms, format: (v) => v.toFixed(3), what: "Norm of each frequency in W_E at every checkpoint." }),
  "wl-fourier-heatmap": stepHeatmap("wl_fourier", { perRow: freqNorms, format: (v) => v.toFixed(3), what: "Norm of each frequency in W_L at every checkpoint." }),
  "svals-over-training": stepHeatmap("we_svals", { perRow: (v) => Array.from(v), format: (v) => v.toFixed(2), limits: { min: 0 }, rows: 113,
    rowName: (i) => `singular value ${i + 1}`, select: () => ({}), selected: () => null, rowsText: "the 113 singular values of W_E, largest at the bottom",
    what: "Singular values of the embedding matrix W_E at every checkpoint." }),
  "head-spectra-over-training": {
    async render(body, ds) { // two views (C_j or OV) of the selected head; the hidden one is not redrawn
      body.innerHTML = "";
      const sel = el(body, "label", "hero-ctl", 'show <select><option value="attn_C_fourier">the attention lookup C_j</option><option value="ov_fourier">what the head writes (OV)</option></select>').querySelector("select");
      const views = {};
      for (const [name, what] of [["attn_C_fourier", "Norm of each frequency in the selected head's attention lookup C_j"], ["ov_fourier", "Norm of each frequency in what the selected head writes, W_O W_V W_E"]]) {
        views[name] = el(body, "div");
        views[name].hidden = name !== sel.value;
        stepHeatmap(name, { perRow: (v, s) => freqNorms(v.subarray(s.head * 113, (s.head + 1) * 113)), keys: ["head"], format: (v) => v.toFixed(3),
          what: `${what} at every checkpoint (pick the head in the control bar).` }).render(views[name], ds);
      }
      sel.onchange = () => { for (const [name, d] of Object.entries(views)) d.hidden = name !== sel.value; };
    },
  },

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

  "attn-C-fourier": headSpectra("attn_C_fourier", "norm of C_j's component", { signed: true }),
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

  "neuron-stars": {
    async render(body, ds) {
      body.innerHTML = "";
      let order = []; // drawing order of neurons (selected frequency's cluster on top); a dot's index → neuron
      const cp = new CanvasPlot(body, { height: 380, onClick: (i) => set({ neuron: order[i] }) });
      const info = note(body);
      const inputs = new WeakMap(), reach = new Map(); // per snapshot: Σ_j W_neur^j rows over a; per run|k: plot radius
      const coeffs = (net, k) => { // each neuron's (cos k, sin k) coefficients of its input from the token a
        if (!inputs.has(net)) inputs.set(net, Array.from({ length: net.p }, (_, a) => net.PA.reduce((acc, ph) => acc.map((v, j) => v + ph[a][j]), new Float32Array(net.n))));
        return freqCoeffs(inputs.get(net), net.p, k);
      };
      const radius = ([x, y]) => x.reduce((m, _, j) => Math.max(m, Math.hypot(x[j], y[j])), 0);
      watch(body, ["run", "step", "freq", "neuron"], async (s) => {
        const id = runOf(ds), r = await run(id), net = await model(id, s.step), k = Number(s.freq) || 1;
        const t = nearestIndex(r.ckpt_steps, net.step), nf = row(await array(id, "neuron_freq"), t), nv = row(await array(id, "neuron_fve"), t);
        const [x, y] = coeffs(net, k), key = `${id}|${k}`;
        if (!reach.has(key)) reach.set(key, radius(coeffs(await model(id, r.weight_steps.at(-1)), k))); // the run's final weights
        const R = 1.1 * Math.max(reach.get(key), radius([x, y])) || 1;
        const inK = (j) => nv[j] > 0.85 && nf[j] === k;
        const sel = Number(s.neuron);
        order = Array.from({ length: net.n }, (_, j) => j).sort((u, v) => inK(u) - inK(v) || (u === sel) - (v === sel)); // drawn last = on top
        cp.draw({ domain: [-R, R, -R, R], equal: true,
          segments: [{ x0: -R, y0: 0, x1: R, y1: 0, color: cssVar("--line"), width: 1 }, { x0: 0, y0: -R, x1: 0, y1: R, color: cssVar("--line"), width: 1 }],
          labels: [{ x: R, y: 0.04 * R, text: `cos ${k}`, color: cssVar("--ink-soft"), align: "right" }, { x: 0.03 * R, y: 0.94 * R, text: `sin ${k}`, color: cssVar("--ink-soft"), align: "left" }],
          points: { x: order.map((j) => x[j]), y: order.map((j) => y[j]),
            color: (i) => (nv[order[i]] > 0.85 ? freqColor(nf[order[i]]) : cssVar("--ink-soft")), r: (i) => (order[i] === sel ? 6 : inK(order[i]) ? 3.5 : 2),
            label: (i) => `neuron ${order[i]}: cluster ${nv[order[i]] > 0.85 ? `k = ${nf[order[i]]}` : "none"}, (${x[order[i]].toFixed(2)}, ${y[order[i]].toFixed(2)})` } });
        const n = order.filter(inK).length;
        info.textContent = `weights from step ${net.step.toLocaleString("en-US")}: each dot is one of the ${net.n} neurons at the cos ${k} and sin ${k} coefficients of its input from the token a (W_neur = W_in·W_O·W_V·W_E summed over the 4 heads, as if every head attended only to a): distance = the size of its frequency-${k} wave, angle = the wave's phase. ${n ? `${n} neurons are in cluster k = ${k} (larger dots)` : `No neuron is in cluster k = ${k} yet`}; colour = a neuron's cluster, grey = none. Scale fixed per run and frequency. Pick k with the frequency control; click a dot to select its neuron.`;
      }, { dedupe: frameKey(ds, "weight_steps", "freq", "neuron") });
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

/**
 * Per-head Fourier spectra ([T, 4, 113]) at the current step as grouped bars of √(cos² + sin²) per frequency. With
 * `signed` (the array holds signed coefficients, e.g. C_j) a selector also offers the signed Const, cos k and sin k
 * coefficients of the run's key frequencies, where the phase of each head shows (two heads can mirror each other).
 */
function headSpectra(arrayName, ytitle, { signed = false } = {}) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      const sel = signed ? el(body, "label", "hero-ctl", 'show <select><option value="norm">size per frequency (all 56)</option><option value="signed">signed cos and sin parts (key frequencies)</option></select>').querySelector("select") : null;
      const plot = new Plot(body, { height: 280 });
      const draw = async (s) => {
        const id = runOf(ds), r = await run(id), a = await array(id, arrayName);
        const t = nearestIndex(r.ckpt_steps, s.step);
        const v = row(a, t), H = a.shape[1], P = a.shape[2];
        const comps = [0, ...r.key_freqs.flatMap((k) => [2 * k - 1, 2 * k])]; // Const, cos k, sin k of the key frequencies
        const traces = [];
        for (let h = 0; h < H; h++) {
          const vh = v.subarray(h * P, (h + 1) * P), comb = freqNorms(vh);
          traces.push(sel?.value === "signed" ? { x: comps.map(basisName), y: comps.map((i) => vh[i]), type: "bar", name: `head ${h}` }
            : { x: comb.map((_, i) => i + 1), y: comb, type: "bar", name: `head ${h}` });
        }
        await plot.draw(traces, sel?.value === "signed"
          ? { barmode: "group", xaxis: { title: { text: "Fourier component over the token (key frequencies)" }, type: "category" }, yaxis: { title: { text: `${ytitle.replace("norm of ", "")} (signed)` }, zeroline: true }, hovermode: "closest" }
          : { barmode: "group", xaxis: { title: { text: "frequency k" }, dtick: 4 }, yaxis: { title: { text: `${ytitle} (√(cos²+sin²))` } }, hovermode: "closest" });
      };
      watch(body, ["run", "step"], draw, { dedupe: frameKey(ds, "ckpt_steps") });
      if (sel) sel.onchange = () => draw(storeOf(body).state);
    },
  };
}
