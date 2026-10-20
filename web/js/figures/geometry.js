// geometry.js — dashboard group D: activation geometry and output-distribution geometry over training.
//
// Local explanation: scatter plots of precomputed low-dimensional projections (src/grokviz/geometry.py):
// token embeddings in each key-frequency plane (circles = the "clock"), embedding PCA, MLP class-mean circles,
// PCA of class-mean logits / predicted distributions, plus grids of the probability of the correct answer and
// of the prediction's entropy over all inputs. Points are coloured by their number on a cyclic colour wheel,
// so a circle traversed in order shows up as a rainbow.
// Global context: the "addition becomes rotation" picture (paper Fig. 1, Sec. 3.1; Welch Labs' linear-probe
// circles) watched as it emerges from noise during training.
// Sources: paper Sec. 3.1 (https://arxiv.org/abs/2301.05217); Welch Labs video (https://www.youtube.com/watch?v=D8GOeCFFby4).
import { array, nearestIndex, row, run, trainMask } from "../core/data.js";
import { Plot } from "../viz/plot.js";
import { Heatmap } from "../viz/heatmap.js";
import { el, note, runOf, set, watch } from "./common.js";

const P = 113;
const hue = (i, n = P) => `hsl(${Math.round((360 * i) / n)}, 70%, 50%)`; // cyclic colour for number i

/** Points [p, 2] (or [p, 3] using the first two columns) as a coloured scatter trace. */
function points(xy, dims, name, idx = 0) {
  const x = [], y = [];
  for (let i = 0; i < P; i++) { x.push(xy[idx + i * dims]); y.push(xy[idx + i * dims + 1]); }
  return { x, y, mode: "markers", type: "scatter", name, text: x.map((_, i) => `number ${i}`), marker: { size: 6, color: x.map((_, i) => hue(i)) }, showlegend: false };
}

/** Small multiples: one scatter per key frequency from an array [F, K, p, 2]. */
function planes(arrayName, what) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      const wrap = el(body, "div", "small-multiples");
      wrap.style.gridTemplateColumns = "repeat(auto-fill, minmax(200px, 1fr))";
      const info = note(body);
      let plots = [];
      let shown = null;
      watch(["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id);
        if (!r.geometry_steps) { info.textContent = "no geometry frames for this run"; return; }
        const a = await array(id, arrayName);
        const f = nearestIndex(r.geometry_steps, s.step);
        const K = a.shape[1];
        if (shown !== id) {
          shown = id;
          wrap.replaceChildren();
          plots = r.key_freqs.map((k) => { const fig = el(wrap, "figure"); el(fig, "figcaption", "", `k = ${k}`); return new Plot(fig, { height: 200 }); });
        }
        const v = row(a, f);
        await Promise.all(plots.map((pl, j) => pl.draw([points(v.subarray(j * P * 2, (j + 1) * P * 2), 2, `k=${r.key_freqs[j]}`)],
          { margin: { l: 30, r: 6, t: 4, b: 24 }, xaxis: { zeroline: false, showticklabels: false }, yaxis: { zeroline: false, showticklabels: false, scaleanchor: "x" }, hovermode: "closest" })));
        info.textContent = `step ${r.geometry_steps[f]}: ${what} — colour runs around the colour wheel from 0 to 112 (K = ${K} key frequencies).`;
      });
    },
  };
}

/** PCA scatter from an array [F, p, 3] + explained variance [F, 3]. */
function pcaScatter(arrayName, varName, what) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 280 });
      const info = note(body);
      watch(["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id);
        if (!r.geometry_steps) return;
        const f = nearestIndex(r.geometry_steps, s.step);
        const v = row(await array(id, arrayName), f), ev = row(await array(id, varName), f);
        await plot.draw([points(v, 3, what)], { xaxis: { title: { text: `PC1 (${(100 * ev[0]).toFixed(0)} %)` }, zeroline: false },
          yaxis: { title: { text: `PC2 (${(100 * ev[1]).toFixed(0)} %)` }, zeroline: false, scaleanchor: "x" }, hovermode: "closest" });
        info.textContent = `step ${r.geometry_steps[f]}: ${what}; first two principal components.`;
      });
    },
  };
}

/** Grid over inputs from an array [F, p*p] (u8-quantised values in 0..1), with train pairs optionally dimmed. */
function inputGrid(arrayName, what, diverging = false) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      const hm = new Heatmap(body, { rows: P, cols: P, size: 300, diverging, format: (v) => v.toFixed(3), onClick: (a, b) => set({ a, b }) });
      const info = note(body);
      watch(["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id);
        if (!r.geometry_steps) return;
        const f = nearestIndex(r.geometry_steps, s.step);
        hm.draw(row(await array(id, arrayName), f), { min: 0, max: 1 });
        const mask = await trainMask(r.split_seed);
        let tr = 0, te = 0, nt = 0, ne = 0;
        const v = row(await array(id, arrayName), f);
        for (let i = 0; i < v.length; i++) if (mask[i]) { tr += v[i]; nt++; } else { te += v[i]; ne++; }
        info.textContent = `step ${r.geometry_steps[f]}: ${what}. Mean on training pairs ${(tr / nt).toFixed(3)}, on test pairs ${(te / ne).toFixed(3)}.`;
      });
    },
  };
}

export const figures = {
  "embed-circles": planes("geo_we_planes", "each number's embedding projected onto the plane that carries cos(w_k a) and sin(w_k a)"),
  "mlp-circles": planes("geo_mlp_planes", "average MLP activation for each answer c = a+b mod p, read along W_L's cos/sin(w_k c) directions"),
  "embed-pca": pcaScatter("geo_we_pca", "geo_we_pca_var", "the 113 number embeddings"),
  "logit-pca": pcaScatter("geo_logit_pca", "geo_logit_pca_var", "mean logit vector for each correct answer"),
  "prob-pca": pcaScatter("geo_prob_pca", "geo_prob_pca_var", "mean predicted probability distribution for each correct answer"),
  "pcorrect-grid": inputGrid("geo_p_correct", "probability the model gives to the correct answer, for every input (rows a, columns b)"),
  "entropy-grid": inputGrid("geo_entropy", "uncertainty of the prediction (normalised entropy: 0 = certain, 1 = uniform guess)"),
  "split-grid": {
    async render(body, ds) {
      body.innerHTML = "";
      const hm = new Heatmap(body, { rows: P, cols: P, size: 300, diverging: false, format: (v) => (v ? "training pair" : "test pair"), onClick: (a, b) => set({ a, b }) });
      const info = note(body);
      watch(["run"], async () => {
        const r = await run(runOf(ds));
        const mask = await trainMask(r.split_seed);
        hm.draw(Float32Array.from(mask), { min: 0, max: 1 });
        const n = mask.reduce((x, y) => x + y, 0);
        info.textContent = `${n} training pairs (yellow) and ${mask.length - n} test pairs (purple); split seed ${r.split_seed}.`;
      });
    },
  },
};
