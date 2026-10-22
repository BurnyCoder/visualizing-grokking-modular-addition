// geometry.js — dashboard group D: activation geometry and output-distribution geometry over training.
//
// Local explanation: scatter plots of precomputed low-dimensional projections (src/grokviz/geometry.py):
// token embeddings in each key-frequency plane (circles = the "clock"), embedding PCA, MLP class-mean circles,
// PCA of class-mean logits / predicted distributions, plus grids of the probability of the correct answer and
// of the prediction's entropy over all inputs. Points are coloured by their position on a cyclic colour map
// (viz/colors.js clockColor), so a circle traversed in order shows one smooth colour cycle.
// Global context: the "addition becomes rotation" picture (paper Fig. 1, Sec. 3.1; Welch Labs' linear-probe
// circles) watched as it emerges from noise during training.
// Sources: paper Sec. 3.1 (https://arxiv.org/abs/2301.05217); Welch Labs video (https://www.youtube.com/watch?v=D8GOeCFFby4).
import { array, nearestIndex, row, run, trainMask } from "../core/data.js";
import { lerpPolar2 } from "../core/anim.js";
import { Plot } from "../viz/plot.js";
import { Heatmap } from "../viz/heatmap.js";
import { CanvasPlot } from "../viz/canvas.js";
import { clockColor, cssVar } from "../viz/colors.js";
import { el, fill, follow, frameKey, note, runOf, set, storeOf, watch } from "./common.js";

const P = 113;

/**
 * Points [p, 2] (or [p, 3] using the first two columns) as a coloured scatter trace. With frequency k, number i is
 * coloured by its predicted clock position (k·i mod p): if the network places i at angle w_k·i, the colours change smoothly once around the circle.
 */
function points(xy, dims, name, k = 1) {
  const x = [], y = [];
  for (let i = 0; i < P; i++) { x.push(xy[i * dims]); y.push(xy[i * dims + 1]); }
  return { x, y, mode: "markers", type: "scatter", name, text: x.map((_, i) => `number ${i} (clock position ${(k * i) % P})`),
    marker: { size: 6, color: x.map((_, i) => clockColor((k * i) % P, P)), line: { width: 0.5, color: cssVar("--ink-soft") } }, showlegend: false }; // outline: light and dark map ends stay visible
}

const radii = new Map(); // "run|array|plane" → largest |coordinate| over all stored frames
/** One fixed scale per plane of an array [F, K, p, 2] for the whole run, so a forming circle visibly grows. */
export function planeRadius(a, j, key) {
  if (!radii.has(key)) {
    const [F, K] = a.shape;
    let m = 0;
    for (let f = 0; f < F; f++) for (let q = 0; q < 2 * P; q++) m = Math.max(m, Math.abs(a.data[(f * K + j) * 2 * P + q]));
    radii.set(key, 1.08 * m || 1);
  }
  return radii.get(key);
}

/**
 * Canvas scene of one clock (viz/canvas.js): the 113 points `xy` (flat x, y pairs) coloured by clock position k·n mod 113,
 * `hands` [[n, colour variable, label]] from the centre to chosen numbers, and number `hi` enlarged and labelled.
 */
export function clockScene(xy, k, R, hands, hi = null, noun = "number") {
  const at = (t) => ({ x: xy[2 * t], y: xy[2 * t + 1] });
  return {
    domain: [-R, R, -R, R], equal: true,
    segments: hands.map(([t, v]) => ({ x0: 0, y0: 0, x1: at(t).x, y1: at(t).y, color: cssVar(v), width: 2 })),
    points: { x: Array.from({ length: P }, (_, t) => xy[2 * t]), y: Array.from({ length: P }, (_, t) => xy[2 * t + 1]), r: (t) => (t === hi ? 6 : 3),
      color: (t) => clockColor((k * t) % P, P), label: (t) => `<span class="tt-term">${noun} ${t}</span>clock position ${k}·${t} mod 113 = ${(k * t) % P}` },
    labels: [...hands.map(([t, v, name]) => ({ x: 1.13 * at(t).x, y: 1.13 * at(t).y, text: name, color: cssVar(v) })),
      ...(hi == null ? [] : [{ x: at(hi).x, y: at(hi).y + 0.12 * R, text: String(hi), color: cssVar("--ink") }])],
  };
}

/**
 * Small multiples, one clock per key frequency, from an array [F, K, p, 2] (canvas, redrawn every played frame).
 * `byAnswer`: points are answers c (MLP class means), so only the a + b hand is meaningful; otherwise numbers n
 * with hands to a, b and a + b. While playing, frames are blended in polar coordinates (points move along the circle).
 */
function planes(arrayName, what, byAnswer = false) {
  return {
    async render(body, ds) {
      body.innerHTML = "";
      const wrap = el(body, "div", "small-multiples");
      wrap.style.gridTemplateColumns = "repeat(auto-fill, minmax(200px, 1fr))";
      const info = note(body);
      let tiles = [], shown = null, hover = null, view = null; // view: what the clocks show, kept for hover redraws
      const paint = () => view && tiles.forEach((t, j) => t.cp.draw(clockScene(t.xy, view.ks[j], view.R[j], view.hands, hover, byAnswer ? "answer" : "number")));
      watch(body, ["run", "step", "stepf", "playing", "a", "b", "freq"], async (s) => {
        const id = runOf(ds), r = await run(id);
        if (!r.geometry_steps) { info.textContent = "no geometry frames for this run"; return; }
        const a = await array(id, arrayName), ks = r.key_freqs;
        if (shown !== id) {
          shown = id;
          wrap.replaceChildren();
          tiles = ks.map((k) => {
            const fig = el(wrap, "figure");
            el(fig, "figcaption", "", `k = ${k}`);
            const cp = new CanvasPlot(fig, { height: 200, onClick: (i) => set(byAnswer ? {} : { a: i }), onHover: (i) => { if (i !== hover) { hover = i; paint(); } } });
            return { fig, k, cp, xy: new Float32Array(2 * P) };
          });
        }
        let label = "";
        tiles.forEach((t, j) => { label = fill(t.xy, a, r.geometry_steps, s, { offset: j * 2 * P, lerp: lerpPolar2 }); t.fig.classList.toggle("selected", t.k === s.freq); });
        const c = (s.a + s.b) % P;
        view = { ks, R: ks.map((_, j) => planeRadius(a, j, `${id}|${arrayName}|${j}`)), hands: byAnswer ? [[c, "--accent", "a+b"]] : [[s.a, "--train", "a"], [s.b, "--restricted", "b"], [c, "--accent", "a+b"]] };
        paint();
        info.textContent = `${label}: ${what}. Colour = clock position k·n mod 113: one smooth colour cycle around the circle means the network places ${byAnswer ? "answer" : "number"} n at angle w_k·n. ${byAnswer ? "Hand: the answer a + b" : "Hands: a, b and a + b; click a dot to make it a"}; hover one to find it on every clock. Outlined: the selected frequency.`;
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
      watch(body, ["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id);
        if (!r.geometry_steps) return;
        const f = nearestIndex(r.geometry_steps, s.step);
        const v = row(await array(id, arrayName), f), ev = row(await array(id, varName), f);
        const tr = points(v, 3, what);
        await plot.draw([tr], { xaxis: { title: { text: `PC1 (${(100 * ev[0]).toFixed(0)} %)` }, zeroline: false },
          yaxis: { title: { text: `PC2 (${(100 * ev[1]).toFixed(0)} %)` }, zeroline: false, scaleanchor: "x" }, hovermode: "closest" });
        info.textContent = `step ${r.geometry_steps[f]}: ${what}; first two principal components.`;
      }, { dedupe: frameKey(ds, "geometry_steps") });
    },
  };
}

/**
 * A 113 × 113 grid over all inputs (rows a, columns b) drawn into `parent` from an array [F, p·p] of stored frames with
 * values in 0…1: blended between stored frames while playing, optionally only the training or only the test pairs, the
 * selected input outlined, a click selecting (a, b). The note reads `describe(frameLabel, meanTrain, meanTest)`.
 * Used by the dashboard's pcorrect/entropy panels and the hero's second tile.
 */
export function inputGridView(parent, ds, arrayName, { size = 300, describe }) {
  const show = el(parent, "label", "hero-ctl", 'show <select><option value="all">all pairs</option><option value="train">training pairs</option><option value="test">test pairs</option></select>').querySelector("select");
  const hm = new Heatmap(parent, { rows: P, cols: P, size, diverging: false, format: (v) => v.toFixed(3), onClick: (a, b) => set({ a, b }) });
  const info = note(parent), grid = new Float32Array(P * P);
  const draw = async (s) => {
    const id = runOf(ds), r = await run(id);
    if (!r.geometry_steps) { info.textContent = "No frames for this run."; return; }
    const [a, mask] = await Promise.all([array(id, arrayName), trainMask(r.split_seed)]);
    const label = fill(grid, a, r.geometry_steps, s);
    let tr = 0, te = 0, nt = 0, ne = 0;
    for (let i = 0; i < grid.length; i++) if (mask[i]) { tr += grid[i]; nt++; } else { te += grid[i]; ne++; }
    if (show.value !== "all") for (let i = 0; i < grid.length; i++) if ((mask[i] === 1) !== (show.value === "train")) grid[i] = NaN; // hide the other pairs
    hm.draw(grid, { min: 0, max: 1 });
    info.textContent = describe(label, tr / nt, te / ne);
  };
  watch(parent, ["run", "step", "stepf", "playing"], draw, { theme: false });
  show.onchange = () => draw(storeOf(parent).state);
  follow(parent, ["a", "b"], (s) => hm.marks.set("input", { row: s.a, col: s.b }));
}

/** Dashboard panel around inputGridView: `what` the grid shows, plus its means on the training and test pairs. */
const inputGrid = (arrayName, what) => ({
  async render(body, ds) {
    body.innerHTML = "";
    inputGridView(body, ds, arrayName, { describe: (label, mt, me) =>
      `${label}: ${what}. Mean on training pairs ${mt.toFixed(3)}, on test pairs ${me.toFixed(3)}. Outlined: the selected input (click a cell to select).` });
  },
});

export const figures = {
  "embed-circles": planes("geo_we_planes", "each number's embedding projected onto the plane that carries cos(w_k a) and sin(w_k a)"),
  "mlp-circles": planes("geo_mlp_planes", "average MLP activation for each answer c = a+b mod p, read along W_L's cos/sin(w_k c) directions", true),
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
      watch(body, ["run"], async () => {
        const r = await run(runOf(ds));
        const mask = await trainMask(r.split_seed);
        hm.draw(Float32Array.from(mask), { min: 0, max: 1 });
        const n = mask.reduce((x, y) => x + y, 0);
        info.textContent = `${n} training pairs (yellow) and ${mask.length - n} test pairs (purple); split seed ${r.split_seed}.`;
      }, { theme: false });
      follow(body, ["a", "b"], (s) => hm.marks.set("input", { row: s.a, col: s.b }));
    },
  },
};
