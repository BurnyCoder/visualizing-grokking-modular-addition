// heads.js — dashboard group C, attention heads: where each head looks, over training and on every input.
//
// Local explanation:
//   attn-over-training     mean attention from '=' to a, b and '=' for each head at every checkpoint (avg_attn);
//   attention-heads-grid   for the current weights, every head's attention from '=' to a, to b and to '=' on all
//                          12,769 inputs (4 × 3 grids, fixed colour scale 0–1, from the model's cached patterns);
//   sigmoid-linearisation  for the selected head, attention to a against the score difference s₀ − s₁ on 2,000
//                          inputs, with σ(s₀ − s₁) (what the attention would be if '=' got none) and the best straight
//                          line, whose variance explained is the paper's "linear range" measure.
// Global context: chapter 6 derives A₀ ≈ σ(C_j[a] − C_j[b]) and chapter 14 checks the linear range (paper App. C.1.3);
// these panels show both on the real network at any step. s₀ − s₁ = C_j[a] − C_j[b] + a per-head constant (the two
// positional embeddings), which shifts the x axis but changes neither σ's fit nor the line's.
// Sources: paper App. C.1.3 (https://arxiv.org/abs/2301.05217); line fit as src/grokviz/interventions.py.
import { array, model, run } from "../core/data.js";
import { Plot, phaseShapes } from "../viz/plot.js";
import { miniHeatmap } from "../viz/heatmap.js";
import { cssVar } from "../viz/colors.js";
import { el, follow, frameKey, note, runOf, set, watch } from "./common.js";

const P = 113, KEYS = ["a", "b", "="];
const keyColor = (j) => cssVar(["--train", "--restricted", "--accent"][j]); // to a, to b, to '='

/** 2,000 fixed input indices (deterministic pseudo-random subset of the 12,769 pairs, Park–Miller generator). */
const SAMPLE = (() => {
  let seed = 12345;
  const out = new Set();
  while (out.size < 2000) { seed = (seed * 16807) % 2147483647; out.add(seed % (P * P)); }
  return [...out];
})();

export const figures = {
  "attn-over-training": {
    async render(body, ds) {
      body.innerHTML = "";
      const grid = el(body, "div", "small-multiples");
      grid.style.gridTemplateColumns = "repeat(auto-fill, minmax(230px, 1fr))";
      const info = note(body);
      let plots = [], shown = null;
      watch(body, ["run", "step", "stepf"], async (s, changed) => {
        const id = runOf(ds), r = await run(id);
        if (id !== shown || changed.includes("theme")) {
          shown = id;
          const a = await array(id, "avg_attn"), [T, H] = a.shape;
          grid.replaceChildren();
          plots = Array.from({ length: H }, (_, h) => {
            const f = el(grid, "figure");
            el(f, "figcaption", "", `head ${h}`);
            return new Plot(f, { height: 190, stepAxis: true }).on("plotly_click", () => set({ head: h }));
          });
          await Promise.all(plots.map((pl, h) => pl.draw(KEYS.map((name, j) => ({
            x: r.ckpt_steps, y: Array.from({ length: T }, (_, t) => a.data[(t * H + h) * 3 + j]), name: `to ${name}`, type: "scatter", mode: "lines",
            line: { color: keyColor(j), width: 1.6 }, showlegend: h === 0,
          })), { yaxis: { range: [0, 1], title: { text: "mean attention" } }, xaxis: { title: { text: "training step" } },
            shapes: phaseShapes(r.phases, r.ckpt_steps.at(-1)), margin: { l: 44, r: 6, t: 28, b: 36 }, hovermode: "x unified" })));
          info.textContent = "Mean attention from '=' over all 12,769 inputs, for each head (the three lines add up to 1 at every step). Click a chart to select that head and step. In the paper's run two heads start attending to '=' again late in training (chapter 18).";
        }
        plots.forEach((pl) => pl.cursor(s.stepf ?? s.step));
      });
    },
  },

  "attention-heads-grid": {
    async render(body, ds) {
      body.innerHTML = "";
      const grid = el(body, "div", "heads-grid");
      const info = note(body);
      const mark = (s) => grid.querySelectorAll("figure").forEach((f) => f.classList.toggle("selected", Number(f.dataset.head) === s.head));
      watch(body, ["run", "step"], async (s) => {
        const net = await model(runOf(ds), s.step);
        grid.replaceChildren();
        el(grid, "span"); // empty corner above the head labels
        KEYS.forEach((k) => el(grid, "p", "fig-note", `to ${k}`));
        for (let h = 0; h < net.H; h++) {
          el(grid, "p", "fig-note", `head ${h}`);
          for (let j = 0; j < 3; j++) {
            const f = el(grid, "figure");
            f.dataset.head = h;
            const c = miniHeatmap(net.attentionGrid(h, j), P, P, false, { min: 0, max: 1 });
            c.title = `head ${h}, attention from '=' to ${KEYS[j]} (click: select head and input)`;
            c.onclick = (e) => { // the clicked cell is the input (a = row, b = column)
              const rect = c.getBoundingClientRect();
              set({ head: h, a: Math.floor(((e.clientY - rect.top) / rect.height) * P), b: Math.floor(((e.clientX - rect.left) / rect.width) * P) });
            };
            f.append(c);
          }
        }
        mark(s);
        info.textContent = `Step ${net.step.toLocaleString("en-US")}: attention from '=' to a, to b and to '=' for every input (rows a, columns b; purple 0, yellow 1). For each head the three grids add up to 1 in every cell. Click a grid to select its head (outlined) and the input under the mouse.`;
      }, { dedupe: frameKey(ds, "weight_steps"), theme: false, minGap: 250 }); // 12 grids
      follow(body, ["head"], mark);
    },
  },

  "sigmoid-linearisation": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 300 });
      const info = note(body);
      watch(body, ["run", "step", "head"], async (s) => {
        const net = await model(runOf(ds), s.step), h = Number(s.head) || 0;
        const A = net.attentionGrid(h, 0), x = new Float32Array(P * P); // attention to a; score difference s₀ − s₁
        for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) x[a * P + b] = net.sA[h][a] - net.sB[h][b];
        let mx = 0, my = 0;
        for (let i = 0; i < x.length; i++) { mx += x[i]; my += A[i]; }
        mx /= x.length; my /= x.length;
        let sxy = 0, sxx = 0, sst = 0, sseLine = 0, sseSig = 0;
        for (let i = 0; i < x.length; i++) { sxy += (x[i] - mx) * (A[i] - my); sxx += (x[i] - mx) ** 2; sst += (A[i] - my) ** 2; }
        const E = sxx ? sxy / sxx : 0, b0 = my - E * mx; // least-squares line A ≈ b0 + E·(s₀ − s₁)
        const sig = (v) => 1 / (1 + Math.exp(-v));
        for (let i = 0; i < x.length; i++) { sseLine += (A[i] - b0 - E * x[i]) ** 2; sseSig += (A[i] - sig(x[i])) ** 2; }
        const lo = Math.min(...SAMPLE.map((i) => x[i])), hi = Math.max(...SAMPLE.map((i) => x[i]));
        const xs = Array.from({ length: 61 }, (_, i) => lo + ((hi - lo) * i) / 60);
        const pct = (sse) => `${(100 * (1 - sse / (sst || 1))).toFixed(1)} %`;
        await plot.draw([
          { x: SAMPLE.map((i) => x[i]), y: SAMPLE.map((i) => A[i]), type: "scatter", mode: "markers", name: "inputs (2,000 of 12,769)",
            marker: { size: 3, color: cssVar("--train"), opacity: 0.45 }, text: SAMPLE.map((i) => `a = ${Math.floor(i / P)}, b = ${i % P}`), hovertemplate: "%{text}<extra></extra>" },
          { x: xs, y: xs.map(sig), type: "scatter", mode: "lines", name: "σ(s₀ − s₁): no attention to '='", line: { color: cssVar("--ink"), dash: "dot", width: 2 } },
          { x: xs, y: xs.map((v) => b0 + E * v), type: "scatter", mode: "lines", name: "best straight line", line: { color: cssVar("--accent"), width: 2 } },
        ], { xaxis: { title: { text: "score difference s₀ − s₁ (= C_j[a] − C_j[b] + const)" } }, yaxis: { title: { text: `head ${h}: attention from '=' to a` }, range: [-0.02, 1.02] }, hovermode: "closest" });
        info.textContent = `Head ${h}, weights from step ${net.step.toLocaleString("en-US")}, over all 12,769 inputs: σ(s₀ − s₁) explains ${pct(sseSig)} of the attention's variance, the best straight line ${pct(sseLine)} (the paper reports 97.5 % for the straight line, averaged over the heads of its final model). Pick another head in the control bar.`;
      }, { dedupe: frameKey(ds, "weight_steps", "head"), minGap: 300 }); // 2,000 SVG points
    },
  },
};
