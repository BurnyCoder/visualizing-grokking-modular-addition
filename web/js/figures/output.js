// output.js — dashboard group E: the model's answer for the selected input (a, b) at the selected step.
//
// Local explanation: runs GrokModel on (a, b), shows the 113 logits next to their best fit by a sum of cosines
// Σ_k α_k cos(w_k(a+b−c)) over the run's key frequencies (least squares over c), the softmax probabilities, and
// each frequency's cosine wave separately so the constructive interference at c = a+b is visible.
// Global context: the last step of the Fourier multiplication algorithm (paper Sec. 3.1, eq. 1; App. B,
// Fig. 8) on any concrete input, at any point of training.
// Sources: paper Sec. 3.1, 4.2 and App. B (https://arxiv.org/abs/2301.05217).
import { model, run } from "../core/data.js";
import { Plot } from "../viz/plot.js";
import { cssVar, freqColor } from "../viz/colors.js";
import { softmax } from "../model/grokmodel.js";
import { note, runOf, watch } from "./common.js";
import { spectrumMap } from "./activations.js";

/** Least-squares α_k (+ intercept) for logits[c] ≈ β + Σ_k α_k cos(w_k(t − c)), t = a + b. */
export function cosFit(logits, t, freqs, p) {
  const cols = freqs.map((k) => Array.from({ length: p }, (_, c) => Math.cos((2 * Math.PI * k * (t - c)) / p)));
  cols.push(Array(p).fill(1));
  const m = cols.length; // normal equations XᵀX α = Xᵀy (tiny: m ≤ 11)
  const A = cols.map((u) => cols.map((v) => u.reduce((s, x, i) => s + x * v[i], 0)));
  const y = cols.map((u) => u.reduce((s, x, i) => s + x * logits[i], 0));
  for (let i = 0; i < m; i++) { // Gauss-Jordan elimination with partial pivoting
    let piv = i;
    for (let r = i + 1; r < m; r++) if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r;
    [A[i], A[piv]] = [A[piv], A[i]]; [y[i], y[piv]] = [y[piv], y[i]];
    for (let r = 0; r < m; r++) if (r !== i) { const f = A[r][i] / A[i][i]; for (let c = i; c < m; c++) A[r][c] -= f * A[i][c]; y[r] -= f * y[i]; }
  }
  const coef = y.map((v, i) => v / A[i][i]);
  const fit = Array.from({ length: p }, (_, c) => cols.reduce((s, col, j) => s + coef[j] * col[c], 0));
  return { alpha: coef.slice(0, -1), beta: coef[m - 1], fit, cols };
}

export const figures = {
  "logit-fourier-2d": spectrumMap("spec_logit_fourier2d", "size of each 2D Fourier component of the logits (norm over the 113 answers c, each answer's mean removed)"),
  "output-dist": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 280 });
      const plot2 = new Plot(body, { height: 200 });
      const info = note(body);
      watch(["run", "step", "a", "b"], async (s) => {
        const id = runOf(ds), r = await run(id), net = await model(id, s.step);
        const p = net.p, L = Array.from(net.logits(s.a, s.b)).slice(0, p);
        const t = (s.a + s.b) % p;
        const { fit } = cosFit(L, s.a + s.b, r.key_freqs, p);
        const cs = Array.from({ length: p }, (_, c) => c);
        await plot.draw([
          { x: cs, y: L, type: "bar", name: "logit (model)", marker: { color: cs.map((c) => (c === t ? cssVar("--accent") : cssVar("--ink-soft"))) } },
          { x: cs, y: fit, type: "scatter", mode: "lines", name: "fit: β + Σ α_k cos(w_k(a+b−c))", line: { color: cssVar("--restricted"), width: 2 } },
        ], { xaxis: { title: { text: "candidate answer c" } }, yaxis: { title: { text: "logit" } }, hovermode: "closest" });
        const prob = softmax(L);
        await plot2.draw([{ x: cs, y: prob, type: "bar", name: "probability", marker: { color: cs.map((c) => (c === t ? cssVar("--accent") : cssVar("--train"))) } }],
          { xaxis: { title: { text: "candidate answer c" } }, yaxis: { title: { text: "softmax probability" }, range: [0, 1] }, hovermode: "closest" });
        info.textContent = `step ${net.step}, input ${s.a} + ${s.b} (mod ${p}) = ${t}: the model gives ${(100 * prob[t]).toFixed(1)} % to the right answer; top answer ${prob.indexOf(Math.max(...prob))}.`;
      });
    },
  },

  "interference": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 300 });
      const info = note(body);
      watch(["run", "step", "a", "b"], async (s) => {
        const id = runOf(ds), r = await run(id), net = await model(id, s.step);
        const p = net.p, L = Array.from(net.logits(s.a, s.b)).slice(0, p);
        const { alpha, cols } = cosFit(L, s.a + s.b, r.key_freqs, p);
        const cs = Array.from({ length: p }, (_, c) => c);
        const traces = r.key_freqs.map((k, j) => ({ x: cs, y: cols[j].map((v) => alpha[j] * v), type: "scatter", mode: "lines", name: `α·cos(w_${k}(a+b−c)), k=${k}`, line: { color: freqColor(k), width: 1.4 } }));
        const sum = cs.map((c) => r.key_freqs.reduce((acc, _, j) => acc + alpha[j] * cols[j][c], 0));
        traces.push({ x: cs, y: sum, type: "scatter", mode: "lines", name: "sum", line: { color: cssVar("--ink"), width: 3 } });
        await plot.draw(traces, { xaxis: { title: { text: "candidate answer c" } }, yaxis: { title: { text: "contribution to the logit" } }, hovermode: "x unified" });
        info.textContent = `Each wave peaks at c = a + b = ${(s.a + s.b) % p} (mod ${p}) but also near other c; only there do all ${r.key_freqs.length} peaks coincide, so the sum stands out (constructive interference).`;
      });
    },
  },
};
