// output.js — dashboard group E: the model's answer for the selected input (a, b) at the selected step.
//
// Local explanation: runs GrokModel on (a, b), shows the 113 logits next to their best fit by a sum of cosines
// Σ_k α_k cos(w_k(a+b−c)) over the run's key frequencies (least squares over c), the softmax probabilities, and
// each frequency's cosine wave separately so the constructive interference at c = a+b is visible. "clock-vs-pizza"
// re-arranges the right answer's logit over all inputs by sum and difference and computes Zhong et al.'s (2023)
// distance irrelevance, which separates their "Clock" from their "Pizza" networks. In "interference"
// the reader can switch frequencies off and sweep the candidate answer c: an inset chains one hand per frequency,
// α_k·(cos θ_k, sin θ_k) with θ_k = w_k(a+b−c), tip to tail (a phasor sum); each hand's horizontal part is exactly that
// frequency's contribution, so the chain stretches straight to the right (all contributions add up) only at c = a + b.
// Global context: the last step of the Fourier multiplication algorithm (paper Sec. 3.1, eq. 1; App. B,
// Fig. 8) on any concrete input, at any point of training.
// Sources: paper Sec. 3.1, 4.2 and App. B (https://arxiv.org/abs/2301.05217).
import { model, run } from "../core/data.js";
import { reducedMotion } from "../core/motion.js";
import { Plot } from "../viz/plot.js";
import { Heatmap } from "../viz/heatmap.js";
import { CanvasPlot } from "../viz/canvas.js";
import { cssVar, freqColor } from "../viz/colors.js";
import { softmax } from "../model/grokmodel.js";
import { el, frameKey, note, runOf, set, watch } from "./common.js";
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
  "clock-vs-pizza": {
    async render(body, ds) {
      body.innerHTML = "";
      const P = 113, inv2 = (P + 1) / 2; // 2·inv2 ≡ 1 (mod p): a = (s − d)/2, b = (s + d)/2
      const row = el(body, "div", "abl-row");
      const hm = new Heatmap(row, { rows: P, cols: P, size: 300, diverging: false, format: (v) => v.toFixed(2),
        label: (r, c) => `b − a ≡ ${P - 1 - r}, a + b ≡ ${c} (mod 113)`, onClick: (r, c) => { const d = P - 1 - r; set({ a: (((c - d) * inv2) % P + P) % P, b: (((c + d) * inv2) % P + P) % P }); } });
      const plot = new Plot(el(row, "div", "cvp-profile"), { height: 260 });
      const info = note(body), grid = new Float32Array(P * P);
      watch(body, ["run", "step"], async (s) => {
        const net = await model(runOf(ds), s.step), L = net.logitsGrid((a, b) => (a + b) % P); // logit of the right answer
        const mean = new Float64Array(P), sd = new Float64Array(P);
        let all = 0, all2 = 0;
        for (let d = 0; d < P; d++) { // inputs with difference b − a ≡ d (Zhong et al. 2023, Def. 4.2: L[i, i + d])
          let m = 0, m2 = 0;
          for (let i = 0; i < P; i++) { const v = L[i * P + ((i + d) % P)]; m += v; m2 += v * v; }
          mean[d] = m / P; sd[d] = Math.sqrt(Math.max(0, m2 / P - mean[d] ** 2)); all += m; all2 += m2;
          for (let sIdx = 0; sIdx < P; sIdx++) { // heatmap row: difference d (drawn 0 at the bottom), column: sum
            const a = (((sIdx - d) * inv2) % P + P) % P, b = (((sIdx + d) * inv2) % P + P) % P;
            grid[(P - 1 - d) * P + sIdx] = L[a * P + b];
          }
        }
        const muAll = all / (P * P), sdAll = Math.sqrt(Math.max(0, all2 / (P * P) - muAll ** 2));
        const q = sd.reduce((x, y) => x + y, 0) / P / (sdAll || 1); // distance irrelevance
        hm.draw(grid);
        const ds_ = Array.from({ length: P }, (_, d) => d);
        await plot.draw([
          { x: ds_, y: Array.from(mean, (m, d) => m + sd[d]), type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
          { x: ds_, y: Array.from(mean, (m, d) => m - sd[d]), type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(127,127,127,0.25)", name: "± 1 standard deviation", hoverinfo: "skip" },
          { x: ds_, y: Array.from(mean), type: "scatter", mode: "lines", name: "mean over all a + b", line: { color: cssVar("--accent"), width: 2 } },
        ], { xaxis: { title: { text: "difference b − a (mod 113)" } }, yaxis: { title: { text: "logit of the right answer" } }, hovermode: "x unified" });
        info.textContent = `Weights from step ${net.step.toLocaleString("en-US")}: the right answer's logit for every input, re-arranged by its sum a + b (columns) and difference b − a (rows); right: its mean and spread at each difference. Distance irrelevance q = ${q.toFixed(2)} (Zhong et al. 2023): low q means the logit depends mainly on the difference, as in their "Pizza" networks (q between 0 and 0.4); a "Clock" network's right-answer logit does not depend on the difference, which keeps q high.`;
      }, { dedupe: frameKey(ds, "weight_steps"), minGap: 500 });
    },
  },

  "logit-fourier-2d": spectrumMap("spec_logit_fourier2d", "size of each 2D Fourier component of the logits (norm over the 113 answers c, each answer's mean removed)"),
  "output-dist": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 280 });
      const plot2 = new Plot(body, { height: 200 });
      const info = note(body);
      watch(body, ["run", "step", "a", "b"], async (s) => {
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
      }, { dedupe: frameKey(ds, "weight_steps", "a", "b") });
    },
  },

  "interference": {
    async render(body, ds) {
      body.innerHTML = "";
      const ctl = el(body, "div", "interf-ctl");
      const chips = el(ctl, "span", "hero-chips");
      const sweep = el(ctl, "button", "btn", "▶ sweep c");
      sweep.type = "button";
      const cIn = el(ctl, "label", "hero-ctl", 'candidate answer c <input type="range" min="0" max="112" step="1" value="0">').querySelector("input");
      const row = el(body, "div", "interf-row");
      const plot = new Plot(row, { height: 300 });
      const cp = new CanvasPlot(el(row, "div", "interf-inset"), { height: 240 });
      const info = note(body);
      let fit = null, ks = [], c = 0, anim = null; // fit: α_k and waves for the current input; anim: sweep in progress
      const off = new Set(); // frequencies switched off by the reader
      const sumAt = (x) => ks.reduce((acc, k, j) => acc + (off.has(k) ? 0 : fit.alpha[j] * fit.cols[j][x]), 0);
      const drawInset = () => { // the hands at the current c, and the plot's cursor
        if (!fit) return;
        const on = ks.map((k, j) => [k, j]).filter(([k]) => !off.has(k)), R = 1.08 * ks.reduce((acc, _, j) => acc + Math.abs(fit.alpha[j]), 0) || 1;
        const segs = [{ x0: -R, y0: 0, x1: R, y1: 0, color: cssVar("--line"), width: 1 }];
        let x = 0, y = 0;
        for (const [k, j] of on) { // hands chained tip to tail (a phasor sum): the end's x is the summed contribution
          const th = (2 * Math.PI * k * (fit.t - c)) / fit.p, nx = x + fit.alpha[j] * Math.cos(th), ny = y + fit.alpha[j] * Math.sin(th);
          segs.push({ x0: x, y0: y, x1: nx, y1: ny, color: freqColor(k), width: 3 });
          [x, y] = [nx, ny];
        }
        segs.push({ x0: x, y0: y, x1: x, y1: 0, color: cssVar("--ink-soft"), width: 1 }); // drop to the axis: the sum
        cp.draw({ domain: [-R, R, -R, R], equal: true, segments: segs, labels: [{ x, y: -0.12 * R, text: "sum", color: cssVar("--ink") }] });
        plot.cursor(c);
        cIn.value = String(c);
        const right = fit.t % fit.p;
        info.textContent = `c = ${c}: the chained hands (one per frequency, coloured as the chips) end at ${sumAt(c).toFixed(1)}, the summed contribution${c === right ? " — every hand points right: c = a + b" : ` (at c = a + b = ${right}: ${sumAt(right).toFixed(1)})`}. Each wave peaks at c = a + b (mod ${fit.p}) but also near other c; only there do the peaks of all switched-on frequencies coincide. Switch frequencies off with the chips: one frequency alone leaves several almost equally good answers.`;
      };
      const drawPlot = async () => {
        if (!fit) return;
        const cs = Array.from({ length: fit.p }, (_, x) => x);
        const traces = ks.map((k, j) => ({ x: cs, y: fit.cols[j].map((v) => fit.alpha[j] * v), type: "scatter", mode: "lines", name: `α·cos(w_${k}(a+b−c)), k=${k}`,
          line: { color: freqColor(k), width: 1.4 }, opacity: off.has(k) ? 0.2 : 1 }));
        traces.push({ x: cs, y: cs.map(sumAt), type: "scatter", mode: "lines", name: "sum of the switched-on frequencies", line: { color: cssVar("--ink"), width: 3 } });
        await plot.draw(traces, { xaxis: { title: { text: "candidate answer c" } }, yaxis: { title: { text: "contribution to the logit" } }, hovermode: "x unified" });
        drawInset();
      };
      watch(body, ["run", "step", "a", "b"], async (s) => {
        const id = runOf(ds), r = await run(id), net = await model(id, s.step);
        const p = net.p, L = Array.from(net.logits(s.a, s.b)).slice(0, p);
        if (ks.join() !== r.key_freqs.join()) { // new run: one toggle chip per key frequency
          ks = r.key_freqs;
          off.clear();
          chips.replaceChildren(...ks.map((k) => Object.assign(document.createElement("button"), { type: "button", className: "chip active", textContent: `k = ${k}`, style: `border-color: ${freqColor(k)}`,
            onclick: (e) => { if (off.has(k)) off.delete(k); else off.add(k); e.target.classList.toggle("active", !off.has(k)); drawPlot(); } })));
        }
        fit = { ...cosFit(L, s.a + s.b, ks, p), t: s.a + s.b, p };
        c = (s.a + s.b) % p;
        await drawPlot();
      }, { dedupe: frameKey(ds, "weight_steps", "a", "b") });
      cIn.oninput = () => { c = Number(cIn.value); drawInset(); };
      sweep.onclick = () => { // sweep c over all answers in 6 s (reduced motion: jump to a + b instead)
        if (anim) { cancelAnimationFrame(anim); anim = null; sweep.textContent = "▶ sweep c"; return; }
        if (!fit) return;
        if (reducedMotion()) { c = fit.t % fit.p; drawInset(); return; }
        const t0 = performance.now();
        sweep.textContent = "❚❚ stop";
        const step = (t) => {
          c = Math.min(fit.p - 1, Math.floor(((t - t0) / 6000) * fit.p));
          drawInset();
          if (c < fit.p - 1) anim = requestAnimationFrame(step); else { anim = null; sweep.textContent = "▶ sweep c"; }
        };
        anim = requestAnimationFrame(step);
      };
    },
  },
};
