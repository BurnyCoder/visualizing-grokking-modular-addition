// fourier.js — teaching widgets for chapters 10 and 14: the 5-point Fourier transform by hand, "periodic means
// sparse" with p = 113, the 2D Fourier basis gallery, and ReLU as a multiplier.
//
// Local explanation: all transforms use the same real orthonormal basis as the analysis code (paper order:
// Const, cos k, sin k), built by fourierBasis(p) in model/fourier.js; 2D transforms use fft2 from
// the same module; the ReLU regression solves ARENA's least-squares fit with normal equations.
// Global context: shows on small, controllable examples the operations later applied to the real network.
// Sources: ARENA 1.5.2 "Fourier Transforms" and the ReLU regression
// (https://learn.arena.education/chapter1_transformer_interp/32_grokking/1-periodicity-fourier-basis/,
// https://learn.arena.education/chapter1_transformer_interp/32_grokking/2-circuit-and-feature-analysis/);
// unitary DFT (https://en.wikipedia.org/wiki/Discrete_Fourier_transform#The_unitary_DFT).
import { Plot } from "../viz/plot.js";
import { Heatmap } from "../viz/heatmap.js";
import { cssVar } from "../viz/colors.js";
import { fft2, fourierBasis } from "../model/fourier.js";
import { basisName as name } from "../figures/common.js";
import { frame, onTheme, slider, sweep } from "./basics.js";

const f3 = (x) => (Math.abs(x) < 5e-4 ? "0.000" : x.toFixed(3));

/** Coefficients F v of a length-p vector. */
function dft(v, p) {
  const F = fourierBasis(p);
  return Array.from({ length: p }, (_, i) => v.reduce((s, x, t) => s + F[i * p + t] * x, 0));
}

export const widgets = {
  "dft-p5": {
    render(el) {
      const { controls, out, work } = frame(el, "The Fourier transform of 5 numbers, by hand");
      const plot = new Plot(out, { height: 220 });
      const v = [3, 1, 0, 0, 1];
      const draw = () => {
        const c = dft(v, 5);
        plot.draw([{ x: [0, 1, 2, 3, 4].map(name), y: c, type: "bar", marker: { color: cssVar("--accent") } }], { yaxis: { title: { text: "coefficient" } }, hovermode: "closest" });
        const F = fourierBasis(5);
        work.textContent = [0, 1, 2, 3, 4].map((i) => `${name(i).padEnd(6)} = ${v.map((x, t) => `${x}·${f3(F[i * 5 + t])}`).join(" + ")} = ${f3(c[i])}`).join("\n") +
          `\nsum of squared coefficients ${f3(c.reduce((s, x) => s + x * x, 0))} = sum of squared entries ${v.reduce((s, x) => s + x * x, 0)}`;
      };
      onTheme(draw);
      v.forEach((x, i) => slider(controls, `v(${i})`, -5, 5, 1, x, (val) => { v[i] = val; draw(); }));
    },
  },

  "periodic-sparse": {
    render(el) {
      const { controls, out, work } = frame(el, "Periodic in the ordinary basis = sparse in the Fourier basis (p = 113)");
      const p1 = new Plot(out, { height: 200 }), p2 = new Plot(out, { height: 200 });
      let k = 14, noise = 0, phase = 0;
      const draw = () => {
        let seed = 7;
        const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; }; // fixed pseudo-random noise
        const xs = Array.from({ length: 113 }, (_, x) => x);
        const v = xs.map((x) => Math.cos((2 * Math.PI * k * x) / 113 + phase) + noise * 2 * rnd());
        const c = dft(v, 113);
        p1.draw([{ x: xs, y: v, type: "scatter", mode: "lines+markers", marker: { size: 3 }, line: { color: cssVar("--train") } }], { xaxis: { title: { text: "x" } }, yaxis: { title: { text: "value" } }, hovermode: "closest" });
        p2.draw([{ x: xs.map(name), y: c, type: "bar", marker: { color: cssVar("--accent") } }], { xaxis: { title: { text: "basis vector" }, showticklabels: false }, yaxis: { title: { text: "coefficient" } }, hovermode: "closest" });
        const top = c.map((x, i) => [Math.abs(x), i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
        work.textContent = `largest coefficients: ${top.map(([x, i]) => `${name(i)} (${f3(c[i])})`).join(", ")}\nwith no noise only cos ${k} and sin ${k} are non-zero; noise spreads over all 113.`;
      };
      onTheme(draw);
      slider(controls, "frequency k", 1, 56, 1, k, (val) => { k = val; draw(); });
      sweep(slider(controls, "phase shift", 0, 6.28, 0.05, phase, (val) => { phase = val; draw(); }), { seconds: 6 }); // the wave moves, its coefficients stay on cos k and sin k
      slider(controls, "noise", 0, 2, 0.05, noise, (val) => { noise = val; draw(); });
    },
  },

  "basis-2d": {
    render(el) {
      const { controls, out, work } = frame(el, "The 2D Fourier basis: wave in a × wave in b");
      const hm = new Heatmap(out, { rows: 113, cols: 113, size: 260, diverging: true, format: (v) => v.toFixed(4) });
      let i = 27, j = 0, combo = 0;
      const F = fourierBasis(113);
      const draw = () => {
        const g = new Float32Array(113 * 113);
        if (combo) { // cos(w_k(a+b)) = (cos⊗cos − sin⊗sin)/√2 at k = ceil(i/2)
          const k = Math.max(1, Math.ceil(i / 2)), c = 2 * k - 1, s = 2 * k;
          for (let a = 0; a < 113; a++) for (let b = 0; b < 113; b++) g[a * 113 + b] = (F[c * 113 + a] * F[c * 113 + b] - F[s * 113 + a] * F[s * 113 + b]) / Math.SQRT2;
          work.textContent = `cos(w_${k}(a+b)) direction = (cos ${k} ⊗ cos ${k} − sin ${k} ⊗ sin ${k}) / √2 — diagonal stripes (depends only on a+b)`;
        } else {
          for (let a = 0; a < 113; a++) for (let b = 0; b < 113; b++) g[a * 113 + b] = F[i * 113 + a] * F[j * 113 + b];
          const kind = i === 0 && j === 0 ? "constant" : i === 0 || j === 0 ? "linear term (depends on one input)" : "quadratic term (product of waves in a and b)";
          work.textContent = `${name(i)} (in a) × ${name(j)} (in b): ${kind}`;
        }
        hm.draw(g);
      };
      sweep(slider(controls, "index in a (0 = Const, 2k−1 = cos k, 2k = sin k)", 0, 112, 1, i, (v) => { i = v; draw(); }), { seconds: 12 });
      slider(controls, "index in b", 0, 112, 1, j, (v) => { j = v; draw(); });
      slider(controls, "show cos(w(a+b)) instead", 0, 1, 1, combo, (v) => { combo = v; draw(); });
    },
  },

  "relu-multiplier": {
    render(el) {
      const { controls, out, work } = frame(el, "ReLU turns a sum of waves into products (ARENA's example)");
      const wrap = document.createElement("div");
      wrap.className = "heatmap-wrap";
      out.append(wrap);
      const left = document.createElement("div"), right = document.createElement("div");
      wrap.append(left, right);
      const hm = new Heatmap(left, { rows: 113, cols: 113, size: 220, diverging: false, format: (v) => v.toFixed(4) });
      const hf = new Heatmap(right, { rows: 113, cols: 113, size: 220, diverging: true, format: (v) => v.toFixed(4), label: (r, c) => `${name(r)} × ${name(c)}` });
      let k = 42, A = 0.5;
      const F = fourierBasis(113);
      const draw = () => {
        const idx = 2 * k - 1, ca = (x) => F[idx * 113 + x];
        const y = new Float32Array(113 * 113);
        for (let a = 0; a < 113; a++) for (let b = 0; b < 113; b++) y[a * 113 + b] = Math.max(0, A / Math.sqrt(113) + ca(a) + ca(b));
        hm.draw(y);
        const mean = y.reduce((s, v) => s + v, 0) / y.length;
        hf.draw(fft2(y.map((v) => v - mean), 113));
        const cols = [() => 1, (a) => ca(a), (a, b) => ca(b), (a, b) => ca(a) * ca(b)]; // design: const, cos a, cos b, cos a·cos b
        const fit = (m) => { // least squares via normal equations on the first m columns
          const X = (a, b) => cols.slice(0, m).map((f) => f(a, b));
          const XtX = Array.from({ length: m }, () => Array(m).fill(0)), Xty = Array(m).fill(0);
          for (let a = 0; a < 113; a++) for (let b = 0; b < 113; b++) { const x = X(a, b), t = y[a * 113 + b]; for (let r = 0; r < m; r++) { Xty[r] += x[r] * t; for (let c = 0; c < m; c++) XtX[r][c] += x[r] * x[c]; } }
          for (let i = 0; i < m; i++) for (let r = 0; r < m; r++) if (r !== i) { const f = XtX[r][i] / XtX[i][i]; for (let c = 0; c < m; c++) XtX[r][c] -= f * XtX[i][c]; Xty[r] -= f * Xty[i]; }
          const coef = Xty.map((v, i) => v / XtX[i][i]);
          let sse = 0, sst = 0;
          for (let a = 0; a < 113; a++) for (let b = 0; b < 113; b++) { const t = y[a * 113 + b], pr = X(a, b).reduce((s, x, r) => s + x * coef[r], 0); sse += (t - pr) ** 2; sst += (t - mean) ** 2; }
          return { coef, r2: 1 - sse / sst };
        };
        const full = fit(4), lin = fit(3);
        work.textContent = `f(a,b) = ReLU(${A}/√113 + cos_${k}(a) + cos_${k}(b))   (cos_k = unit-length basis wave)\nfit with product term: coefficient of cos(a)·cos(b) = ${full.coef[3].toFixed(3)}, R² = ${full.r2.toFixed(3)}\nfit without it: R² = ${lin.r2.toFixed(3)}\nRight: the 2D Fourier transform — the quadratic (cos ${k} × cos ${k}) term appears.`;
      };
      slider(controls, "frequency k", 1, 56, 1, k, (v) => { k = v; draw(); });
      slider(controls, "offset A", -2, 2, 0.1, A, (v) => { A = v; draw(); });
    },
  },
};
