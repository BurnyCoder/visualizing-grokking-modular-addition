// ablation.js — the paper's logit-frequency ablation on every input, computed in the browser.
//
// Local explanation: each answer c's logits form a p × p grid over the inputs (a, b); in the 2D Fourier basis, the 8
// components that involve frequency k are (Const, cos k, sin k) ⊗ (Const, cos k, sin k) without Const ⊗ Const.
// "Remove k" subtracts those components from every answer's grid; "keep only K" keeps the components of the listed
// frequencies and zeroes everything else, Const ⊗ Const included — exactly src/grokviz/interventions.py
// `ablate_logit_freqs` (paper Sec. 4.4, Fig. 6). `logitBlocks` precomputes, for every answer and frequency, the 3 × 3
// coefficients those components need (one 2D transform per answer); `ablate` then rebuilds the ablated logits of every
// input from them and returns the loss on all pairs, the accuracy and each input's probability of the right answer.
// Global context: the dashboard's ablation playground (figures/interventions.js); its parity with the Python ablation is
// tested in tests/js/grokmodel.test.mjs. Pure module; heavy parts run in model/worker.js.
// Sources: paper Sec. 4.4 (https://arxiv.org/abs/2301.05217); log-sum-exp for a stable cross-entropy
// (https://en.wikipedia.org/wiki/LogSumExp).
import { fourierBasis } from "./fourier.js";

/**
 * 3 × 3 Fourier blocks of the logits: for answer c and frequency k, C_c[i, j] for i, j ∈ {Const, cos k, sin k}.
 * `L`: logits [p·p, p] (row a·p + b, column c). Returns Float64Array [p (c), K (k − 1), 9].
 */
export function logitBlocks(L, p) {
  const F = fourierBasis(p), K = (p - 1) / 2, out = new Float64Array(p * K * 9);
  const G = new Float64Array(p * p), M = new Float64Array(p * p); // answer c's grid, and F·G (transform along a)
  for (let c = 0; c < p; c++) {
    for (let i = 0; i < p * p; i++) G[i] = L[i * p + c];
    for (let i = 0; i < p; i++) {
      M.fill(0, i * p, (i + 1) * p);
      for (let a = 0; a < p; a++) { const f = F[i * p + a], g = a * p; for (let b = 0; b < p; b++) M[i * p + b] += f * G[g + b]; }
    }
    for (let k = 1; k <= K; k++) {
      const rows = [0, 2 * k - 1, 2 * k], o = (c * K + k - 1) * 9;
      for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) { // C[i, j] = Σ_b M[i, b] F[j, b] (transform along b)
        let s = 0;
        for (let b = 0; b < p; b++) s += M[rows[u] * p + b] * F[rows[v] * p + b];
        out[o + u * 3 + v] = s;
      }
    }
  }
  return out;
}

/**
 * Ablate the logits of every input: `remove` = frequencies whose 8 components are subtracted, or `keep` = the only
 * frequencies whose components are kept. `labels[i]` is the right answer of input i = a·p + b.
 * Returns {loss (mean cross-entropy over all pairs), acc, pCorrect: Float32Array(p·p)}.
 */
export function ablate(L, blocks, p, labels, { remove = null, keep = null }) {
  const F = fourierBasis(p), K = (p - 1) / 2, ks = keep ?? remove ?? [], row = new Float64Array(p), pCorrect = new Float32Array(p * p);
  const basisAt = (k, x) => [F[x], F[(2 * k - 1) * p + x], F[2 * k * p + x]]; // Const, cos k, sin k at x
  let loss = 0, right = 0;
  for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
    const i = a * p + b, terms = ks.map((k) => [k, basisAt(k, a), basisAt(k, b)]);
    for (let c = 0; c < p; c++) {
      let part = 0; // the selected frequencies' components of answer c's logit at (a, b)
      for (const [k, fa, fb] of terms) {
        const o = (c * K + k - 1) * 9;
        for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) if (u || v) part += blocks[o + u * 3 + v] * fa[u] * fb[v];
      }
      row[c] = keep ? part : L[i * p + c] - part;
    }
    let mx = -Infinity, best = 0;
    for (let c = 0; c < p; c++) if (row[c] > mx) { mx = row[c]; best = c; }
    let z = 0;
    for (let c = 0; c < p; c++) z += Math.exp(row[c] - mx);
    const lse = mx + Math.log(z), y = labels[i];
    loss += lse - row[y];
    pCorrect[i] = Math.exp(row[y] - lse);
    if (best === y) right++;
  }
  return { loss: loss / (p * p), acc: right / (p * p), pCorrect };
}
