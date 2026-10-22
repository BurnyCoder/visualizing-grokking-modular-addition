// fourier.js — the real orthonormal Fourier basis over Z_p and the 2D transform of a p × p grid.
//
// Local explanation: `fourierBasis(p)` returns the p × p matrix F whose rows are the constant vector and, for each
// frequency k = 1 … (p − 1)/2, cos(2πkx/p) and sin(2πkx/p), each scaled to length 1 (paper order: Const, cos 1,
// sin 1, cos 2, …); `fft2(G, p)` returns C = F G Fᵀ, the coefficients of a grid G over (a, b) on the 2D basis F_i ⊗ F_j.
// Global context: the same basis as src/grokviz/fourier.py, used by every Fourier view on the site, the ablation
// playground (model/ablation.js) and the teaching widgets. Pure module, testable with Node.
// Sources: paper Sec. 4 and code `fourier_basis` (https://arxiv.org/abs/2301.05217); unitary DFT
// (https://en.wikipedia.org/wiki/Discrete_Fourier_transform#The_unitary_DFT).

const cache = new Map(); // p → basis
/** Real orthonormal Fourier basis (paper order) as a flat p×p Float32Array (row i = basis vector i), cached. */
export function fourierBasis(p) {
  if (cache.has(p)) return cache.get(p);
  const F = new Float32Array(p * p);
  for (let x = 0; x < p; x++) F[x] = 1 / Math.sqrt(p); // constant row
  for (let k = 1; k <= (p - 1) / 2; k++) {
    for (let x = 0; x < p; x++) {
      F[(2 * k - 1) * p + x] = Math.cos((2 * Math.PI * k * x) / p) * Math.sqrt(2 / p);
      F[2 * k * p + x] = Math.sin((2 * Math.PI * k * x) / p) * Math.sqrt(2 / p);
    }
  }
  cache.set(p, F);
  return F;
}

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
