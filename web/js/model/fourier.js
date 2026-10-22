// fourier.js — the real orthonormal Fourier basis over Z_p and the 2D transform of a p × p grid.
//
// Local explanation: `fourierBasis(p)` returns the p × p matrix F whose rows are the constant vector and, for each
// frequency k = 1 … (p − 1)/2, cos(2πkx/p) and sin(2πkx/p), each scaled to length 1 (paper order: Const, cos 1,
// sin 1, cos 2, …); `fft2(G, p)` returns C = F G Fᵀ, the coefficients of a grid G over (a, b) on the 2D basis F_i ⊗ F_j;
// `freqCoeffs` projects many units on one frequency's cos/sin pair; `torusPoints` places inputs at their frequency-k angles.
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

/**
 * Coefficients of cos k and sin k over the token a, for many units at once: `vectors[a][j]` is unit j's value at
 * token a. Returns [x, y] with x[j] = Σ_a F_cos k[a]·vectors[a][j] (y likewise for sin k): a unit that is one wave
 * r·cos(w_k a + φ) sits at r·√(p/2)·(cos φ, −sin φ), so a frequency cluster forms a ring (figure `neuron-stars`).
 */
export function freqCoeffs(vectors, p, k) {
  const F = fourierBasis(p), n = vectors[0].length, x = new Float32Array(n), y = new Float32Array(n);
  for (let a = 0; a < p; a++) {
    const c = F[(2 * k - 1) * p + a], s = F[2 * k * p + a], v = vectors[a]; // basis rows cos k and sin k at a
    for (let j = 0; j < n; j++) { x[j] += c * v[j]; y[j] += s * v[j]; }
  }
  return [x, y];
}

/**
 * Every input (a, b) on a torus (flat [x, y, z] per input, index a·p + b): angle w_k·a around the ring (radius `ring`),
 * angle w_k·b around the tube (radius `tube`), so inputs with one answer a + b lie on one helix (figure `torus-3d`).
 * Source: torus parametrisation (https://en.wikipedia.org/wiki/Torus#Geometry).
 */
export function torusPoints(p, k, ring, tube) {
  const xyz = new Float32Array(3 * p * p), w = (2 * Math.PI * k) / p;
  for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
    const i = 3 * (a * p + b), r = ring + tube * Math.cos(w * b); // distance from the torus axis
    xyz[i] = r * Math.cos(w * a); xyz[i + 1] = tube * Math.sin(w * b); xyz[i + 2] = r * Math.sin(w * a);
  }
  return xyz;
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
