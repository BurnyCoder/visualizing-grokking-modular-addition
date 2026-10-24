// grokmodel.js — the paper's one-layer transformer, re-implemented in plain JavaScript for the browser.
//
// Local explanation: loads a flat weight vector (in the order of the manifest's param_layout) and computes
// the forward pass exactly like src/grokviz/model.py (paper code layout: W_E [d, V], W_K/W_Q/W_V [H, dh, d],
// W_O [d, H*dh], W_in [n, d], W_out [d, n], W_U [d, V]). Because inputs are one-hot tokens, everything before
// the softmax/ReLU depends on one token at a time, so we precompute per-token tables once per checkpoint and
// then evaluate any (a, b) cheaply — fast enough to redraw 113×113 grids while the user scrubs through training.
// The tables can be computed elsewhere (model/worker.js, off the main thread) and handed over with
// `toTables()` / `GrokModel.fromTables()`.
// Global context: powers every per-input view on the site (forward walkthrough, attention/neuron grids,
// output distributions, activation patching); its numbers are tested against PyTorch in tests/js/.
// Sources: paper App. A equations (https://arxiv.org/abs/2301.05217); src/grokviz/model.py (same maths).

/** Decode IEEE-754 half-precision values (Uint16Array) into a Float32Array. */
export function decodeF16(u16) {
  const out = new Float32Array(u16.length); // destination, one float32 per half
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i]; // 16 raw bits: 1 sign, 5 exponent, 10 mantissa
    const s = h & 0x8000 ? -1 : 1; // sign bit
    const e = (h >> 10) & 0x1f; // biased exponent
    const f = h & 0x3ff; // fraction bits
    if (e === 0) out[i] = s * Math.pow(2, -14) * (f / 1024); // subnormal numbers (and ±0)
    else if (e === 31) out[i] = f ? NaN : s * Infinity; // NaN / infinity
    else out[i] = s * Math.pow(2, e - 15) * (1 + f / 1024); // normal numbers
  }
  return out;
}

/** Split a flat weight vector into named Float32Array views using the manifest layout. */
export function unpack(flat, layout) {
  const params = {}; // name → {data, shape}
  let offset = 0; // running position inside the flat vector
  for (const { name, shape } of layout) {
    const size = shape.reduce((x, y) => x * y, 1); // number of elements of this parameter
    params[name] = { data: flat.subarray(offset, offset + size), shape };
    offset += size;
  }
  return params;
}

/** Numerically stable softmax of a small array. */
export function softmax(xs) {
  const m = Math.max(...xs); // subtract the max so exp() never overflows
  const ex = xs.map((x) => Math.exp(x - m));
  const z = ex.reduce((s, x) => s + x, 0);
  return ex.map((x) => x / z);
}

export class GrokModel {
  /**
   * @param {Float32Array} flat  weights in param_layout order
   * @param {Array<{name:string, shape:number[]}>} layout  parameter layout from manifest.json
   * @param {number} p  modulus (113); token p is '='
   */
  constructor(flat, layout, p, tables = null) {
    const P = unpack(flat, layout); // named parameter views
    this.p = p;
    this.WE = P["embed.W_E"]; // [d, V]
    this.Wpos = P["pos_embed.W_pos"]; // [3, d]
    this.WK = P["blocks.0.attn.W_K"]; // [H, dh, d]
    this.WQ = P["blocks.0.attn.W_Q"];
    this.WV = P["blocks.0.attn.W_V"];
    this.WO = P["blocks.0.attn.W_O"]; // [d, H*dh]
    this.Win = P["blocks.0.mlp.W_in"]; // [n, d]
    this.bin = P["blocks.0.mlp.b_in"]; // [n]
    this.Wout = P["blocks.0.mlp.W_out"]; // [d, n]
    this.bout = P["blocks.0.mlp.b_out"]; // [d]
    this.WU = P["unembed.W_U"]; // [d, V]
    [this.d, this.V] = this.WE.shape; // residual width, vocabulary size
    [this.H, this.dh] = this.WK.shape; // heads, head size
    this.n = this.Win.shape[0]; // MLP neurons
    if (tables) Object.assign(this, tables); // computed elsewhere (web worker)
    else this._precompute();
  }

  /** A model from weights plus precomputed tables (no precompute on this thread). */
  static fromTables(flat, layout, p, tables) {
    return new GrokModel(flat, layout, p, tables);
  }

  /** The per-checkpoint tables every fast method reads (plain typed arrays: structured-clone friendly). */
  toTables() {
    return Object.fromEntries(TABLES.map((k) => [k, this[k]]));
  }

  /** Residual vector of token t at position pos: W_E[:, t] + W_pos[pos]. */
  resid0(t, pos) {
    const { d, V } = this;
    const x = new Float32Array(d);
    for (let i = 0; i < d; i++) x[i] = this.WE.data[i * V + t] + this.Wpos.data[pos * d + i];
    return x;
  }

  /** y = M x for a row-major matrix block M [rows, cols] starting at `off` inside `data`. */
  static matvec(data, off, rows, cols, x) {
    const y = new Float32Array(rows);
    for (let r = 0; r < rows; r++) {
      let s = 0;
      const base = off + r * cols;
      for (let c = 0; c < cols; c++) s += data[base + c] * x[c];
      y[r] = s;
    }
    return y;
  }

  /** Per-head query/key/value of a residual vector x: {q, k, v} each [H][dh]. */
  qkv(x) {
    const { H, dh, d } = this;
    const out = { q: [], k: [], v: [] };
    for (let h = 0; h < H; h++) {
      out.q.push(GrokModel.matvec(this.WQ.data, h * dh * d, dh, d, x)); // W_Q[h] x
      out.k.push(GrokModel.matvec(this.WK.data, h * dh * d, dh, d, x)); // W_K[h] x
      out.v.push(GrokModel.matvec(this.WV.data, h * dh * d, dh, d, x)); // W_V[h] x
    }
    return out;
  }

  /** OV output of head h for value vector v: W_O[:, h*dh:(h+1)*dh] v → [d]. */
  ov(h, v) {
    const { d, dh, H } = this;
    const y = new Float32Array(d);
    for (let i = 0; i < d; i++) {
      let s = 0;
      for (let j = 0; j < dh; j++) s += this.WO.data[i * H * dh + h * dh + j] * v[j];
      y[i] = s;
    }
    return y;
  }

  /**
   * Precompute token tables for the '=' position (everything the output at position 2 depends on). Each head's
   * output reaches the neurons and the logits linearly, so W_in·W_O^h [n, dh] and W_U^T·W_O^h [V, dh] are formed once
   * per head and applied to each token's dh-wide value vector — about half the multiply-adds of computing the d-wide
   * head output W_O^h v first and then W_in and W_U^T of it.
   */
  _precompute() {
    const { p, H, dh, d, n, V } = this;
    const scale = 1 / Math.sqrt(dh);
    const WO = this.WO.data, Win = this.Win.data, WU = this.WU.data;
    const WUt = new Float32Array(V * d); // W_U transposed [V, d], so logit rows are contiguous
    for (let i = 0; i < d; i++) for (let c = 0; c < V; c++) WUt[c * d + i] = WU[i * V + c];
    const inO = [], uO = []; // per head: W_in W_O^h [n, dh] and W_U^T W_O^h [V, dh]
    for (let h = 0; h < H; h++) {
      const A = new Float32Array(n * dh), B = new Float32Array(V * dh);
      for (let j = 0; j < dh; j++) {
        const col = new Float32Array(d); // column h*dh + j of W_O
        for (let i = 0; i < d; i++) col[i] = WO[i * H * dh + h * dh + j];
        for (let m = 0; m < n; m++) { let s = 0; for (let i = 0; i < d; i++) s += Win[m * d + i] * col[i]; A[m * dh + j] = s; }
        for (let c = 0; c < V; c++) { let s = 0; for (let i = 0; i < d; i++) s += WUt[c * d + i] * col[i]; B[c * dh + j] = s; }
      }
      inO.push(A); uO.push(B);
    }
    const reach = (h, v) => [GrokModel.matvec(inO[h], 0, n, dh, v), GrokModel.matvec(uO[h], 0, V, dh, v)]; // to neurons, to logits
    const x2 = this.resid0(p, 2); // '=' token at position 2 — the same for every input
    const qkv2 = this.qkv(x2);
    const q2 = qkv2.q; // queries from '=' (constant)
    this.s2 = q2.map((q, h) => dot(q, qkv2.k[h]) * scale); // score '=' → '='
    this.P2 = []; this.U2 = []; // what each head adds to the neuron inputs / logits when attending to '='
    for (let h = 0; h < H; h++) { const [a, b] = reach(h, qkv2.v[h]); this.P2.push(a); this.U2.push(b); }
    this.sA = []; this.sB = []; this.PA = []; this.PB = []; this.UA = []; this.UB = []; // per head, per token at pos 0 / 1
    for (let h = 0; h < H; h++) {
      this.sA.push(new Float32Array(p)); this.sB.push(new Float32Array(p));
      this.PA.push([]); this.PB.push([]); this.UA.push([]); this.UB.push([]);
    }
    for (let t = 0; t < p; t++) {
      for (const [pos, S, P, U] of [[0, this.sA, this.PA, this.UA], [1, this.sB, this.PB, this.UB]]) {
        const kv = this.qkv(this.resid0(t, pos));
        for (let h = 0; h < H; h++) {
          S[h][t] = dot(q2[h], kv.k[h]) * scale; // pre-softmax score from '=' to this token
          const [a, b] = reach(h, kv.v[h]); // what head h adds to the neuron inputs / logits if it attends fully here
          P[h].push(a); U[h].push(b);
        }
      }
    }
    this.c0 = GrokModel.matvec(Win, 0, n, d, x2).map((v, i) => v + this.bin.data[i]); // W_in x2 + b_in
    const U0 = new Float32Array(V); // W_U^T (x2 + b_out): the residual's direct contribution to the logits
    for (let c = 0; c < V; c++) { let s = 0; for (let i = 0; i < d; i++) s += WU[i * V + c] * (x2[i] + this.bout.data[i]); U0[c] = s; }
    this.U0 = U0;
    const WL = new Float64Array(V * n); // W_U^T W_out [V, n]: the neuron-logit map (incl. the '=' row)
    for (let i = 0; i < d; i++) for (let c = 0; c < V; c++) { // i outermost: the inner loop runs along contiguous rows
      const u = WU[i * V + c], row = i * n, out = c * n;
      for (let m = 0; m < n; m++) WL[out + m] += u * this.Wout.data[row + m];
    }
    this.WL = Float32Array.from(WL);
  }

  /** Attention pattern from '=' to (a, b, '=') for every head: [H][3]. */
  pattern(a, b) {
    const out = [];
    for (let h = 0; h < this.H; h++) out.push(softmax([this.sA[h][a], this.sB[h][b], this.s2[h]]));
    return out;
  }

  /** Neuron pre-activations at '=' for input (a, b): Float32Array(n). */
  pre(a, b, pat = this.pattern(a, b)) {
    const y = Float32Array.from(this.c0);
    for (let h = 0; h < this.H; h++) {
      const [pa, pb, p2] = pat[h];
      const A = this.PA[h][a], B = this.PB[h][b], C = this.P2[h];
      for (let i = 0; i < this.n; i++) y[i] += pa * A[i] + pb * B[i] + p2 * C[i];
    }
    return y;
  }

  /** Logits at '=' over all V outputs for input (a, b) (index p is the '=' output). */
  logits(a, b) {
    const pat = this.pattern(a, b);
    const post = this.pre(a, b, pat).map((v) => (v > 0 ? v : 0)); // ReLU
    const L = Float32Array.from(this.U0);
    for (let h = 0; h < this.H; h++) {
      const [pa, pb, p2] = pat[h];
      const A = this.UA[h][a], B = this.UB[h][b], C = this.U2[h];
      for (let c = 0; c < this.V; c++) L[c] += pa * A[c] + pb * B[c] + p2 * C[c];
    }
    for (let c = 0; c < this.V; c++) { let s = 0; const off = c * this.n; for (let m = 0; m < this.n; m++) s += this.WL[off + m] * post[m]; L[c] += s; }
    return L;
  }

  /** p×p grid (row a, column b) of attention from '=' to position `key` (0 = a, 1 = b, 2 = '=') for head h. */
  attentionGrid(h, key = 0) {
    const { p } = this, pat = this.patterns(), g = new Float32Array(p * p), off = h * p * p;
    for (let i = 0; i < p * p; i++) g[i] = pat[3 * (off + i) + key];
    return g;
  }

  /**
   * Attention from '=' to (a, b, '=') for every head and input, Float32Array [H, p·p, 3], computed once per model:
   * every attention and neuron grid reads it, so drawing many grids per frame costs no softmax at all.
   */
  patterns() {
    if (this._pat) return this._pat;
    const { p, H } = this, pat = new Float32Array(H * p * p * 3);
    for (let h = 0; h < H; h++) for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
      const x = this.sA[h][a], y = this.sB[h][b], z = this.s2[h], mx = Math.max(x, y, z); // stable softmax of 3 scores
      const ex = Math.exp(x - mx), ey = Math.exp(y - mx), ez = Math.exp(z - mx), s = ex + ey + ez, o = 3 * ((h * p + a) * p + b);
      pat[o] = ex / s; pat[o + 1] = ey / s; pat[o + 2] = ez / s;
    }
    return (this._pat = pat);
  }

  /** p×p grid of one neuron's activation (post-ReLU by default) at '='. */
  neuronGrid(m, post = true) {
    const { p, H } = this, pat = this.patterns(), g = new Float32Array(p * p);
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
      let v = this.c0[m];
      for (let h = 0; h < H; h++) {
        const o = 3 * ((h * p + a) * p + b);
        v += pat[o] * this.PA[h][a][m] + pat[o + 1] * this.PB[h][b][m] + pat[o + 2] * this.P2[h][m];
      }
      g[a * p + b] = post ? Math.max(v, 0) : v;
    }
    return g;
  }

  /**
   * Logits of the answers c = 0 … p−1 for every input (row a·p + b), from the cached attention patterns: [p·p·p].
   * `only(a, b)` restricts each row to one answer (returns [p·p] instead), e.g. the right answer for correct logits.
   */
  logitsGrid(only = null) {
    const { p, H, n } = this, pat = this.patterns(), post = new Float32Array(n);
    const out = new Float32Array(only ? p * p : p * p * p);
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
      for (let m = 0; m < n; m++) { // neuron pre-activation, then ReLU (as `pre` + `logits`, without per-input allocations)
        let v = this.c0[m];
        for (let h = 0; h < H; h++) { const o = 3 * ((h * p + a) * p + b); v += pat[o] * this.PA[h][a][m] + pat[o + 1] * this.PB[h][b][m] + pat[o + 2] * this.P2[h][m]; }
        post[m] = v > 0 ? v : 0;
      }
      const cs = only ? [only(a, b)] : null;
      for (let j = 0; j < (cs ? 1 : p); j++) {
        const c = cs ? cs[0] : j;
        let L = this.U0[c];
        for (let h = 0; h < H; h++) { const o = 3 * ((h * p + a) * p + b); L += pat[o] * this.UA[h][a][c] + pat[o + 1] * this.UB[h][b][c] + pat[o + 2] * this.U2[h][c]; }
        const w = c * n;
        for (let m = 0; m < n; m++) L += this.WL[w + m] * post[m];
        out[cs ? a * p + b : (a * p + b) * p + c] = L;
      }
    }
    return out;
  }

  /**
   * Full forward pass with every intermediate at every position (for the step-by-step walkthrough).
   * Mirrors src/grokviz/model.py; returns plain arrays for display.
   */
  forwardDetailed(a, b) {
    const { p, H, dh, d, n, V } = this;
    const scale = 1 / Math.sqrt(dh);
    const toks = [a, b, p];
    const x = toks.map((t, pos) => this.resid0(t, pos)); // residual stream after embeddings
    const qkvs = x.map((xi) => this.qkv(xi));
    const patterns = []; // [H][3 queries][3 keys]
    const attnOut = x.map(() => new Float32Array(d));
    for (let h = 0; h < H; h++) {
      const rows = [];
      for (let q = 0; q < 3; q++) {
        const scores = [0, 1, 2].map((k) => (k <= q ? dot(qkvs[q].q[h], qkvs[k].k[h]) * scale : -Infinity)); // causal mask
        const pat = softmax(scores.map((s) => (s === -Infinity ? -1e30 : s)));
        rows.push(pat);
        const z = new Float32Array(dh);
        for (let k = 0; k <= q; k++) for (let j = 0; j < dh; j++) z[j] += pat[k] * qkvs[k].v[h][j];
        const o = this.ov(h, z);
        for (let i = 0; i < d; i++) attnOut[q][i] += o[i];
      }
      patterns.push(rows);
    }
    const residMid = x.map((xi, q) => xi.map((v, i) => v + attnOut[q][i]));
    const pre = GrokModel.matvec(this.Win.data, 0, n, d, residMid[2]).map((v, i) => v + this.bin.data[i]);
    const post = pre.map((v) => (v > 0 ? v : 0));
    const mlpOut = GrokModel.matvec(this.Wout.data, 0, d, n, post).map((v, i) => v + this.bout.data[i]);
    const residPost = residMid[2].map((v, i) => v + mlpOut[i]);
    const logits = new Float32Array(V);
    for (let c = 0; c < V; c++) { let s = 0; for (let i = 0; i < d; i++) s += this.WU.data[i * V + c] * residPost[i]; logits[c] = s; }
    return { tokens: toks, resid0: x, qkv: qkvs, patterns, attnOut, residMid, pre, post, mlpOut, residPost, logits };
  }
}

/** Names of the precomputed tables (see `_precompute`), the state `toTables` / `fromTables` hand over. */
const TABLES = ["s2", "sA", "sB", "c0", "PA", "PB", "P2", "U0", "UA", "UB", "U2", "WL"];

/** Dot product of two equal-length arrays. */
export function dot(u, v) {
  let s = 0;
  for (let i = 0; i < u.length; i++) s += u[i] * v[i];
  return s;
}
