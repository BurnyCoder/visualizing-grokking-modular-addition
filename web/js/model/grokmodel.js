// grokmodel.js — the paper's one-layer transformer, re-implemented in plain JavaScript for the browser.
//
// Local explanation: loads a flat weight vector (state_dict order, layout from manifest.json) and computes
// the forward pass exactly like src/grokviz/model.py (paper code layout: W_E [d, V], W_K/W_Q/W_V [H, dh, d],
// W_O [d, H*dh], W_in [n, d], W_out [d, n], W_U [d, V]). Because inputs are one-hot tokens, everything before
// the softmax/ReLU depends on one token at a time, so we precompute per-token tables once per checkpoint and
// then evaluate any (a, b) cheaply — fast enough to redraw 113×113 grids while the user scrubs through training.
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
   * @param {Float32Array} flat  weights in state_dict order
   * @param {Array<{name:string, shape:number[]}>} layout  parameter layout from manifest.json
   * @param {number} p  modulus (113); token p is '='
   */
  constructor(flat, layout, p) {
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
    this._precompute();
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

  /** Precompute token tables for the '=' position (everything the output at position 2 depends on). */
  _precompute() {
    const { p, H, dh, d, n, V } = this;
    const scale = 1 / Math.sqrt(dh);
    this.x2 = this.resid0(p, 2); // '=' token at position 2 — the same for every input
    const qkv2 = this.qkv(this.x2);
    this.q2 = qkv2.q; // queries from '=' (constant)
    this.s2 = qkv2.q.map((q, h) => dot(q, qkv2.k[h]) * scale); // score '=' → '='
    this.OV2 = qkv2.v.map((v, h) => this.ov(h, v)); // what each head writes when attending to '='
    this.sA = []; this.sB = []; this.OVA = []; this.OVB = []; // per head: scores and OV outputs for tokens at pos 0 / 1
    for (let h = 0; h < H; h++) {
      this.sA.push(new Float32Array(p)); this.sB.push(new Float32Array(p));
      this.OVA.push([]); this.OVB.push([]);
    }
    for (let t = 0; t < p; t++) {
      for (const [pos, S, OV] of [[0, this.sA, this.OVA], [1, this.sB, this.OVB]]) {
        const x = this.resid0(t, pos);
        const kv = this.qkv(x);
        for (let h = 0; h < H; h++) {
          S[h][t] = dot(this.q2[h], kv.k[h]) * scale; // pre-softmax score from '=' to this token
          OV[h].push(this.ov(h, kv.v[h])); // what head h writes if it attends fully to this token
        }
      }
    }
    // Neuron-input contributions (linear, so they can be pre-multiplied by W_in).
    const Win = (x) => GrokModel.matvec(this.Win.data, 0, n, d, x);
    this.c0 = Win(this.x2).map((v, i) => v + this.bin.data[i]); // W_in x2 + b_in
    this.PA = this.OVA.map((row) => row.map(Win)); // W_in OV_A[h][a]
    this.PB = this.OVB.map((row) => row.map(Win));
    this.P2 = this.OV2.map(Win);
    // Logit read-out: W_U^T (x2 + b_out + attention + W_out post).
    const WUT = (x) => { // [V] = W_U^T x
      const y = new Float32Array(V);
      for (let c = 0; c < V; c++) { let s = 0; for (let i = 0; i < d; i++) s += this.WU.data[i * V + c] * x[i]; y[c] = s; }
      return y;
    };
    this.U0 = WUT(this.x2.map((v, i) => v + this.bout.data[i]));
    this.UA = this.OVA.map((row) => row.map(WUT));
    this.UB = this.OVB.map((row) => row.map(WUT));
    this.U2 = this.OV2.map(WUT);
    this.WL = new Float32Array(V * n); // W_U^T W_out [V, n]: the neuron-logit map (incl. the '=' row)
    for (let c = 0; c < V; c++) for (let m = 0; m < n; m++) {
      let s = 0; for (let i = 0; i < d; i++) s += this.WU.data[i * V + c] * this.Wout.data[i * n + m];
      this.WL[c * n + m] = s;
    }
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
    const { p } = this;
    const g = new Float32Array(p * p);
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) g[a * p + b] = softmax([this.sA[h][a], this.sB[h][b], this.s2[h]])[key];
    return g;
  }

  /** p×p grid of one neuron's activation (post-ReLU by default) at '='. */
  neuronGrid(m, post = true) {
    const { p, H } = this;
    const g = new Float32Array(p * p);
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
      let v = this.c0[m];
      for (let h = 0; h < H; h++) {
        const [pa, pb, p2] = softmax([this.sA[h][a], this.sB[h][b], this.s2[h]]);
        v += pa * this.PA[h][a][m] + pb * this.PB[h][b][m] + p2 * this.P2[h][m];
      }
      g[a * p + b] = post ? Math.max(v, 0) : v;
    }
    return g;
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

/** Dot product of two equal-length arrays. */
export function dot(u, v) {
  let s = 0;
  for (let i = 0; i < u.length; i++) s += u[i] * v[i];
  return s;
}
