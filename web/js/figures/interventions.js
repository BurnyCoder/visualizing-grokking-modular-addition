// interventions.js — dashboard group G: causal tests (ablations, patching) on the trained network.
//
// Local explanation: bar charts of exported final-checkpoint experiments (src/grokviz/hypotheses.py and
// rchecks.py): loss after removing each Fourier frequency from the logits, with the random-control range;
// recovered logit difference when patching the key-frequency part of a's embedding with a′'s (vs controls);
// plus a live patching demo for the selected (a, a′, b) at the selected step, computed in the browser.
// Global context: the paper's ablation evidence (Sec. 4.4, Fig. 6) and our counterfactual swap test.
// Sources: paper Sec. 4.4 (https://arxiv.org/abs/2301.05217); interchange interventions, Geiger et al. 2021
// (https://arxiv.org/abs/2106.02997).
import { model, run } from "../core/data.js";
import { Plot } from "../viz/plot.js";
import { cssVar } from "../viz/colors.js";
import { softmax } from "../model/grokmodel.js";
import { fourierBasis } from "./weights.js";
import { note, runOf, watch } from "./common.js";

export const figures = {
  "freq-ablation-bars": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 280 });
      const info = note(body);
      watch(["run"], async () => {
        const r = await run(runOf(ds));
        const fin = r.final;
        if (!fin) { info.textContent = "No final analysis for this run yet."; return; }
        const a = fin.ablation, ks = a.single_freq.map((_, i) => i + 1);
        const rand = a.random_nonkey_removed.slice().sort((x, y) => x - y);
        await plot.draw([
          { x: ks, y: a.single_freq, type: "bar", name: "remove one frequency", marker: { color: ks.map((k) => (fin.key_freqs.includes(k) ? cssVar("--test") : cssVar("--train"))) } },
        ], { yaxis: { type: "log", title: { text: "loss after ablation (log)" }, exponentformat: "power" }, xaxis: { title: { text: "frequency k removed from the logits" }, dtick: 4 }, hovermode: "closest",
          shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: a.original, y1: a.original, line: { dash: "dot", color: cssVar("--ink") } }] });
        info.innerHTML = `Red bars = key frequencies (${fin.key_freqs.join(", ")}). Dotted line = unablated loss ${a.original.toExponential(2)}. ` +
          `Keeping only the key frequencies' 8·${fin.key_freqs.length} components: loss ${a.only_key_kept.toExponential(2)}. ` +
          `Removing ${fin.key_freqs.length} random non-key frequencies (100 draws): loss between ${rand[0].toExponential(2)} and ${rand.at(-1).toExponential(2)}.`;
      });
    },
  },

  "cluster-ablations": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 260 });
      const info = note(body);
      watch(["run"], async () => {
        const r = await run(runOf(ds));
        const c = r.final?.cluster_ablation;
        if (!c) { info.textContent = "No cluster analysis for this run yet."; return; }
        // one bar per neuron cluster (its frequency k, or "-1" = neurons with no dominant frequency), sorted by effect
        const keys = Object.keys(c.loss_excluding_cluster).filter((k) => c.cluster_sizes[k] > 0)
          .sort((x, y) => c.loss_excluding_cluster[y] - c.loss_excluding_cluster[x]);
        const name = (k) => (k === "-1" ? "other neurons" : `k = ${k}`);
        const times = (x) => `${Number(x.toPrecision(2)).toLocaleString("en-US")}×`; // 803.4 → "800×"
        await plot.draw([{ x: keys.map(name), y: keys.map((k) => c.loss_excluding_cluster[k]), type: "bar",
          marker: { color: keys.map((k) => (k === "-1" ? cssVar("--train") : cssVar("--test"))) },
          hovertemplate: keys.map((k) => `${name(k)}: ${c.cluster_sizes[k]} neurons<br>loss %{y:.3g}<extra></extra>`) }],
        { yaxis: { type: "log", title: { text: "loss without this cluster (log)" }, exponentformat: "power" }, xaxis: { type: "category" },
          shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: c.loss_key_projection, y1: c.loss_key_projection, line: { dash: "dot", color: cssVar("--ink") } }] });
        const base = c.loss_key_projection, ratio = (k) => c.loss_excluding_cluster[k] / base;
        const keyRatios = keys.filter((k) => k !== "-1").map(ratio), rest = keys.includes("-1") ? ` and removing the other neurons (no dominant frequency) by ${times(ratio("-1"))}` : "";
        info.textContent = `Each neuron cluster's contribution to the logits is removed in turn (clusters: neurons whose activations are > 85 % one frequency; each restricted to that frequency's terms, ARENA 1.5.2). Dotted line: loss with all clusters kept, ${base.toExponential(2)}. ` +
          (keyRatios.length ? `Removing one frequency's cluster multiplies the loss by ${times(Math.min(...keyRatios))} to ${times(Math.max(...keyRatios))}${rest}.` : "");
      });
    },
  },

  "patching-result": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 240 });
      watch(["run"], async () => {
        const r = await run(runOf(ds));
        const pt = r.final?.patching;
        if (!pt) return;
        const names = ["key", "random", "complement"], labels = [`key-frequency subspace (${pt.key.dims} dims)`, `random subspace (${pt.random.dims} dims)`, `everything else (${pt.complement.dims} dims)`];
        await plot.draw([{ x: labels, y: names.map((n) => pt[n].mean), type: "bar", marker: { color: [cssVar("--restricted"), cssVar("--ink-soft"), cssVar("--excluded")] },
          error_y: { type: "data", symmetric: false, array: names.map((n) => pt[n].ci95[1] - pt[n].mean), arrayminus: names.map((n) => pt[n].mean - pt[n].ci95[0]) } }],
        { yaxis: { title: { text: "fraction of logit difference recovered" }, range: [-0.1, 1.1] }, hovermode: "closest" });
      });
    },
  },

  "patching-live": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 260 });
      const info = note(body);
      watch(["run", "step", "a", "b", "ap"], async (s) => {
        const id = runOf(ds), r = await run(id), net = await model(id, s.step);
        const p = net.p;
        const clean = Array.from(net.logits(s.a, s.b)).slice(0, p);
        const patched = patchedLogits(net, r.key_freqs, s.a, s.ap, s.b).slice(0, p);
        const cs = Array.from({ length: p }, (_, c) => c);
        await plot.draw([
          { x: cs, y: softmax(clean), type: "scatter", mode: "lines", name: `clean input (${s.a}, ${s.b})`, line: { color: cssVar("--train") } },
          { x: cs, y: softmax(patched), type: "scatter", mode: "lines", name: `key part of a swapped with a′ = ${s.ap}`, line: { color: cssVar("--test") } },
        ], { xaxis: { title: { text: "answer c" } }, yaxis: { title: { text: "probability" }, range: [0, 1.02] }, hovermode: "x unified" });
        info.textContent = `Expected answer after the swap: ${s.ap} + ${s.b} = ${(s.ap + s.b) % p} (mod ${p}); original answer ${(s.a + s.b) % p}. Step ${net.step}.`;
      });
    },
  },
};

/** Logits after replacing the key-frequency component of token a's embedding by token a′'s (JS version of patching.py). */
export function patchedLogits(net, freqs, a, ap, b) {
  const { d, V, p } = net;
  const F = fourierBasis(p);
  const cols = []; // residual-space directions W_E F_i^T for the key cos/sin rows
  for (const k of freqs) for (const i of [2 * k - 1, 2 * k]) {
    const v = new Float32Array(d);
    for (let r = 0; r < d; r++) { let s = 0; for (let t = 0; t < p; t++) s += net.WE.data[r * V + t] * F[i * p + t]; v[r] = s; }
    cols.push(v);
  }
  const Q = gramSchmidt(cols); // orthonormal basis of the key subspace
  const e = (t) => Float32Array.from({ length: d }, (_, r) => net.WE.data[r * V + t]);
  const ea = e(a), eap = e(ap);
  const x = Float32Array.from(ea);
  for (const q of Q) { // x = e_a − P e_a + P e_a′
    let ca = 0, cp = 0;
    for (let r = 0; r < d; r++) { ca += q[r] * ea[r]; cp += q[r] * eap[r]; }
    for (let r = 0; r < d; r++) x[r] += (cp - ca) * q[r];
  }
  const backup = new Float32Array(d); // temporarily overwrite column a of W_E, run the exact forward pass, restore
  for (let r = 0; r < d; r++) { backup[r] = net.WE.data[r * V + a]; net.WE.data[r * V + a] = x[r]; }
  const out = Array.from(net.forwardDetailed(a, b).logits);
  for (let r = 0; r < d; r++) net.WE.data[r * V + a] = backup[r];
  return out;
}

function gramSchmidt(vs) {
  const out = [];
  for (const v of vs) {
    const u = Float32Array.from(v);
    for (const q of out) { let c = 0; for (let i = 0; i < u.length; i++) c += q[i] * u[i]; for (let i = 0; i < u.length; i++) u[i] -= c * q[i]; }
    const n = Math.hypot(...u);
    if (n > 1e-8) out.push(u.map((x) => x / n));
  }
  return out;
}
