// interventions.js — dashboard group G: causal tests (ablations, patching) on the trained network.
//
// Local explanation: an ablation playground (choose frequencies to remove from, or keep in, the logits of every input
// and see the loss, accuracy and per-input confidence, computed in the browser; model/ablation.js); bar charts of exported final-checkpoint experiments (src/grokviz/hypotheses.py and
// rchecks.py): loss after removing each Fourier frequency from the logits, with the random-control range;
// recovered logit difference when patching the key-frequency part of a's embedding with a′'s (vs controls);
// plus a live patching demo for the selected (a, a′, b) at the selected step, computed in the browser.
// Global context: the paper's ablation evidence (Sec. 4.4, Fig. 6) and our counterfactual swap test.
// Sources: paper Sec. 4.4 (https://arxiv.org/abs/2301.05217); interchange interventions, Geiger et al. 2021
// (https://arxiv.org/abs/2106.02997).
import { ablationData, model, run } from "../core/data.js";
import { Plot } from "../viz/plot.js";
import { Heatmap } from "../viz/heatmap.js";
import { cssVar } from "../viz/colors.js";
import { softmax } from "../model/grokmodel.js";
import { fourierBasis } from "../model/fourier.js";
import { ablate } from "../model/ablation.js";
import { el, frameKey, note, runOf, set, watch } from "./common.js";

export const figures = {
  "ablation-playground": {
    async render(body, ds) {
      body.innerHTML = "";
      const P = 113, ctl = el(body, "div", "interf-ctl");
      const mode = el(ctl, "label", "hero-ctl", 'the chosen frequencies are <select><option value="remove">removed</option><option value="keep">the only ones kept</option></select>').querySelector("select");
      const presets = el(ctl, "span", "hero-chips");
      const chips = el(body, "div", "hero-chips abl-chips");
      const row = el(body, "div", "abl-row"), stats = el(row, "div", "abl-stats");
      const hm = new Heatmap(row, { rows: P, cols: P, size: 280, diverging: false, format: (v) => v.toFixed(3), onClick: (a, b) => set({ a, b }) });
      const info = note(body), chosen = new Set();
      const labels = Uint8Array.from({ length: P * P }, (_, i) => (Math.floor(i / P) + (i % P)) % P); // right answer a + b
      let data = null, keys = [];
      const compute = () => {
        if (!data) return;
        const base = ablate(data.L, data.blocks, P, labels, { remove: [] });
        const res = ablate(data.L, data.blocks, P, labels, mode.value === "keep" ? { keep: [...chosen] } : { remove: [...chosen] });
        stats.innerHTML = `<p class="abl-big">loss ${res.loss.toExponential(2)}</p><p class="fig-note">unablated ${base.loss.toExponential(2)} · ×${(res.loss / base.loss).toPrecision(3)}</p>`
          + `<p class="abl-big">accuracy ${(100 * res.acc).toFixed(1)} %</p><p class="fig-note">unablated ${(100 * base.acc).toFixed(1)} %</p>`;
        hm.draw(res.pCorrect, { min: 0, max: 1 });
        [...chips.children].forEach((c) => c.classList.toggle("active", chosen.has(Number(c.dataset.k))));
        const what = mode.value === "keep" ? `kept only ${chosen.size ? [...chosen].sort((x, y) => x - y).join(", ") : "nothing"}` : `removed ${chosen.size ? [...chosen].sort((x, y) => x - y).join(", ") : "nothing"}`;
        info.textContent = `Weights from step ${data.step.toLocaleString("en-US")}, ${what}. Grid: the probability of the right answer for every input after the ablation (rows a, columns b). Computed in your browser from float16 weights, so tiny losses differ a little from the analysis (frequency-ablation panel).`;
      };
      const choose = (ks) => { chosen.clear(); ks.forEach((k) => chosen.add(k)); compute(); };
      for (const [name, pick] of [["the key frequencies", () => keys], ["one key frequency", () => keys.slice(0, 1)], ["five other frequencies", () => Array.from({ length: 56 }, (_, i) => i + 1).filter((k) => !keys.includes(k)).slice(0, 5)], ["none", () => []]]) {
        presets.append(Object.assign(document.createElement("button"), { type: "button", className: "chip", textContent: name, onclick: () => choose(pick()) }));
      }
      mode.onchange = compute;
      watch(body, ["run", "step"], async (s) => {
        const id = runOf(ds), r = await run(id);
        if (r.label_seed != null) { data = null; info.textContent = "This run was trained on random labels, so there is no task answer to ablate against."; return; }
        if (keys.join() !== r.key_freqs.join()) { // one chip per frequency; the run's key frequencies outlined
          keys = r.key_freqs;
          chips.replaceChildren(...Array.from({ length: 56 }, (_, i) => Object.assign(document.createElement("button"), { type: "button", className: `chip${keys.includes(i + 1) ? " key" : ""}`, textContent: String(i + 1),
            title: `frequency k = ${i + 1}${keys.includes(i + 1) ? " (a key frequency of this run)" : ""}`, onclick: () => { if (chosen.has(i + 1)) chosen.delete(i + 1); else chosen.add(i + 1); compute(); } })));
          chips.querySelectorAll("button").forEach((b, i) => { b.dataset.k = i + 1; });
        }
        info.textContent = "Computing the logits of all 113 answers for all 12,769 inputs (in the background)…";
        data = await ablationData(id, s.step);
        compute();
      }, { dedupe: frameKey(ds, "weight_steps"), theme: false, minGap: 1000 });
    },
  },

  "freq-ablation-bars": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 280 });
      const info = note(body);
      watch(body, ["run"], async () => {
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
      watch(body, ["run"], async () => {
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
      watch(body, ["run"], async () => {
        const r = await run(runOf(ds));
        const pt = r.final?.patching;
        if (!pt) return;
        const names = ["key", "random", "complement"], labels = [`key frequencies<br>(${pt.key.dims} dims)`, `random<br>(${pt.random.dims} dims)`, `everything else<br>(${pt.complement.dims} dims)`]; // <br>: two-line tick labels fit narrow panels
        await plot.draw([{ x: labels, y: names.map((n) => pt[n].mean), type: "bar", marker: { color: [cssVar("--restricted"), cssVar("--ink-soft"), cssVar("--excluded")] },
          error_y: { type: "data", symmetric: false, array: names.map((n) => pt[n].ci95[1] - pt[n].mean), arrayminus: names.map((n) => pt[n].mean - pt[n].ci95[0]) } }],
        { yaxis: { title: { text: "fraction of logit difference recovered" }, range: [-0.1, 1.1] }, xaxis: { tickangle: 0, automargin: true }, hovermode: "closest" });
      });
    },
  },

  "patching-live": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 260 });
      const info = note(body);
      watch(body, ["run", "step", "a", "b", "ap"], async (s) => {
        const id = runOf(ds), r = await run(id), net = await model(id, s.step);
        const p = net.p;
        const clean = Array.from(net.logits(s.a, s.b)).slice(0, p);
        const patched = patchedLogits(net, r.key_freqs, s.a, s.ap, s.b).slice(0, p);
        const cs = Array.from({ length: p }, (_, c) => c);
        await plot.draw([
          { x: cs, y: softmax(clean), type: "scatter", mode: "lines", name: `clean input (${s.a}, ${s.b})`, line: { color: cssVar("--train") } },
          { x: cs, y: softmax(patched), type: "scatter", mode: "lines", name: `key part of a swapped with a′ = ${s.ap}`, line: { color: cssVar("--test") } },
        ], { xaxis: { title: { text: "answer c" } }, yaxis: { title: { text: "probability" }, range: [0, 1.02] }, hovermode: "x unified" });
        info.textContent = `Expected answer after the swap: ${s.ap} + ${s.b} = ${(s.ap + s.b) % p} (mod ${p}); original answer ${(s.a + s.b) % p}. Step ${net.step}.${s.a === s.b ? " Here b = a, so b's embedding is patched too (the demo edits a's column of W_E; patching.py patches position 0 only)." : ""}`;
      }, { dedupe: frameKey(ds, "weight_steps", "a", "b", "ap") });
    },
  },
};

/**
 * Logits after replacing the key-frequency component of token a's embedding by token a′'s (JS version of patching.py).
 * It edits column a of W_E, so when b = a the token at position 1 is patched as well; patching.py patches position 0 only.
 */
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
