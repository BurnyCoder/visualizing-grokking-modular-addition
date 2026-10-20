// basics.js — teaching widgets for chapters 1–7 and 11: clock arithmetic, dot products, softmax/cross-entropy,
// gradient descent with weight decay, and the Gini coefficient.
//
// Local explanation: each widget builds its own small UI (sliders + SVG or Plotly) and recomputes everything
// from first principles in the browser, showing the intermediate numbers ("workings") so readers can follow the
// arithmetic of the worked examples in the text.
// Global context: these widgets carry the "simplest example" side of each explanation; the figures carry the
// real network. Sources: softmax/cross-entropy (https://en.wikipedia.org/wiki/Softmax_function,
// https://en.wikipedia.org/wiki/Cross-entropy); AdamW (https://arxiv.org/abs/1711.05101); Gini index
// (Hurley & Rickard 2009, https://arxiv.org/abs/0811.4706).
import { Plot } from "../viz/plot.js";
import { cssVar } from "../viz/colors.js";
import { softmax } from "../model/grokmodel.js";

/** Build a widget frame: title, controls row, output area. */
export function frame(el, title) {
  el.innerHTML = `<p class="widget-title">${title}</p><div class="row controls"></div><div class="out"></div><div class="workings"></div>`;
  return { controls: el.querySelector(".controls"), out: el.querySelector(".out"), work: el.querySelector(".workings") };
}

/** Add a labelled range slider; returns the input element. */
export function slider(parent, label, min, max, step, value, onInput) {
  const lab = document.createElement("label");
  lab.innerHTML = `${label} <input type="range" min="${min}" max="${max}" step="${step}" value="${value}"> <output class="mono"></output>`;
  const input = lab.querySelector("input"), out = lab.querySelector("output");
  const update = () => { out.textContent = input.value; onInput(Number(input.value)); };
  input.addEventListener("input", update);
  parent.append(lab);
  queueMicrotask(update);
  return input;
}

const f3 = (x) => (Math.abs(x) < 5e-4 ? "0.000" : x.toFixed(3));
const NS = "http://www.w3.org/2000/svg";
/** Create an SVG element with attributes. */
export function svg(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
}

export const widgets = {
  clock: {
    render(el) {
      const { controls, out, work } = frame(el, "A clock with p hours");
      const s = svg("svg", { viewBox: "-120 -120 240 240", width: 260, height: 260 });
      out.append(s);
      let p = 12, a = 9, b = 4;
      const draw = () => {
        s.replaceChildren();
        svg("circle", { r: 100, fill: "none", stroke: cssVar("--line"), "stroke-width": 2 }, s);
        const step = p <= 24 ? 1 : Math.ceil(p / 24);
        for (let i = 0; i < p; i += step) { // tick labels around the face, 0 at the top
          const t = (2 * Math.PI * i) / p - Math.PI / 2;
          svg("text", { x: 86 * Math.cos(t), y: 86 * Math.sin(t) + 4, "text-anchor": "middle", "font-size": p > 24 ? 9 : 11, fill: cssVar("--ink-soft") }, s).textContent = i;
        }
        const hand = (v, color, w) => { const t = (2 * Math.PI * v) / p - Math.PI / 2; svg("line", { x1: 0, y1: 0, x2: 70 * Math.cos(t), y2: 70 * Math.sin(t), stroke: color, "stroke-width": w, "stroke-linecap": "round" }, s); };
        hand(a, cssVar("--train"), 3);
        hand((a + b) % p, cssVar("--accent"), 5);
        const q = Math.floor((a + b) / p);
        work.textContent = `${a} + ${b} = ${a + b} = ${q} × ${p} + ${(a + b) % p}\nso (${a} + ${b}) mod ${p} = ${(a + b) % p}   (blue hand: a, orange hand: the answer)`;
      };
      const pa = slider(controls, "p", 2, 113, 1, p, (v) => { p = v; pa2.max = p - 1; pb.max = p - 1; a = Math.min(a, p - 1); b = Math.min(b, p - 1); draw(); });
      const pa2 = slider(controls, "a", 0, p - 1, 1, a, (v) => { a = v; draw(); });
      const pb = slider(controls, "b", 0, p - 1, 1, b, (v) => { b = v; draw(); });
      void pa;
    },
  },

  "dot-product": {
    render(el) {
      const { controls, out, work } = frame(el, "Dot product, length and angle of two vectors");
      const s = svg("svg", { viewBox: "-6 -6 12 12", width: 260, height: 260 });
      out.append(s);
      const v = { ux: 1, uy: 2, vx: 3, vy: 4 };
      const draw = () => {
        s.replaceChildren();
        svg("line", { x1: -6, y1: 0, x2: 6, y2: 0, stroke: cssVar("--line"), "stroke-width": 0.05 }, s);
        svg("line", { x1: 0, y1: -6, x2: 0, y2: 6, stroke: cssVar("--line"), "stroke-width": 0.05 }, s);
        svg("line", { x1: 0, y1: 0, x2: v.ux, y2: -v.uy, stroke: cssVar("--train"), "stroke-width": 0.15 }, s);
        svg("line", { x1: 0, y1: 0, x2: v.vx, y2: -v.vy, stroke: cssVar("--accent"), "stroke-width": 0.15 }, s);
        const dot = v.ux * v.vx + v.uy * v.vy, nu = Math.hypot(v.ux, v.uy), nv = Math.hypot(v.vx, v.vy);
        const cos = nu && nv ? dot / (nu * nv) : 0;
        const pr = nv ? dot / (nv * nv) : 0; // projection of u onto v's direction: (u·v/|v|²) v
        svg("line", { x1: 0, y1: 0, x2: pr * v.vx, y2: -pr * v.vy, stroke: cssVar("--restricted"), "stroke-width": 0.08, "stroke-dasharray": "0.2 0.15" }, s);
        work.textContent = `u = (${v.ux}, ${v.uy})  v = (${v.vx}, ${v.vy})\nu·v = ${v.ux}·${v.vx} + ${v.uy}·${v.vy} = ${dot}\n‖u‖ = ${f3(nu)}  ‖v‖ = ${f3(nv)}  cos(angle) = u·v/(‖u‖‖v‖) = ${f3(cos)}\nprojection of u on v (green, dashed) = (u·v/‖v‖²)·v = ${f3(pr)}·v`;
      };
      for (const k of ["ux", "uy", "vx", "vy"]) slider(controls, k.replace("x", "₁").replace("y", "₂"), -5, 5, 1, v[k], (x) => { v[k] = x; draw(); });
    },
  },

  softmax: {
    render(el) {
      const { controls, out, work } = frame(el, "From logits to probabilities to loss");
      const plot = new Plot(out, { height: 220 });
      const L = [2, 1, 0, -1];
      let correct = 0, temp = 1;
      const draw = () => {
        const p = softmax(L.map((x) => x * temp));
        plot.draw([{ x: ["answer 0", "answer 1", "answer 2", "answer 3"], y: p, type: "bar", marker: { color: p.map((_, i) => (i === correct ? cssVar("--accent") : cssVar("--train"))) } }],
          { yaxis: { range: [0, 1], title: { text: "probability" } }, hovermode: "closest" });
        const ex = L.map((x) => Math.exp(x * temp)), z = ex.reduce((a, b) => a + b, 0);
        work.textContent = `logits × ${temp}: (${L.map((x) => f3(x * temp)).join(", ")})\nexp: (${ex.map(f3).join(", ")}), sum ${f3(z)}\nprobabilities: (${p.map(f3).join(", ")})  — they add up to 1\ncross-entropy for correct answer ${correct}: −ln(${f3(p[correct])}) = ${f3(-Math.log(p[correct]))}   (uniform guess over 4: ln 4 = 1.386)`;
      };
      L.forEach((x, i) => slider(controls, `logit ${i}`, -4, 4, 0.5, x, (v) => { L[i] = v; draw(); }));
      slider(controls, "scale all logits", 0, 5, 0.25, 1, (v) => { temp = v; draw(); });
      slider(controls, "correct answer", 0, 3, 1, 0, (v) => { correct = v; draw(); });
    },
  },

  "gd-toy": {
    render(el) {
      const { controls, out, work } = frame(el, "Gradient descent on a one-parameter loss, with and without weight decay");
      const plot = new Plot(out, { height: 260 });
      let lr = 0.25, wd = 0, start = 5, steps = 12;
      const loss = (w) => (w - 3) ** 2; // toy loss with its minimum at w = 3
      const draw = () => {
        const path = [start];
        for (let i = 0; i < steps; i++) { const w = path.at(-1); path.push(w * (1 - lr * wd) - lr * 2 * (w - 3)); } // decoupled decay + gradient step
        const xs = Array.from({ length: 121 }, (_, i) => -1 + i * 0.07);
        plot.draw([
          { x: xs, y: xs.map(loss), type: "scatter", mode: "lines", name: "loss (w − 3)²", line: { color: cssVar("--ink-soft") } },
          { x: path, y: path.map(loss), type: "scatter", mode: "lines+markers", name: "steps", line: { color: cssVar("--accent") } },
        ], { xaxis: { title: { text: "parameter w" } }, yaxis: { title: { text: "loss" } }, hovermode: "closest" });
        work.textContent = path.slice(0, 5).map((w, i) => `step ${i}: w = ${f3(w)}, slope 2(w−3) = ${f3(2 * (w - 3))}`).join("\n") +
          `\n…\nafter ${steps} steps: w = ${f3(path.at(-1))}` + (wd ? `   (weight decay pulls the answer below 3, toward 0: it settles at 3/(1 + λ/2) = ${f3(3 / (1 + wd / 2))})` : "");
      };
      slider(controls, "learning rate γ", 0.05, 0.95, 0.05, lr, (v) => { lr = v; draw(); });
      slider(controls, "weight decay λ", 0, 2, 0.1, wd, (v) => { wd = v; draw(); });
      slider(controls, "start w", -1, 7, 0.5, start, (v) => { start = v; draw(); });
      slider(controls, "steps", 1, 40, 1, steps, (v) => { steps = v; draw(); });
    },
  },

  "gini-toy": {
    render(el) {
      const { controls, out, work } = frame(el, "The Gini coefficient of eight numbers");
      const plot = new Plot(out, { height: 200 });
      const c = [1, 1, 1, 1, 1, 1, 1, 1];
      const draw = () => {
        const sorted = [...c].sort((x, y) => x - y), N = c.length, tot = sorted.reduce((a, b) => a + b, 0) || 1;
        const G = 1 - 2 * sorted.reduce((acc, ck, k) => acc + (ck / tot) * ((N - (k + 1) + 0.5) / N), 0);
        plot.draw([{ x: c.map((_, i) => `c${i + 1}`), y: c, type: "bar", marker: { color: cssVar("--train") } }], { yaxis: { range: [0, 10] }, hovermode: "closest" });
        work.textContent = `sorted: (${sorted.join(", ")}), total ${tot}\nG = 1 − 2 Σ_k (c_(k)/total)·(N − k + ½)/N = ${f3(G)}   (0 = all equal, max 1 − 1/N = 0.875)`;
      };
      c.forEach((x, i) => slider(controls, `c${i + 1}`, 0, 10, 1, x, (v) => { c[i] = v; draw(); }));
    },
  },
};
