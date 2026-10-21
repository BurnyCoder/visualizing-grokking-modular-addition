// trig.js — teaching widgets for chapters 9 and 12: numbers as angles on the unit circle, the angle-addition
// identities, and constructive interference of cosine waves.
//
// Local explanation: `unit-circle` places a, b and (a + b) mod p at angles w_k·a, w_k·b, w_k·(a+b) and shows
// that the third is the sum of the first two; `angle-addition` evaluates both sides of the identities;
// `interference-toy` sums cos(w_k(a+b−c)) over a chosen set of frequencies and shows the single sharp peak.
// Global context: the mathematical core of the Fourier multiplication algorithm, before seeing it in the network.
// Sources: angle-sum identities (https://en.wikipedia.org/wiki/List_of_trigonometric_identities); paper Sec. 3.1
// and App. B (https://arxiv.org/abs/2301.05217).
import { Plot } from "../viz/plot.js";
import { cssVar, freqColor } from "../viz/colors.js";
import { frame, slider, svg } from "./basics.js";

const f4 = (x) => x.toFixed(4);

export const widgets = {
  "unit-circle": {
    render(el) {
      const { controls, out, work } = frame(el, "Numbers as angles: a, b and a + b on the unit circle (p = 113)");
      const s = svg("svg", { viewBox: "-1.35 -1.35 2.7 2.7", width: 280, height: 280 });
      out.append(s);
      let k = 14, a = 100, b = 50;
      const P = 113;
      const draw = () => {
        s.replaceChildren();
        svg("circle", { r: 1, fill: "none", stroke: cssVar("--line"), "stroke-width": 0.015 }, s);
        for (let x = 0; x < P; x++) { const t = (2 * Math.PI * k * x) / P; svg("circle", { cx: Math.cos(t), cy: -Math.sin(t), r: 0.012, fill: cssVar("--ink-soft") }, s); }
        const point = (v, color, label) => {
          const t = (2 * Math.PI * k * v) / P;
          svg("line", { x1: 0, y1: 0, x2: Math.cos(t), y2: -Math.sin(t), stroke: color, "stroke-width": 0.03 }, s);
          svg("circle", { cx: Math.cos(t), cy: -Math.sin(t), r: 0.045, fill: color }, s);
          svg("text", { x: 1.15 * Math.cos(t), y: -1.15 * Math.sin(t) + 0.04, "font-size": 0.12, "text-anchor": "middle", fill: color }, s).textContent = label;
        };
        point(a, cssVar("--train"), `a=${a}`);
        point(b, cssVar("--restricted"), `b=${b}`);
        point((a + b) % P, cssVar("--accent"), `a+b=${(a + b) % P}`);
        const w = (2 * Math.PI * k) / P;
        work.textContent = `w_k = 2π·${k}/113 = ${f4(w)} rad\na: angle w·a ↔ position (${k}·${a}) mod 113 = ${(k * a) % P} of 113 → (cos, sin) = (${f4(Math.cos(w * a))}, ${f4(Math.sin(w * a))})\nb: position (${k}·${b}) mod 113 = ${(k * b) % P} → (${f4(Math.cos(w * b))}, ${f4(Math.sin(w * b))})\na+b: position ${((k * a) % P + (k * b) % P) % P} = (${(k * a) % P} + ${(k * b) % P}) mod 113 — the angles add.  Answer (a+b) mod 113 = ${(a + b) % P}`;
      };
      slider(controls, "frequency k", 1, 56, 1, k, (v) => { k = v; draw(); });
      slider(controls, "a", 0, 112, 1, a, (v) => { a = v; draw(); });
      slider(controls, "b", 0, 112, 1, b, (v) => { b = v; draw(); });
    },
  },

  "angle-addition": {
    render(el) {
      const { controls, work } = frame(el, "Check the angle-addition identities");
      let x = 0.5, y = 0.3;
      const draw = () => {
        const lc = Math.cos(x + y), rc = Math.cos(x) * Math.cos(y) - Math.sin(x) * Math.sin(y);
        const ls = Math.sin(x + y), rs = Math.sin(x) * Math.cos(y) + Math.cos(x) * Math.sin(y);
        work.textContent = `cos(x+y) = cos(${x + y >= 0 ? "" : ""}${(x + y).toFixed(2)}) = ${f4(lc)}\ncos x·cos y − sin x·sin y = ${f4(Math.cos(x))}·${f4(Math.cos(y))} − ${f4(Math.sin(x))}·${f4(Math.sin(y))} = ${f4(rc)}\n\nsin(x+y) = ${f4(ls)}\nsin x·cos y + cos x·sin y = ${f4(rs)}`;
      };
      slider(controls, "x", -3.14, 3.14, 0.01, x, (v) => { x = v; draw(); });
      slider(controls, "y", -3.14, 3.14, 0.01, y, (v) => { y = v; draw(); });
    },
  },

  "interference-toy": {
    render(el) {
      const { controls, out, work } = frame(el, "Constructive interference: add cos(w_k(a+b−c)) over several frequencies (all weights 1)");
      const plot = new Plot(out, { height: 280 });
      let a = 100, b = 50, kset = "14,35,41,42,52";
      const input = document.createElement("label");
      input.innerHTML = 'frequencies <input type="text" value="14,35,41,42,52" size="16">';
      controls.append(input);
      const P = 113;
      const draw = () => {
        const ks = kset.split(",").map((t) => parseInt(t, 10)).filter((k) => k >= 1 && k <= 56);
        const cs = Array.from({ length: P }, (_, c) => c);
        const waves = ks.map((k) => cs.map((c) => Math.cos((2 * Math.PI * k * (a + b - c)) / P)));
        const sum = cs.map((c) => waves.reduce((acc, w) => acc + w[c], 0));
        const traces = ks.map((k, j) => ({ x: cs, y: waves[j], type: "scatter", mode: "lines", name: `k=${k}`, line: { color: freqColor(k), width: 1 } }));
        traces.push({ x: cs, y: sum, type: "scatter", mode: "lines", name: "sum", line: { color: cssVar("--ink"), width: 3 } });
        plot.draw(traces, { xaxis: { title: { text: "candidate answer c" } }, yaxis: { title: { text: "score" } } });
        const star = (a + b) % P;
        const best = sum.map((v, c) => [v, c]).filter(([, c]) => c !== star).sort((u, v) => v[0] - u[0])[0];
        work.textContent = `correct answer c* = ${star}: sum = ${sum[star].toFixed(3)} (every wave equals 1 there)\nbest wrong answer: c = ${best[1]} with sum ${best[0].toFixed(3)}\nTry a single frequency (e.g. 14): several answers score almost as high as c*.`;
      };
      input.querySelector("input").addEventListener("change", (e) => { kset = e.target.value; draw(); });
      slider(controls, "a", 0, 112, 1, a, (v) => { a = v; draw(); });
      slider(controls, "b", 0, 112, 1, b, (v) => { b = v; draw(); });
    },
  },
};
