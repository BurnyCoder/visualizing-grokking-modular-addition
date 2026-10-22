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
import { Heatmap } from "../viz/heatmap.js";
import { cssVar, freqColor } from "../viz/colors.js";
import { frame, onTheme, slider, svg, sweep } from "./basics.js";

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
      onTheme(draw);
      slider(controls, "frequency k", 1, 56, 1, k, (v) => { k = v; draw(); });
      sweep(slider(controls, "a", 0, 112, 1, a, (v) => { a = v; draw(); }), { seconds: 10 });
      slider(controls, "b", 0, 112, 1, b, (v) => { b = v; draw(); });
    },
  },

  "add-angles": {
    render(el) {
      const { controls, out, work } = frame(el, "Adding by turning: a, b and a + b as angles, and a candidate answer c");
      const s = svg("svg", { viewBox: "-1.45 -1.45 2.9 3.25", width: 290, height: 325 });
      out.append(s);
      let k = 1, a = 30, b = 50, c = 0;
      const P = 113, deg = (x) => ((360 * k * x) / P) % 360;
      const draw = () => {
        s.replaceChildren();
        const w = (2 * Math.PI * k) / P, tip = (x, r = 1) => [r * Math.cos(w * x), -r * Math.sin(w * x)];
        svg("circle", { r: 1, fill: "none", stroke: cssVar("--line"), "stroke-width": 0.015 }, s);
        for (let x = 0; x < P; x++) { const [u, v] = tip(x); svg("circle", { cx: u, cy: v, r: 0.012, fill: cssVar("--ink-soft") }, s); }
        const hand = (x, color, width, label, dash = "") => {
          const [u, v] = tip(x, 0.92), [lu, lv] = tip(x, 1.17);
          svg("line", { x1: 0, y1: 0, x2: u, y2: v, stroke: color, "stroke-width": width, "stroke-linecap": "round", "stroke-dasharray": dash }, s);
          svg("text", { x: lu, y: lv + 0.04, "font-size": 0.11, "text-anchor": "middle", fill: color }, s).textContent = label;
        };
        hand(c, cssVar("--ink-soft"), 0.025, `c=${c}`, "0.06 0.04");
        hand(a, cssVar("--train"), 0.035, `a=${a}`);
        hand(b, cssVar("--restricted"), 0.035, `b=${b}`);
        hand(a + b, cssVar("--accent"), 0.055, `a+b=${(a + b) % P}`);
        const val = Math.cos(w * (a + b - c)); // the score this frequency gives candidate c
        svg("line", { x1: -1, y1: 1.62, x2: 1, y2: 1.62, stroke: cssVar("--line"), "stroke-width": 0.01 }, s);
        svg("rect", { x: Math.min(0, val), y: 1.55, width: Math.abs(val), height: 0.14, fill: val > 0.99 ? cssVar("--accent") : cssVar("--ink-soft") }, s);
        svg("text", { x: 0, y: 1.5, "font-size": 0.1, "text-anchor": "middle", fill: cssVar("--ink") }, s).textContent = `cos(w_k(a+b−c)) = ${f4(val)}`;
        const gap = ((((deg(a + b) - deg(c)) % 360) + 360) % 360), apart = Math.min(gap, 360 - gap); // angle between the hands
        work.textContent = `w_k = 2π·${k}/113: a turns the hand by ${deg(a).toFixed(1)}°, b by ${deg(b).toFixed(1)}°; together ${((deg(a) + deg(b)) % 360).toFixed(1)}° — exactly the angle of a + b = ${(a + b) % P} (mod 113).
Candidate c = ${c} sits at ${deg(c).toFixed(1)}°, ${apart.toFixed(1)}° away from a + b, so cos(w_k(a+b−c)) = ${f4(val)}. It reaches 1 only at c = a + b; sweep c to watch.`;
      };
      onTheme(draw);
      slider(controls, "frequency k", 1, 56, 1, k, (v) => { k = v; draw(); });
      slider(controls, "a", 0, 112, 1, a, (v) => { a = v; draw(); });
      slider(controls, "b", 0, 112, 1, b, (v) => { b = v; draw(); });
      sweep(slider(controls, "c", 0, 112, 1, c, (v) => { c = v; draw(); }), { seconds: 10 });
    },
  },

  "trig-outer-product": {
    render(el) {
      const { controls, out, work } = frame(el, "cos(w a)·cos(w b) − sin(w a)·sin(w b) depends only on a + b");
      const wrap = Object.assign(document.createElement("div"), { className: "tiles3" }); // three grids side by side
      out.append(wrap);
      const P = 113, maps = ["cos(w a)·cos(w b)", "sin(w a)·sin(w b)", "difference = cos(w(a + b))"].map((title) => {
        const d = document.createElement("div");
        d.innerHTML = `<p class="fig-note">${title}</p>`;
        wrap.append(d);
        return new Heatmap(d, { rows: P, cols: P, size: 190, diverging: true, format: (v) => v.toFixed(3) });
      });
      let k = 3;
      const draw = () => {
        const w = (2 * Math.PI * k) / P, cc = new Float32Array(P * P), ss = new Float32Array(P * P), dd = new Float32Array(P * P);
        for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) {
          const i = a * P + b;
          cc[i] = Math.cos(w * a) * Math.cos(w * b); ss[i] = Math.sin(w * a) * Math.sin(w * b); dd[i] = cc[i] - ss[i];
        }
        [cc, ss, dd].forEach((g, j) => maps[j].draw(g, { min: -1, max: 1 }));
        work.textContent = `k = ${k}: each grid has a row for every a and a column for every b (red +1, blue −1). The first two are checkerboards that depend on a and b separately; their difference is cos(w_k(a + b)), so its stripes run along the lines a + b = constant. This is the product the network's neurons approximate (chapter 14).`;
      };
      sweep(slider(controls, "frequency k", 1, 56, 1, k, (v) => { k = v; draw(); }), { seconds: 12 });
    },
  },

  "phasor-wave": {
    render(el) {
      const { controls, out, work } = frame(el, "A turning hand draws cos and sin");
      const s = svg("svg", { viewBox: "0 0 560 230", width: "100%", height: 230 });
      out.append(s);
      let th = 60; // degrees
      const R = 80, cx = 100, cy = 115, x0 = 220, x1 = 545, X = (t) => x0 + ((x1 - x0) * t) / 720; // 0…720° on the wave axis
      const draw = () => {
        s.replaceChildren();
        const r = (th * Math.PI) / 180, hx = cx + R * Math.cos(r), hy = cy - R * Math.sin(r);
        svg("circle", { cx, cy, r: R, fill: "none", stroke: cssVar("--line"), "stroke-width": 1.5 }, s);
        svg("line", { x1: cx - R - 10, y1: cy, x2: x1, y2: cy, stroke: cssVar("--line"), "stroke-width": 1 }, s);
        for (const [f, col, name] of [[Math.sin, "--restricted", "sin θ"], [Math.cos, "--train", "cos θ"]]) {
          const pts = Array.from({ length: 145 }, (_, i) => `${X(i * 5)},${cy - R * f((i * 5 * Math.PI) / 180)}`).join(" ");
          svg("polyline", { points: pts, fill: "none", stroke: cssVar(col), "stroke-width": 1.2, opacity: 0.45 }, s);
          const y = cy - R * f(r);
          svg("circle", { cx: X(th), cy: y, r: 5, fill: cssVar(col) }, s);
          svg("text", { x: x1 - 2, y: cy - R * f((720 * Math.PI) / 180) - 8, "text-anchor": "end", "font-size": 12, fill: cssVar(col) }, s).textContent = name;
        }
        svg("line", { x1: hx, y1: hy, x2: X(th), y2: hy, stroke: cssVar("--restricted"), "stroke-width": 1, "stroke-dasharray": "4 3" }, s); // same height = sin θ
        svg("line", { x1: cx, y1: cy, x2: hx, y2: cy, stroke: cssVar("--train"), "stroke-width": 4, "stroke-linecap": "round" }, s); // along the axis: the horizontal position, cos θ
        svg("line", { x1: cx, y1: cy, x2: hx, y2: hy, stroke: cssVar("--accent"), "stroke-width": 3, "stroke-linecap": "round" }, s);
        svg("line", { x1: X(th), y1: cy - R - 8, x2: X(th), y2: cy + R + 8, stroke: cssVar("--ink-soft"), "stroke-width": 1, "stroke-dasharray": "2 3" }, s);
        work.textContent = `θ = ${th}°: the hand's tip is at (cos θ, sin θ) = (${f4(Math.cos(r))}, ${f4(Math.sin(r))}). Its height traces the sin wave and its horizontal position the cos wave; after 360° both repeat.`;
      };
      onTheme(draw);
      sweep(slider(controls, "angle θ (degrees)", 0, 720, 1, th, (v) => { th = v; draw(); }), { seconds: 8 });
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
      onTheme(draw);
      input.querySelector("input").addEventListener("change", (e) => { kset = e.target.value; draw(); });
      sweep(slider(controls, "a", 0, 112, 1, a, (v) => { a = v; draw(); }), { seconds: 10 }); // the peak follows a + b
      slider(controls, "b", 0, 112, 1, b, (v) => { b = v; draw(); });
    },
  },
};
