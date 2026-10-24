// flow.js — dashboard group E: "forward-flow", one forward pass of the network drawn as a data-flow diagram.
//
// Local explanation: for the selected input (a, b) and the nearest weight snapshot, the diagram shows every stage of
// GrokModel.forwardDetailed with its actual numbers: the three tokens; their embeddings x⁽⁰⁾ (128 cells each, blue
// negative, red positive); each head's attention from '=' to a, b and '='; the residual x⁽¹⁾ at '='; the 512 neuron
// activations (sorted by frequency cluster, as in the neuron panels); the 113 logits (the correct answer marked) and
// the answer with its probability. Small pulses run along the edges stage by stage (none under reduced motion);
// cell colours ease to the new values when a, b or the step changes (CSS transitions). The state key `stage` (set by
// clicking a stage, or by chapter 5's scenes while its step sections are read) highlights a stage and shows its
// equation (chapter 5 notation) below the diagram; clicking a neuron selects it.
// Global context: the dashboard's picture of "what the network does with one input", tying chapter 5's equations to
// the live numbers that every other panel aggregates.
// Sources: paper App. A (https://arxiv.org/abs/2301.05217); SVG animateMotion (https://developer.mozilla.org/docs/Web/SVG/Element/animateMotion).
import { array, model, nearestIndex, row, run } from "../core/data.js";
import { reducedMotion } from "../core/motion.js";
import { softmax } from "../model/grokmodel.js";
import { rdbu, viridis } from "../viz/colors.js";
import { svg } from "../widgets/basics.js";
import { clusterOrder, el, follow, frameKey, note, runOf, set, watch } from "./common.js";

const Y = [90, 200, 310]; // rows of positions 0 (a), 1 (b), 2 (=)
const HY = [45, 130, 215, 300].map((y) => y + 35); // centres of the four head boxes
const STAGES = {
  embed: { tex: "x^{(0)}_i = W_E\\, t_i + W_{pos}[i]", say: "Each token is looked up in the embedding matrix and gets its position's vector added: 128 numbers per position." },
  attention: { tex: "A^j = \\softmax(s_0, s_1, s_2), \\quad \\text{head}_j = W_O^j W_V^j \\sum_{i=0}^{2} A^j_i\\, x^{(0)}_i", say: "Each head decides how much '=' reads from a, b and itself (bars) and writes a mix of their values back." },
  residual: { tex: "x^{(1)} = x^{(0)}_2 + \\sum_{j=0}^{3} \\text{head}_j", say: "The heads' outputs are added to the '=' position's own vector (the dashed skip connection)." },
  mlp: { tex: "\\mathrm{MLP}(a,b) = \\ReLU\\big(W_{in}x^{(1)} + b_{in}\\big)", say: "512 neurons each take a weighted sum of x⁽¹⁾ and keep it only if positive (grid: brighter = more active; neurons in frequency-cluster order, read row by row)." },
  logits: { tex: "x^{(2)} = x^{(1)} + W_{out}\\,\\mathrm{MLP}(a,b) + b_{out}, \\qquad \\Logits(a,b) = W_U\\,x^{(2)}", say: "The neurons' outputs are added back and read out as one score per candidate answer c = 0 … 112." },
  answer: { tex: "P(c) = \\softmax\\big(\\Logits(a,b)\\big)_c, \\qquad \\text{answer} = \\argmax_c P(c)", say: "Softmax turns the scores into probabilities; the prediction is the most likely answer." },
};

/** Fill colour of value v on a diverging scale ±m (blue negative, red positive) or a sequential scale 0…m. */
const colour = (v, m, diverging = true) => `rgb(${(diverging ? rdbu(0.5 + v / (2 * (m || 1))) : viridis(v / (m || 1))).map(Math.round).join(",")})`;

/** One animated pulse along `path`, moving during the window [t0, t1] (fractions of a `total`-second cycle). */
function pulse(parent, path, t0, t1, total) {
  const c = svg("circle", { r: 3.2, class: "flow-pulse" }, parent);
  const e = 0.001; // keyTimes must be non-decreasing; tiny steps make the pulse appear and vanish
  svg("animateMotion", { dur: `${total}s`, repeatCount: "indefinite", path, keyPoints: "0;0;1;1", keyTimes: `0;${t0};${t1};1`, calcMode: "linear" }, c);
  svg("animate", { attributeName: "opacity", dur: `${total}s`, repeatCount: "indefinite", values: "0;0;1;1;0;0", keyTimes: `0;${t0};${t0 + e};${t1 - e};${t1};1` }, c);
}

export const figures = {
  "forward-flow": {
    async render(body, ds) {
      body.innerHTML = "";
      const s0 = svg("svg", { viewBox: "0 0 1000 400", class: "flow", role: "img", "aria-label": "Data flow of one forward pass" }, body);
      const eq = el(body, "div", "flow-eq");
      const info = note(body);
      const edges = svg("g", { class: "flow-edges" }, s0), pulses = svg("g", {}, s0), zoneG = svg("g", {}, s0), nodes = svg("g", { class: "flow-nodes" }, s0);
      const curve = (x0, y0, x1, y1) => `M${x0},${y0} C${(x0 + x1) / 2},${y0} ${(x0 + x1) / 2},${y1} ${x1},${y1}`;
      const paths = [ // [path, stage window start] — the pulse schedule follows the computation
        ...Y.map((y) => [curve(82, y, 110, y), 0]),
        ...Y.flatMap((y) => HY.map((hy) => [curve(270, y, 330, hy), 1])),
        ...HY.map((hy) => [curve(450, hy, 510, 200), 2]), [curve(270, Y[2], 510, 208), 2, "skip"], // x⁽⁰⁾₂ skips the heads
        [curve(670, 200, 700, 200), 3], [curve(828, 200, 860, 200), 4], [curve(882, 200, 905, 200), 5],
      ];
      paths.forEach(([d, , cls]) => svg("path", { d, class: cls || "" }, edges));
      if (!reducedMotion()) paths.forEach(([d, stage]) => pulse(pulses, d, stage / 6, (stage + 0.8) / 6, 4.2));
      const strip = (x, y, n, w) => Array.from({ length: n }, (_, i) => svg("rect", { x: x + (i * w) / n, y: y - 10, width: w / n + 0.2, height: 20, class: "flow-cell" }, nodes));
      const tokens = Y.map((y) => { svg("rect", { x: 10, y: y - 16, width: 72, height: 32, rx: 6, class: "flow-box" }, nodes); return svg("text", { x: 46, y: y + 5, "text-anchor": "middle", class: "flow-text" }, nodes); });
      const emb = Y.map((y) => strip(110, y, 128, 160));
      const heads = HY.map((hy, h) => {
        svg("rect", { x: 330, y: hy - 35, width: 120, height: 70, rx: 6, class: "flow-box" }, nodes);
        svg("text", { x: 338, y: hy - 20, class: "flow-small" }, nodes).textContent = `head ${h}`;
        return [0, 1, 2].map((k) => { svg("text", { x: 338, y: hy - 3 + 14 * k, class: "flow-small" }, nodes).textContent = ["a", "b", "="][k]; return svg("rect", { x: 352, y: hy - 11 + 14 * k, width: 0, height: 10, class: `flow-bar k${k}` }, nodes); });
      });
      const res = strip(510, 200, 128, 160);
      const titled = (r) => { svg("title", {}, r); return r; }; // native hover text, updated with the values
      const neu = Array.from({ length: 512 }, (_, i) => titled(svg("rect", { x: 700 + (i % 32) * 4, y: 136 + Math.floor(i / 32) * 8, width: 4.2, height: 8.2, class: "flow-cell clickable" }, nodes)));
      const logit = Array.from({ length: 113 }, (_, c) => titled(svg("rect", { x: 860, y: 53 + c * 2.6, width: 22, height: 2.8, class: "flow-cell hover" }, nodes)));
      const marker = svg("path", { d: "", class: "flow-marker" }, nodes);
      const answer = [0, 1, 2].map((k) => svg("text", { x: 908, y: 185 + 20 * k, class: k ? "flow-small" : "flow-text" }, nodes));
      [["tokens", 82], ["embedding x⁽⁰⁾", 190], ["attention", 390], ["residual x⁽¹⁾", 590], ["neurons", 764], ["logits", 871], ["answer", 940]].forEach(([t, x]) => {
        svg("text", { x, y: 22, "text-anchor": "middle", class: "flow-head" }, nodes).textContent = t;
      });
      const zones = [["embed", 100, 280], ["attention", 320, 460], ["residual", 500, 680], ["mlp", 690, 838], ["logits", 850, 890], ["answer", 895, 1000]];
      zones.forEach(([stage, x0, x1]) => { // transparent click targets per stage
        const z = svg("rect", { x: x0, y: 30, width: x1 - x0, height: 365, class: "flow-zone", "data-stage": stage }, zoneG); // under the cells: neurons stay clickable
        z.addEventListener("click", () => set({ stage }));
      });
      follow(body, ["stage"], (s) => { // highlight the selected stage and show its equation (none: nothing selected)
        s0.querySelectorAll(".flow-zone").forEach((q) => q.classList.toggle("on", q.dataset.stage === s.stage));
        eq.innerHTML = STAGES[s.stage] ? `<div class="math block">${STAGES[s.stage].tex}</div><p class="fig-note">${STAGES[s.stage].say}</p>` : "";
        if (STAGES[s.stage]) window.grokvizRenderMath?.(eq);
      });
      let order = [];
      neu.forEach((cell, i) => cell.addEventListener("click", () => set({ neuron: order[i] })));
      watch(body, ["run", "step", "a", "b"], async (s) => {
        const id = runOf(ds), r = await run(id), net = await model(id, s.step), f = net.forwardDetailed(s.a, s.b), p = net.p;
        const t = nearestIndex(r.ckpt_steps, net.step);
        order = clusterOrder(row(await array(id, "neuron_freq"), t), row(await array(id, "neuron_fve"), t), net.n);
        [`a = ${s.a}`, `b = ${s.b}`, "="].forEach((txt, i) => { tokens[i].textContent = txt; });
        const m0 = Math.max(...f.resid0.flatMap((x) => Array.from(x, Math.abs)));
        emb.forEach((cells, i) => cells.forEach((c, j) => c.setAttribute("fill", colour(f.resid0[i][j], m0))));
        heads.forEach((bars, h) => bars.forEach((b, k) => b.setAttribute("width", 92 * f.patterns[h][2][k])));
        const m1 = Math.max(...Array.from(f.residMid[2], Math.abs));
        res.forEach((c, j) => c.setAttribute("fill", colour(f.residMid[2][j], m1)));
        const mn = Math.max(...f.post) || 1;
        neu.forEach((c, i) => { c.setAttribute("fill", colour(f.post[order[i]], mn, false)); c.firstChild.textContent = `neuron ${order[i]}: activation ${f.post[order[i]].toFixed(2)} (click to select)`; });
        const L = Array.from(f.logits).slice(0, p), ml = Math.max(...L.map(Math.abs)), P = softmax(L), right = (s.a + s.b) % p, top = L.indexOf(Math.max(...L));
        logit.forEach((c, k) => { c.setAttribute("fill", colour(L[k], ml)); c.firstChild.textContent = `c = ${k}: logit ${L[k].toFixed(2)}, P = ${(100 * P[k]).toFixed(2)} %`; });
        const ym = 53 + right * 2.6 + 1.4;
        marker.setAttribute("d", `M884,${ym} l8,-5 v10 z`); // ◀ beside the correct answer's logit
        answer[0].textContent = `answer ${top}`;
        answer[1].textContent = `P = ${(100 * P[top]).toFixed(1)} %`;
        answer[2].textContent = top === right ? "✓ correct" : `✗ correct is ${right}`;
        s0.setCurrentTime?.(0); // restart the pulses: a new input flows through
        info.textContent = `Input ${s.a} + ${s.b} (mod ${p}), weights from step ${net.step.toLocaleString("en-US")}. Blue cells are negative, red positive (each column scaled to its largest value); neurons: brighter = more active. Click a stage for its equation, a neuron to select it.`;
      }, { dedupe: frameKey(ds, "weight_steps", "a", "b") });
    },
  },
};
