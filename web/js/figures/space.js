// space.js — dashboard group D in 3D: every input on a torus, and the representations' first three principal components.
//
// Local explanation:
//   torus-3d  each input (a, b) is a point on a torus at angles θ_a = w_k·a (around the ring) and θ_b = w_k·b (around
//             the tube) for a key frequency k. Since the network's answer depends on θ_a + θ_b, inputs with the same
//             answer a + b lie on one helix winding around the torus. Colours: the probability of the right answer
//             (blended between stored frames while playing), training vs test pairs, or the answer's clock position.
//   pca-3d    the first three principal components of the number embeddings, of the mean logits per answer, or of the
//             mean predicted distribution per answer (src/grokviz/geometry.py), on one fixed scale for the whole run.
//             Principal axes have no inherent sign, so while blending two stored frames the second is first flipped to
//             agree with the first (core/anim.js alignSigns); paused, the stored frame is shown as computed.
// Global context: the circle pictures of chapter 13 extended to pairs of numbers, and the PCA pictures in 3D.
// Sources: torus parametrisation (https://en.wikipedia.org/wiki/Torus#Geometry); paper Sec. 3.1 (angles add,
// https://arxiv.org/abs/2301.05217); PCA sign indeterminacy (https://en.wikipedia.org/wiki/Principal_component_analysis).
import { array, nearestIndex, row, run, trainMask } from "../core/data.js";
import { alignSigns, lerpInto } from "../core/anim.js";
import { View3D } from "../viz/view3d.js";
import { torusPoints } from "../model/fourier.js";
import { clockColor, cssVar, viridis } from "../viz/colors.js";
import { el, fill, note, runOf, set, storeOf, watch } from "./common.js";

const P = 113, RING = 0.62, TUBE = 0.3; // torus radii (whole shape within ±1)
const VIRIDIS_CSS = Array.from({ length: 256 }, (_, i) => `rgb(${viridis(i / 255).map(Math.round).join(",")})`); // lookup: probability → colour

const select = (parent, label, options) => el(parent, "label", "hero-ctl", `${label} <select>${options.map(([v, t]) => `<option value="${v}">${t}</option>`).join("")}</select>`).querySelector("select");

export const figures = {
  "torus-3d": {
    async render(body, ds) {
      body.innerHTML = "";
      const ctl = el(body, "div", "interf-ctl"), chips = el(ctl, "span", "hero-chips");
      const mode = select(ctl, "colour", [["p", "probability of the right answer"], ["split", "training / test pair"], ["answer", "the answer a + b (clock colour)"]]);
      const view = new View3D(body, { height: 380, onPick: (i) => set({ a: Math.floor(i / P), b: i % P }) });
      const info = note(body), tori = new Map(), pc = new Float32Array(P * P), colors = new Array(P * P); // tori: k → torus coordinates
      let chipsFor = null;
      const draw = async (s) => {
        const id = runOf(ds), r = await run(id), ks = r.key_freqs, k = ks.includes(s.freq) ? s.freq : ks[0];
        if (chipsFor !== id) {
          chipsFor = id;
          chips.replaceChildren(...ks.map((f) => Object.assign(document.createElement("button"), { type: "button", className: "chip", textContent: `k = ${f}`, onclick: () => set({ freq: f }) })));
        }
        [...chips.children].forEach((c, i) => c.classList.toggle("active", ks[i] === k));
        if (!tori.has(k)) tori.set(k, torusPoints(P, k, RING, TUBE)); // a around the ring, b around the tube
        const xyz = tori.get(k);
        let label = "";
        if (mode.value === "p" && r.geometry_steps) {
          label = fill(pc, await array(id, "geo_p_correct"), r.geometry_steps, s);
          for (let i = 0; i < pc.length; i++) colors[i] = VIRIDIS_CSS[Math.round(255 * Math.min(1, Math.max(0, pc[i])))];
        } else if (mode.value === "split") {
          const mask = await trainMask(r.split_seed), tr = cssVar("--train"), te = cssVar("--test");
          for (let i = 0; i < colors.length; i++) colors[i] = mask[i] ? tr : te;
        } else for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) colors[a * P + b] = clockColor((k * (a + b)) % P, P);
        colors[s.a * P + s.b] = cssVar("--ink"); // the selected input
        view.set({ xyz, colors, label: (i) => `<span class="tt-term">a = ${Math.floor(i / P)}, b = ${i % P}</span>answer ${(Math.floor(i / P) + (i % P)) % P}${mode.value === "p" ? `, P(right answer) = ${pc[i].toFixed(3)}` : ""}` });
        info.textContent = `${label ? `${label}. ` : ""}Each of the 12,769 inputs (a, b) sits at angle w_k·a around the ring and w_k·b around the tube, for k = ${k}; inputs with the same answer a + b lie on one helix (choose "the answer a + b" to see them). Drag to turn it; click a point to select that input (dark point: a = ${s.a}, b = ${s.b}).`;
      };
      mode.onchange = () => draw(storeOf(body).state);
      watch(body, ["run", "step", "stepf", "playing", "freq", "a", "b"], draw, { minGap: 60 });
    },
  },

  "pca-3d": {
    async render(body, ds) {
      body.innerHTML = "";
      const what = select(body, "show", [["geo_we_pca", "the 113 number embeddings"], ["geo_logit_pca", "the mean logits for each answer"], ["geo_prob_pca", "the mean predicted distribution for each answer"]]);
      const view = new View3D(body, { height: 360 });
      const info = note(body), xyz = new Float32Array(3 * P), scales = new Map();
      const aligned = (out, A, B, t) => lerpInto(out, A, alignSigns(A, B, 3), t); // flip B's axes to match A, then blend
      const draw = async (s) => {
        const id = runOf(ds), r = await run(id);
        if (!r.geometry_steps) { info.textContent = "No geometry frames for this run."; return; }
        const a = await array(id, what.value), ev = await array(id, `${what.value}_var`), key = `${id}|${what.value}`;
        if (!scales.has(key)) { let m = 0; for (const v of a.data) m = Math.max(m, Math.abs(v)); scales.set(key, m || 1); } // one scale for the whole run
        const label = fill(xyz, a, r.geometry_steps, s, { lerp: aligned }), m = scales.get(key), v = row(ev, nearestIndex(r.geometry_steps, s.step));
        view.set({ xyz: xyz.map((x) => x / m), colors: Array.from({ length: P }, (_, n) => clockColor(n, P)),
          label: (n) => `<span class="tt-term">${what.value === "geo_we_pca" ? "number" : "answer"} ${n}</span>` });
        info.textContent = `${label}: principal components 1–3 explain ${[0, 1, 2].map((j) => `${(100 * v[j]).toFixed(0)} %`).join(", ")} of the variance. Colour = the number's position on the 113-hour clock (0 … 112). Drag to turn it.`;
      };
      what.onchange = () => draw(storeOf(body).state);
      watch(body, ["run", "step", "stepf", "playing"], draw);
    },
  },
};
