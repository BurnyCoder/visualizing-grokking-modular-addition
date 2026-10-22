// hero.js — the dashboard's opening panel, "Grokking in one view": four linked views of one moment of training.
//
// Local explanation: one ▶ button plays the selected run through grokking (0 to 1.1·t3, looping) on the page's
// Player (or the figure's own, when placed with {{anim:hero-grokking}}), and a one-line narration names the current
// phase. The four tiles follow the same step:
//   1. train and test loss (log scale) with the phase bands and a cursor;
//   2. the probability of the correct answer on all 12,769 inputs (rows a, columns b), optionally only the training
//      or only the test pairs;
//   3. the 113 number embeddings projected on one key frequency's plane — the clock — coloured by clock position
//      k·n mod 113, with hands to a, b and a + b; chips pick the frequency;
//   4. the selected neuron's activation on all inputs, computed in the browser from the nearest weight snapshot.
// Tiles 2 and 3 blend the two stored frames around the play position while playing (linear blend for the grid,
// polar blend for the clock so points move along the circle); paused, every tile shows a stored frame and says which.
// Global context: the dashboard's first impression and chapter 0's opening animation; every tile is also a full
// dashboard panel further down (loss-curves, pcorrect-grid, embed-circles, neuron-grid).
// Sources: paper Fig. 1–4 and Sec. 3–5 (https://arxiv.org/abs/2301.05217).
import { array, model, run, trainMask } from "../core/data.js";
import { lerpPolar2, phaseOf } from "../core/anim.js";
import { Plot, phaseShapes } from "../viz/plot.js";
import { Heatmap } from "../viz/heatmap.js";
import { CanvasPlot } from "../viz/canvas.js";
import { clockColor, cssVar } from "../viz/colors.js";
import { pagePlayer } from "../ui/page-player.js";
import { lossTraces } from "./curves.js";
import { el, fill, follow, frameKey, note, runOf, set, storeOf, watch } from "./common.js";

const P = 113;
const SAY = { // one line per phase (phases detected by src/grokviz/phases.py)
  memorisation: "Memorisation: the training loss plunges while the test loss rises — the network fits its training sums only.",
  "circuit formation": "Circuit formation: the test loss barely moves, yet the numbers spread onto circles — the Fourier circuit is being built.",
  cleanup: "Cleanup: weight decay removes the memorising weights; the test loss collapses and every input lights up.",
  "after cleanup": "After cleanup: the network adds on every input with a few frequencies.",
};

export const figures = {
  "hero-grokking": {
    async render(body, ds) {
      body.innerHTML = '<div class="hero-head"><button type="button" class="btn hero-play">▶ Play grokking</button><p class="hero-say"></p></div><div class="hero-tiles"></div>';
      const tiles = body.querySelector(".hero-tiles"), say = body.querySelector(".hero-say"), btn = body.querySelector(".hero-play");
      const tile = (title) => { const t = el(tiles, "div", "hero-tile"); el(t, "h4", "", title); return t; };
      const player = () => body.closest("[data-scope]")?.player || pagePlayer(); // figure's own player, else the page's
      btn.onclick = () => {
        const p = player();
        if (p.playing) { p.pause(); return; }
        p.setRange("grokking");
        p.setLoop("loop");
        p.play();
      };
      watch(body, ["run", "step", "playing"], async (s) => {
        const r = await run(runOf(ds));
        btn.textContent = s.playing ? "❚❚ Pause" : "▶ Play grokking";
        say.textContent = SAY[phaseOf(s.step, r.phases)] || "Press ▶ to play this run from the start to just after grokking.";
      }, { theme: false });

      // 1. losses with phase bands and a cursor
      const t1 = tile("Loss (log scale)");
      const plot = new Plot(t1, { height: 230, stepAxis: true });
      let shown = null;
      watch(t1, ["run", "step", "stepf"], async (s, changed) => {
        const id = runOf(ds), r = await run(id);
        if (id !== shown || changed.includes("theme")) {
          shown = id;
          await plot.draw(await lossTraces(id), { yaxis: { type: "log", exponentformat: "power", title: { text: "loss" } }, xaxis: { title: { text: "training step" } },
            shapes: phaseShapes(r.phases, r.ckpt_steps.at(-1)), margin: { l: 48, r: 8, t: 30, b: 38 } });
        }
        plot.cursor(s.stepf ?? s.step);
      });

      // 2. confidence on every input
      const t2 = tile("Confidence on every input");
      const show = el(t2, "label", "hero-ctl", 'show <select><option value="all">all pairs</option><option value="train">training pairs</option><option value="test">test pairs</option></select>').querySelector("select");
      const hm = new Heatmap(t2, { rows: P, cols: P, size: 260, diverging: false, format: (v) => v.toFixed(3), onClick: (a, b) => set({ a, b }) });
      const n2 = note(t2), grid = new Float32Array(P * P);
      const draw2 = async (s) => {
        const id = runOf(ds), r = await run(id);
        if (!r.geometry_steps) { n2.textContent = "No frames for this run."; return; }
        const [a, mask] = await Promise.all([array(id, "geo_p_correct"), trainMask(r.split_seed)]);
        const label = fill(grid, a, r.geometry_steps, s);
        if (show.value !== "all") for (let i = 0; i < grid.length; i++) if ((mask[i] === 1) !== (show.value === "train")) grid[i] = NaN; // hide the other pairs
        hm.draw(grid, { min: 0, max: 1 });
        n2.textContent = `${label}. Yellow: right answer with probability 1; purple: 0. Click a cell to select that input.`;
      };
      watch(t2, ["run", "step", "stepf", "playing"], draw2, { theme: false });
      show.onchange = () => draw2(storeOf(t2).state);
      follow(t2, ["a", "b"], (s) => hm.marks.set("input", { row: s.a, col: s.b }));

      // 3. the clock: embeddings on one key frequency's plane
      const t3 = tile("Numbers on a circle");
      const chips = el(t3, "div", "hero-chips");
      const cp = new CanvasPlot(t3, { height: 260, onClick: (i) => set({ a: i }) });
      const n3 = note(t3), xy = new Float32Array(2 * P), radius = new Map(); // "run|plane" → largest |coordinate| over training
      let chipsFor = null;
      watch(t3, ["run", "step", "stepf", "playing", "freq", "a", "b"], async (s) => {
        const id = runOf(ds), r = await run(id);
        if (!r.geometry_steps) { n3.textContent = "No frames for this run."; return; }
        const ks = r.key_freqs, j = Math.max(0, ks.indexOf(s.freq)), k = ks[j];
        if (chipsFor !== id) { // one chip per key frequency of this run
          chipsFor = id;
          chips.replaceChildren(...ks.map((f) => Object.assign(document.createElement("button"), { type: "button", className: "chip", textContent: `k = ${f}`, onclick: () => set({ freq: f }) })));
        }
        [...chips.children].forEach((c, i) => c.classList.toggle("active", i === j));
        const a = await array(id, "geo_we_planes"), K = a.shape[1];
        const key = `${id}|${j}`;
        if (!radius.has(key)) { // fixed scale for the whole run, so the circle visibly grows
          let m = 0;
          for (let f = 0; f < a.shape[0]; f++) for (let q = 0; q < 2 * P; q++) m = Math.max(m, Math.abs(a.data[(f * K + j) * 2 * P + q]));
          radius.set(key, 1.08 * m || 1);
        }
        const R = radius.get(key), label = fill(xy, a, r.geometry_steps, s, { offset: j * 2 * P, lerp: lerpPolar2 });
        const at = (t) => ({ x: xy[2 * t], y: xy[2 * t + 1] }), c = (s.a + s.b) % P;
        const hands = [[s.a, "--train"], [s.b, "--restricted"], [c, "--accent"]];
        cp.draw({
          domain: [-R, R, -R, R], equal: true,
          segments: hands.map(([t, v]) => ({ x0: 0, y0: 0, x1: at(t).x, y1: at(t).y, color: cssVar(v), width: 2.2 })),
          points: { x: Array.from({ length: P }, (_, t) => xy[2 * t]), y: Array.from({ length: P }, (_, t) => xy[2 * t + 1]), r: 3.2,
            color: (t) => clockColor((k * t) % P, P), label: (t) => `<span class="tt-term">number ${t}</span>clock position ${k}·${t} mod 113 = ${(k * t) % P}` },
          labels: hands.map(([t, v], h) => ({ x: 1.13 * at(t).x, y: 1.13 * at(t).y, text: ["a", "b", "a+b"][h], color: cssVar(v) })),
        });
        n3.textContent = `${label}. Each dot is a number n, projected on the plane of cos and sin of frequency k = ${k} and coloured by its clock position k·n mod 113; hands point at a = ${s.a}, b = ${s.b} and a + b = ${c} (mod 113). Click a dot to make it a.`;
      });

      // 4. one neuron on every input (from the nearest weight snapshot)
      const t4 = tile("One neuron on every input");
      const hm4 = new Heatmap(t4, { rows: P, cols: P, size: 260, diverging: false, format: (v) => v.toFixed(3), onClick: (a, b) => set({ a, b }) });
      const n4 = note(t4);
      watch(t4, ["run", "step", "neuron"], async (s) => {
        const net = await model(runOf(ds), s.step);
        hm4.draw(net.neuronGrid(s.neuron));
        n4.textContent = `Neuron ${s.neuron} (pick another in the control bar), weights from step ${net.step.toLocaleString("en-US")}: stripes appear when the neuron becomes a wave of one frequency.`;
      }, { dedupe: frameKey(ds, "weight_steps", "neuron"), theme: false });
      follow(t4, ["a", "b"], (s) => hm4.marks.set("input", { row: s.a, col: s.b }));
    },
  },
};
