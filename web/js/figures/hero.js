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
import { array, model, run } from "../core/data.js";
import { lerpPolar2, phaseOf } from "../core/anim.js";
import { Plot, phaseShapes } from "../viz/plot.js";
import { Heatmap } from "../viz/heatmap.js";
import { CanvasPlot } from "../viz/canvas.js";
import { pagePlayer } from "../ui/page-player.js";
import { lossTraces } from "./curves.js";
import { clockScene, inputGridView, planeRadius } from "./geometry.js";
import { el, fill, follow, frameKey, note, runOf, set, watch } from "./common.js";

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
      inputGridView(tile("Confidence on every input"), ds, "geo_p_correct", { size: 260,
        describe: (label) => `${label}. Yellow: right answer with probability 1; purple: 0. Click a cell to select that input.` });

      // 3. the clock: embeddings on one key frequency's plane
      const t3 = tile("Numbers on a circle");
      const chips = el(t3, "div", "hero-chips");
      const cp = new CanvasPlot(t3, { height: 260, onClick: (i) => set({ a: i }) });
      const n3 = note(t3), xy = new Float32Array(2 * P);
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
        const a = await array(id, "geo_we_planes");
        const label = fill(xy, a, r.geometry_steps, s, { offset: j * 2 * P, lerp: lerpPolar2 }), c = (s.a + s.b) % P;
        cp.draw(clockScene(xy, k, planeRadius(a, j, `${id}|geo_we_planes|${j}`), [[s.a, "--train", "a"], [s.b, "--restricted", "b"], [c, "--accent", "a+b"]]));
        n3.textContent = `${label}. Each dot is a number n, projected on the plane of cos and sin of frequency k = ${k} and coloured by its clock position k·n mod 113; hands point at a = ${s.a}, b = ${s.b} and a + b = ${c} (mod 113). Click a dot to make it a.`;
      });

      // 4. one neuron on every input (from the nearest weight snapshot)
      const t4 = tile("One neuron on every input");
      const hm4 = new Heatmap(t4, { rows: P, cols: P, size: 260, diverging: false, format: (v) => v.toFixed(3), onClick: (a, b) => set({ a, b }) });
      const n4 = note(t4);
      watch(t4, ["run", "step", "neuron"], async (s) => {
        const net = await model(runOf(ds), s.step);
        hm4.draw(net.neuronGrid(s.neuron));
        n4.textContent = `Neuron ${s.neuron} (the dashboard's control bar picks another), weights from step ${net.step.toLocaleString("en-US")}: stripes appear when the neuron becomes a wave of one frequency.`;
      }, { dedupe: frameKey(ds, "weight_steps", "neuron"), theme: false });
      follow(t4, ["a", "b"], (s) => hm4.marks.set("input", { row: s.a, col: s.b }));
    },
  },
};
