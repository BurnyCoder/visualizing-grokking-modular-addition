// seeds.js — dashboard group H: every run side by side (our seeds, the paper's run, the controls).
//
// Local explanation: test-loss curves of all runs on one log axis (the selected run highlighted), the key
// frequencies each run chose, and the preregistered hypothesis results with their statistics (from
// site/data/summary.json).
// Global context: turns the single-model story into a claim about the setup — does the same mechanism appear
// in every seed, and do the controls (no weight decay, random labels) behave as predicted?
// Sources: docs/preregistration.md (tests and thresholds); paper App. C.2 (other seeds), D.1 (weight decay).
import { array, manifest, run } from "../core/data.js";
import { Plot } from "../viz/plot.js";
import { cssVar, freqColor, runColor } from "../viz/colors.js";
import { el, note, runOf, set, watch } from "./common.js";

const style = (g) => ({ wd1: "solid", paper: "solid", wd0: "dash", rand: "dot", split0: "dashdot" })[g] || "solid";

export const figures = {
  "seed-test-loss": {
    async render(body, ds) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 320, stepAxis: true });
      el(body, "p", "fig-note", "Solid: weight decay 1 (paper settings); dashed: no weight decay; dotted: random labels; dash-dot: the paper's data split with a new initialisation. Click a legend entry to hide it; click the curve area to jump to that step.");
      watch(["run", "step"], async (s, changed) => {
        const m = await manifest();
        if (!changed.length || changed.includes("run") || changed.includes("theme")) {
          const traces = [];
          for (const r of m.runs) {
            const te = await array(r.id, "curve_test_loss");
            const every = te.entry.step_every ?? 10; // spacing recorded by export.py
            const x = Array.from({ length: te.data.length }, (_, i) => i * every);
            const sel = r.id === runOf(ds);
            traces.push({ x, y: Array.from(te.data), type: "scatter", mode: "lines", name: r.id,
              line: { color: sel ? cssVar("--accent") : runColor(r.id), width: sel ? 3 : 1, dash: style(r.group) }, opacity: sel ? 1 : 0.6 });
          }
          await plot.draw(traces, { yaxis: { type: "log", title: { text: "test loss (log)" }, exponentformat: "power" }, xaxis: { title: { text: "training step" } }, legend: { orientation: "v", x: 1.02, y: 1, font: { size: 10 } }, margin: { r: 120 } });
          plot.el.on?.("plotly_legendclick", () => true);
        }
        plot.cursor(s.step);
      });
    },
  },

  "seed-key-freqs": {
    async render(body) {
      body.innerHTML = "";
      const plot = new Plot(body, { height: 360 });
      const m = await manifest();
      const runs = m.runs;
      const traces = [];
      runs.forEach((r, i) => r.key_freqs.forEach((k) => traces.push({ x: [k], y: [r.id], type: "scatter", mode: "markers", marker: { size: 10, color: freqColor(k) }, showlegend: false, hovertemplate: `${r.id}: k=${k}<extra></extra>` })));
      await plot.draw(traces, { xaxis: { title: { text: "key frequency k" }, range: [0, 57], dtick: 4 }, yaxis: { automargin: true, categoryorder: "array", categoryarray: runs.map((r) => r.id).reverse() }, hovermode: "closest", margin: { l: 110 } });
      plot.el.on?.("plotly_click", (ev) => set({ run: ev.points[0].y }));
      el(body, "p", "fig-note", "Each dot is a key frequency of that run (frequencies whose W_L Fourier norm is at least 25 % of the largest). Click a run to select it everywhere.");
    },
  },

  "hypothesis-table": {
    async render(body) {
      const summary = await fetch(`${window.GROKVIZ.root}data/summary.json`).then((r) => r.json());
      const h = summary["hypotheses.verdict"]?.value;
      if (!h) { body.innerHTML = '<p class="fig-note">Hypothesis tests not computed yet.</p>'; return; }
      const rows = Object.entries(h).map(([k, v]) => `<tr><td>${k}</td><td>${v.successes !== undefined ? `${v.successes}/${v.n} seeds` : `groked: ${v.wd1_grokked}/${v.wd1_n} (λ=1) vs ${v.wd0_grokked}/${v.wd0_n} (λ=0)`}</td><td>${v.p.toPrecision(2)}</td><td>${v.p_holm.toPrecision(2)}</td><td>${v.supported ? "supported" : "<strong>not supported</strong>"}</td></tr>`).join("");
      body.innerHTML = `<table><tr><th>Hypothesis</th><th>Result</th><th>p</th><th>Holm p</th><th>Verdict</th></tr>${rows}</table>`;
    },
  },
};
void note;
