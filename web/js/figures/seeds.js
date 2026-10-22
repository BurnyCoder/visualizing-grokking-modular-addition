// seeds.js — dashboard group H: every run side by side (our seeds, the paper's run, the controls).
//
// Local explanation: test-loss curves of all runs on one log axis (the selected run highlighted), the key
// frequencies each run chose (the selected run's row shaded), and the preregistered hypothesis results with their
// statistics (from site/data/summary.json).
// Global context: turns the single-model story into a claim about the setup — does the same mechanism appear
// in every seed, and do the controls (no weight decay, random labels) behave as predicted?
// Sources: docs/preregistration.md (tests and thresholds); paper App. C.2 (other seeds), D.1 (weight decay).
import { array, manifest } from "../core/data.js";
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
      watch(body, ["run", "step"], async (s, changed) => {
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
        }
        plot.cursor(s.step);
      });
    },
  },

  "seed-key-freqs": {
    async render(body) {
      body.innerHTML = "";
      const m = await manifest();
      // runs where (almost) every frequency passes the 25 % rule have no sparse key set (no weight decay, random
      // labels): drawing 56 "key" dots for them would suggest structure that is not there, so they are listed instead
      const natural = (a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }); // wd1_s2 before wd1_s10
      const sparse = m.runs.filter((r) => r.key_freqs.length <= 20).sort(natural), dense = m.runs.filter((r) => r.key_freqs.length > 20);
      const fewest = Math.min(...dense.map((r) => r.key_freqs.length));
      const rows = sparse.map((r) => r.id).reverse(); // category order bottom → top
      const plot = new Plot(body, { height: 90 + 24 * sparse.length }).on("plotly_click", (ev) => set({ run: ev.points[0].y }));
      const traces = [];
      sparse.forEach((r) => r.key_freqs.forEach((k) => traces.push({ x: [k], y: [r.id], type: "scatter", mode: "markers", marker: { size: 10, color: freqColor(k) }, showlegend: false, hovertemplate: `${r.id}: k=${k}<extra></extra>` })));
      const info = note(body);
      watch(body, ["run"], async (s) => {
        const i = rows.indexOf(s.run); // shade the selected run's row (category axes place row i at y = i)
        await plot.draw(traces, { xaxis: { title: { text: "key frequency k" }, range: [0, 57], dtick: 4 },
          yaxis: { automargin: true, type: "category", dtick: 1, categoryorder: "array", categoryarray: rows }, hovermode: "closest", margin: { l: 110 },
          shapes: i < 0 ? [] : [{ type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y", y0: i - 0.5, y1: i + 0.5, fillcolor: cssVar("--accent"), opacity: 0.15, line: { width: 0 }, layer: "below" }] });
        info.textContent = `Each dot is a key frequency of that run (frequencies whose W_L Fourier norm is at least 25 % of the largest); shaded: the selected run${i < 0 ? ` (${s.run} is not drawn, see below)` : ""}. Click a run to select it everywhere.`
          + (dense.length ? ` Not shown: ${dense.map((r) => r.id).join(", ")} — no frequency stands out in these runs (${fewest === 56 ? "all 56" : `${fewest}–56 of 56`} frequencies pass the rule), i.e. they have no sparse Fourier structure.` : "");
      });
    },
  },

  "hypothesis-table": {
    async render(body) {
      const summary = await fetch(`${window.GROKVIZ.root}data/summary.json`).then((r) => r.json());
      const h = summary["hypotheses.verdict"]?.value;
      if (!h) { body.innerHTML = '<p class="fig-note">Hypothesis tests not computed yet.</p>'; return; }
      // one-line statements of the preregistered hypotheses (docs/preregistration.md)
      const claim = { H1: "restricted loss falls before test loss", H2: "excluded loss rises ≥ 10× during circuit formation",
        H3: "Fourier sparsity is learned and specific to the Fourier basis", H4: "key frequencies necessary, all others dispensable",
        H5: "weight decay is necessary for grokking" };
      const rows = Object.entries(h).map(([k, v]) => `<tr><td>${k}: ${claim[k] ?? ""}</td><td>${v.successes !== undefined ? `${v.successes}/${v.n} seeds` : `grokked: ${v.wd1_grokked}/${v.wd1_n} (λ=1) vs ${v.wd0_grokked}/${v.wd0_n} (λ=0)`}</td><td>${v.p.toPrecision(2)}</td><td>${v.p_holm.toPrecision(2)}</td><td>${v.supported ? "supported" : "<strong>not supported</strong>"}</td></tr>`).join("");
      body.innerHTML = `<table><tr><th>Hypothesis</th><th>Result</th><th>p (one-sided)</th><th>Holm p</th><th>Verdict</th></tr>${rows}</table>`;
    },
  },
};
