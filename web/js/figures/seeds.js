// seeds.js — dashboard group H: every run side by side (our seeds, the paper's run, the controls).
//
// Local explanation: test-loss curves of all runs on one log axis (the selected run highlighted), the key
// frequencies each run chose (the selected run's row shaded), the preregistered hypothesis results with their
// statistics (from site/data/summary.json), each seed's value of every hypothesis metric against its preregistered
// threshold, initial → final Gini against the random-basis null, every run's phases on one time axis, and the
// embedding clock of every weight-decay seed played together.
// Global context: turns the single-model story into a claim about the setup — does the same mechanism appear
// in every seed, and do the controls (no weight decay, random labels) behave as predicted?
// Sources: docs/preregistration.md (tests and thresholds); paper App. C.2 (other seeds), D.1 (weight decay).
import { array, manifest, summary } from "../core/data.js";
import { lerpPolar2 } from "../core/anim.js";
import { Plot } from "../viz/plot.js";
import { CanvasPlot } from "../viz/canvas.js";
import { cssVar, freqColor, runColor } from "../viz/colors.js";
import { clockScene, planeRadius } from "./geometry.js";
import { el, fill, note, runOf, set, watch } from "./common.js";

const style = (g) => ({ wd1: "solid", paper: "solid", wd0: "dash", rand: "dot", split0: "dashdot" })[g] || "solid";
const natural = (a, b) => a.localeCompare(b, undefined, { numeric: true }); // wd1_s2 before wd1_s10

/** Preregistered per-seed metrics and pass conditions (docs/preregistration.md; values from reports/tables/hypotheses.json). */
const METRICS = [
  { key: "H1_delta", title: "H1: test-loss drop minus restricted-loss drop (steps)", pass: (v) => v > 0, line: 0, log: false, side: "right of the line" },
  { key: "H2_ratio", title: "H2: excluded loss at t2 ÷ at t1", pass: (v) => v >= 10, line: 10, log: true, side: "right of the line" },
  { key: "H4_key_min_ratio", title: "H4 (i): loss ratio, removing the least important key frequency", pass: (v) => v >= 100, line: 100, log: true, side: "right of the line" },
  { key: "H4_nonkey_max_ratio", title: "H4 (ii): loss ratio, removing the most important other frequency", pass: (v) => v <= 2, line: 2, log: true, side: "left of the line" },
  { key: "H4_keep_ratio", title: "H4 (iii): loss ratio, keeping only the key frequencies", pass: (v) => v <= 1.5, line: 1.5, log: true, side: "left of the line" },
];

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

  "hypothesis-dots": {
    async render(body) {
      body.innerHTML = "";
      const per = (await summary())["hypotheses.per_run"]?.value;
      if (!per) { body.innerHTML = '<p class="fig-note">Hypothesis tests not computed yet.</p>'; return; }
      const ids = Object.keys(per).sort(natural), grid = el(body, "div", "small-multiples");
      grid.style.gridTemplateColumns = "repeat(auto-fill, minmax(260px, 1fr))";
      const plots = METRICS.map((m) => { const f = el(grid, "figure"); el(f, "figcaption", "", m.title); return new Plot(f, { height: 70 + 16 * ids.length }).on("plotly_click", (ev) => set({ run: ev.points[0].y })); });
      note(body).textContent = "One dot per weight-decay seed; filled dots meet the preregistered condition (docs/preregistration.md), hollow ones do not; the dotted line is the threshold. Click a dot to select that run everywhere.";
      watch(body, ["run"], async (s) => {
        await Promise.all(METRICS.map((m, j) => {
          const xs = ids.map((id) => per[id][m.key]), ok = xs.map(m.pass);
          return plots[j].draw([{ x: xs, y: ids, type: "scatter", mode: "markers", hovertemplate: "%{y}: %{x:.3g}<extra></extra>",
            marker: { size: ids.map((id) => (id === s.run ? 13 : 9)), color: ids.map((id) => (id === s.run ? cssVar("--accent") : cssVar("--train"))),
              symbol: ok.map((o) => (o ? "circle" : "circle-open")), line: { width: 2, color: ids.map((id) => (id === s.run ? cssVar("--accent") : cssVar("--train"))) } } }],
          { xaxis: { type: m.log ? "log" : "linear", exponentformat: "power", title: { text: `pass: ${m.side}` } }, yaxis: { type: "category", categoryorder: "array", categoryarray: [...ids].reverse(), automargin: true },
            shapes: [{ type: "line", x0: m.line, x1: m.line, yref: "paper", y0: 0, y1: 1, line: { dash: "dot", width: 1.5, color: cssVar("--ink") } }], hovermode: "closest", margin: { l: 70, r: 10, t: 6, b: 40 } });
        }));
      });
    },
  },

  "gini-vs-null": {
    async render(body, ds) {
      body.innerHTML = "";
      const sel = el(body, "label", "hero-ctl", 'matrix <select><option value="we">W_E (embedding)</option><option value="wl">W_L (neuron → logit map)</option></select>').querySelector("select");
      const m = await manifest(), runs = m.runs.filter((r) => r.final?.gini_we != null).sort((a, b) => natural(a.id, b.id));
      const plot = new Plot(body, { height: 90 + 18 * runs.length }).on("plotly_click", (ev) => set({ run: ev.points[0].y }));
      const info = note(body);
      const draw = async (s) => {
        const w = sel.value, ids = runs.map((r) => r.id), f = (r, k) => r.final[`gini_${w}${k}`];
        const need = runs.map((r) => Math.max(f(r, "_init") + 0.3, r.final[`gini_null_${w}_p99`])); // preregistered bar (H3)
        await plot.draw([
          { x: runs.flatMap((r) => [f(r, "_init"), f(r, ""), null]), y: ids.flatMap((id) => [id, id, null]), type: "scatter", mode: "lines", line: { color: cssVar("--line"), width: 2 }, hoverinfo: "skip", showlegend: false },
          { x: runs.map((r) => f(r, "_init")), y: ids, type: "scatter", mode: "markers", name: "at initialisation", marker: { symbol: "circle-open", size: 8, color: cssVar("--ink-soft") } },
          { x: need, y: ids, type: "scatter", mode: "markers", name: "needed: max(initial + 0.3, random-basis 99th percentile)", marker: { symbol: "line-ns-open", size: 14, color: cssVar("--ink") } },
          { x: runs.map((r) => f(r, "")), y: ids, type: "scatter", mode: "markers", name: "final", marker: { size: ids.map((id) => (id === s.run ? 13 : 9)), color: runs.map((r, i) => (f(r, "") > need[i] ? cssVar("--restricted") : cssVar("--test"))) } },
        ], { xaxis: { title: { text: `Gini of the ${w === "we" ? "W_E" : "W_L"} frequency norms` }, range: [0, 1] },
          yaxis: { type: "category", categoryorder: "array", categoryarray: [...ids].reverse(), automargin: true }, hovermode: "closest", legend: { orientation: "h", y: 1.02, yanchor: "bottom" } });
        info.textContent = "Each row is one run: hollow circle = Gini at initialisation, dot = final Gini (green when it clears the preregistered bar of H3, red when not), tick = the bar itself; 0 = spread over all frequencies, 1 = a single frequency. Runs with weight decay become sparse in the Fourier basis; runs without weight decay or with random labels stay near their initial value.";
      };
      sel.onchange = () => draw({ run: runOf(ds) });
      watch(body, ["run"], draw);
    },
  },

  "seed-phase-strips": {
    async render(body) {
      body.innerHTML = "";
      const m = await manifest(), runs = m.runs.filter((r) => r.phases?.t1).sort((a, b) => natural(a.id, b.id)), ids = runs.map((r) => r.id);
      const plot = new Plot(body, { height: 90 + 20 * runs.length, stepAxis: true }).on("plotly_click", (ev) => { if (typeof ev.points[0].y === "string") set({ run: ev.points[0].y }); });
      const info = note(body);
      let shownRun = null;
      watch(body, ["run", "step", "stepf"], async (s, changed) => {
        if (shownRun !== s.run || changed.includes("theme")) {
          shownRun = s.run;
          const band = (name, a, b, c) => ({ type: "bar", orientation: "h", name, y: ids, base: runs.map((r) => (a ? r.phases[a] : 0)), x: runs.map((r) => (r.phases[b] ?? NaN) - (a ? r.phases[a] : 0)),
            marker: { color: cssVar(c).replace(/0\.\d+\)$/, "0.75)"), line: { width: runs.map((r) => (r.id === s.run ? 2 : 0)), color: cssVar("--ink") } }, hovertemplate: `%{y}: ${name}<extra></extra>` });
          const pb = m.runs.find((r) => r.id === "paper")?.phases?.paper_boundaries; // the paper's own boundaries, drawn on its row
          await plot.draw([band("memorisation", null, "t1", "--phase1"), band("circuit formation", "t1", "t2", "--phase2"), band("cleanup", "t2", "t3", "--phase3"),
            { x: runs.map((r) => r.phases.t_grok ?? null), y: ids, type: "scatter", mode: "markers", name: "test accuracy ≥ 99 % (t_grok)", marker: { symbol: "diamond", size: 9, color: cssVar("--accent") } },
            ...(pb ? [{ x: [pb.t1, pb.t2, pb.t3], y: ["paper", "paper", "paper"], type: "scatter", mode: "markers", name: "the paper's own boundaries", marker: { symbol: "line-ns-open", size: 18, color: cssVar("--ink") } }] : [])],
          { barmode: "overlay", xaxis: { title: { text: "training step" } }, yaxis: { type: "category", categoryorder: "array", categoryarray: [...ids].reverse(), automargin: true }, hovermode: "closest",
            legend: { orientation: "h", y: 1.02, yanchor: "bottom" }, margin: { l: 100 } });
        }
        plot.cursor(s.stepf ?? s.step);
        info.textContent = `Every run whose phases were detected (docs/operationalizations.md), on one time axis; outlined: ${s.run}. Click a row to select the run; click elsewhere to set the step.`;
      });
    },
  },

  "seed-clocks": {
    async render(body) {
      body.innerHTML = "";
      const m = await manifest(), runs = m.runs.filter((r) => (r.group === "wd1" || r.id === "paper") && r.geometry_steps).sort((a, b) => natural(a.id, b.id));
      const grid = el(body, "div", "small-multiples");
      grid.style.gridTemplateColumns = "repeat(auto-fill, minmax(150px, 1fr))";
      const tiles = runs.map((r) => {
        const f = el(grid, "figure");
        f.dataset.run = r.id;
        el(f, "figcaption", "", `${r.id} · k = ${r.key_freqs[0]}`);
        return { r, f, xy: new Float32Array(2 * 113), cp: new CanvasPlot(f, { height: 150, onClick: () => set({ run: r.id }) }) };
      });
      const info = note(body);
      watch(body, ["run", "step", "stepf", "playing", "a", "b"], async (s) => {
        let label = "";
        for (const t of tiles) { // every seed at the same training step, each on the plane of its first key frequency
          const a = await array(t.r.id, "geo_we_planes"), c = (s.a + s.b) % 113;
          label = fill(t.xy, a, t.r.geometry_steps, s, { lerp: lerpPolar2 });
          t.cp.draw(clockScene(t.xy, t.r.key_freqs[0], planeRadius(a, 0, `${t.r.id}|geo_we_planes|0`), [[c, "--accent", "a+b"]]));
          t.f.classList.toggle("selected", t.r.id === s.run);
        }
        info.textContent = `${label}: every weight-decay seed (and the paper's run) at the same step, each number's embedding on the plane of the run's first key frequency, coloured by its clock position. Different seeds choose different frequencies, but all build the same circle. Click a clock to select that run.`;
      }, { minGap: 100 });
    },
  },

  "hypothesis-table": {
    async render(body) {
      const h = (await summary())["hypotheses.verdict"]?.value;
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
