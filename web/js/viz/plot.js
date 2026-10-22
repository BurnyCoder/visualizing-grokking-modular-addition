// plot.js — thin wrappers around Plotly for line and bar charts with the site's conventions.
//
// Local explanation: `Plot` owns one Plotly chart; `draw(traces, layout)` re-renders with Plotly.react (cheap
// updates); `cursor(step)` moves a dotted line marking the current training step (an overlay <div>, viz/overlay.js,
// so moving it costs no Plotly call); `on(event, fn)` binds a Plotly event once, after the first draw (before it the
// graph div has no `.on`); `phaseShapes(phases)` shades memorisation / circuit formation / cleanup. Clicking a line
// chart with a step axis sets the step of the figure's store. Theme colours come from main.css.
// Global context: every loss curve, spectrum bar chart and metric-over-training plot on the site.
// Sources: Plotly.react (https://plotly.com/javascript/plotlyjs-function-reference/#plotlyreact);
// layout shapes (https://plotly.com/javascript/shapes/); events (https://plotly.com/javascript/plotlyjs-events/).
import { cssVar } from "./colors.js";
import { StepCursor } from "./overlay.js";
import { storeOf } from "../core/state.js";

/** Wait until the deferred Plotly script has loaded. */
export function plotlyReady() {
  return new Promise((resolve) => {
    const check = () => (window.Plotly ? resolve(window.Plotly) : setTimeout(check, 30));
    check();
  });
}

/** Base layout following the current theme. */
export function baseLayout(extra = {}) {
  const ink = cssVar("--ink"), soft = cssVar("--ink-soft"), line = cssVar("--line");
  const axis = { color: soft, gridcolor: line, zerolinecolor: line, linecolor: line, tickfont: { size: 11 } };
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { family: "Inter, sans-serif", size: 12, color: ink },
    margin: { l: 56, r: 12, t: 34, b: 42 }, legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { size: 10.5 } }, hovermode: "x unified", // legend above the plot area so it never covers the x-axis title
    ...extra, xaxis: { ...axis, ...(extra.xaxis || {}) }, yaxis: { ...axis, ...(extra.yaxis || {}) },
  };
}

/** Rectangles shading the training phases (paper-style bands) between boundaries t1, t2, t3. */
export function phaseShapes(phases, xmax) {
  if (!phases || !phases.t1) return [];
  const bands = [[0, phases.t1, "--phase1"], [phases.t1, phases.t2, "--phase2"], [phases.t2, phases.t3, "--phase3"]];
  return bands.filter(([a, b]) => a != null && b != null).map(([a, b, c]) => ({
    type: "rect", xref: "x", yref: "paper", x0: a, x1: Math.min(b, xmax ?? b), y0: 0, y1: 1, fillcolor: cssVar(c), line: { width: 0 }, layer: "below",
  }));
}

export class Plot {
  /** @param {HTMLElement} parent  @param {object} o {height, stepAxis (click sets step)} */
  constructor(parent, o = {}) {
    this.o = { height: 260, stepAxis: false, ...o };
    this.el = document.createElement("div");
    this.el.className = "plot";
    this.el.style.height = `${this.o.height}px`;
    parent.append(this.el);
    this.handlers = []; // [event, fn] pairs, bound after the first draw
    this.drawn = false;
    this.ready = plotlyReady();
    if (this.o.stepAxis) this.on("plotly_click", (ev) => { const x = ev.points?.[0]?.x; if (typeof x === "number") storeOf(this.el).set({ step: Math.round(x) }); });
  }

  /** Bind a Plotly event handler exactly once (now if the chart exists, else right after its first draw). */
  on(event, fn) {
    this.handlers.push([event, fn]);
    if (this.drawn) this.el.on(event, fn);
    return this;
  }

  /** Render traces with a layout (merged into the themed base layout). */
  async draw(traces, layout = {}) {
    const Plotly = await this.ready;
    await Plotly.react(this.el, traces, baseLayout(layout), { displayModeBar: false, responsive: true });
    if (!this.drawn) {
      this.drawn = true;
      for (const [e, f] of this.handlers) this.el.on(e, f);
      this.el.on("plotly_afterplot", () => this.stepCursor?.place()); // follow resizes and zooms
    }
    this.stepCursor?.place();
  }

  /** Move the dotted "current step" line to x (training step on the x axis). */
  cursor(x) {
    (this.stepCursor ||= new StepCursor(this.el)).move(x);
  }
}
