// plot.js — thin wrappers around Plotly for line and bar charts with the site's conventions.
//
// Local explanation: `Plot` owns one Plotly chart; `draw(traces, layout)` re-renders with Plotly.react (cheap
// updates); `cursor(step)` moves a vertical line marking the current training step; `phaseShapes(phases)`
// shades memorisation / circuit formation / cleanup. Clicking a line chart with a step axis sets the
// global step. Theme colours come from main.css so charts match light and dark mode.
// Global context: every loss curve, spectrum bar chart and metric-over-training plot on the site.
// Sources: Plotly.react (https://plotly.com/javascript/plotlyjs-function-reference/#plotlyreact);
// layout shapes (https://plotly.com/javascript/shapes/).
import { cssVar } from "./colors.js";
import { set } from "../core/state.js";

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
    margin: { l: 56, r: 12, t: 10, b: 42 }, legend: { orientation: "h", y: -0.22, font: { size: 11 } }, hovermode: "x unified",
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
    this.shapes = [];
    this.cursorShape = null;
    this.ready = plotlyReady();
  }

  /** Render traces with a layout (merged into the themed base layout). */
  async draw(traces, layout = {}) {
    const Plotly = await this.ready;
    this.layout = baseLayout(layout);
    this.shapes = layout.shapes || [];
    this.layout.shapes = [...this.shapes, ...(this.cursorShape ? [this.cursorShape] : [])];
    await Plotly.react(this.el, traces, this.layout, { displayModeBar: false, responsive: true });
    if (this.o.stepAxis && !this._clickBound) {
      this._clickBound = true;
      this.el.on("plotly_click", (ev) => { const x = ev.points?.[0]?.x; if (typeof x === "number") set({ step: Math.round(x) }); });
    }
  }

  /** Move the vertical "current step" line. */
  async cursor(x) {
    const Plotly = await this.ready;
    if (!this.layout) return;
    this.cursorShape = { type: "line", xref: "x", yref: "paper", x0: x, x1: x, y0: 0, y1: 1, line: { color: cssVar("--accent"), width: 2, dash: "dot" } };
    await Plotly.relayout(this.el, { shapes: [...this.shapes, this.cursorShape] });
  }
}
