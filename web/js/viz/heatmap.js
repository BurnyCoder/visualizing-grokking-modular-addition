// heatmap.js — fast canvas heatmaps for 113×113 grids (and other small matrices) with hover and click.
//
// Local explanation: `Heatmap` draws a rows×cols array into an offscreen canvas one pixel per cell, scales it up
// with pixelated rendering, shows a tooltip "(row, col): value" on hover and calls `onClick(row, col)` on click;
// `marks` (viz/overlay.js GridMarks) outlines a selected cell, column or row on top without redrawing the canvas.
// Colour scale: diverging (centred at 0, symmetric limits) for signed data or sequential for non-negative data,
// with a small colour bar. Convention throughout the site (as in the paper's Fig. 4): rows = a, columns = b.
// Global context: neuron grids, attention grids, logit grids, Fourier-coefficient maps and train/test masks
// all use this; canvas instead of Plotly keeps redraws fast while scrubbing through training.
// Source: Canvas ImageData (https://developer.mozilla.org/docs/Web/API/ImageData).
import { rdbu, viridis } from "./colors.js";
import { GridMarks } from "./overlay.js";

const tooltip = () => document.getElementById("tooltip");

export class Heatmap {
  /**
   * @param {HTMLElement} parent  container
   * @param {object} o  {rows, cols, size (CSS px), diverging, rowLabel, colLabel, format, onClick, label(r,c)}
   */
  constructor(parent, o) {
    this.o = { size: 260, diverging: true, rowLabel: "a", colLabel: "b", format: (v) => v.toPrecision(3), ...o };
    const wrap = document.createElement("div");
    wrap.className = "heatmap-wrap";
    const stage = document.createElement("div"); // positioned box exactly around the canvas, holds the marks
    stage.className = "heatmap-stage";
    stage.style.width = `${this.o.size}px`;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.o.cols; // one canvas pixel per cell
    this.canvas.height = this.o.rows;
    if (this.o.height) this.canvas.style.height = `${this.o.height}px`; // very wide matrices (steps × frequencies): fixed height
    else this.canvas.style.aspectRatio = `${this.o.cols} / ${this.o.rows}`; // otherwise keep cells square when the width shrinks
    stage.append(this.canvas);
    this.marks = new GridMarks(stage, this.o.rows, this.o.cols);
    this.bar = document.createElement("div");
    this.bar.className = "colorbar";
    wrap.append(stage, this.bar);
    parent.append(wrap);
    this.ctx = this.canvas.getContext("2d");
    this.values = null;
    this.canvas.addEventListener("mousemove", (e) => this._hover(e));
    this.canvas.addEventListener("mouseleave", () => { tooltip().hidden = true; });
    this.canvas.addEventListener("click", (e) => { const rc = this._cell(e); if (rc && this.o.onClick) this.o.onClick(...rc); });
  }

  /** Cell (row, col) under the mouse, or null. */
  _cell(e) {
    const r = this.canvas.getBoundingClientRect();
    const col = Math.floor(((e.clientX - r.left) / r.width) * this.o.cols);
    const row = Math.floor(((e.clientY - r.top) / r.height) * this.o.rows);
    return row >= 0 && row < this.o.rows && col >= 0 && col < this.o.cols ? [row, col] : null;
  }

  _hover(e) {
    const rc = this._cell(e);
    const tt = tooltip();
    if (!rc || !this.values) { tt.hidden = true; return; }
    const [r, c] = rc;
    const v = this.values[r * this.o.cols + c];
    if (Number.isNaN(v)) { tt.hidden = true; return; } // hidden cell
    const head = this.o.label ? this.o.label(r, c) : `${this.o.rowLabel} = ${r}, ${this.o.colLabel} = ${c}`;
    tt.innerHTML = `<span class="tt-term">${head}</span>value ${this.o.format(v)}`;
    tt.hidden = false;
    tt.style.left = `${e.clientX + 14}px`;
    tt.style.top = `${e.clientY + 14}px`;
  }

  /** Draw `values` (length rows*cols, row-major; NaN cells stay transparent). Optional fixed limits {min, max}. */
  draw(values, limits = null) {
    this.values = values;
    const { lo, hi, cmap } = paint(this.ctx, values, this.o.rows, this.o.cols, this.o.diverging, limits);
    this._colorbar(lo, hi, cmap);
  }

  _colorbar(lo, hi, cmap) {
    const c = document.createElement("canvas");
    c.width = 120; c.height = 1;
    const cx = c.getContext("2d"), img = cx.createImageData(120, 1);
    for (let i = 0; i < 120; i++) { const [r, g, b] = cmap(i / 119); img.data.set([r, g, b, 255], 4 * i); }
    cx.putImageData(img, 0, 0);
    this.bar.replaceChildren(Object.assign(document.createElement("span"), { textContent: this.o.format(lo) }), c,
      Object.assign(document.createElement("span"), { textContent: this.o.format(hi) }));
  }
}

/**
 * Paint `values` (rows × cols, row-major) one pixel per cell: diverging map centred at 0 or sequential map, colour
 * limits from the data unless `limits` {min, max} fixes them; NaN cells stay transparent. Returns {lo, hi, cmap}.
 */
function paint(ctx, values, rows, cols, diverging, limits = null) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; } // NaN never compares true: skipped
  if (limits) ({ min: lo, max: hi } = { min: limits.min ?? lo, max: limits.max ?? hi });
  if (diverging) { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; lo = -m; hi = m; } // symmetric around 0
  const span = hi - lo || 1, cmap = diverging ? rdbu : viridis, img = ctx.createImageData(cols, rows);
  for (let i = 0; i < values.length; i++) {
    const [r, g, b] = cmap((values[i] - lo) / span);
    img.data[4 * i] = r; img.data[4 * i + 1] = g; img.data[4 * i + 2] = b; img.data[4 * i + 3] = Number.isNaN(values[i]) ? 0 : 255;
  }
  ctx.putImageData(img, 0, 0);
  return { lo, hi, cmap };
}

/** Draw a grid into a bare canvas (for small multiples), optionally with fixed colour limits; returns the canvas. */
export function miniHeatmap(values, rows, cols, diverging = true, limits = null) {
  const c = document.createElement("canvas");
  c.width = cols; c.height = rows;
  paint(c.getContext("2d"), values, rows, cols, diverging, limits);
  return c;
}
