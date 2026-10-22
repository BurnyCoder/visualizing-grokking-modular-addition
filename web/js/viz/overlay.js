// overlay.js — cheap marks drawn over charts and heatmaps: the current-step cursor and grid crosshairs.
//
// Local explanation: `stepToPixel` maps a data value to a pixel offset exactly as Plotly does for linear and log axes;
// `StepCursor` is a thin absolutely-positioned <div> moved with a CSS transform over a Plotly chart (no Plotly call
// per frame); `GridMarks` places column / row / cell outlines over a canvas heatmap using percentages, so they follow
// the canvas when it is resized and need no redraw of the heatmap itself.
// Global context: playing a run moves the cursor of every metric-over-training chart and the step column of every
// step × frequency heatmap on every frame; before, each move was a full `Plotly.relayout`.
// Plotly internals used (isolated here): `gd._fullLayout._size` (plot-area margins and size in px) and
// `gd._fullLayout.xaxis.range/type`. Sources: Plotly layout xaxis `range` (log axes in log10 units,
// https://plotly.com/javascript/reference/layout/xaxis/#layout-xaxis-range); `plotly_afterplot` event
// (https://plotly.com/javascript/plotlyjs-events/); CSS transforms (https://developer.mozilla.org/docs/Web/CSS/transform).

/**
 * Pixel offset of value `x` on an axis {range: [r0, r1], type: "linear"|"log", offset, length}, or null when the
 * value is not drawable (outside the range, missing, or ≤ 0 on a log axis).
 */
export function stepToPixel(x, { range, type, offset, length }) {
  if (x == null || !Number.isFinite(x) || (type === "log" && x <= 0)) return null;
  const v = type === "log" ? Math.log10(x) : x; // Plotly stores log-axis ranges in log10 units
  const [r0, r1] = range;
  const f = (v - r0) / (r1 - r0); // fraction along the axis (works for reversed ranges too)
  return f < 0 || f > 1 ? null : offset + f * length;
}

/** Dotted vertical line marking the current training step over a Plotly chart (`gd` = the chart's div). */
export class StepCursor {
  constructor(gd) {
    this.gd = gd;
    this.x = null;
    this.el = document.createElement("div");
    this.el.className = "step-cursor";
    this.el.hidden = true;
    gd.append(this.el); // a child of the graph div: Plotly.react leaves foreign children alone
  }

  /** Move to step x (null hides it). */
  move(x) {
    this.x = x;
    this.place();
  }

  /** Recompute the position from the chart's current axes (after a draw, resize or zoom). */
  place() {
    const fl = this.gd._fullLayout, ax = fl?.xaxis, size = fl?._size;
    const px = ax && size ? stepToPixel(this.x, { range: ax.range, type: ax.type, offset: size.l, length: size.w }) : null;
    this.el.hidden = px == null;
    if (px == null) return;
    this.el.style.top = `${size.t}px`;
    this.el.style.height = `${size.h}px`;
    this.el.style.transform = `translateX(${px}px)`;
  }
}

/** Column / row / cell outlines over a heatmap canvas of rows × cols cells (`stage` = positioned wrapper). */
export class GridMarks {
  constructor(stage, rows, cols) {
    Object.assign(this, { stage, rows, cols });
    this.marks = new Map(); // name → div
  }

  /** Show mark `name` at {row, col}: both = one cell, only col = a full column, only row = a full row; null hides. */
  set(name, at) {
    let m = this.marks.get(name);
    if (!m) {
      m = document.createElement("div");
      m.className = `grid-mark grid-mark-${name}`;
      this.stage.append(m);
      this.marks.set(name, m);
    }
    const row = at?.row ?? null, col = at?.col ?? null;
    m.hidden = row == null && col == null;
    if (m.hidden) return;
    const pct = (i, n) => `${(100 * i) / n}%`;
    Object.assign(m.style, {
      left: col == null ? "0" : pct(col, this.cols), width: col == null ? "100%" : pct(1, this.cols),
      top: row == null ? "0" : pct(row, this.rows), height: row == null ? "100%" : pct(1, this.rows),
    });
  }
}
