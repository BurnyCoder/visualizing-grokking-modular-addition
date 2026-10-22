// canvas.js — a small canvas plot for pictures redrawn on every animation frame (points, segments, labels).
//
// Local explanation: `CanvasPlot` owns one <canvas> sized to its CSS box at the device pixel ratio (sharp on high-DPI
// screens) and redrawn on resize; `draw(scene)` maps a data domain onto the box (optionally with equal x/y scale, for
// circles), draws rectangles (bars), line segments, filled points with a thin outline and text labels; hovering shows
// the label of the point (nearest, within a few px) or rectangle under the mouse in the shared #tooltip, and clicking
// reports the point's index or the rectangle's `id`.
// Global context: the dashboard hero's embedding clock and other per-frame scatters, where a Plotly redraw per frame
// would be too slow (docs/architecture.md, render pipeline).
// Sources: canvas scaling for devicePixelRatio (https://developer.mozilla.org/docs/Web/API/Window/devicePixelRatio);
// ResizeObserver (https://developer.mozilla.org/docs/Web/API/ResizeObserver).
import { cssVar } from "./colors.js";

const tooltip = () => document.getElementById("tooltip");
const HIT = 8; // px: hover / click radius

export class CanvasPlot {
  /** @param {HTMLElement} parent  @param {{height?: number, onClick?: (i) => void, onHover?: (i) => void}} o */
  constructor(parent, { height = 240, onClick = null, onHover = null } = {}) {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "canvas-plot";
    this.canvas.style.height = `${height}px`;
    parent.append(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.scene = null;
    this.onClick = onClick;
    this.onHover = onHover; // called with the point index / rectangle id under the mouse (null when none)
    new ResizeObserver(() => { if (this.scene) this.draw(this.scene); }).observe(this.canvas);
    this.canvas.addEventListener("mousemove", (e) => this.hover(e));
    this.canvas.addEventListener("mouseleave", () => { tooltip().hidden = true; this.onHover?.(null); });
    this.canvas.addEventListener("click", (e) => { const i = this.hit(e); if (i != null) this.onClick?.(i); });
  }

  /**
   * Draw a scene: {domain: [x0, x1, y0, y1], equal (same scale on both axes), rects: [{x0, y0, x1, y1, fill, stroke,
   * id, label}], segments: [{x0, y0, x1, y1, color, width}], points: {x, y, color(i), r (number or r(i)), label(i)},
   * labels: [{x, y, text, color, align ("center" | "left" | "right")}]}.
   */
  draw(scene) {
    this.scene = scene;
    const dpr = devicePixelRatio || 1, w = this.canvas.clientWidth, h = this.canvas.clientHeight, ctx = this.ctx;
    if (!w || !h) return; // not laid out yet (hidden): the ResizeObserver redraws later
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels
    ctx.clearRect(0, 0, w, h);
    const [x0, x1, y0, y1] = scene.domain, pad = 6;
    let sx = (w - 2 * pad) / (x1 - x0), sy = (h - 2 * pad) / (y1 - y0);
    if (scene.equal) sx = sy = Math.min(sx, sy);
    const cx = w / 2, cy = h / 2, mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
    this.px = (x, y) => [cx + (x - mx) * sx, cy - (y - my) * sy]; // data → canvas pixels (y up)
    for (const q of scene.rects || []) {
      const [a, b] = this.px(q.x0, q.y0), [c, d] = this.px(q.x1, q.y1);
      if (q.fill) { ctx.fillStyle = q.fill; ctx.fillRect(Math.min(a, c), Math.min(b, d), Math.abs(c - a), Math.abs(d - b)); }
      if (q.stroke) { ctx.strokeStyle = q.stroke; ctx.lineWidth = 1; ctx.strokeRect(Math.min(a, c) + 0.5, Math.min(b, d) + 0.5, Math.abs(c - a), Math.abs(d - b)); }
    }
    for (const s of scene.segments || []) {
      const [a, b] = this.px(s.x0, s.y0), [c, d] = this.px(s.x1, s.y1);
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.5; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke();
    }
    const pts = scene.points;
    if (pts) {
      ctx.lineWidth = 0.6;
      ctx.strokeStyle = cssVar("--ink-soft"); // outline keeps very light / very dark colours visible in both themes
      for (let i = 0; i < pts.x.length; i++) {
        const [u, v] = this.px(pts.x[i], pts.y[i]);
        ctx.fillStyle = pts.color(i);
        ctx.beginPath(); ctx.arc(u, v, (typeof pts.r === "function" ? pts.r(i) : pts.r) || 3.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      }
    }
    ctx.font = "600 11px Inter, sans-serif";
    for (const l of scene.labels || []) {
      const [u, v] = this.px(l.x, l.y);
      ctx.textAlign = l.align || "center";
      ctx.fillStyle = l.color;
      ctx.fillText(l.text, u, v + 4);
    }
  }

  /** What is under the mouse: the nearest point within HIT px ({i, label}), else a rectangle ({i: its id, label}), or null. */
  target(e) {
    if (!this.scene || !this.px) return null;
    const r = this.canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, pts = this.scene.points;
    let best = null, bd = HIT * HIT;
    for (let i = 0; pts && i < pts.x.length; i++) {
      const [u, v] = this.px(pts.x[i], pts.y[i]), d = (u - mx) ** 2 + (v - my) ** 2;
      if (d < bd) { bd = d; best = { i, label: pts.label?.(i) }; }
    }
    if (best) return best;
    for (const q of this.scene.rects || []) {
      const [a, b] = this.px(q.x0, q.y0), [c, d] = this.px(q.x1, q.y1);
      if (q.id != null && mx >= Math.min(a, c) && mx <= Math.max(a, c) && my >= Math.min(b, d) && my <= Math.max(b, d)) return { i: q.id, label: q.label };
    }
    return null;
  }

  /** Point index or rectangle id under the mouse, or null. */
  hit(e) { return this.target(e)?.i ?? null; }

  hover(e) {
    const t = this.target(e), tt = tooltip();
    this.onHover?.(t ? t.i : null);
    if (!t?.label) { tt.hidden = true; return; }
    tt.innerHTML = t.label;
    tt.hidden = false;
    tt.style.left = `${e.clientX + 14}px`;
    tt.style.top = `${e.clientY + 14}px`;
  }
}
