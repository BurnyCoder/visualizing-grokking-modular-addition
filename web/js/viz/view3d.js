// view3d.js — a small canvas renderer for rotating 3D point clouds (no WebGL needed).
//
// Local explanation: `View3D` draws points given in 3D (flat x, y, z triples) after rotating them by a yaw and a
// pitch angle and projecting them orthographically onto the canvas; points are drawn back to front (sorted by depth)
// and slightly larger and brighter when nearer, which is enough to read the shape. Dragging rotates the cloud, with
// inertia after release; when nobody touches it, it turns slowly by itself (never under reduced motion, never while
// off screen). Hovering shows the nearest point's label in the shared #tooltip; clicking reports its index.
// Global context: the dashboard's torus of all inputs and the 3D principal-component views. A full WebGL engine
// (Three.js) was planned, but point clouds of at most 12,769 points render at full frame rate with the 2D canvas,
// without a 670 KB download, WebGL context limits or a GPU in headless capture.
// Sources: rotation matrices (https://en.wikipedia.org/wiki/Rotation_matrix#Basic_3D_rotations); painter's algorithm
// (https://en.wikipedia.org/wiki/Painter%27s_algorithm); canvas devicePixelRatio scaling
// (https://developer.mozilla.org/docs/Web/API/Window/devicePixelRatio).
import { reducedMotion } from "../core/motion.js";
import { onVisibility } from "../core/visibility.js";

const tooltip = () => document.getElementById("tooltip");

export class View3D {
  /** @param {HTMLElement} parent  @param {{height?: number, onPick?: (i: number) => void}} o */
  constructor(parent, { height = 320, onPick = null } = {}) {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "view3d";
    this.canvas.style.height = `${height}px`;
    parent.append(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    Object.assign(this, { onPick, yaw: 0.6, pitch: 0.5, vyaw: 0, vpitch: 0, touched: false, pts: null, proj: null });
    this.gate = onVisibility(this.canvas, () => this.loop());
    let drag = null;
    this.canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; this.touched = true; this.canvas.setPointerCapture(e.pointerId); });
    this.canvas.addEventListener("pointermove", (e) => {
      if (!drag) { this.hover(e); return; }
      this.vyaw = (e.clientX - drag.x) * 0.01; this.vpitch = (e.clientY - drag.y) * 0.01; // radians per pixel
      this.yaw += this.vyaw; this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + this.vpitch));
      drag = { x: e.clientX, y: e.clientY };
      this.draw();
    });
    this.canvas.addEventListener("pointerup", () => { drag = null; this.loop(); }); // inertia continues in loop()
    this.canvas.addEventListener("mouseleave", () => { tooltip().hidden = true; });
    this.canvas.addEventListener("click", (e) => { const i = this.pick(e); if (i != null && Math.abs(this.vyaw) < 0.02) this.onPick?.(i); });
    new ResizeObserver(() => this.draw()).observe(this.canvas);
  }

  /** Points to draw: xyz (flat triples, roughly within ±1), colors (CSS strings), optional label(i). */
  set(points) {
    this.pts = points;
    this.draw();
    this.loop();
  }

  /** Animate inertia, then a slow turn when untouched; stops off screen, when still, and under reduced motion. */
  loop() {
    if (this.raf) return;
    let last = 0; // time of the last idle auto-turn frame
    const tick = (t) => {
      this.raf = null;
      if (!this.gate.visible() || !this.pts) return;
      const spinning = Math.abs(this.vyaw) > 1e-4 || Math.abs(this.vpitch) > 1e-4;
      if (!spinning && (this.touched || reducedMotion())) return;
      if (spinning) { this.yaw += this.vyaw; this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + this.vpitch)); this.vyaw *= 0.94; this.vpitch *= 0.94; this.draw(); }
      else if (t - last > 50) { last = t; this.yaw += 0.012; this.draw(); } // idle auto-turn: 20 redraws per second
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  draw() {
    const pts = this.pts, dpr = devicePixelRatio || 1, w = this.canvas.clientWidth, h = this.canvas.clientHeight, ctx = this.ctx;
    if (!pts || !w || !h) return;
    if (this.canvas.width !== Math.round(w * dpr)) { this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const n = pts.xyz.length / 3, cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const scale = 0.47 * Math.min(w, h), proj = this.proj = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) { // yaw about the vertical axis, then pitch about the horizontal one
      const x = pts.xyz[3 * i], y = pts.xyz[3 * i + 1], z = pts.xyz[3 * i + 2];
      const x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
      const y2 = cp * y - sp * z1, z2 = sp * y + cp * z1;
      proj[3 * i] = w / 2 + scale * x1; proj[3 * i + 1] = h / 2 - scale * y2; proj[3 * i + 2] = z2;
    }
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => proj[3 * a + 2] - proj[3 * b + 2]); // far → near
    const base = pts.r ?? (n > 2000 ? 1.6 : 3.2);
    for (const i of order) {
      const depth = proj[3 * i + 2]; // about −1 (far) … +1 (near)
      ctx.globalAlpha = 0.55 + 0.45 * Math.max(0, Math.min(1, (depth + 1) / 2));
      ctx.fillStyle = pts.colors[i];
      const r = base * (1 + 0.25 * depth);
      if (n > 2000) ctx.fillRect(proj[3 * i] - r, proj[3 * i + 1] - r, 2 * r, 2 * r); // big clouds: squares are much faster
      else { ctx.beginPath(); ctx.arc(proj[3 * i], proj[3 * i + 1], r, 0, 2 * Math.PI); ctx.fill(); }
    }
    ctx.globalAlpha = 1;
  }

  /** Index of the nearest drawn point within 6 px of the mouse (the nearest to the viewer wins ties), or null. */
  pick(e) {
    if (!this.proj) return null;
    const rect = this.canvas.getBoundingClientRect(), mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = null, bd = 36;
    for (let i = 0; i < this.proj.length / 3; i++) {
      const d = (this.proj[3 * i] - mx) ** 2 + (this.proj[3 * i + 1] - my) ** 2;
      if (d < bd || (d === bd && best != null && this.proj[3 * i + 2] > this.proj[3 * best + 2])) { bd = d; best = i; }
    }
    return best;
  }

  hover(e) {
    const i = this.pick(e), tt = tooltip(), label = this.pts?.label;
    if (i == null || !label) { tt.hidden = true; return; }
    tt.innerHTML = label(i);
    tt.hidden = false;
    tt.style.left = `${e.clientX + 14}px`;
    tt.style.top = `${e.clientY + 14}px`;
  }
}
