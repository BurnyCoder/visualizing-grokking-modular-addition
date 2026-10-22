// anim.js — pure animation maths: playback clock, ranges from the training phases, frame lookup, interpolation.
//
// Local explanation: playback moves a fractional checkpoint index `pos` (0 … number of checkpoints − 1) at speed ×
// RATE checkpoints per second; `advance` also handles the end of the range (hold, then loop / ping-pong / stop).
// `rangePresets` turns the detected phase boundaries t1, t2, t3 into index ranges (e.g. "grokking" = 0 … 1.1·t3);
// `stepAt` maps a fractional index to the nearest stored step and the interpolated step; `frameAt` locates a
// fractional step between two stored frames (geometry or weight snapshots are sparser than checkpoints); `lerpInto`,
// `lerpPolar2` and `alignSigns` interpolate between two stored frames without artefacts.
// Global context: core/player.js drives every animation with these; figures interpolate only while playing, and a
// paused view always shows a stored frame (docs/content-style.md).
// Pure module (no DOM) so tests/js/unit/anim.test.mjs can test it with Node.
// Sources: linear interpolation (https://en.wikipedia.org/wiki/Linear_interpolation); shortest signed angle
// difference via atan2 (https://en.wikipedia.org/wiki/Atan2); PCA axes are defined only up to sign
// (https://en.wikipedia.org/wiki/Principal_component_analysis#Properties).

export const RATE = 10; // checkpoints per second at 1×
export const SPEEDS = [0.25, 0.5, 1, 2, 4, 8]; // speed choices offered by the player bar

/**
 * Playback state after dt seconds. `st` = {pos, dir (+1/−1), hold (seconds left waiting at an end)};
 * options {lo, hi, speed, loop: "loop" | "pingpong" | "once", holdEnd (seconds to wait at an end)}.
 */
export function advance(st, dt, { lo, hi, speed = 1, loop = "loop", holdEnd = 1 }) {
  let { pos, dir = 1, hold = 0 } = st;
  if (hold > 0) { // waiting at an end of the range
    hold = Math.max(0, hold - dt);
    if (hold > 0) return { pos, dir, hold, done: false };
    return loop === "loop" ? { pos: lo, dir: 1, hold: 0, done: false } : { pos, dir, hold: 0, done: false };
  }
  pos += dir * speed * RATE * dt;
  const wait = Math.max(holdEnd, 1e-6); // > 0 so the next call leaves the end
  if (dir > 0 && pos >= hi) {
    if (loop === "once") return { pos: hi, dir, hold: 0, done: true };
    return { pos: hi, dir: loop === "pingpong" ? -1 : 1, hold: wait, done: false };
  }
  if (dir < 0 && pos <= lo) return { pos: lo, dir: 1, hold: wait, done: false };
  return { pos, dir, hold: 0, done: false };
}

/** Index of the stored step closest to `step` in a sorted list (same rule as core/data.js nearestIndex). */
export function nearest(steps, step) {
  let lo = 0, hi = steps.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (steps[mid] < step) lo = mid + 1; else hi = mid; } // first ≥ step
  if (lo > 0 && Math.abs(steps[lo - 1] - step) <= Math.abs(steps[lo] - step)) lo -= 1;
  return lo;
}

/** Named index ranges [lo, hi] into `steps` from phase boundaries {t1, t2, t3} (whole run when none were detected). */
export function rangePresets(steps, phases) {
  const n = steps.length - 1, at = (s) => nearest(steps, s);
  const whole = [0, n];
  if (!phases?.t3) return { grokking: whole, whole };
  const { t1, t2, t3 } = phases;
  return {
    grokking: [0, at(1.1 * t3)], whole, // the interesting part: memorisation, circuit formation, cleanup and a margin
    memorisation: [0, at(t1)], circuit: [at(t1), at(t2)], cleanup: [at(t2), at(t3)], late: [at(t3), n],
  };
}

/** Name of the phase containing `step` ("" when phases were not detected). */
export function phaseOf(step, phases) {
  if (!phases?.t1) return "";
  if (step < phases.t1) return "memorisation";
  if (phases.t2 == null || step < phases.t2) return "circuit formation";
  if (phases.t3 == null || step < phases.t3) return "cleanup";
  return "after cleanup";
}

/** Nearest stored step and interpolated (fractional) step at fractional index `pos` of `steps`. */
export function stepAt(steps, pos) {
  const i = Math.min(Math.max(Math.floor(pos), 0), steps.length - 1), j = Math.min(i + 1, steps.length - 1);
  const f = Math.min(Math.max(pos - i, 0), 1);
  return { step: steps[Math.min(Math.max(Math.round(pos), 0), steps.length - 1)], stepf: steps[i] + (steps[j] - steps[i]) * f };
}

/** Stored frames around step `s` in sorted `frameSteps`: indices i ≤ j and the fraction t ∈ [0, 1] from i to j. */
export function frameAt(frameSteps, s) {
  const n = frameSteps.length;
  let i = 0;
  while (i + 1 < n && frameSteps[i + 1] <= s) i++; // last frame at or before s (frame lists are short)
  const j = Math.min(i + 1, n - 1);
  const t = j === i || s <= frameSteps[i] ? 0 : Math.min((s - frameSteps[i]) / (frameSteps[j] - frameSteps[i]), 1);
  return { i, j, t };
}

/** out = (1 − t)·a + t·b elementwise; returns out. */
export function lerpInto(out, a, b, t) {
  for (let k = 0; k < out.length; k++) out[k] = a[k] + (b[k] - a[k]) * t;
  return out;
}

/**
 * Interpolate 2D points (flat x, y pairs) in polar coordinates: radius linearly, angle along the shorter arc. Points on
 * a rotating circle stay on (near) the circle instead of cutting through its centre as straight-line interpolation does.
 */
export function lerpPolar2(out, a, b, t) {
  for (let k = 0; k < out.length; k += 2) {
    const ra = Math.hypot(a[k], a[k + 1]), rb = Math.hypot(b[k], b[k + 1]);
    const fa = Math.atan2(a[k + 1], a[k]), fb = Math.atan2(b[k + 1], b[k]);
    const d = Math.atan2(Math.sin(fb - fa), Math.cos(fb - fa)); // signed shortest angle from a to b
    const r = ra + (rb - ra) * t, f = fa + d * t;
    out[k] = r * Math.cos(f);
    out[k + 1] = r * Math.sin(f);
  }
  return out;
}

/** Copy of points x [n, dims] with each axis's sign flipped where it disagrees with the same axis of `ref`. */
export function alignSigns(ref, x, dims) {
  const out = Float32Array.from(x);
  for (let j = 0; j < dims; j++) {
    let dot = 0;
    for (let k = j; k < x.length; k += dims) dot += ref[k] * x[k];
    if (dot < 0) for (let k = j; k < x.length; k += dims) out[k] = -x[k];
  }
  return out;
}

/** Smooth start and stop for seek tweens (cubic ease-in-out on t ∈ [0, 1]). */
export const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
