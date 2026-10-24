// anim.test.mjs — unit tests of the pure animation maths in web/js/core/anim.js and the Player in core/player.js.
//
// Local explanation: playback advances a fractional checkpoint index by speed × 10 checkpoints per second, holds at
// the end of the range and loops / ping-pongs / stops; ranges come from the detected phases; interpolation helpers
// must reproduce the stored frames exactly at t = 0 and t = 1 and never pull points through the origin. The Player
// is driven by a fake animation-frame clock so its timing is deterministic.
// Global context: every animated figure and the dashboard's play button rely on these functions; "paused = the stored
// frame" (docs/content-style.md) is checked here too.
// Source: Node.js test runner (https://nodejs.org/api/test.html).
import { test } from "node:test"; // built-in test runner
import assert from "node:assert/strict"; // assertions
import { advance, RATE, rangePresets, phaseOf, frameAt, lerpInto, lerpPolar2, alignSigns, stepAt } from "../../../web/js/core/anim.js";
import { Player } from "../../../web/js/core/player.js";
import { createStore } from "../../../web/js/core/store.js";

const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

test("advance moves speed × RATE checkpoints per second and holds, then wraps, at the end", () => {
  const o = { lo: 0, hi: 10, speed: 2, loop: "loop", holdEnd: 1 };
  let st = advance({ pos: 0 }, 0.1, o);
  assert.ok(close(st.pos, 2 * RATE * 0.1));
  st = advance({ pos: 9.5, dir: 1 }, 0.1, o); // overshoots → clamps to the end and holds
  assert.deepEqual([st.pos, st.hold > 0, st.done], [10, true, false]);
  st = advance(st, 0.5, o);
  assert.equal(st.pos, 10, "still holding");
  st = advance(st, 0.6, o);
  assert.equal(st.pos, 0, "wrapped to the start after the hold");
});

test("ping-pong reverses at both ends; once stops at the end", () => {
  const pp = { lo: 2, hi: 4, speed: 1, loop: "pingpong", holdEnd: 0 };
  let st = advance({ pos: 3.9, dir: 1 }, 0.1, pp);
  assert.deepEqual([st.pos, st.dir], [4, -1]);
  st = advance(advance(st, 0.01, pp), 0.15, pp);
  assert.ok(st.pos < 4 && st.dir === -1, "moving back down");
  st = advance({ pos: 2.05, dir: -1 }, 0.1, pp);
  assert.deepEqual([st.pos, st.dir], [2, 1]);
  const once = advance({ pos: 9.9, dir: 1 }, 0.1, { lo: 0, hi: 10, speed: 1, loop: "once" });
  assert.deepEqual([once.pos, once.done], [10, true]);
});

test("range presets come from the detected phases (indices into the checkpoint list)", () => {
  const steps = Array.from({ length: 501 }, (_, i) => i * 100); // 0 … 50,000
  const r = rangePresets(steps, { t1: 1500, t2: 9300, t3: 14000 });
  assert.deepEqual(r.grokking, [0, 154], "0 to 1.1 · t3 = 15,400");
  assert.deepEqual(r.whole, [0, 500]);
  assert.deepEqual(r.memorisation, [0, 15]);
  assert.deepEqual(r.circuit, [15, 93]);
  assert.deepEqual(r.cleanup, [93, 140]);
  assert.deepEqual(r.late, [140, 500]);
  assert.deepEqual(rangePresets(steps, null).grokking, [0, 500], "no phases → the whole run");
  assert.equal(phaseOf(1000, { t1: 1500, t2: 9300, t3: 14000 }), "memorisation");
  assert.equal(phaseOf(20000, { t1: 1500, t2: 9300, t3: 14000 }), "after cleanup");
  assert.equal(phaseOf(20000, null), "");
});

test("each phase boundary closes the phase it ends (t2 = end of circuit formation, as in phases.py)", () => {
  const ph = { t1: 1567, t2: 9301, t3: 14001 }; // the paper run's detected boundaries
  assert.equal(phaseOf(ph.t1, ph), "memorisation");
  assert.equal(phaseOf(ph.t1 + 1, ph), "circuit formation");
  assert.equal(phaseOf(ph.t2, ph), "circuit formation");
  assert.equal(phaseOf(ph.t2 + 1, ph), "cleanup");
  assert.equal(phaseOf(ph.t3, ph), "cleanup");
  assert.equal(phaseOf(ph.t3 + 1, ph), "after cleanup");
});

test("a run without a detected t2 is never labelled circuit formation", () => {
  const ph = { t1: 11817, t2: null, t3: null }; // wd0_s0: memorises, never reaches the t2 rule
  assert.equal(phaseOf(ph.t1, ph), "memorisation");
  assert.equal(phaseOf(ph.t1 + 1, ph), "no circuit formation detected");
  assert.equal(phaseOf(40000, ph), "no circuit formation detected");
});

test("stepAt and frameAt locate fractional positions between stored frames", () => {
  const steps = [0, 100, 300];
  assert.deepEqual(stepAt(steps, 1.5), { step: 300, stepf: 200 }, "nearest stored step and the interpolated step");
  assert.deepEqual(stepAt(steps, 1.4), { step: 100, stepf: 180 });
  assert.deepEqual(frameAt(steps, 200), { i: 1, j: 2, t: 0.5 });
  assert.deepEqual(frameAt(steps, 300), { i: 2, j: 2, t: 0 });
  assert.deepEqual(frameAt(steps, -5), { i: 0, j: 1, t: 0 });
});

test("interpolation reproduces stored frames and keeps circle points off the origin", () => {
  const a = Float32Array.from([1, 0, 0, 1]), b = Float32Array.from([-1, 0, 0, -1]);
  const out = new Float32Array(4);
  assert.deepEqual([...lerpInto(out, a, b, 0)], [...a]);
  assert.deepEqual([...lerpInto(out, a, b, 1)], [...b]);
  lerpPolar2(out, a, b, 0.5); // half-way along the shorter arc: radius stays 1
  assert.ok(close(Math.hypot(out[0], out[1]), 1, 1e-6) && close(Math.hypot(out[2], out[3]), 1, 1e-6));
  lerpPolar2(out, a, b, 1);
  assert.ok(close(out[0], -1, 1e-6) && close(out[3], -1, 1e-6));
});

test("alignSigns flips principal axes to agree with the previous frame", () => {
  const ref = Float32Array.from([1, 2, 3, 4]); // 2 points × 2 dims
  const x = Float32Array.from([-1, 2, -3, 4]); // first axis flipped
  assert.deepEqual([...alignSigns(ref, x, 2)], [1, 2, 3, 4]);
});

/** A Player on a fresh store, driven by a hand-cranked animation clock. */
function rig(opts = {}) {
  const store = createStore({ step: 0, stepf: null, playing: false }, { schedule: (f) => f() }); // synchronous flushes
  let t = 0, cb = null;
  const player = new Player(store, { raf: (f) => { cb = f; }, now: () => t, ...opts });
  const frame = (ms) => { t += ms; const f = cb; cb = null; f?.(t); };
  player.setSteps(Array.from({ length: 101 }, (_, i) => i * 100), { t1: 1000, t2: 4000, t3: 6000 });
  return { store, player, frame };
}

test("the Player publishes the nearest stored step while playing and the stored frame when paused", () => {
  const { store, player, frame } = rig();
  assert.deepEqual(player.range, [0, 66], "default range: 0 to 1.1 · t3");
  player.play();
  frame(0);
  frame(250); // 1× = 10 checkpoints / s → 2.5 checkpoints
  assert.equal(store.state.playing, true);
  assert.equal(store.state.step, 300, "nearest stored checkpoint to index 2.5");
  assert.ok(close(store.state.stepf, 250));
  player.pause();
  assert.deepEqual([store.state.playing, store.state.stepf, store.state.step], [false, null, 300], "paused = the stored frame");
});

test("an outside step change (a click) pauses the Player and moves it there", () => {
  const { store, player, frame } = rig();
  player.play();
  frame(0);
  frame(100);
  store.set({ step: 5000 });
  assert.equal(player.playing, false);
  assert.equal(player.pos, 50);
});

test("reduced motion: discrete steps, no interpolation, instant seeks", () => {
  const { store, player, frame } = rig({ reduced: () => true });
  player.seek(10, { tween: true });
  assert.equal(store.state.step, 1000, "seek is immediate");
  player.play();
  frame(0);
  for (let i = 0; i < 3; i++) frame(250); // 0.75 s at 2 checkpoints / s (one long frame would be capped at 0.25 s)
  assert.ok(Number.isInteger(player.pos) && player.pos > 10, "moves by whole checkpoints");
  assert.equal(store.state.stepf, store.state.step, "no fractional position");
});
