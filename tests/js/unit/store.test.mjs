// store.test.mjs — unit tests of the update pipeline primitives in web/js/core/store.js.
//
// Local explanation: `createStore` must merge every `set` immediately but notify listeners only once per flush,
// with the union of the keys that changed; child ("scoped") stores own some keys and forward the rest to their
// parent; `latestWins` must never run two renders at once and must collapse queued triggers into one run.
// Global context: every figure redraw on the site goes store → flush → latestWins → figure, so these properties
// are what keeps playback smooth and frames in order (bugs "out-of-order renders" and "one redraw per set").
// Source: Node.js test runner (https://nodejs.org/api/test.html).
import { test } from "node:test"; // built-in test runner
import assert from "node:assert/strict"; // assertions
import { createStore, latestWins } from "../../../web/js/core/store.js"; // modules under test

/** A store whose flushes run only when the test calls `run()` (instead of on the next animation frame). */
function manual(initial, extra = {}) {
  const queue = []; // scheduled flush callbacks
  const store = createStore(initial, { schedule: (fn) => queue.push(fn), ...extra });
  return { store, run: () => { while (queue.length) queue.shift()(); }, queue };
}

test("set merges at once but notifies once per flush with the union of changed keys", () => {
  const { store, run, queue } = manual({ a: 1, b: 2, c: 3 });
  const calls = [];
  store.subscribe(["a", "b"], (s, changed) => calls.push([{ ...s }, changed]), { immediate: false });
  store.set({ a: 10 });
  store.set({ b: 20 });
  assert.equal(store.state.a, 10, "state is updated immediately");
  assert.equal(queue.length, 1, "one flush scheduled for several sets");
  assert.equal(calls.length, 0, "no notification before the flush");
  run();
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0][1].sort(), ["a", "b"]);
  assert.equal(calls[0][0].b, 20);
});

test("unchanged values and unrelated keys do not notify", () => {
  const { store, run } = manual({ a: 1, b: 2 });
  let n = 0;
  store.subscribe(["a"], () => n++, { immediate: false });
  store.set({ a: 1 }); // same value
  store.set({ b: 5 }); // other key
  run();
  assert.equal(n, 0);
});

test("subscribe calls immediately with no changed keys; null keys listen to everything", () => {
  const { store, run } = manual({ a: 1 });
  const seen = [];
  store.subscribe(null, (_, changed) => seen.push(changed));
  store.set({ a: 2 });
  run();
  assert.deepEqual(seen, [[], ["a"]]);
});

test("a listener error does not stop the others, and unsubscribing during a flush is safe", () => {
  const { store, run } = manual({ a: 1 });
  const errors = [];
  const orig = console.error;
  console.error = (...x) => errors.push(x);
  let second = 0;
  const off = store.subscribe(["a"], () => { off(); throw new Error("boom"); }, { immediate: false });
  store.subscribe(["a"], () => second++, { immediate: false });
  store.set({ a: 2 });
  run();
  console.error = orig;
  assert.equal(second, 1);
  assert.equal(errors.length, 1);
  store.set({ a: 3 });
  run();
  assert.equal(second, 2, "the unsubscribed listener is gone, the other still runs");
});

test("a scoped child store owns its keys and forwards the rest to the parent", () => {
  const { store: parent, run } = manual({ run: "paper", step: 100, a: 1 });
  const queue = [];
  const child = createStore({ step: 0, playing: false }, { parent, schedule: (fn) => queue.push(fn) });
  assert.equal(child.state.run, "paper", "inherits the parent's keys");
  assert.equal(child.state.step, 0, "owns its step");
  const childCalls = [], parentCalls = [];
  child.subscribe(null, (_, c) => childCalls.push(c), { immediate: false });
  parent.subscribe(null, (_, c) => parentCalls.push(c), { immediate: false });
  child.set({ step: 500, a: 7 }); // step stays local, a goes to the parent
  assert.equal(parent.state.step, 100, "the parent's step is untouched");
  assert.equal(parent.state.a, 7);
  while (queue.length) queue.shift()();
  run();
  assert.deepEqual(childCalls.flat().sort(), ["a", "step"], "the child hears its own step and the forwarded a");
  assert.deepEqual(parentCalls, [["a"]]);
  childCalls.length = 0;
  parent.set({ step: 900, run: "wd1_s0" }); // the parent's step must not leak into the scope
  run();
  assert.equal(child.state.step, 500);
  assert.equal(child.state.run, "wd1_s0");
  assert.deepEqual(childCalls, [["run"]], "parent changes reach the child in the same flush");
});

test("latestWins runs one call at a time and collapses queued triggers into the latest merged one", async () => {
  const log = [];
  let release;
  const gate = () => new Promise((r) => { release = r; });
  const draw = latestWins(async (keys) => { log.push(keys); await gate(); }, (a, b) => [...new Set([...a, ...b])]);
  const first = draw(["step"]);
  draw(["a"]);
  draw(["b"]);
  assert.deepEqual(log, [["step"]], "only the first call has started");
  release();
  await new Promise((r) => setImmediate(r));
  assert.deepEqual(log, [["step"], ["a", "b"]], "the queued calls ran once, merged");
  release();
  await first;
  assert.equal(log.length, 2);
});

test("latestWins keeps going after a failing call", async () => {
  const orig = console.error;
  console.error = () => {};
  let n = 0;
  const draw = latestWins(async (x) => { n++; if (x === 1) throw new Error("fail"); });
  await draw(1);
  await draw(2);
  console.error = orig;
  assert.equal(n, 2);
});
