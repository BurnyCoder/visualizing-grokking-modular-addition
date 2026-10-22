// overlay.test.mjs — unit tests of stepToPixel in web/js/viz/overlay.js.
//
// Local explanation: the step cursor is a translated <div> drawn over a Plotly chart instead of a Plotly shape, so
// its pixel position must be computed exactly as Plotly maps data to pixels: linearly in the value on linear axes and
// in log10(value) on log axes, whose `range` Plotly stores in log10 units.
// Global context: every metric-over-training chart moves this cursor on every played frame (bug: Plotly.relayout per
// frame), so the mapping has to be right on both axis types.
// Sources: Plotly axis `range` for log axes is in log10 units (https://plotly.com/javascript/reference/layout/xaxis/#layout-xaxis-range);
// Node.js test runner (https://nodejs.org/api/test.html).
import { test } from "node:test"; // built-in test runner
import assert from "node:assert/strict"; // assertions
import { stepToPixel } from "../../../web/js/viz/overlay.js"; // function under test

test("linear axis: offset + fraction of the range × length", () => {
  const ax = { range: [0, 100], type: "linear", offset: 50, length: 200 };
  assert.equal(stepToPixel(0, ax), 50);
  assert.equal(stepToPixel(50, ax), 150);
  assert.equal(stepToPixel(100, ax), 250);
});

test("log axis: the range is in log10 units", () => {
  const ax = { range: [0, 2], type: "log", offset: 10, length: 100 };
  assert.equal(stepToPixel(10, ax), 60);
  assert.ok(Math.abs(stepToPixel(100, ax) - 110) < 1e-9);
});

test("values outside the visible range or invalid on a log axis give null", () => {
  assert.equal(stepToPixel(150, { range: [0, 100], type: "linear", offset: 0, length: 100 }), null);
  assert.equal(stepToPixel(-1, { range: [0, 100], type: "linear", offset: 0, length: 100 }), null);
  assert.equal(stepToPixel(0, { range: [0, 2], type: "log", offset: 0, length: 100 }), null);
  assert.equal(stepToPixel(null, { range: [0, 2], type: "linear", offset: 0, length: 100 }), null);
});

test("reversed ranges map the larger value to the left edge", () => {
  assert.equal(stepToPixel(100, { range: [100, 0], type: "linear", offset: 0, length: 100 }), 0);
  assert.equal(stepToPixel(25, { range: [100, 0], type: "linear", offset: 0, length: 100 }), 75);
});
