// grokmodel.test.mjs — checks the browser model against PyTorch reference outputs.
//
// Local explanation: tests/test_js_model.py writes weights (float32 and float16), the layout and reference
// activations computed by grokviz.model.GrokTransformer into a temporary folder given by GROKVIZ_JS_FIXTURE,
// then runs this file with `node --test`. Every per-input view on the website depends on these equalities.
// Global context: the site's "pick a and b", attention/neuron grids and patching all call GrokModel.
// Source: Node.js test runner (https://nodejs.org/api/test.html).
import { test } from "node:test"; // built-in test runner
import assert from "node:assert/strict"; // assertions
import { readFileSync } from "node:fs"; // fixture files
import { join } from "node:path"; // path joining
import { GrokModel, decodeF16 } from "../../web/js/model/grokmodel.js"; // module under test

const dir = process.env.GROKVIZ_JS_FIXTURE; // folder prepared by the Python test
const ref = JSON.parse(readFileSync(join(dir, "reference.json"), "utf8")); // layout + reference outputs
const f32 = new Float32Array(readFileSync(join(dir, "weights_f32.bin")).buffer.slice(0)); // exact weights
const u16 = new Uint16Array(readFileSync(join(dir, "weights_f16.bin")).buffer.slice(0)); // half-precision weights
const model = new GrokModel(f32, ref.layout, ref.p); // float32 model

const maxAbs = (xs, ys) => xs.reduce((m, x, i) => Math.max(m, Math.abs(x - ys[i])), 0); // max |difference|

test("logits match PyTorch (float32 weights)", () => {
  for (const { a, b, logits } of ref.samples) assert.ok(maxAbs(model.logits(a, b), logits) < 1e-4, `logits (${a},${b})`);
});

test("detailed forward pass matches the fast path and PyTorch", () => {
  for (const { a, b, logits, pre } of ref.samples.slice(0, 8)) {
    const det = model.forwardDetailed(a, b);
    assert.ok(maxAbs(det.logits, logits) < 1e-4, "detailed logits");
    assert.ok(maxAbs(det.pre, pre) < 1e-4, "pre-activations");
    assert.ok(maxAbs(Array.from(model.pre(a, b)), pre) < 1e-4, "fast pre-activations");
  }
});

test("attention and neuron grids match PyTorch", () => {
  assert.ok(maxAbs(model.attentionGrid(0, 0), ref.attn_grid_h0) < 1e-5, "attention grid head 0");
  assert.ok(maxAbs(model.neuronGrid(7), ref.neuron_grid_7) < 1e-4, "neuron 7 grid");
});

test("float16 decoding reproduces the half-precision model", () => {
  const m16 = new GrokModel(decodeF16(u16), ref.layout, ref.p);
  for (const { a, b, logits_f16 } of ref.samples) assert.ok(maxAbs(m16.logits(a, b), logits_f16) < 1e-4, `f16 logits (${a},${b})`);
});

test("tables round trip: a model rebuilt from toTables() (as the web worker returns them) gives the same outputs", () => {
  const copy = GrokModel.fromTables(f32, ref.layout, ref.p, structuredClone(model.toTables())); // structured clone = postMessage
  for (const { a, b } of ref.samples.slice(0, 8)) {
    assert.ok(maxAbs(copy.logits(a, b), model.logits(a, b)) === 0, `logits (${a},${b})`);
    assert.ok(maxAbs(copy.forwardDetailed(a, b).logits, model.logits(a, b)) < 1e-4, `detailed (${a},${b})`);
  }
  assert.ok(maxAbs(copy.neuronGrid(7), model.neuronGrid(7)) === 0, "neuron grid");
});
