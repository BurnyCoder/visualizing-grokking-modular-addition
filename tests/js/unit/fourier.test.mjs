// fourier.test.mjs — unit tests of the per-neuron frequency coefficients in web/js/model/fourier.js.
//
// Local explanation: a neuron whose input over the token a is a single wave r·cos(w_k a + φ) must land at
// r·√(p/2)·(cos φ, −sin φ) in the (cos k, sin k) plane of the orthonormal basis — on a ring of radius r·√(p/2), at
// its phase — and at the origin for every other frequency.
// Global context: the `neuron-stars` panel plots every MLP neuron at these coordinates, so neurons of one frequency
// cluster form a ring; this is the identity cos(θ + φ) = cos θ cos φ − sin θ sin φ projected on the basis.
// Sources: Node.js test runner (https://nodejs.org/api/test.html); angle-sum identity
// (https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Angle_sum_and_difference_identities).
import { test } from "node:test"; // built-in test runner
import assert from "node:assert/strict"; // assertions
import { freqCoeffs } from "../../../web/js/model/fourier.js";

const p = 113, k = 14;
const neurons = [[2, 0.3], [0.5, -2.1], [1.2, 3]]; // [r, φ] per neuron
const vectors = Array.from({ length: p }, (_, a) => Float32Array.from(neurons, ([r, ph]) => r * Math.cos((2 * Math.PI * k * a) / p + ph)));

test("a single wave of frequency k lands on its ring, at its phase", () => {
  const [x, y] = freqCoeffs(vectors, p, k);
  neurons.forEach(([r, ph], j) => {
    assert.ok(Math.abs(x[j] - r * Math.sqrt(p / 2) * Math.cos(ph)) < 1e-4, `x[${j}] = ${x[j]}`);
    assert.ok(Math.abs(y[j] + r * Math.sqrt(p / 2) * Math.sin(ph)) < 1e-4, `y[${j}] = ${y[j]}`);
  });
});

test("other frequencies see nothing", () => {
  const [x, y] = freqCoeffs(vectors, p, 35);
  for (let j = 0; j < neurons.length; j++) assert.ok(Math.abs(x[j]) < 1e-4 && Math.abs(y[j]) < 1e-4);
});
