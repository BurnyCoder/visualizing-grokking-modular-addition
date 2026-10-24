// colors.test.mjs — unit test of the frequency colours in web/js/viz/colors.js.
//
// Local explanation: `freqColor(k)` must be a pure function of k (same frequency, same colour on every page) and must
// give the key frequencies of any one run different colours, or two lines / bars of one chart become indistinguishable.
// The key sets below are copied from runs/*/analysis/key_freqs.json (key frequencies at the reference checkpoint);
// rand_s0–2 and wd0_s0–4 are left out because every one of their 56 frequencies passes the key rule (no sparse key set;
// figures/seeds.js draws only runs with ≤ 20 key frequencies), and 10 colours cannot separate 56.
// Global context: activations, curves, output, seeds, weights and the trig widget colour frequencies with freqColor.
// Source: Node.js test runner (https://nodejs.org/api/test.html).
import { test } from "node:test"; // built-in test runner
import assert from "node:assert/strict"; // assertions
import { freqColor } from "../../../web/js/viz/colors.js";

const KEY_FREQS = { // run id → key frequencies (runs/<id>/analysis/key_freqs.json)
  paper: [14, 35, 41, 42, 52],
  split0_init100: [2, 14, 27, 30, 37, 52],
  split0_init101: [9, 14, 15, 27],
  split0_init102: [7, 38, 49, 53],
  wd1_s0: [14, 35, 41, 42, 52],
  wd1_s1: [2, 3, 23, 37, 39],
  wd1_s2: [6, 7, 15, 27, 38],
  wd1_s3: [9, 27, 53, 55],
  wd1_s4: [11, 25, 49],
  wd1_s5: [9, 34, 38, 40],
  wd1_s6: [7, 26, 35, 41],
  wd1_s7: [12, 18, 20, 48, 55],
  wd1_s8: [37, 44, 45, 46, 47, 51],
  wd1_s9: [19, 34, 35],
  wd1_s10: [19, 31, 45, 47],
};

test("no two key frequencies of one run share a colour", () => {
  for (const [id, ks] of Object.entries(KEY_FREQS)) {
    const colours = ks.map(freqColor);
    assert.equal(new Set(colours).size, ks.length, `${id}: ${ks.map((k, i) => `k=${k} ${colours[i]}`).join(", ")}`);
  }
});

test("freqColor is a pure function of k: a hex colour for every frequency 1 … 56, the same on every call", () => {
  for (let k = 1; k <= 56; k++) {
    assert.match(freqColor(k), /^#[0-9a-f]{6}$/);
    assert.equal(freqColor(k), freqColor(k));
  }
});
