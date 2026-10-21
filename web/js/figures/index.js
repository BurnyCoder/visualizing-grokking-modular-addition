// index.js — registry: figure id → lazy loader of its implementation.
//
// Local explanation: each line maps one figure id (as used in content/figures.yaml, chapters and
// content/dashboard.yaml) to a function that imports the module implementing it; modules are loaded only
// when a figure scrolls into view. The site build parses the quoted ids below to verify that every figure
// referenced in the content has an implementation (src/grokviz/site_build.py: js_ids).
// Global context: one implementation per figure, shared by the chapters and the dashboard.
const mod = (path, id) => () => import(path).then((m) => {
  if (!m.figures[id]) throw new Error(`figure ${id} missing in ${path}`);
  return m.figures[id];
});

export const FIGURES = {
  "loss-curves": mod("./curves.js", "loss-curves"),
  "loss-curves-linear": mod("./curves.js", "loss-curves-linear"),
  "acc-curves": mod("./curves.js", "acc-curves"),
  "progress-measures": mod("./curves.js", "progress-measures"),
  "excluded-per-freq": mod("./curves.js", "excluded-per-freq"),
  "gini-curves": mod("./curves.js", "gini-curves"),
  "ssw-curves": mod("./curves.js", "ssw-curves"),
  "trig-loss-ratio": mod("./curves.js", "trig-loss-ratio"),
  "trig-fraction": mod("./curves.js", "trig-fraction"),
  "cos-coeffs": mod("./curves.js", "cos-coeffs"),
  "fve-curves": mod("./curves.js", "fve-curves"),
  "we-fourier-bars": mod("./weights.js", "we-fourier-bars"),
  "wl-fourier-bars": mod("./weights.js", "wl-fourier-bars"),
  "we-fourier-heatmap": mod("./weights.js", "we-fourier-heatmap"),
  "wl-fourier-heatmap": mod("./weights.js", "wl-fourier-heatmap"),
  "we-svals": mod("./weights.js", "we-svals"),
  "attn-C-fourier": mod("./weights.js", "attn-C-fourier"),
  "ov-fourier": mod("./weights.js", "ov-fourier"),
  "wlogit-sorted": mod("./weights.js", "wlogit-sorted"),
  "pos-embed": mod("./weights.js", "pos-embed"),
  "attn-grid": mod("./activations.js", "attn-grid"),
  "attn-grids-all": mod("./activations.js", "attn-grids-all"),
  "avg-attn": mod("./activations.js", "avg-attn"),
  "neuron-grid": mod("./activations.js", "neuron-grid"),
  "neuron-small-multiples": mod("./activations.js", "neuron-small-multiples"),
  "neuron-freq-scatter": mod("./activations.js", "neuron-freq-scatter"),
  "quad-scatter": mod("./activations.js", "quad-scatter"),
  "cluster-sizes": mod("./activations.js", "cluster-sizes"),
  "embed-circles": mod("./geometry.js", "embed-circles"),
  "mlp-circles": mod("./geometry.js", "mlp-circles"),
  "embed-pca": mod("./geometry.js", "embed-pca"),
  "logit-pca": mod("./geometry.js", "logit-pca"),
  "prob-pca": mod("./geometry.js", "prob-pca"),
  "pcorrect-grid": mod("./geometry.js", "pcorrect-grid"),
  "entropy-grid": mod("./geometry.js", "entropy-grid"),
  "split-grid": mod("./geometry.js", "split-grid"),
  "output-dist": mod("./output.js", "output-dist"),
  "interference": mod("./output.js", "interference"),
  "grad-norms": mod("./gradients.js", "grad-norms"),
  "grad-alignment": mod("./gradients.js", "grad-alignment"),
  "grad-vs-decay": mod("./gradients.js", "grad-vs-decay"),
  "grad-we-fourier-heatmap": mod("./gradients.js", "grad-we-fourier-heatmap"),
  "freq-ablation-bars": mod("./interventions.js", "freq-ablation-bars"),
  "patching-result": mod("./interventions.js", "patching-result"),
  "patching-live": mod("./interventions.js", "patching-live"),
  "seed-test-loss": mod("./seeds.js", "seed-test-loss"),
  "seed-key-freqs": mod("./seeds.js", "seed-key-freqs"),
  "hypothesis-table": mod("./seeds.js", "hypothesis-table"),
};
