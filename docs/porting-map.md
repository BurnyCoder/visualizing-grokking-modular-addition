# Porting map — where each function came from and what changed

Abbreviations, with pinned versions:

| Code | Source |
|---|---|
| **P** | `grokking-original.ipynb` in the submodule @ `38776db`. This is the paper's own analysis notebook; "cell N" is the 0-based nbformat index. |
| **T** | `grokking-transformerlens-explained.ipynb` in the submodule @ `38776db`. |
| **PC** | Paper training code [`transformers.py` / `helpers.py`](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper) @ `23d2c64`. |
| **A** | ARENA [`solutions.py` / `utils.py` / `tests.py`](https://github.com/ARENA-education/ARENA_materials/tree/85d10112344b68addebacb04909392e515dd7068/chapter1_transformer_interp/exercises/part52_grokking_and_modular_arithmetic) @ `85d1011`. |

All functions were re-implemented. None was copied verbatim: P, PC and A have no license; T is MIT-derived.

| grokviz function | Source | Change / fix |
|---|---|---|
| `task.all_pairs`, `make_labels` | P34, T21–25, PC `fns_dict` | Row order a·p+b. The `rand` table is seeded; the paper's was unseeded. |
| `task.paper_split` | PC `gen_train_test`, P34, A `gen_train_test` | Uses a private `random.Random(seed)`. Verified to equal the paper's split: the losses reproduce. |
| `model.GrokTransformer` | PC `Embed`…`Transformer`, P24 | Same parameter names, layouts, initialisation and causal-mask arithmetic. Hooks are written as a small dict-based mechanism instead of HookPoint modules. Initialisation is seeded with a private generator. |
| `losses.cross_entropy_f64` | P26, A `cross_entropy_high_precision` | float64, as the notebook and ARENA do. PC `helpers.py` casts to float32 despite its comment. |
| `losses.bias_correct` | P26 `test_logits(bias_correction=True)` | Factored out. |
| `fourier.fourier_basis` | P30, PC `make_fourier_basis`, A `make_fourier_basis` | cos at 2k−1, sin at 2k. T74 uses the opposite order. |
| `fourier.fft2d`, `basis_term_2d`, `cos_apb_dir`, `sin_apb_dir`, `extract_freq_2d` | P31, PC helpers, A `fft2d`, `get_trig_sum_directions` | Adds `ifft2d`. `extract_freq_2d` covers k = 56, which A's `arrange_by_2d_freqs` skips. |
| `fourier.gini` | Hurley & Rickard 2009 | Not in any notebook; the paper cites Hurley & Rickard (p. 8) without writing out the formula. |
| `metrics.key_frequencies` | Paper p.18 text; PC `calculate_key_freqs` | Threshold τ on the W_L Fourier norms (docs/operationalizations.md). |
| `metrics.restricted_loss` | PC `calculate_trig_loss`, P66, A `trig_loss` | Same definition. |
| `metrics.excluded_loss(_per_freq)` | All K at once: P61 `calculate_excluded_loss_2D_full`. One frequency at a time: P60 (every k), PC `calculate_excluded_loss` and A `excl_loss` (k ∈ K) | Same definitions. The per-frequency version covers all 56 frequencies. |
| `metrics.cos_coefficients` | PC `calculate_coefficients`, P50 | Loops over k so memory stays small. |
| `metrics.logit_cos_fit` | Paper Sec. 4.2; T131–136 | OLS with an intercept; reports the centred FVE. T133's "~2 %" comment is wrong: its uncentred residual was 18.8 %. |
| `metrics.neuron_frequencies` | PC `calculate_key_freqs`, P47–49, A `find_neuron_freqs`, `get_frac_explained` | Covers k = 56. Also returns the quadratic share. |
| `metrics.trig_fraction` | A `tensor_trig_ratio` | Same definition. |
| `metrics.attention_lookup` | Paper App. C.1.3, P97, A `W_attn` | Includes W_pos[2] in the query, as A's code does. |
| `metrics.ov_fourier_norms` | Paper Fig. 11 | — |
| `metrics.sum_sq_weights` | PC `take_metrics`, P68, A `sum_sq_weights` | Same definition. |
| `interventions.freq_ablation_table` | Paper Sec. 4.4, Fig. 6 | Adds a random-frequency control distribution. |
| `interventions.neuron_polynomial_replacement`, `cluster_ablations` | Paper Sec. 4.4; A "validate neuron clusters" | MLP-only path through W_L, bias-corrected (reproduces 2.48e-7). |
| `interventions.fit_attention_heads` | Paper Table 2, App. C.1.3 | — |
| `interventions.table1_readoff` | Paper Table 1; A W_logit read-off | — |
| `interventions.relu_multiplier_regression` | A regression cell | Uses A's exact constant 0.5/√p. A's printed label says 0.5. |
| `patching.patch_recovery` | New: counterfactual interchange (Geiger et al. 2021) | — |
| `gradients.*`, `geometry.*` | New; motivated by ARENA §3 "Analysis During Training" and its bonus discussion of memorising vs. generalising gradients | — |

## Bugs found in the sources (documented on the site's errata page)
- **T112:** `W_logit @ fourier_basis` should be `@ fourier_basis.T`.
- **T127:** the phase boundaries 1500/13300/16600 are hardcoded from a different run.
- **T43, T129, T141:** a 99-epoch offset between the loss curves and the checkpoint metrics.
- **A, `arrange_by_2d_freqs`:** it iterates `range(1, p//2)` and so drops k = 56.
- **PC `helpers.cross_entropy_high_precision`:** the code casts to float32 while the comment says float64.
- **P17 `plot_embed_bars`:** labels row 0 of its cos-first basis as "sin", which swaps the sin/cos labels. Whether the
  paper's Fig. 3 inherits the swap cannot be told from the weights, because cos and sin norms differ by ≤ 2.2 % at every
  key frequency (reports/exp-01, item 6). It is therefore not listed as a paper erratum.
