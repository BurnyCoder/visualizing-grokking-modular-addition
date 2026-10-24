# Operationalizations — exact definitions of every measured quantity (frozen)

**Status:** frozen at tag `prereg-v1`. These definitions were fixed after being checked on the paper's
own model only (`runs/paper`, epoch 40,000), and **before** any of our own training runs was analysed.
Later changes must be logged in the "Deviations" section of `docs/preregistration.md`.

## Notation

| Symbol | Meaning |
|---|---|
| p | 113, the prime modulus |
| a, b, c | inputs a, b ∈ {0..p−1}; c indexes the p possible answers |
| L | logits at the `=` position for all p² inputs, shape p²×p. The `=` output is dropped, as in the paper's code (`[:, -1, :-1]`). |
| F | p×p orthonormal Fourier basis. Row 0 is Const; row 2k−1 is cos k; row 2k is sin k; k = 1..56. Same ordering as the paper's code. |
| w_k | 2πk/p |
| d_cos,k | (F_{2k−1}⊗F_{2k−1} − F_{2k}⊗F_{2k})/√2, the unit vector over the (a,b) grid proportional to cos(w_k(a+b)) |
| d_sin,k | (F_{2k}⊗F_{2k−1} + F_{2k−1}⊗F_{2k})/√2, the unit vector proportional to sin(w_k(a+b)) |
| W_E | d×p embedding matrix, without the `=` column |
| W_U | p×d unembedding matrix |
| W_out | d×n MLP output matrix |
| W_L | W_U W_out, the p×n neuron-logit map (paper Sec. 3) |
| K | key frequencies: computed once per run from a reference checkpoint and held fixed over training |

**Reference checkpoint for K:** the paper run uses `state_dicts[400]`, i.e. step 40,001. Our runs use their final checkpoint.

## Definitions

### Losses
Cross-entropy, computed in float64.
- **Training objective:** uses all 114 outputs, as the paper's training code does.
- **Every analysis metric:** uses the 113 real answers.
- **Masks:** "train" and "test" select rows of the full grid with the run's split. "All" means all p² rows.

### Key frequencies K
K = {k : n_k ≥ τ·max_j n_j}, where n_k = √(‖F_{2k−1} W_L‖² + ‖F_{2k} W_L‖²) and τ = **0.25**.
- On the paper model, every τ in (0.060, 0.826] gives {14, 35, 41, 42, 52}.
- This matches the paper's text: take the DFT of W_L, keep frequencies with nontrivial coefficients (p.18).

### Restricted loss
L_r = bias_correct(P_K L), evaluated on **all pairs**.
- P_K projects each logit column onto span{d_cos,k, d_sin,k : k∈K}.
- bias_correct adds back mean_rows(L − P_K L), which is the Const⊗Const term.
- Source: paper code `calculate_trig_loss` and ARENA `trig_loss`.
- Variants also stored: train-only and test-only.
- Check on the paper model: 5.547e-8, ARENA's value.

### Excluded loss
L − P_K L, evaluated on the **train** pairs, with no bias correction.
- Source: paper code `calculate_excluded_loss` and ARENA `excl_loss`.
- Per-frequency variant: remove one k at a time, for k = 1..56.
- K-free variant: the maximum over k (paper footnote 3). This is the only excluded loss used for the wd=0 and random-label controls.
- Check on the paper model: ARENA's test values, exactly.

### Logit-frequency ablation (paper Fig. 6)
Take the 2D DFT of each logit column. "Ablate k" zeroes the 8 components of k, (Const, cos k, sin k)² minus Const⊗Const. "Keep only K" keeps those 8·|K| = 40 components and zeroes everything else, including Const⊗Const.
- Check on the paper model: 7.236e-8 vs the paper's 7.24e-8.

### Sum of squared weights (SSW)
Σ θ² over **every** named parameter. This includes the `=` row/column of W_E and W_U, W_pos, all attention matrices, W_in, b_in, W_out and b_out.
- Source: paper code `take_metrics.sum_sq_weights`.

### Gini
G(c) = 1 − 2 Σ_{k=1..N} (c_(k)/‖c‖₁)(N − k + ½)/N, with c sorted ascending (Hurley & Rickard 2009, arXiv:0811.4706).
- For W_E, c is the 113 per-component norms ‖(W_E Fᵀ)_{:,i}‖.
- For W_L, c is the 113 norms ‖(F W_L)_{i,:}‖.
- Paper model at 40k: G(W_E) = 0.736, G(W_L) = 0.801, read against Fig. 7.

### Neuron frequency and FVE
- Centre each neuron's activations over the grid and take its 2D DFT.
- For each k, energy_k is the sum of squares of the 3×3 block (Const, cos k, sin k)². Take the argmax k.
- FVE = energy_k / total.
- A neuron is "clustered" if FVE > 0.85.
- Source: paper code `calculate_key_freqs` and ARENA `find_neuron_freqs`, but covering k = 1..56; ARENA stops at 55.
- Check on the paper model: 433 neurons, with clusters 44/93/145/87/64, plus 79 always-firing neurons.

### Logit cosine fit
OLS of the logit cube on {cos(w_k(a+b−c))}_{k∈K} plus an intercept. FVE is centred.
- Paper model: 0.961 vs the paper's 0.95.

### Trig fraction (ARENA `tensor_trig_ratio`)
For each column, the fraction of centred variance lying in span{d_cos,k, d_sin,k}, then averaged over columns. Computed for neuron pre-activations, post-activations and logits.

### Attention lookup C_j
C_j[a] = (W_Q^j x₂)·(W_K^j W_E[:,a]) / √d_head, where x₂ = W_E[:, '='] + W_pos[2].
- Its Fourier fit: Const + F cos + G sin at the dominant k.
- Table 2 fit: A₀ ≈ 0.5 + α(cos w_k a − cos w_k b) + β(sin w_k a − sin w_k b).
- Softmax linearisation: A₀ ≈ β₀ + E(C_j[a] − C_j[b]), fitted by least squares; attention to `=` is set to 0.

## Phase boundaries
These use only losses, accuracy and SSW, never the progress measures. Constants were calibrated on the paper run.

- **t1**: first step after which train loss falls by < 1 % over the next 100 steps.
  - Paper run: 1,567. Paper: ≈1.4k, drawn by eye.
- **t2**: first checkpoint after t1 with test loss ≤ 0.2 × median(test loss over [t1, t1+2000]).
  - Paper run: 9,301. Paper: 9.4k.
- **t3**: first checkpoint after t2 with SSW ≤ SSW_final + 0.05·(SSW(t2) − SSW_final), where SSW_final is the median over the last 5,000 steps.
  - Paper run: 14,001. Paper: 14k.
- **t_grok**: first checkpoint with test accuracy ≥ 0.99.
  - Paper run: 10,601.

## Step semantics
A checkpoint at step s holds the parameters after s optimizer steps. For the paper's file this was established empirically:
- The checkpoint labelled "epoch e" is at step e+1.
- train_losses[j] is the loss after j steps.
- test_losses[j] is the test loss after 10j steps.
- The first optimizer step has learning rate 0, from the warm-up min(step/10, 1).

## Counterfactual patching
- For n = 1000 random triples (a, a′ ≠ a, b), replace the component of token a's embedding inside a subspace S with the corresponding component of a′'s embedding.
- recovered = (D_patched − D(a)) / (D(a′) − D(a)), where D = L[a′+b] − L[a+b].
- **Key subspace S:** the column space of W_E Fᵀ restricted to the cos/sin rows of K (2|K| dimensions).
- **Controls:** a random 2|K|-dimensional subspace, and the orthogonal complement of the key subspace.
- Paper model: key 0.997 [0.994, 1.000], complement 0.002, random 0.06.
