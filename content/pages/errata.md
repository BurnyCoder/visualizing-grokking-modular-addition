# Errata and nuances

Small errors and ambiguities we found while reproducing the sources. None of them changes the paper's conclusions; we list them so readers comparing this site with the sources are not confused.

## In the paper (arXiv:2301.05217 v3)

- **The cos-addition identity on p. 4** is printed as cos(w_k a)cos(w_k a) − sin(w_k a)sin(w_k b); the first product should be cos(w_k a)cos(w_k **b**). The correct form appears on p. 6 and in Fig. 1 {{cite:nanda2023}}.
- **“Test loss” 2.41 × 10⁻⁷.** The loss the paper reports for its model at epoch 40,000 equals the loss over *all* 12,769 inputs (113 answers), {{num:rc.R1}} in our computation; the loss on test inputs only is {{num:paper.loss40k_test}}.
- **W_L residual “under 0.55 %”** (p. 6) matches the *squared* Frobenius-norm ratio ({{num:paper.wl_resid_sq}}); the plain norm ratio is {{num:paper.wl_resid_norm}}.
- **Restricting neurons to cos/sin(w_k(a+b)) (p. 7)** gives 5.54 × 10⁻⁸ in the paper; we obtain this value when restricting the *logits* ({{num:rc.R19a}}) and {{num:rc.R10b}} when restricting each neuron separately.
- **Not reproduced:** projecting the MLP onto the 10 W_L directions (paper 1.19 × 10⁻⁷) or onto their null space (paper 5.27), and zero-ablating the skip connection around attention (paper 19.1); our closest variants are listed on the [Reproduce](reproduce.html) page. The paper does not specify these procedures in enough detail to know which variant it used.
- **Gini of “W_U and W_L”** (p. 9) should read W_E and W_L, as in Sec. 5.1 and Fig. 7. The Gini formula itself is not given; we use Hurley & Rickard's {{cite:hurley2009}}.
- **Minor:** “d × p” vs “P”; “intepretability” (p. 3); “INTEREFERENCE” (App. B title); “Table 1” should be “Table 2” in App. C.1.3; App. D.1 states that smaller weight decay leads to slower grokking but then gives “around 3k epochs” for λ = 0.3 and “20k epochs” for λ = 3.0 — the two numbers contradict the sentence (they look swapped); Fig. 24's caption calls weight decay γ.
- **Training details not in the text but in the code:** a learning-rate warm-up min(step/10, 1) whose first step has learning rate 0, and checkpoints that hold the weights after epoch + 1 updates (verified from the released losses) {{cite:paperrepo}}.

## In the paper's released code

- `helpers.cross_entropy_high_precision` casts logits to float32 although its comment says float64; the paper's analysis notebook uses float64 {{cite:paperrepo}}.

## In ARENA 1.5.2 (commit 85d1011)

- `arrange_by_2d_freqs` loops over `range(1, p//2)` and therefore omits frequency 56 {{cite:arena152code}}.
- The displayed Fourier basis lists sin before cos while the code puts cos first; this site follows the code.
- The embedding text lists 31 among the key frequencies; the neuron clusters (and the paper) give {14, 35, 41, 42, 52} — 31 appears in the embedding only.
- The ReLU-regression printout says “ReLU(0.5 + …)” but the code uses 0.5/√p with unit-normalised cosines.

## In the notebooks of this project's submodule

- The TransformerLens-based notebook orders the Fourier basis sin-first, multiplies W_logit by the basis instead of its transpose in one cell, hard-codes phase boundaries from a different run, and plots loss curves and checkpoint metrics with a 99-epoch offset; all are fixed in this site's code (see `docs/porting-map.md`).
