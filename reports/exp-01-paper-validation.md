# Experiment 1 — Validating our analysis code on the paper's own network

**Question.** Does our re-implementation of the paper's (and ARENA's) analysis reproduce their published numbers when
applied to the paper's published checkpoints? If yes, the same code is trustworthy on other checkpoints and runs.

**Status:** done. Phases `acquire_paper_sources → convert_paper_run → validate_paper_run → validate_against_paper`;
outputs `runs/paper/analysis/{validation,rchecks}.json`, table [`tables/exp01_rchecks.md`](tables/exp01_rchecks.md).

## Setup

- **Model and data.** The paper's mainline run, from Hugging Face `callummcdougall/grokking_full_run_data`:
  - revision `80e5dbb`, sha256 `a151f021…` (both verified on download);
  - 500 state dicts saved every 100 epochs;
  - train losses for every epoch (50,000 values) and test losses every 10 epochs (5,000 values).
- **Code.** Our own re-implementation in `src/grokviz` (see `docs/porting-map.md`):
  - `GrokTransformer` mirrors the paper's parameter layout, so the checkpoints load with `strict=True` and need no conversion;
  - metrics are ported from the paper's analysis notebook and code and from ARENA 1.5.2, each with source links.
- **Device.** RTX 5070 Laptop GPU, torch 2.14.1+cu130; CPU gives identical numbers to within float rounding.

## Step 1 — Checkpoint and split alignment

**Method.** Rebuild the paper's train/test split (`random.seed(0); random.shuffle(pairs)`, first 3,830 pairs train). Then
recompute the 114-way training loss of checkpoints between step 100 and 49,900 and compare it with the logged
`train_losses`.

**Result.**
- **Train loss.** The checkpoint labelled "epoch e" reproduces `train_losses[e+1]` to 6–7 significant digits at all ten
  checked checkpoints. For example, epoch 100 gives 2.211129 against a logged 2.211129, while `train_losses[100]` is 2.273161.
- **Step offset.** The checkpoint therefore holds the weights after e + 1 optimizer updates. The paper's code saves *after*
  the update of epoch e, and its LambdaLR warm-up starts at learning rate 0, which is why `train_losses[0] == train_losses[1]`.
- **Test loss.** `test_losses[j]` is bracketed by the recomputed values around step 10j, so it is the test loss after 10j steps.

**Learning.** Without this check every training-time plot would be off by one step, and every figure would carry an
unknown split. The match also proves that we reconstructed the exact training split: the training loss sits around
10⁻⁷ while the test loss is around 20 during the plateau, so a wrong split could not reproduce it.

## Step 2 — Reproduction checks (R-checks)

All 27 rows are in [`tables/exp01_rchecks.md`](tables/exp01_rchecks.md). **23 of 27 match** within the stated
tolerances, including:
- loss 2.412 × 10⁻⁷;
- key frequencies {14, 35, 41, 42, 52};
- 433 single-frequency neurons, and 44 neurons in the k = 14 cluster;
- the paper's Table 2 head frequencies;
- loss 7.236 × 10⁻⁸ when keeping only the 40 key 2D components;
- loss 2.482 × 10⁻⁷ after the polynomial neuron replacement;
- the zero/mean skip ablations;
- 97.6 % variance explained by the attention-softmax linearisation, with loss 2.124 × 10⁻⁷;
- ARENA's test values for the per-frequency excluded loss, matched to every printed digit;
- ARENA's cluster sizes 44/93/145/87/64/79;
- ARENA's ReLU regression r² = 0.966 / 0.849.

**What the checks taught us about the paper's definitions (learnings):**

1. **"Test loss" 2.41 × 10⁻⁷ is the all-pairs loss.** On test pairs only, the loss is 3.02 × 10⁻⁷. We found this
   because our first test-only comparison missed by 25 %.
2. **"W_L residual under 0.55 %" is a squared ratio.** The Frobenius norm ratio is 7.43 %; its square is 0.552 %.
3. **The restricted loss must use the logits.** It is defined on the logits with a bias correction (paper code
   `calculate_trig_loss`); this gives 5.5474 × 10⁻⁸, identical to ARENA. The paper's text describes the strict ablation
   (5.54 × 10⁻⁸) as restricting each *neuron*. Restricting neurons individually gives 1.12 × 10⁻⁷, so we conclude the
   number was computed on logits (R10b, listed as not reproduced, with this explanation on the site).
4. **The neuron polynomial replacement uses the MLP-only path.** It reproduces 2.48 × 10⁻⁷ only through the MLP path
   W_L · MLP with bias correction, which is ARENA's method. Including the skip connection gives 1.52 × 10⁻⁷. The paper's
   own text approximates Logits ≈ W_L·MLP (App. A.1), consistent with this.
5. **The Gini definition is unspecified in the paper.** Hurley & Rickard's definition on the 113 per-component norms gives
   final values 0.736 (W_E) and 0.801 (W_L). These are consistent with the paper's Fig. 7, which reports no numeric
   mainline values.

**Not reproduced.** In each case we tried several plausible variants (listed in `rchecks.json → details`):

| Row | Ablation | Paper | Ours (closest variant) |
|---|---|---|---|
| R9b | MLP projected onto the 10 W_L directions | 1.19 × 10⁻⁷ | 2.38 × 10⁻⁷ |
| R9c | MLP projected onto the null space | 5.27 | 4.80 |
| R10b | per-neuron trig restriction | 5.54 × 10⁻⁸ | 1.12 × 10⁻⁷ (explained in item 3 above) |
| R11d | zero-ablating the skip around attention | 19.1 | 1.09 |

The paper does not specify these procedures in enough detail to know which variant was used. None of the four is
used as evidence on the site without this caveat.

## Step 3 — New causal test (not in the paper)

**Method.** Counterfactual patching of the key-frequency subspace of a's embedding (definition in
`docs/operationalizations.md`), on 1,000 random triples.

**Result.** The key subspace (10 of 128 dimensions) recovers **0.997 [0.994, 1.000]** of the logit difference. The two
controls recover almost nothing:
- the complement subspace (118 dimensions): 0.002;
- a random 10-dimensional subspace: 0.06.

This is strong interventional evidence that the network reads a only through cos/sin(w_k a) at the key frequencies.

## Why these choices

- **Re-implementation instead of copying.** We re-implemented rather than copied because neither ARENA nor the paper code
  has a license (NOTICE.md). Each function was validated against the published numbers instead.
- **Freezing the ambiguous constants.** The calibration constants (key-frequency threshold τ = 0.25, phase-rule
  constants) were chosen on this run only and frozen in `docs/operationalizations.md` before our own runs were analysed.
  This avoids tuning analysis choices on the data the hypotheses are tested on.
