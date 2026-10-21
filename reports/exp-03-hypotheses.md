# Experiment 3: Do the paper's mechanistic claims hold across seeds? (preregistered)

**Question.** Nanda et al. (2023) reverse-engineered one network. Do its central claims hold, with pre-specified tests,
for 11 independently trained seeds with weight decay? The claims are:
- hidden progress before grokking;
- dismantling of memorisation;
- learned Fourier sparsity;
- necessary key frequencies;
- weight decay is necessary.

**Status:** done. The phases, in order, are `compute_checkpoint_metrics`, `compute_gradient_dynamics`,
`compute_geometry`, `detect_phases`, `analyse_final_models` and `test_hypotheses`. Outputs:
- per run: `runs/<id>/analysis/{ckpt_metrics.npz, phases.json, final.json}`;
- the tests: [`tables/hypotheses.json`](tables/hypotheses.json) and [`tables/hypotheses.md`](tables/hypotheses.md).

## Pre-specification

Hypotheses, metrics, thresholds, the seed count and the multiple-comparison correction are in
[`docs/preregistration.md`](../docs/preregistration.md) (git tag `prereg-v1`, pushed before training). Exact metric
definitions and the phase rules are in [`docs/operationalizations.md`](../docs/operationalizations.md). These were
calibrated on the paper's run only, and frozen before any of our runs was analysed.

Everything that happened after the tag is listed under *Deviations* in the preregistration. In short:
- the analysis code was written one minute after the tag, before any training output;
- training logs showed the test accuracy of six seeds while training ran;
- no test, threshold or run was changed.

## Methods (summary)

- **Runs.** 11 `wd1` seeds (confirmatory) and 5 `wd0` seeds (H5), plus the descriptive controls `rand` and `split0`
  (exp-02).
- **Per checkpoint (401 per run).** We measure:
  - train and test loss and accuracy;
  - restricted loss and excluded loss, computed with each run's own key frequencies K (from its final checkpoint);
  - Gini of the W_E and W_L Fourier spectra;
  - the sum of squared weights.
- **Phase boundaries.** These are objective rules computed from losses and weights only:
  - t1: the train loss stops falling (relative drop < 1 % per 100 steps);
  - t2: the test loss falls to ≤ 20 % of its median over [t1, t1 + 2,000];
  - t3: the sum of squared weights has completed 95 % of its decline from t2 to its final level (the median over the
    last 5,000 steps).
- **Final checkpoint (step 40,000).** We run:
  - single-frequency logit ablations;
  - keep-only-key ablations;
  - Gini under 100 random orthonormal bases (the null for H3);
  - neuron FVE;
  - the logit cosine fit;
  - counterfactual patching on 1,000 random (a, a′, b) triples.
- **Tests:**
  - H1–H4: one-sided exact sign tests over the 11 seeds;
  - H5: one-sided Fisher exact test, wd1 vs wd0;
  - Holm correction over the five tests;
  - "supported" means fewer than two failing seeds *and* a Holm-adjusted p < 0.05.
- **Secondary statistics:** 95 % t-intervals over seeds, with bootstrap intervals for the patching fractions.

## Results

### Preregistered tests

The table below is [`tables/hypotheses.md`](tables/hypotheses.md), produced by `test_hypotheses`.

| Hypothesis | Seeds passing | p (one-sided) | Holm-adjusted p | Verdict |
|---|---|---|---|---|
| H1: restricted loss falls before test loss | 11/11 | 4.9 × 10⁻⁴ | 0.0020 | supported |
| H2: excluded loss rises ≥ 10× during circuit formation | 11/11 | 4.9 × 10⁻⁴ | 0.0020 | supported |
| H3: Fourier sparsity is learned and basis-specific | 11/11 | 4.9 × 10⁻⁴ | 0.0020 | supported |
| H4: key frequencies necessary, others dispensable | 6/11 | 0.50 | 0.50 | **not supported** |
| H5: weight decay is necessary | wd1 11/11 vs wd0 0/5 grokked | 2.3 × 10⁻⁴ | 0.0011 | supported |

Effect sizes behind each pass, as ranges over the 11 seeds (from `tables/hypotheses.json` and
`runs/wd1_s*/analysis/final.json`):

- **H1.** The restricted loss reaches 10 % of its value at t1 between 700 and 2,100 steps before the test loss does.
- **H2.** The excluded loss at t2 is 12.6× to 7.9 × 10⁵× its value at t1. The smallest ratios are 12.6 (s2) and 38.7 (s7).
- **H3.** The final Gini is 0.61–0.77 for W_E and 0.78–0.88 for W_L, against:
  - initial values of 0.03–0.04;
  - 99th percentiles of the random-basis null of 0.12–0.18 (W_E) and 0.13–0.19 (W_L).
- **H5.** Every wd1 run reaches 100 % test accuracy. The wd0 runs end at 1.9–21.5 % test accuracy (100 % train).

### Why H4 fails (post-hoc decomposition of the preregistered criterion; counts only, no new test)

| Part of H4 | Seeds passing | Worst seed |
|---|---|---|
| (i) every key frequency necessary (single ablation ≥ 100×) | 9/11 | 5.9× (s1, frequency 2) and 70× (s8, frequency 46) |
| (ii) every non-key frequency dispensable (single ablation ≤ 2×) | 11/11 | 1.84× (s10) |
| (iii) keeping only the key frequencies costs ≤ 1.5× | 8/11 | 2.78× (s4), 1.99× (s9), 1.64× (s10) |

- **Parts (i) and (iii)** concern the rule that names the key frequencies (W_L Fourier norm ≥ 25 % of the largest), not
  the claim that a few frequencies carry the computation.
  - **(i): weight norm is not importance.** The rule picks frequency 2 in s1, which has a large W_L norm but a small
    effect when ablated. Every other key frequency of s1 costs ≥ 1.5 × 10⁵× when removed.
  - **(iii): small contributions add up.** In the three seeds with 3–4 key frequencies, no single non-key frequency
    matters (each ≤ 1.84×), yet all of them together multiply a loss of ~10⁻⁷ by 1.6–2.8.
- **Our reading.** The paper's "few key frequencies, everything else irrelevant" is approximately true in every seed.
  It is not exact enough to meet thresholds calibrated on the paper's own model, whose keep-only-key ratio is 0.30.

### Secondary statistics (95 % t-interval over the 11 seeds; bootstrap for patching)

| Quantity | Ours | Paper's model |
|---|---|---|
| S1: neurons with single-frequency FVE > 0.85 | 80.2 % [75.3, 85.2] (range 62.7–90.4 %) | 84.6 % (433/512) |
| S2: logit cosine-fit FVE | 96.5 % [95.3, 97.7] | 96.1 % (ours on the paper model; paper reports 95 %) |
| S3: W_L squared residual outside the key directions | 0.58 % [0.30, 0.86] | 0.55 % |
| S4: patching recovery, key subspace | 0.997 [0.995, 0.998] | 0.997 |
| S4: complement subspace | 0.003 [0.002, 0.005] | 0.002 |
| S4: random subspace of equal dimension | 0.058 [0.049, 0.067] | 0.062 |
| S5: number of key frequencies | 4.36 [3.74, 4.99] (range 3–6) | 5 |
| S6: t1 (end of memorisation) | 1,576 [1,567, 1,586] | 1,567 |
| S6: t2 (end of circuit formation) | 7,445 [5,874, 9,017] | 9,301 |
| S6: t_grok (test accuracy ≥ 99 %) | 8,455 [6,816, 10,093] | 10,601 |

Per-run values (accuracies, phase boundaries, key frequencies) are tabulated in [exp-02](exp-02-training.md).

### Controls

- **No weight decay (wd0).** All five memorise (100 % train) and none generalises.
  - Test accuracy ends at 1.9–21.5 %, and the W_E Gini stays at 0.07–0.08 (initially 0.03).
  - All 56 frequencies pass the 25 % rule, i.e. there is no sparse Fourier structure.
- **Random labels (rand).** They memorise a table with nothing to generalise.
  - Test accuracy is 0.8–0.9 %, which is chance (1/113 = 0.9 %).
  - The Gini stays at 0.08–0.09.
- **Paper's split, new initialisations (split0, exploratory).** All three grok, with key frequencies {2, 14, 27, 30, 37, 52},
  {9, 14, 15, 27} and {7, 38, 49, 53}. They share at most two frequencies (14, 52) with the paper's {14, 35, 41, 42, 52}.

### Exploratory observations (not preregistered)

1. **Same initialisation, same frequencies.**
   - **The setup.** Our `wd1_s0` starts from the paper's exact initial weights (max |Δ| = 3.0 × 10⁻⁸) with the paper's split.
   - **Divergence.** The two training losses agree to 5 significant digits for 52 steps, then diverge. Weight cosine
     similarity at step 40,000 is 0.74.
   - **Same outcome.** Both runs end with key frequencies {14, 35, 41, 42, 52}, and t_grok is 10,600 vs 10,601.
   - **Interpretation.** Together with split0, this suggests that the initialisation, not rounding noise, selects the
     frequencies. This rests on n = 1 same-init pair.
2. **Late loss spikes.**
   - **Where.** In 2 of the 14 grokked runs (`wd1_s4`: 4 spikes; `split0_init102`: 6 spikes), the test loss repeatedly jumps
     more than 100× above its running minimum after grokking. It peaks at 0.25 and 0.64, with test accuracy dipping to 93 %
     and 87 %, and recovers each time.
   - **Similar reports.** Spikes like these with adaptive optimisers have been called the slingshot mechanism
     (Thilak et al. 2022, arXiv:2206.04817).
   - **Not investigated.** We did not investigate their cause.
   - **Effect on the tests.** None: the tests use step 40,000, where both runs are at 100 % test accuracy.

## Learnings

- **A failed preregistered test is a result, not a bug to fix.**
  - **What we did.** We report H4 as not supported. Changing τ or the thresholds after seeing the data would turn the
    test into a fitted description.
  - **What the failure teaches.** The decomposition shows that H4 failed on the *operationalisation of "key frequency"*.
    A future preregistration should define K by causal importance (ablation), not by weight norm, or test the three
    parts separately.
- **Thresholds calibrated on one model are brittle.** H4's thresholds (≥ 100×, ≤ 2×, ≤ 1.5×) were set by looking only at
  the paper's model, which has 5 well-separated key frequencies. Seeds with 3 frequencies (s4, s9) or 6 crowded ones
  (s8: 44–47) sit near the boundaries. Calibrating on a single model hides between-seed variance.
- **Controls make the key-frequency rule's limits visible.** For wd0 and rand runs every frequency passes the 25 % rule,
  because the rule is relative to the maximum. A relative threshold always returns *something*, so "K" is only
  meaningful when the spectrum is sparse. This is why the site lists those runs instead of plotting 56 "key" frequencies.
- **Phase rule under λ = 0.** The memorisation rule (train loss stops falling) fires at about 11,800 steps for wd0 runs,
  versus about 1,570 with weight decay. Without decay the train loss keeps creeping down instead of reaching a decay
  floor. Rules calibrated on λ = 1 runs should not be read as "memorisation ends later" for λ = 0.
- **Where the effort went.** Computing every metric at every checkpoint took 72 minutes for 23 runs (≈ 0.5 s per
  checkpoint on GPU). The controls were slowest: with all 56 "key" frequencies, the restricted-loss projection covers
  112 directions.
