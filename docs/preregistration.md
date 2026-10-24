# Preregistration: replicating the grokking mechanism across seeds, with controls

- **Frozen at:** git tag `prereg-v1`, pushed **before** any full training run.
- **Definitions:** exact definitions of every metric and phase boundary are in [`operationalizations.md`](operationalizations.md). They were checked only on the paper's published model (`runs/paper`).
- **Analysis code:** `src/grokviz/hypotheses.py` implements exactly the tests below.
- **Changes after the freeze:** any change made after this tag is listed under *Deviations*, with its reason.

## Question
Nanda et al. (2023) reverse-engineered one model, their mainline run. They report that four other seeds and several other settings behave "broadly" the same (paper Sec. 4, App. C.2). We ask whether the paper's central mechanistic claims hold, quantitatively and with pre-specified tests, across independently trained seeds and against controls.

## Runs (the run matrix)
- **Architecture, initialisation, optimiser, warm-up and data fraction:** identical to the paper's released code.
- **Training length:** 40,000 full-batch steps.
- **Logging:** dense; checkpoints every 100 steps.
- **Batching:** all runs are trained together (vectorised over runs). Runs do not interact; this is tested in `tests/test_train.py`.

| Group | Runs | Init seed | Split seed | λ (weight decay) | Labels | Role |
|---|---|---|---|---|---|---|
| `wd1` | 11 | s = 0..10 | s | 1 | (a+b) mod 113 | confirmatory; s=0 uses the paper's exact split |
| `wd0` | 5 | s = 0..4 | s | 0 | (a+b) mod 113 | control: no weight decay (paper App. D.1: no grokking) |
| `rand` | 3 | s = 0..2 | s | 1 | random table (seed s) | control: nothing generalisable to learn |
| `split0` | 3 | 100..102 | 0 (paper split) | 1 | (a+b) mod 113 | exploratory: same data, different init |

## Hypotheses, primary metrics and falsification criteria

All are evaluated on the `wd1` runs unless stated otherwise. Write t1 for the end of memorisation and t2 for the end of circuit formation (objective rules in operationalizations.md). K is each run's own set of key frequencies.

### H1: hidden progress
Restricted loss falls before test loss.
- **Metric:** Δ = t_test − t_restr. Here t_X is the first checkpoint after t1 at which X is at most 10 % of X(t1). X ranges over the test loss and the restricted loss.
- **Test:** one-sided exact sign test that Δ > 0.
- **Falsified if:** Δ ≤ 0 in 2 or more of the 11 runs, or the Holm-adjusted p is ≥ 0.05.

### H2: memorisation is dismantled during circuit formation
Excluded loss rises during circuit formation.
- **Metric:** excl(t2)/excl(t1), the excluded loss at t2 divided by its value at t1.
- **Test:** one-sided sign test that the ratio exceeds 10.
- **Falsified if:** the ratio is < 10 in 2 or more runs, or the Holm-adjusted p is ≥ 0.05.

### H3: Fourier sparsity is learned, and specific to the Fourier basis
Final Gini of the W_E and W_L component norms is high, and high only in the Fourier basis.
- **Metric:**
  - (i) Final Gini(W_E) and Gini(W_L), each compared with its own initial value.
  - (ii) The same Gini computed in 100 random orthonormal bases, as a null distribution.
- **Test:** sign test over runs on "final Gini > max(initial + 0.3, 99th percentile of the random-basis null)", required for both matrices.
- **Falsified if:** the condition fails in 2 or more runs, or the Holm-adjusted p is ≥ 0.05.

### H4: key frequencies are necessary, the others dispensable
Logit-frequency ablations at the final checkpoint.
- **Metric:**
  - (i) The smallest loss ratio, ablated/original, over single key-frequency ablations.
  - (ii) The largest loss ratio over single non-key ablations.
  - (iii) The loss ratio after removing every non-key frequency.
- **Test:** sign test over runs on "(i) ≥ 100 and (ii) ≤ 2 and (iii) ≤ 1.5".
- **Falsified if:** the condition fails in 2 or more runs, or the Holm-adjusted p is ≥ 0.05.

### H5: weight decay is necessary for grokking
- **Metric:** "grokked" = test accuracy ≥ 0.99 at step 40,000.
- **Test:** one-sided Fisher exact test comparing wd1 (11 runs) with wd0 (5 runs).
- **Falsified if:** any wd1 run fails to grok, or any wd0 run groks, or Fisher p ≥ 0.05.

**Multiplicity:** the five p-values (H1 to H5) are Holm-corrected at family-wise α = 0.05.

### Secondary (descriptive, reported with 95 % intervals, no hypothesis test)
- **S1.** Fraction of neurons with single-frequency FVE > 0.85. The paper reports 84.6 % for its model.
- **S2.** Logit cosine-fit FVE. The paper reports 95 %.
- **S3.** Squared residual of W_L after the 2|K|-direction approximation. The paper reports < 0.55 %.
- **S4.** Counterfactual patching on 1,000 random triples per run (interval: bootstrap over runs).
  - Predicted: swapping the key-frequency subspace of a's embedding with that of a′ recovers ≥ 0.9 of the logit difference.
  - Controls: the complementary subspace and a random subspace of equal dimension should each recover ≤ 0.1.
- **S5.** Number of key frequencies per run. Distribution of key frequencies across runs, compared with the paper's seeds 1 to 4 (3–4 key frequencies each).
- **S6.** Phase timings t1, t2, t3 and t_grok, as a mean with a t-interval. The paper reports memorisation completing near 1.4k "for all five runs".

### Controls (descriptive)
- **wd0:** test accuracy over training, and the K-free maximum excluded loss. The paper reports that without weight decay "excluded loss does not increase".
- **rand:** train accuracy reaches 1 with test accuracy near chance (1/113). Gini of W_E and W_L stays near its initial value.
- **split0:** do the three runs that share the paper's split choose the paper's key frequencies {14, 35, 41, 42, 52}? This is exploratory; no hypothesis.

## Power and sample size
Computed with `grokviz.stats.power_table`. The numbers in the table are reproduced in the logs.

| Seeds n | P(all n agree by chance) | Sign-test power if the true agreement rate is 0.95 (α = 0.01, the strictest Holm level) | Smallest effect d with 80 % power, one-sided t-test at α = 0.01 |
|---|---|---|---|
| 5 | 0.031 | 0 (cannot reach 0.01) | 2.19 |
| 8 | 0.0039 | 0.66 | 1.40 |
| **11** | **0.00049** | **0.90** | **1.11** |
| 15 | 3e-5 | 0.96 | 0.91 |

Why n = 11:
- It is the smallest n at which one disagreeing seed (10 of 11) still passes the strictest Holm threshold: P(X ≥ 10) = 12/2048 = 0.0059 < 0.01.
- It reaches 90 % power if 95 % of seeds behave as claimed.
- For H5, perfect separation of 11 wd1 runs from 5 wd0 runs gives Fisher p = 2.3e-4.

The paper's own mechanistic effects are orders of magnitude larger than these detectable effect sizes. For example, ablating one key frequency raises the loss by 4,000× to 300,000× on the paper model. The binding constraint is therefore the robustness across seeds, not the effect size.

## What would change our conclusions
- **Paper's account incomplete:** if H1 or H2 fail, the account of "hidden progress before grokking" does not generalise across seeds.
- **Fourier picture not universal:** if H3 or H4 fail, the Fourier-multiplication picture is not universal for this architecture and setup.
- **Weight-decay role not established:** if H5 fails, the claimed necessity of weight decay is not supported under these conditions.

## Deviations
*(None at the time of tagging. Any later change to definitions, runs or tests is listed here with date and reason.)*

No hypothesis, metric, threshold, run or test was changed after the tag. The implementation, however, was completed
after it. For transparency, here is everything that happened after the tag, with the information available at each point:

- **Analysis code written after the tag** (2026-10-18 18:04:39 +02:00, one minute after the tag).
  - **What.** `src/grokviz/hypotheses.py` was written, implementing the tests exactly as worded above. The tag points
    at the whole repository as it was then, which did not yet contain `hypotheses.py`.
  - **One case the text leaves open.** In H1, a run whose restricted loss falls tenfold but whose test loss never does
    counts as Δ = +∞, a pass (already in the first committed version, `ad8e69d`). No run hit this case.
  - **When relative to results.** The first training launch started at 18:03:59. Its first log line, at 18:07:50,
    came after the file was written.
  - **When committed.** Only on 2026-10-21 (`ad8e69d`), because an OS restart interrupted the session.
- **Information seen during training.** The training log prints test accuracy for `wd1_s0`–`wd1_s5` every 2,000
  steps. This is the input of H5 on the weight-decay side. No metric used by H1–H4 was computed for any of our runs
  before the analysis code was final.
- **Later code edits.** These did not change the test logic, a threshold or any outcome:
  - `158e527` (2026-10-21 00:35): `analyse_final` uses the paper's analysis checkpoint (epoch 40k) for the paper run,
    which is not part of any test; runs without per-checkpoint metrics are skipped;
  - `c872ae1` (2026-10-21 01:18, before training finished at 01:20): H1's Δ = +∞ is stored as null so the JSON stays
    standard, and `H1_ok` keeps the +∞ meaning; `test_hypotheses` reads every run whatever `--runs` says, writes its
    table atomically and skips when no wd1 run is analysed;
  - `9f66e15` (2026-10-21 10:32): `analyse_final` also stores a descriptive cluster ablation for the site; no test
    reads it;
  - `2f56492` (2026-10-23 00:37): the tenfold-drop helper moved unchanged to `phases.drop_time`, so the prose can
    quote the paper run's drop steps.
- **Training code changes after the tag.** These do not change what is trained:
  - `train.py` gained crash-resume support (`38d0510`); the resumed and uninterrupted trajectories are bitwise identical
    (`tests/test_train.py`);
  - all runs were relaunched from scratch after the interruption.
- **Logging cadence.** During training, test loss and accuracy are logged every 10 steps (`GROKVIZ_TEST_EVERY`), not
  every step. This does not affect the tests: the phase rules and the hypotheses read test loss and accuracy from the
  per-checkpoint metrics (every 100 steps), as stated.
- **Corrected note (2026-10-24).** The note under *Key frequencies K* in `operationalizations.md` gave the range of τ
  that yields the paper model's five key frequencies as (0.06, 0.83); recomputed, it is (0.060, 0.826]. Only the note
  changed; the definition and τ = 0.25 are unchanged. Likewise, S3 above attributes "< 0.55 %" to the squared
  residual, but the paper (p. 6) states it for the residual's Frobenius norm; S3's metric (the squared residual) is unchanged.
