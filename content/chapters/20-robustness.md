Everything so far came from **one** trained network. The paper reports four more seeds and many other settings in its appendix and finds the same algorithm with different key frequencies {{cite:nanda2023}} (App. C.2). We retrained the paper's architecture 22 times to test the main claims with pre-specified statistics.

## Testing claims, not stories

A [[dfn:hypothesis-test]] asks how surprising the data would be if the claimed effect were absent. The answer is a [[dfn:p-value]]: the probability of results at least this extreme under “no effect”; small values are evidence for the effect. A [[dfn:confidence-interval]] gives a range of plausible values for a quantity; a 95 % interval is built so that, over many repetitions of the experiment, 95 % of such intervals would contain the true value.

Our main tool is the [[dfn:sign-test]]: if each of n independently trained networks either shows an effect or not, and “no effect” would make each outcome a coin flip, then all n showing it has probability $0.5^n$. Testing several hypotheses at once raises the chance of a false alarm, so we apply a [[dfn:multiple-comparisons|multiple-comparison correction]] (Holm's method) to the five tests.

To avoid choosing tests after seeing the results, we wrote the hypotheses, metrics and pass/fail thresholds down *before training* — a [[dfn:preregistration]], frozen in the repository as git tag `prereg-v1` ([docs/preregistration.md](https://github.com/BurnyCoder/visualizing-grokking-modular-addition/blob/main/docs/preregistration.md)). The number of runs was chosen for [[dfn:statistical-power]]: with 11 seeds, even one disagreeing seed (10 of 11) still passes the strictest corrected threshold, and a 95 %-reliable effect is detected with 90 % probability.

::: example
**Why 11 seeds and not 5?** With 5 seeds the smallest possible sign-test p-value is $0.5^5 = 0.031$ — not small enough for the strictest Holm threshold (0.05/5 = 0.01), even if all five agree. With 11 seeds, 11/11 gives $0.5^{11} = 0.00049$ and 10/11 gives 0.0059 — both below 0.01.
:::

## The runs

| Group | Runs | Weight decay | Task | Purpose |
|---|---|---|---|---|
| wd1 | 11 (seeds 0–10) | 1 | addition | the paper's setting; seed 0 uses the paper's exact data split |
| wd0 | 5 | 0 | addition | control: no weight decay |
| rand | 3 | 1 | random answer table | control: nothing to generalise |
| split0 | 3 | 1 | addition | the paper's data split with new initial weights |

All runs use the paper's architecture, optimiser and initialisation rule for 40,000 full-batch steps (chapter 7).

{{fig:seed-test-loss}}

The phases of every run on one time axis, and every weight-decay seed's clock at the same moment (it plays while on screen): different seeds grok at different times and with different frequencies, but in the same order and into the same circle.

{{fig:seed-phase-strips}}

{{anim:seed-clocks}}

Grokking is not always the end of the story. In {{num:seeds.spike_n}} of our 14 runs that grokked, the test loss later jumps more than 100-fold above its lowest value so far, again and again, and falls back each time: {{num:seeds.spikes}}. Similar loss spikes late in training with adaptive optimisers like Adam were described as the "slingshot mechanism" {{cite:thilak2022}}; we did not investigate their cause, and none of the preregistered tests depends on them (all are measured at step 40,000, between spikes).

## Results

{{fig:hypothesis-table}}

Each seed's value of every preregistered metric against its threshold, and the sparsity test (H3) run by run:

{{fig:hypothesis-dots}}

{{fig:gini-vs-null}}

{{num:hyp.summary}}

Four of the five claims pass. **H4 does not**: it required, in at least 10 of 11 seeds, all three of: every key frequency necessary (ablating it multiplies the loss by ≥ 100), every other frequency dispensable (≤ 2×), and all other frequencies jointly dispensable (keeping only the key frequencies costs ≤ 1.5×). Splitting the preregistered criterion into its parts (a description after the fact, not a new test): the second part held in {{num:hyp.h4_nonkey_ok}} seeds (worst case {{num:hyp.h4_nonkey_max}}×), the first in {{num:hyp.h4_key_ok}}, the third in {{num:hyp.h4_keep_ok}} (worst case {{num:hyp.h4_keep_max}}×).

::: nuance
**Why H4 failed.** Both failing parts concern *which* frequencies count as "key", not whether the network uses a few frequencies. Our preregistered rule calls a frequency "key" if its $W_L$ Fourier norm is at least 25 % of the largest. Twice it included a frequency that matters much less than that seed's other key frequencies: {{num:hyp.h4_key_fails}}. So a large weight is not the same as causal importance. Of the {{num:hyp.n_few_keys}} seeds with at most four key frequencies, three have the opposite problem: every other frequency is individually dispensable, but removing all of them together still multiplies a loss of about $10^{-7}$ by more than 1.5: {{num:hyp.h4_keep_fails}}. The preregistered verdict stands: the paper's model of "a few key frequencies, all else irrelevant" is a good description of every seed, but not as exact a one as our thresholds demanded.
:::

The phase boundaries vary from seed to seed: across the 11 weight-decay runs, memorisation ends at step {{num:seeds.t1_ci}} (mean and 95 % interval), circuit formation at {{num:seeds.t2_ci}}, and test accuracy reaches 99 % at {{num:seeds.tgrok_ci}}.

## Different seeds, different frequencies, same algorithm

Each run picks its own key frequencies:

{{fig:seed-key-freqs}}

The weight-decay runs use {{num:seeds.nkey_ci}} key frequencies on average (mean and 95 % interval over the 11 seeds); on the same footing, single-frequency neurons make up {{num:seeds.neurons085_ci}} of the MLP and the cosine formula of chapter 15 explains {{num:seeds.logitfve_ci}} of the logit variance. The counterfactual patching test of chapter 16 recovers {{num:seeds.patch_key_ci}} of the logit difference for the key-frequency subspace versus {{num:seeds.patch_random_ci}} for a random subspace of the same size.

## Controls

- **No weight decay:** {{num:controls.wd0}}
- **Random labels:** {{num:controls.rand}}
- **Same data, new initial weights:** {{num:controls.split0}}

## Who chooses the frequencies?

Seed 0 turned out to start from *exactly* the paper's initial weights (largest difference {{num:seeds.same_init_diff}}, i.e. float32 rounding) and to use the paper's data split — our code evidently draws the initial weights from the same random generator in the same order as the paper's. It is therefore a re-run of the paper's network, with our code on our GPU. The two training losses agree to five significant digits for the first {{num:seeds.same_init_loss_steps}} steps and then drift apart — tiny rounding differences are amplified by training (see the experiment report) — and by step 40,000 the two weight vectors have cosine similarity only {{num:seeds.same_init_cos40k}}. Yet the re-run ended with {{num:seeds.same_init_keys_equal}} key frequencies: {{num:seeds.same_init_keys}}. The three runs with the same data but *new* initial weights chose different sets (above). This is an exploratory observation on one pair of runs, not a tested hypothesis, but it suggests that the initial weights, not rounding noise, decide which frequencies a network uses.

::: nuance
**Scope of these checks.** All 22 runs share the paper's architecture, data fraction and optimiser; robustness to other settings (data fraction, depth, modulus, dropout) is reported in the paper's appendix, not re-tested here. A pre-specified threshold that a run misses by a little counts as a miss — that is the price of not tuning criteria after the fact.
:::
