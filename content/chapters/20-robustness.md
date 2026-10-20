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

## Results

{{fig:hypothesis-table}}

{{num:hyp.summary}}

The phase boundaries vary from seed to seed: across the 11 weight-decay runs, memorisation ends at step {{num:seeds.t1_ci}} (mean and 95 % interval), circuit formation at {{num:seeds.t2_ci}}, and test accuracy reaches 99 % at {{num:seeds.tgrok_ci}}.

## Different seeds, different frequencies, same algorithm

Each run picks its own key frequencies:

{{fig:seed-key-freqs}}

The weight-decay runs use {{num:seeds.nkey_ci}} key frequencies on average; in all of them, single-frequency neurons make up {{num:seeds.neurons085_ci}} of the MLP and the cosine formula of chapter 15 explains {{num:seeds.logitfve_ci}} of the logit variance. The counterfactual patching test of chapter 16 recovers {{num:seeds.patch_key_ci}} of the logit difference for the key-frequency subspace versus {{num:seeds.patch_random_ci}} for a random subspace of the same size.

## Controls

- **No weight decay:** {{num:controls.wd0}}
- **Random labels:** {{num:controls.rand}}
- **Same data, new initial weights:** {{num:controls.split0}}

::: nuance
**Scope of these checks.** All 22 runs share the paper's architecture, data fraction and optimiser; robustness to other settings (data fraction, depth, modulus, dropout) is reported in the paper's appendix, not re-tested here. A pre-specified threshold that a run misses by a little counts as a miss — that is the price of not tuning criteria after the fact.
:::
