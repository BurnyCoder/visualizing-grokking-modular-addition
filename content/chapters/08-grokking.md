## The phenomenon

In 2022, Power and colleagues trained small transformers on tables of modular arithmetic and saw something odd {{cite:power2022}}. The networks first fitted their training data perfectly while doing no better than chance on held-out data — and then, long afterwards, test performance jumped, in their words

{{quote:power-chance}}

and

{{quote:power-overfitting}}

They called this [[dfn:grokking]]. Nanda et al. describe it as follows {{cite:nanda2023}}:

{{quote:nanda-grokking}}

Here is the paper's own training run. Both curves are plotted on a [[dfn:log-scale]]: each horizontal grid line is ten times the one below it, so we can see losses from ~10 down to ~10⁻⁷ at once.

{{fig:loss-curves}}

::: example
**Reading the plot** (paper's run; numbers from its logs):

- **Steps 0 – {{num:paper.train_acc_full_step}}:** both losses start at ln 113 ≈ 4.73 (guessing). Training accuracy reaches 100 % by step {{num:paper.train_acc_full_step}}.
- **{{num:paper.train_acc_full_step}} – {{num:paper.t1}}:** training loss keeps falling by orders of magnitude while test loss *rises* to about {{num:paper.loss_test_peak}} — worse than guessing: the network confidently gives memorised-looking answers to questions it has never seen.
- **{{num:paper.t1}} – {{num:paper.t2}}:** a long plateau. Training loss is tiny and flat; test loss decreases only slowly.
- **Around {{num:paper.t2}}:** test loss collapses by seven orders of magnitude; test accuracy reaches 99 % at step {{num:paper.t_grok}}.
:::

{{fig:acc-curves}}

The curves average over many questions. The next picture shows every question at once: each pixel is one of the 12,769 sums a + b (a = row, b = column), coloured by the probability the network gives to the correct answer (yellow = certain and right, dark purple = no chance). While it is on screen it plays the paper's run from the start to just after grokking; pause it or drag its slider to study one moment. Watch the [[term:training-set|training questions]], a random speckle, turn yellow early while the [[term:test-set|test questions]] stay mostly dark through the plateau, until the whole square lights up after step {{num:paper.t2}}.

{{anim:pcorrect-grid}}

## Why it looks sudden

On a linear axis the same data looks like an abrupt [[dfn:phase-transition]] — nothing, then a cliff:

{{fig:loss-curves-linear}}

The log plot already hints that something *is* changing during the plateau. Neel Nanda points out in his replication walkthrough that the choice of axis changes how sudden grokking appears {{cite:video_replication}}. The question this site answers, following the paper, is: **what is the network doing during the plateau, and why does the test loss eventually collapse?** The paper's answer in one sentence:

{{quote:nanda-gradual}}

## When does grokking happen?

Grokking in this setup depends on a few ingredients, all reported in the paper {{cite:nanda2023}}:

- **Limited data.** With 30 % of the pairs the network groks; with ≥ 60 % it generalises immediately; with 10–20 % it had not generalised after 40,000 steps (paper App. C.2, Fig. 20).
- **Regularisation.** With no regulariser at all the networks memorise but never grok; weight decay works, and so does dropout, but an ℓ₁ penalty did not (paper Sec. 5.3, App. D.1).
- **Time.** Smaller weight decay makes grokking take longer, according to the paper's text (Sec. 5.3, App. D.1). The average times App. D.1 itself gives point the other way (about 3k epochs at λ = 0.3, 5–10k at λ = 1, 20k at λ = 3); see the [errata](../errata.html).

In the paper's words:

{{quote:nanda-no-wd}}

We test the weight-decay claim ourselves in chapter 20, with five runs without weight decay.

::: nuance
**Not “after zero loss”.** Grokking is sometimes summarised as “the model generalises after reaching zero training loss”. As chapter 4 showed, cross-entropy is never zero; the training loss keeps decreasing throughout the plateau, and the paper explicitly warns against this summary (footnote 4) {{cite:nanda2023}}.
:::
