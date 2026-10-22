The progress measures describe *what* changes during training. This chapter looks at the forces that change it: the gradient of the loss and the pull of weight decay. Because training is full-batch (chapter 7), the gradient the optimizer saw at any checkpoint can be recomputed exactly from the saved weights — which we did for every checkpoint.

{{widget:scrubber}}

## Two forces

Every AdamW step combines two pulls on the weights (chapter 7): the loss gradient, rescaled by Adam, and weight decay, $-\gamma\lambda\theta$, which always points straight back toward zero. The paper argues that weight decay drives both circuit formation and cleanup {{cite:nanda2023}} (Sec. 5.2–5.3): the generalising circuit can reach the same training loss with much smaller weights than the memorised lookup table, so under constant pressure toward small weights it wins.

::: nuance
**Why we do not compare raw sizes.** After memorisation the raw gradient is tiny — about 10⁻⁸ of λ‖θ‖ in the paper's run — but Adam divides each coordinate by the root of its recent squared gradients, so even a tiny but consistent gradient produces updates of size ~γ. Raw gradient norms therefore say little about how strongly the loss pushes; *directions* (cosines) do, and they are what we plot.
:::

## Where the gradient points

The cosine between the loss gradient and the weights tells us how the two forces relate. A gradient-descent step moves along *minus* the gradient, so a cosine near −1 means the loss “wants” the weights to grow along exactly the direction weight decay shrinks them:

{{fig:grad-alignment}}

On the paper's run, cos(gradient, weights) is {{num:paper.cos_gw_40k}} at step 40,001. Once the network is perfectly accurate, the cheapest way to lower cross-entropy is to scale all logits up (chapter 4), i.e. to grow the weights uniformly; weight decay shrinks them uniformly. The two cancel — the network sits in a balance between them, which is why its loss stops at ~10⁻⁷ instead of going to zero.

The second curve compares the gradient of the *training* loss with the gradient of the *test* loss. At the very start they are aligned (cosine {{num:paper.cos_tt_start}}): any improvement is a general improvement. During memorisation and circuit formation the cosine is **negative** (e.g. {{num:paper.cos_tt_mid}} at step {{num:paper.mid_step}}, midway through circuit formation): a step that lowers the training loss *raises* the test loss — memorisation actively hurts generalisation. Only around grokking does it turn positive again ({{num:paper.cos_tt_40k}} at step 40,001).

## Which frequencies the gradient builds

Finally, which frequencies does the gradient push into the embedding at each moment? The share of the embedding gradient's energy per frequency, over training:

{{fig:grad-we-fourier-heatmap}}

{{fig:grad-norms}}

## What Adam actually does

The raw gradient is not the step. For our retrained runs we saved Adam's two running averages every 1,000 steps, so we can rebuild the direction Adam really moves in, $\hat m/(\sqrt{\hat v}+\epsilon)$, and compare it with the raw gradient and with the weights:

{{fig:adam-alignment}}

Adam's loss step and the weight-decay step are multiplied by the same learning rate, so their sizes can be compared directly:

{{fig:adam-vs-decay}}

The ratio stays a little below 1 during memorisation and dips lowest right when the network groks: its minimum, {{num:seeds.adam_dip_ci}} (mean and 95 % interval over seeds), falls within 1,000 steps of the grokking step in {{num:seeds.adam_dip_near_grok}} seeds. At that moment weight decay outweighs the loss by about 10 %, so the weights shrink — the cleanup that the paper attributes to weight decay, seen on the actual steps. (This pattern was noticed after the experiments, not predicted in advance, and the Adam state was saved only every 1,000 steps.)

At step 40,000, across the 11 weight-decay seeds, the loss step is {{num:seeds.adam_ratio_end_ci}} times the size of the weight-decay step (mean and 95 % interval), and its cosine with the weights is {{num:seeds.adam_cos_w_end_ci}} (bootstrap intervals). The two steps are nearly equal in size and nearly opposite in direction: the balance described above, measured on the steps that are actually taken. The raw gradient alone would not show this reliably — its cosine with the weights at the last checkpoint ranges from {{num:seeds.cos_gw_end_range}} across seeds, because Adam averages the gradient over many steps and rescales every coordinate. How much each weight matrix moves per step:

{{fig:update-norms}}

Where these forces take the weights: each run's path through (weight norm, loss) during training. With weight decay the norm first grows, then shrinks while the test loss collapses; without it the norm keeps growing and the test loss stays high.

{{anim:loss-vs-norm}}

## Why does grokking happen? The paper's account

Combining the measurements, the paper's explanation (Sec. 5 and the speculative App. E) runs as follows {{cite:nanda2023}}:

1. Memorisation is fast but its weight cost grows with the amount of data; the Fourier circuit has a fixed cost.
2. Weight decay therefore slowly favours the circuit, which grows while train loss stays flat (circuit formation).
3. Once the circuit carries the training data, the memorising weights are pure cost and weight decay removes them (cleanup) — only then does the test loss drop.

ARENA's bonus chapter adds intuition: gradients from individual memorised examples point in unrelated directions and largely cancel, while gradients that serve the general pattern agree across examples and add up {{cite:arena152}}. These accounts are explanations of the measurements, not proofs; the paper labels parts of its App. E as speculative.

::: nuance
**What we can and cannot measure here.** For the paper's run we can recompute gradients but not Adam's internal running averages (they were not saved), so the size of the actual updates is unknown. For our own runs we saved Adam's state every 1,000 steps and logged each update's size, which the panels in “What Adam actually does” use; the gradient panels work for both.
:::
