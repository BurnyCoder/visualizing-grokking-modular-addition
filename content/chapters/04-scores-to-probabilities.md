The network's final output is not a number like “37” but a list of 113 scores, one per possible answer. This chapter explains how scores become probabilities and how wrongness is measured.

## Exponential and logarithm

The [[dfn:exponential]] $\exp(x) = e^x$ (with $e \approx 2.718$) turns any number into a positive one: $e^0 = 1$, $e^1 \approx 2.718$, $e^{-1} \approx 0.368$, $e^{5} \approx 148.4$. Adding to x *multiplies* the result: $e^{x+1} = e\cdot e^x$.

The [[dfn:logarithm]] undoes it: $\ln(e^x) = x$. So $\ln 1 = 0$, $\ln 113 \approx 4.727$, and $\ln$ of a number between 0 and 1 is negative ($\ln 0.5 \approx -0.693$).

## From scores to a probability distribution

A [[dfn:probability-distribution]] over the 113 answers is a list of 113 non-negative numbers that add up to 1. The network instead outputs one unconstrained score per answer — a [[dfn:logit]]. The [[dfn:softmax]] function converts logits into probabilities:

$$\softmax(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}.$$

Exponentials make every score positive; dividing by their total makes them add up to 1. Adding the same number to all logits changes nothing (it multiplies numerator and denominator by the same factor) — only *differences* between logits matter.

::: example
**Three answers with logits (2, 1, 0).** $e^2 \approx 7.39$, $e^1 \approx 2.72$, $e^0 = 1$, total 11.11. Probabilities: 7.39/11.11 ≈ **0.665**, 2.72/11.11 ≈ **0.245**, 1/11.11 ≈ **0.090**. Doubling all logits to (4, 2, 0) sharpens the distribution to ≈ (0.867, 0.117, 0.016).
:::

Press ▶ next to “scale all logits” to multiply every logit by a growing factor: the probabilities pile up on the largest logit, so the loss shrinks when the correct answer has the largest logit and grows when it does not.

{{widget:softmax}}

The predicted answer is the [[dfn:argmax]] — the position of the largest logit (equivalently, of the largest probability). With only two options, softmax has a simple form: the probability of the first is $\frac{e^{x_1}}{e^{x_1}+e^{x_2}} = \frac{1}{1+e^{-(x_1-x_2)}}$, the [[dfn:sigmoid]] $\sigma$ of the difference. We will meet this when an [[preview:attention-head|attention head]] chooses between a and b.

## Measuring wrongness: the cross-entropy loss

A [[dfn:loss]] is a single number saying how wrong the outputs are. The network is trained with the [[dfn:cross-entropy]] loss: for one example it is $-\ln p$, where $p$ is the probability given to the correct answer, and for a set of examples we average {{cite:wiki_crossentropy}} {{cite:torch_ce}}:

$$\mathcal L = -\frac{1}{N}\sum_{\text{examples}} \ln \softmax(\Logits)_{c^*}.$$

::: example
**Loss values to remember.**

- Correct answer gets probability 1 → loss $-\ln 1 = 0$ (perfect).
- Correct answer gets probability 0.5 → loss ≈ 0.693.
- Uniform guessing over 113 answers (each gets 1/113) → loss $\ln 113 \approx$ **4.727**. This is the “chance level”; a freshly initialised network starts near it.
- Correct answer gets probability $10^{-10}$ → loss ≈ 23 — far *worse* than guessing. Confidently wrong answers are punished hard.
:::

The fraction of examples whose argmax is the correct answer is the [[dfn:accuracy]]. Accuracy only cares whether the right answer is ranked first; loss also cares *how* confident the network is. The paper stresses a consequence:

{{quote:nanda-zero-loss}}

Even with 100 % accuracy, the loss can always be lowered by making the logits more extreme, so [[preview:training|training]] never “finishes” — which will matter when we look at what [[preview:weight-decay|weight decay]] does in chapter 19.

::: nuance
**Precision.** When the network is very confident, probabilities like $1 - 10^{-9}$ cannot be represented accurately in standard 32-bit floating point (it cannot distinguish numbers closer than about $1.2\times10^{-7}$ to 1). The paper's analysis code therefore computes the loss in 64-bit precision; this avoids artificial loss spikes {{cite:paperrepo}} {{cite:arena152code}}. We do the same.
:::
