A small [[preview:neural-network|neural network]] was trained to answer one kind of question: *what is a + b, if you count on a clock with 113 hours?* For thousands of [[preview:training|training]] steps it seemed only to have memorised the answers it was shown, failing on most others. Then — long after it had stopped improving on its training questions — it suddenly started answering *every* question correctly. Researchers reverse-engineered what it had learned: it turns numbers into angles and adds them by rotating. In the words of the paper that did this {{cite:nanda2023}}, the network's algorithm

{{quote:nanda-algorithm}}

Here is that story in one view: the paper's own network, from its first training step to just after it learns the task. It plays while it is on screen; pause it or drag its slider to look at one moment, and watch the four views change together.

{{anim:hero-grokking}}

This site explains that result from the ground up, with the real network's numbers on every page. You do not need prior knowledge of machine learning, linear algebra or trigonometry: every word and every symbol is explained the first time it is used, with a small worked example. Hover any dotted word or symbol for a one-sentence reminder.

## The objects this site explains

Here they are, all at once. You are **not** expected to understand them yet — by the last chapter you will understand every symbol. Each symbol has a card: hover it (or read the table below).

::: object
**1. What the network computes.** For an input question (a, b), with t₀, t₁, t₂ standing for the three input tokens a, b and “=”:

$$x^{(0)}_i = W_E\, t_i + W_{pos}[i] \qquad (i = 0, 1, 2)$$

$$A^j = \softmax\!\Big(\tfrac{1}{\sqrt{32}}\, \big(W_Q^j x^{(0)}_2\big)^{\!\top} W_K^j\, x^{(0)}_i \Big)_{i=0,1,2} \qquad (j = 0,1,2,3)$$

$$x^{(1)} = x^{(0)}_2 + \sum_{j} W_O^j W_V^j \sum_{i} A^j_i\, x^{(0)}_i$$

$$\mathrm{MLP}(a,b) = \ReLU\big(W_{in}\, x^{(1)} + b_{in}\big), \qquad x^{(2)} = x^{(1)} + W_{out}\,\mathrm{MLP}(a,b) + b_{out}$$

$$\Logits(a,b) = W_U\, x^{(2)}, \qquad \text{answer} = \argmax_c \Logits(a,b)_c$$
:::

The same computation as a picture, with the real network's numbers for one question. Each column is one stage; click a stage for its formula (chapter 5 explains every one).

{{fig:forward-flow}}

::: object
**2. What those numbers turn out to mean.** For the trained network, to a very good approximation:

$$\Logits(a,b)_c \;\approx\; \beta + \sum_{k \in \mathcal K} \alpha_k \cos\!\big(w_k\,(a+b-c)\big), \qquad w_k = \frac{2\pi k}{p}$$

where each term is built from the inputs with two [[preview:angle-addition|angle-addition identities]],

$$\cos(w_k(a{+}b{-}c)) = \cos(w_k(a{+}b))\cos(w_k c) + \sin(w_k(a{+}b))\sin(w_k c),$$

$$\begin{aligned}\cos(w_k(a{+}b)) &= \cos(w_k a)\cos(w_k b) - \sin(w_k a)\sin(w_k b),\\ \sin(w_k(a{+}b)) &= \sin(w_k a)\cos(w_k b) + \cos(w_k a)\sin(w_k b),\end{aligned}$$

and the sum over the set 𝒦 of a few “key” frequencies is largest exactly when c = (a + b) mod p.
:::

The trick at the heart of it, on a clock with 113 hours: turning by a and then by b lands exactly where a + b is. Move the candidate answer c (or press ▶ next to it) and watch its score reach its top only where c meets a + b.

{{widget:add-angles}}

::: object
**3. How it got there.** Training repeats, tens of thousands of times, the update

$$\theta \;\leftarrow\; \theta\,(1 - \gamma\lambda) \;-\; \gamma\cdot \text{AdamStep}\big(\nabla_\theta \mathcal{L}\big)$$

and along the way passes through three phases — memorisation, circuit formation, cleanup — that can be measured before the sudden jump in test [[preview:accuracy|accuracy]] happens.
:::

| Symbol | Meaning (one sentence) |
|---|---|
| [[symdef:p]] | The prime the numbers wrap around at; p = 113, so the possible answers are 0, 1, …, 112. |
| [[symdef:a]], [[symdef:b]] | The two numbers of the question “a + b = ?”, each from 0 to p − 1. |
| [[symdef:c]], [[symdef:cstar]] | A candidate answer c that the network scores; the correct one is c* = (a + b) mod p. |
| [[symdef:eq]] | The third input token; the answer is read off at its [[preview:position|position]]. |
| [[symdef:WE]] | The [[preview:embedding|embedding]] [[preview:matrix|matrix]]: column t is the list of 128 numbers that represents token t. |
| [[symdef:Wpos]] | Positional embeddings: one list of 128 numbers per position, added to the token's list. |
| [[symdef:x0]] | The list of numbers at position i before any processing. |
| [[symdef:WQ]], [[symdef:WK]], [[symdef:WV]], [[symdef:WO]] | The four matrices of [[preview:attention-head|attention head]] j (query, key, value, output). |
| [[symdef:A]] | How strongly the “=” position reads from positions a, b and “=” in head j (three numbers summing to 1). |
| [[symdef:x1]] | The “=” position's list of numbers after the [[preview:attention|attention]] step. |
| [[symdef:Win]], [[symdef:bin]] | Input weights and offsets of the 512 hidden “neurons”. |
| [[symdef:MLP]] | The 512 [[preview:neuron|neuron]] values for input (a, b). |
| [[symdef:Wout]], [[symdef:bout]] | How the neurons write back into the list of numbers. |
| [[symdef:x2]] | The final list of 128 numbers at the “=” position. |
| [[symdef:WU]] | The [[preview:unembedding|unembedding]] matrix that turns the final list into one score per answer. |
| [[symdef:logits]] | The 113 scores, one per candidate answer c. |
| [[symdef:k]], [[symdef:wk]] | A [[preview:frequency|frequency]] k (a whole number 1–56) and its angle step w_k = 2πk/p. |
| [[symdef:K]] | The few [[preview:key-frequencies|key frequencies]] the trained network uses. |
| [[symdef:alpha]] | How strongly frequency k contributes to the scores. |
| [[symdef:theta]] | All 226,816 learned numbers of the network. |
| [[symdef:L]] | The loss: how wrong the network's scores are. |
| [[symdef:gamma]], [[symdef:lam]] | The [[preview:learning-rate|learning rate]] (0.001) and the weight-decay strength (1). |
| [[symdef:d]], [[symdef:n]], [[symdef:H]] | Sizes: 128 numbers per position, 512 neurons, 4 attention heads. |

## How to read this site

- **Chapters 1–11** build the background: clock arithmetic, the [[preview:dataset|dataset]], vectors and matrices, probabilities, the network itself, how it learns, what [[preview:grokking|grokking]] is, waves, the [[preview:fourier-basis|Fourier basis]], and how to measure a fit. Each chapter only uses what earlier chapters defined.
- **Chapters 12–16** derive the algorithm and test it against the trained network in four ways, following the paper {{cite:nanda2023}} and the ARENA course {{cite:arena152}}.
- **Chapters 17–20** go back in time: how the algorithm forms during training, what drives it, and whether it happens the same way for other random seeds (we retrained 22 networks to check).
- **Chapter 21** returns to the three objects above, now with every piece explained, plus the limitations and open questions.
- **The [dashboard](../dashboard.html)** shows every visualization on one page, linked together and scrubbable through training.

Every number on this site is computed from real networks: by default the paper's own published training run {{cite:paperckpt}}, with our replications one click away. Our analysis code reproduces most of the paper's reported numbers on the paper's own network, and the few it does not are listed with explanations (see the [reproduction page](../reproduce.html)). Where a step or claim comes from somewhere, it is cited; quotations are exact and short.

This model is a toy: one task, one layer, ~227 thousand learned numbers. Its value is that its algorithm has been reverse-engineered in unusual detail — the paper's authors used it as a proof of concept for measuring hidden progress in training. The three videos that inspired this site tell the same story from different angles {{cite:video_replication}} {{cite:video_paper}} {{cite:video_welch}}.
