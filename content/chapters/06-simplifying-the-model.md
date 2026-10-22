Before asking *what* the trained network computes, it helps to find out which of its parts matter at all. The ARENA course does this first, deriving a simpler “functional form” of the network {{cite:arena152}}; the paper reports the same facts in its appendix {{cite:nanda2023}}. All numbers below are measured on the paper's trained network at step 40,001.

## Ablations: switch a part off and see what breaks

An [[dfn:ablation]] removes or replaces one part of the computation and measures how much the output changes. A [[dfn:zero-ablation]] replaces the part's output with zeros; a [[dfn:mean-ablation]] replaces it with its average over all inputs, which removes only the information that depends on the input (zeroing can also remove a useful constant offset).

A [[dfn:skip-connection]] is a path that carries a layer's input around the layer and adds it back — the “x +” in $x^{(1)} = x^{(0)}_2 + \ldots$ and $x^{(2)} = x^{(1)} + \ldots$ of chapter 5.

::: example
**What matters in the trained network** (loss on all 12,769 inputs; unablated: {{num:rc.R1}}; uniform guessing: 4.727):

| Ablation | Loss | Interpretation |
|---|---|---|
| Zero the attention output | {{num:rc.R11c}} | attention is essential (worse than guessing) |
| Zero the skip connection around the MLP | {{num:rc.R11a}} | negligible: accuracy unchanged |
| Mean-ablate the skip connection around the MLP | {{num:rc.R11b}} | negligible |
| Attention from “=” to itself (per head) | {{num:paper.attn_eq_pct}} | tiny, can be ignored |

The paper reports 24.3, 9.12 × 10⁻⁷ and 7.25 × 10⁻⁷ for the first three and 0.1–0.4 % for the last {{cite:nanda2023}} (App. A.1).
:::

{{quote:nanda-skip}}

## Consequence 1: logits come from the neurons through one matrix

If the skip connection around the MLP is ignored, $x^{(2)} \approx W_{out}\,\mathrm{MLP}(a,b)$ (plus constants), so

$$\Logits(a,b) \approx W_U W_{out}\,\mathrm{MLP}(a,b) = W_L\,\mathrm{MLP}(a,b).$$

The paper calls $W_L = W_U W_{out}$ (a 113 × 512 matrix) the [[dfn:neuron-logit-map]] [[symdef:WL]]: column m says how much one unit of neuron m's activation adds to each answer's logit. ARENA calls the same matrix (transposed) $W_{logit}$. A product of matrices that maps one meaningful quantity directly to another is an [[dfn:effective-weights|effective weight]] matrix; we will study $W_L$ in the [[preview:fourier-basis|Fourier basis]] in chapter 15.

## Consequence 2: each head is a two-way choice between a and b

Attention to “=” is negligible, so each head's softmax is effectively over two scores, which (chapter 4) is a sigmoid of their difference:

$$A^j_0 \approx \sigma\big(s_0 - s_1\big), \qquad A^j_1 \approx 1 - A^j_0 .$$

Moreover the query comes from the “=” position, whose vector $x^{(0)}_2$ is the same for every input. So the score of position 0 depends only on the token there: it is a fixed lookup table [[symdef:Cj]] over the 113 numbers, $s_0 = C_j[a]$, and likewise $s_1 \approx C_j[b]$ (up to the small difference between the positional embeddings of positions 0 and 1, see below). Hence

$$A^j_0(a, b) \approx \sigma\big(C_j[a] - C_j[b]\big).$$

This is the paper's App. C.1.3 derivation {{cite:nanda2023}}; ARENA calls $C_j$ the effective attention weights $W_{attn}$ {{cite:arena152}}.

{{fig:avg-attn}}

Here is that two-way choice in the real network. Each dot is one input: its attention to a against the difference of the two scores, with the sigmoid drawn on top (pick another head in the dashboard's control bar).

{{fig:sigmoid-linearisation}}

And the choice on every input at once: for each head, the attention from “=” to a, to b and to itself over all 12,769 questions.

{{fig:attention-heads-grid}}

## Consequence 3: the input is symmetric in a and b

The problem has [[dfn:commutativity]]: a + b = b + a. A network that has learned the task should treat a and b alike. In the trained network the positional embeddings of positions 0 and 1 are both **small** — length {{num:paper.pos_norm0}} and {{num:paper.pos_norm1}}, against {{num:paper.we_norm_mean}} for a typical token embedding — and point in similar directions ([[preview:cosine|cosine]] similarity {{num:paper.pos_cos01}}), whereas the “=” position's embedding is large ({{num:paper.pos_norm2}}). So whether a number sits at position 0 or 1 barely changes its vector, and a and b are processed (almost) identically:

{{fig:pos-embed}}

## The simplified network

Putting it together, the trained network behaves like

$$\begin{aligned}\Logits(a,b) \approx W_L\,\ReLU\Big(W_{in}\big[\,&x_2 + \textstyle\sum_j W_O^j W_V^j\big(\sigma(C_j[a]{-}C_j[b])\,E_a\\ &+ (1-\sigma(\cdot))\,E_b\big)\big] + b_{in}\Big),\end{aligned}$$

where $E_t$ is token t's embedding plus its positional vector. Everything that depends on the input enters through the embeddings $E_a$, $E_b$, the lookup tables $C_j$, a ReLU, and the neuron–logit map $W_L$. The rest of the site explains what each of these does.

::: nuance
These simplifications are justified *empirically for the trained network*. Early in [[preview:training|training]], the skip connections and attention to “=” can matter; the dashboard's attention panels show attention to “=” jumping back up around step 43,000 — a late change that neither the paper nor ARENA explains {{cite:arena152}}.
:::
