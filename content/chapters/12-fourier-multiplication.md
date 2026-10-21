We now have every ingredient to state the paper's central claim. Nanda et al. write {{cite:nanda2023}}:

{{quote:nanda-claim}}

— and the algorithm, which they call [[dfn:fourier-multiplication|Fourier multiplication]], is the one below. This chapter derives it step by step and runs it by hand on our example; chapters 13–16 then check it against the trained network.

## The algorithm, step by step

Fix one frequency k and write $w = w_k = 2\pi k/113$.

**Step 1 — embed: numbers become angles.** Map a to $\cos(w a)$ and $\sin(w a)$, and b to $\cos(w b)$ and $\sin(w b)$. In the network this is done by the embedding matrix $W_E$ (chapter 13 shows its columns *are* such waves).

**Step 2 — combine: add the angles with the identities of chapter 9.**

$$\begin{aligned}\cos(w(a+b)) &= \cos(wa)\cos(wb) - \sin(wa)\sin(wb),\\ \sin(w(a+b)) &= \sin(wa)\cos(wb) + \cos(wa)\sin(wb).\end{aligned}$$

These need *products* of two numbers; in the network they are computed by the attention layer and the MLP (chapter 14 shows how a ReLU multiplies).

**Step 3 — score each candidate answer c** with one more identity (the subtraction form of chapter 9):

$$\cos(w(a+b-c)) = \cos(w(a+b))\cos(wc) + \sin(w(a+b))\sin(wc).$$

For fixed (a, b) this is a *linear* function of the two numbers $\cos(w(a+b))$ and $\sin(w(a+b))$, with weights $\cos(wc)$ and $\sin(wc)$ that depend only on c — exactly what one matrix can do. In the network it is the neuron–logit map $W_L$ (chapter 15). The score is largest, equal to 1, when $a + b - c$ is a multiple of 113, i.e. when c = (a + b) mod 113.

**Step 4 — add up several frequencies.** The network does steps 1–3 for a few [[dfn:key-frequencies|key frequencies]] k ∈ [[sym:K]] at once and adds the results with weights [[sym:alpha]]:

$$\Logits(a,b)_c \approx \sum_{k\in\mathcal K} \alpha_k \cos\big(w_k(a+b-c)\big).$$

## Running it by hand

::: example
**(a, b) = (100, 50), frequency k = 14, so w ≈ 0.7785.**

Step 1: $\cos(100w) = -0.7680$, $\sin(100w) = 0.6404$, $\cos(50w) = 0.3406$, $\sin(50w) = 0.9402$.

Step 2: $\cos(150w) = (-0.7680)(0.3406) - (0.6404)(0.9402) = -0.2616 - 0.6021 = -0.8637$, and $\sin(150w) = (0.6404)(0.3406) + (-0.7680)(0.9402) = 0.2181 - 0.7221 = -0.5040$. (Directly: cos(150w) = −0.8637, sin(150w) = −0.5040. ✓)

Step 3: for c = 37, $\cos(37w) = -0.8637$ and $\sin(37w) = -0.5040$ (37 sits on the same clock position as 150, since 150 − 113 = 37), so the score is $(-0.8637)^2 + (-0.5040)^2 = 1.0000$. For c = 38 the score is 0.7120, for c = 0 it is −0.8637.

But for c = 45 the score is **0.9985** — almost as high as the right answer! At frequency 14, a step of 8 numbers turns the clock by $14\cdot8 = 112 \equiv -1$ steps out of 113, nearly a full turn, so 45 looks almost like 37.
:::

## Why several frequencies: constructive interference

One wave alone cannot single out the answer reliably: it has near-peaks elsewhere (above, c = 45). Different frequencies have their near-peaks in *different* places, but all of them peak *exactly* at c = (a + b) mod 113. Summing them is [[dfn:constructive-interference]] at the right answer and destructive interference (partial cancellation) everywhere else {{cite:nanda2023}} (App. B, Fig. 8).

::: example
**Interference with the paper's key frequencies {14, 35, 41, 42, 52}** (all weights 1): at c = c* every cosine equals 1, so the sum is **5.0**; at every other c the five waves disagree and the largest sum anywhere else is only **2.59**. Softmax then turns this gap into a confident answer.
:::

{{widget:interference-toy}}

On the real network, for the selected input and step (try different a, b in the dashboard), the logits and their best fit by a sum of cosines over the run's key frequencies look like this:

{{fig:output-dist}}

{{fig:interference}}

## Where each step lives in the network

| Step | Mathematical operation | Network component | Evidence |
|---|---|---|---|
| 1 | a ↦ cos(w_k a), sin(w_k a) | embedding $W_E$ | ch. 13: $W_E$ sparse in the Fourier basis |
| 2 | products → cos/sin(w_k(a+b)) | attention + MLP (ReLU) | ch. 14: neurons are degree-2 polynomials of one frequency |
| 3 | cos(w_k(a+b−c)) for every c | neuron–logit map $W_L$ | ch. 15: $W_L$ has rank ≈ 10 and reads cos/sin(w_k(a+b)) |
| 4 | sum over key frequencies | unembedding (adding into the logits) | ch. 15–16: logits ≈ Σ α_k cos(w_k(a+b−c)); ablations |

::: nuance
**Other algorithms exist.** Fourier multiplication is what *this* architecture with *these* settings learned, consistently across seeds in the paper and in all 11 of our weight-decay seeds (chapter 20). Zhong et al. (2023) show that slightly different architectures can learn a different, “pizza” algorithm that also uses circles but combines them differently {{cite:zhong2023}}; Chughtai et al. (2023) generalise the Fourier story to other finite groups using representation theory {{cite:chughtai2023}}.
:::
