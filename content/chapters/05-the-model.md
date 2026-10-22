## What a neural network is

A [[dfn:neural-network]] is a function assembled from simple steps — mostly matrix–vector products and a few simple nonlinear functions — whose numbers are not written by hand but tuned from data. Each tunable number is a [[dfn:parameter]] (also called a weight). Some parameters are a [[dfn:bias|bias]]: a number added regardless of the input.

The network studied here is a [[dfn:transformer]], the architecture behind today's language models {{cite:vaswani2017}}, shrunk to a single [[dfn:layer]] (one attention step followed by one MLP step) {{cite:nanda2023}}. It has {{num:model.n_params}} parameters. Running it on an input from start to finish is a [[dfn:forward-pass]]. Below we follow one forward pass on our running example (a, b) = (100, 50); the widget computes every intermediate number with the paper's trained weights.

{{widget:forward-walkthrough}}

The same pass as a picture: every stage with its numbers for the selected question, and small pulses running from the tokens to the answer. Click a stage to see its formula; the sections below explain each one.

{{fig:forward-flow}}

## Step 1 — embeddings: tokens become vectors

Each token is turned into a vector of dimension [[sym:d]] = 128 by its [[dfn:embedding]]: column t of the matrix [[sym:WE]] (via the one-hot product of chapter 3). A [[dfn:positional-embedding]] — row i of [[sym:Wpos]] — is added so the network knows where each token sits:

$$x^{(0)}_i = W_E\, t_i + W_{pos}[i].$$

For (100, 50, =): $x^{(0)}_0$ = column 100 of $W_E$ + position-0 vector, $x^{(0)}_1$ = column 50 + position-1 vector, $x^{(0)}_2$ = column 113 + position-2 vector.

These three vectors start the [[dfn:residual-stream]]: a running vector at each position that every later step reads from and *adds* its result to {{cite:elhage2021}}.

## Step 2 — attention: the “=” position gathers information about a and b

[[dfn:attention|Attention]] lets each position read from positions at or before it. The answer is read at the “=” position (position 2), and only its result matters, so we follow position 2. The model has [[sym:H]] = 4 [[dfn:attention-head|attention heads]], each with its own four matrices. Head j:

1. computes a [[dfn:query]] for position 2, $q = W_Q^j x^{(0)}_2$ (32 numbers), and a [[dfn:key]] for every position i, $k_i = W_K^j x^{(0)}_i$;
2. computes an [[dfn:attention-score]] $s_i = q\cdot k_i/\sqrt{32}$ for each position — the dot product measures how well the key “matches” the query; dividing by $\sqrt{32}$ keeps the numbers in a reasonable range {{cite:vaswani2017}};
3. turns the three scores into the [[dfn:attention-pattern]] $A^j = \softmax(s_0, s_1, s_2)$ — three weights that sum to 1;
4. computes a [[dfn:value]] $v_i = W_V^j x^{(0)}_i$ for each position, mixes them with the pattern, and writes the result back to dimension 128 with $W_O^j$:

$$\text{head}_j = W_O^j W_V^j \sum_{i=0}^{2} A^j_i\, x^{(0)}_i .$$

The [[dfn:causal-mask]] forbids positions 0 and 1 from reading later positions (it sets those scores to a huge negative number, so their softmax weight is 0); position 2 may read all three. Two matrix products summarise a head {{cite:elhage2021}}: the [[dfn:qk-circuit]] $W_Q^{j\top}W_K^j$ decides *where* to look, and the [[dfn:ov-circuit]] $W_O^jW_V^j$ decides *what* gets written. The outputs of the four heads are added to the residual stream:

$$x^{(1)} = x^{(0)}_2 + \sum_{j=0}^{3} \text{head}_j .$$

::: example
**Attention in miniature.** Suppose head 0's scores at the “=” position are $s = (1.0, 0.6, -5)$ for (a, b, =). Softmax gives weights ≈ (0.598, 0.401, 0.001): the head reads 60 % from a, 40 % from b and almost nothing from “=”. If a's value vector after $W_O$ is $u_a$ and b's is $u_b$, the head writes ≈ $0.598\,u_a + 0.401\,u_b$. For the trained network the weights to “=” are indeed tiny (about 0.1–0.4 %) {{cite:nanda2023}}.
:::

## Step 3 — the MLP: 512 neurons

The [[dfn:mlp|MLP]] multiplies $x^{(1)}$ by $W_{in}$ (512 × 128), adds the bias $b_{in}$, and applies the [[dfn:relu|ReLU]], $\ReLU(z) = \max(0, z)$, to each of the 512 entries. Each entry is a [[dfn:neuron]]: its value before the ReLU is its [[dfn:pre-activation]], after the ReLU its [[dfn:activation]].

$$\mathrm{MLP}(a,b) = \ReLU\big(W_{in}x^{(1)} + b_{in}\big), \qquad x^{(2)} = x^{(1)} + W_{out}\,\mathrm{MLP}(a,b) + b_{out}.$$

::: example
**ReLU on four pre-activations** (2.3, −0.7, 0.0, 5.1) gives activations (2.3, 0, 0, 5.1). It is the only nonlinearity in the MLP, and — chapter 14 shows — the network uses it to *multiply* two waves together.
:::

## Step 4 — unembedding: one logit per answer

Finally the [[dfn:unembedding]] matrix $W_U$ turns the 128 numbers of $x^{(2)}$ into 113 logits (the 114th, for “=”, is ignored), and the answer is the argmax:

$$\Logits(a,b) = W_U\,x^{(2)}.$$

## All the parameters

| Matrix | Shape (as stored) | Count | Role |
|---|---|---|---|
| $W_E$ | 128 × 114 | 14,592 | token embeddings |
| $W_{pos}$ | 3 × 128 | 384 | positional embeddings |
| $W_Q, W_K, W_V$ | 3 × (4 × 32 × 128) | 49,152 | queries, keys, values of 4 heads |
| $W_O$ | 128 × 128 | 16,384 | head outputs |
| $W_{in}$, $b_{in}$ | 512 × 128, 512 | 66,048 | neuron inputs |
| $W_{out}$, $b_{out}$ | 128 × 512, 128 | 65,664 | neuron outputs |
| $W_U$ | 128 × 114 | 14,592 | unembedding |
| **Total** | | **226,816** | |

There is no layer normalisation and the embedding and unembedding are separate matrices, as in the paper {{cite:nanda2023}}; the only biases are those of the MLP {{cite:paperrepo}}.
