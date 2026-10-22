Step 2 of the algorithm needs products such as $\cos(wa)\cos(wb)$. This chapter shows where the network computes them.

## Neurons compute degree-2 polynomials of one frequency

The paper approximates each neuron's activation, as a function of (a, b), by a [[dfn:degree-2-polynomial]] of the waves of a single frequency: a constant, the four linear terms (cos/sin of w a, cos/sin of w b) and the four quadratic terms (products like cos(w a)·cos(w b)) — exactly the 3 × 3 block of chapter 10's 2D basis. Measured by the fraction of variance explained:

{{quote:nanda-neurons}}

Our measurement on the paper's network gives the same count, **{{num:paper.neurons_085}}** neurons (the definition and code follow the paper's analysis code {{cite:paperrepo}}). Neurons sharing a dominant frequency form a [[dfn:neuron-cluster|frequency cluster]]:

{{fig:neuron-freq-scatter}}

The cluster sizes are {{num:paper.cluster_sizes}} — the paper reports 44 neurons for k = 14 and ARENA the full list {{cite:arena152}}. The remaining {{num:paper.always_firing}} neurons are positive on almost every input (colour), so their ReLU never “switches”; they act linearly and carry little frequency structure (ARENA calls them the always-firing cluster).

The scatter shows only each neuron's best frequency. The full table — every neuron's share of variance at every frequency — makes the clusters visible as bright cells in a few columns:

{{fig:neuron-freq-map}}

How does a cluster form? Below, every neuron is a dot placed by the frequency-k part of its input from the token a — the $W_{neur}$ of chapter 13, summed over the four heads. Distance from the centre is the size of that wave, the angle is its phase. Watch the dots: they start at the centre, spread into a grey cloud while the network memorises, and then most of them shrink back while the neurons of cluster k move out, each at its own angle.

{{anim:neuron-stars}}

## The ReLU does the multiplication

How can a network built from matrix products and ReLUs multiply two numbers? A ReLU applied to a *sum* of waves produces products. ARENA's example {{cite:arena152}}: take $f(a,b) = \ReLU\big(A + \cos(wa) + \cos(wb)\big)$ with a small constant A, and fit it by least squares with a constant, $\cos(wa)$, $\cos(wb)$ and the product $\cos(wa)\cos(wb)$:

::: example
**ReLU as a multiplier** (k = 42, unit-normalised waves, A = 0.5/√113 as in ARENA's code). Least squares gives a positive coefficient on the product term, and the fit explains {{num:paper.relu_r2_full}} of the variance with the product term but only {{num:paper.relu_r2_lin}} without it (ARENA: 0.966 and 0.849). Intuition: ReLU passes the sum through where it is positive — mostly where both cosines are large — so its output is larger when cos(wa) and cos(wb) are large *together*, which is what a product term measures.
:::

{{widget:relu-multiplier}}

The dashboard shows that the quadratic share of the neurons' variance grows sharply *after* the ReLU compared with before it:

{{fig:quad-scatter}}

## Attention heads: lookup tables of one frequency

Chapter 6 showed $A^j_0(a, b) \approx \sigma\big(C_j[a] - C_j[b]\big)$ with a lookup table $C_j$ over the 113 numbers. In the trained network each $C_j$ is a single wave:

{{fig:attn-C-fourier}}

How each head's lookup table got there, over the whole run (pick the head in the dashboard's control bar):

{{fig:head-spectra-over-training}}

The paper fits $C_j[a] \approx F_j\cos(w_k a) + G_j\sin(w_k a)$ and finds 92–99 % variance explained for the four heads {{cite:nanda2023}} (App. C.1.3). Two more facts make attention useful for step 2:

- **The sigmoid is in its almost-linear range**: replacing it by its best straight-line fit explains {{num:paper.linear_fve}} of the attention variance (paper: 97.5 %), and the network still works with the linearised softmax — loss {{num:paper.linearised_loss}} (paper: 2.12 × 10⁻⁷).
- **So a head's output is roughly** $\big(0.5 + E(C_j[a] - C_j[b])\big)\cdot\text{(value of a)} + \ldots$ — attention weight (a wave in a and b) times value (a wave in a): a product of waves, i.e. degree-2 terms.

The paper finds that two heads compute such products at a single frequency, while the other two have nearly opposite attention coefficients and OV circuits containing all key frequencies — they effectively amplify the embedding in the residual stream; the authors offer a speculative explanation via weight decay {{cite:nanda2023}} (App. C.1.3):

{{fig:ov-fourier}}

::: nuance
**“Most”, not “all”.** About 15 % of neurons are not single-frequency, and the attention analysis is described by the paper as a “high-level sketch”. The polynomial description is an approximation that is good enough to replace the neurons without hurting the network (chapter 16) — that, not a perfect fit, is the standard of evidence.
:::
