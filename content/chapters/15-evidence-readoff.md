Step 3 says the logits are a *linear* read-out of $\cos(w_k(a+b))$ and $\sin(w_k(a+b))$ with weights $\cos(w_k c)$ and $\sin(w_k c)$. Because the skip connection around the MLP barely matters (chapter 6), the logits are essentially $W_L\,\mathrm{MLP}(a,b)$, so the claim is about the 113 × 512 matrix $W_L$.

## W_L contains only the key frequencies

If $\Logits_c \approx \sum_k \cos(w_k c)\,[\ldots] + \sin(w_k c)\,[\ldots]$, then every column of $W_L$, read as a function of c, must be a mix of the waves $\cos(w_k c)$, $\sin(w_k c)$ for k ∈ 𝒦 — ten waves in total. The Fourier transform of $W_L$ along c:

{{fig:wl-fourier-bars}}

Over the whole run: how strongly each frequency's wave appears in the logits (or how much the training loss rises without it), with the selected frequency drawn below.

{{fig:alpha-growth}}

Only the ten key-frequency components are visible: $W_L$ has [[term:rank]] about 10. Approximating $W_L$ by its projection onto those ten waves leaves a residual whose squared [[dfn:frobenius-norm]] is {{num:paper.wl_resid_sq}} of $W_L$'s at step 40,001 ({{num:paper.wl_resid_sq_last}} at the released run's last checkpoint, step {{num:paper.last_step}}); its norm itself is {{num:paper.wl_resid_norm}} of $W_L$'s. The paper reports a residual norm “under 0.55 %”: that is close to our *squared* ratio (marginally above 0.55 % at step 40,001), not to the norm ratio; we flag the wording on the [errata page](../errata.html).

Writing the ten rows as $W_L \approx \sum_{k}\cos(w_k)\,u_k^\top + \sin(w_k)\,v_k^\top$ (paper eq. 2) defines, for each key frequency, two directions [[symdef:uk]] in the 512-dimensional neuron space: whatever lies along $u_k$ is written into the logits as $\cos(w_k c)$, whatever lies along $v_k$ as $\sin(w_k c)$.

## The neurons hold cos(w_k(a+b)) and sin(w_k(a+b))

The read-off $u_k^\top\,\mathrm{MLP}(a,b)$ is one number per input (a, b) — a 113 × 113 grid. The paper's Table 1 shows that this grid is almost exactly a multiple of $\cos(w_k(a+b))$, and $v_k^\top\mathrm{MLP}(a,b)$ a multiple of $\sin(w_k(a+b))$, with 93–98 % of the variance explained {{cite:nanda2023}}. On the paper's network we measure an average of {{num:paper.table1_fve_mean}} over the ten directions (range {{num:paper.table1_fve_range}}). This is the MLP *outputting* the result of step 2.

ARENA shows the same thing from the neurons' side {{cite:arena152}}: each neuron writes only the two waves of its own frequency into the logits — a block structure once the neurons are sorted by cluster:

{{fig:wlogit-sorted}}

## Linear terms in the neurons, none in the logits

The neurons hold more than the products: chapter 13's 3 × 3 blocks also contain the linear terms cos(w_k a), sin(w_k b), … Averaging the squared 2D Fourier coefficients over all 512 neurons shows both kinds — on the paper's network {{num:paper.neuron_linear_share}} of the neurons' variance sits in linear terms (row and column 0 of the map):

{{fig:neuron-b2-map}}

The same picture for the logits keeps only the products. In the logits, linear terms carry {{num:paper.logit_linear_share}} of the variance: the read-off $W_L$ adds the neurons up so that their linear parts cancel, leaving cos(w_k(a+b)) and sin(w_k(a+b)) — exactly the terms that step 3 of the algorithm needs. The products of the key frequencies hold {{num:paper.logit_key_product_share}} of the logits' variance:

{{anim:logit-fourier-2d}}

Played through training, watch which components light up, and when.

## The logits are a sum of cos(w_k(a+b−c))

Putting the pieces together, the paper fits all 113 × 113 × 113 logits with five numbers, $\Logits(a,b)_c \approx \sum_k \alpha_k\cos(w_k(a+b-c))$, by least squares (we add a constant). On the paper's network this simple formula explains {{num:paper.logit_fve}} of the variance of the 1,442,897 logits (paper: 95 %), with coefficients α = {{num:paper.logit_alpha}} for k = {{num:paper.key_freqs}}. Replacing the network's logits by the fitted formula gives loss {{num:paper.logit_fit_loss}} — *lower* than the network's own {{num:rc.R1}} (paper: 4.7 × 10⁻⁸ vs 2.4 × 10⁻⁷).

::: example
**What a fit of five numbers means.** For the input (100, 50) the formula predicts that the logit of c is a sum of five cosines peaking at c = 37 (chapter 12's interference picture), and nothing else about the input matters — not whether (100, 50) was a training question, not any memorised detail. 96 % of the variance of 1.4 million logits is captured that way.
:::

## The skip connection: why we could ignore it

The paper reasons with $\Logits \approx W_L\,\mathrm{MLP}$, ignoring the residual path that carries $x^{(1)}$ (the attention output plus the “=” embedding) straight to the unembedding. Chapter 6 measured that zero-ablating that path raises the loss only from {{num:rc.R1}} to {{num:rc.R11a}} with unchanged accuracy, so the approximation is justified for the trained network.
