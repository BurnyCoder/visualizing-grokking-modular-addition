This chapter is a guided walk through training. Use the bar below to pick a run and move through its checkpoints (or press ▶); every figure on the page follows. The [dashboard](../dashboard.html) shows all of these side by side.

{{widget:scrubber}}

## The spectrum sharpens

First, the embedding's Fourier spectrum over the whole run at once — each column is one checkpoint, each row one frequency:

{{fig:we-fourier-heatmap}}

Early on every frequency has some weight. During memorisation the picture stays noisy; during circuit formation a few rows brighten while the rest fade; during cleanup everything except the key frequencies disappears. ARENA observes the same and notes that the singular values of $W_E$ go from a random-matrix profile to a few dominant ones {{cite:arena152}}. The neuron–logit map follows the same pattern, slightly earlier:

{{fig:wl-fourier-heatmap}}

## Circles emerge from noise

Project each number's embedding onto the plane that carries cos(w_k a) and sin(w_k a), for each key frequency. Move the bar from step ~1,000 to ~15,000:

{{fig:embed-circles}}

At initialisation the 113 points are a random cloud. As the circuit forms they spread into a ring whose colours (each number coloured by its predicted clock position k·a mod 113) run smoothly around it — **the number a has become the angle w_k·a**, the clock of chapter 9. The same happens for the answer: the MLP's average activation for each answer c = (a + b) mod 113, read along $W_L$'s cos/sin directions, also traces circles:

{{fig:mlp-circles}}

## The logits learn the trig terms first, the neurons later

How much of the variance of the neurons and of the logits lies in the cos/sin(w_k(a+b)) directions? ARENA finds that the logits acquire this structure smoothly and early, the neurons later and more abruptly {{cite:arena152}}:

{{fig:trig-fraction}}

{{fig:cos-coeffs}}

{{fig:cluster-sizes}}

{{fig:fve-curves}}

## Confidence on every input

Each cell below is one question (a, b); its brightness is the probability the network gives the correct answer. During memorisation the training questions are bright and the test questions dark — the speckle *is* the random train/test split of chapter 2. During cleanup the test questions light up, all together:

{{fig:pcorrect-grid}}

{{fig:entropy-grid}}

## The geometry of the output

Averaging the network's predicted distribution over all inputs with the same correct answer gives 113 points; PCA draws them in two dimensions. As the network becomes certain, each average distribution approaches a one-hot vector, and the points move to the corners of the probability simplex:

{{fig:prob-pca}}

{{fig:logit-pca}}

## A late surprise

Training does not end at step 14,000. Around step 43,000, in the paper's run, two attention heads suddenly start attending to “=” again (on average {{num:paper.attn_eq_45k}} for heads 0–3 at step 45,001, up from {{num:paper.attn_eq_pct}} at 40,001), and the sum of squared weights shows a small kink (955 → 942 in ARENA's description) {{cite:arena152}}. Neither the paper nor ARENA explains it; it is a reminder that even this “fully understood” network has unexplained behaviour.

{{fig:avg-attn}}
