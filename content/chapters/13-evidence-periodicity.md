[[dfn:mechanistic-interpretability|Mechanistic interpretability]] tries to reverse-engineer the algorithm inside a trained network from its weights and activations. The paper supports its claim with four lines of evidence {{cite:nanda2023}}; this chapter is the first: everything in the trained network is periodic, and sparse in the Fourier basis. All figures show the paper's network at the selected step (default: step 40,001, the model the paper analyses) — use the dashboard to move through training.

## The embedding uses only a few frequencies

Each of the 128 rows of $W_E$ is a list of 113 numbers — one per input number. Taking the Fourier transform of every row and measuring, for each frequency, the total size across the 128 rows gives the embedding's spectrum:

{{fig:we-fourier-bars}}

At the end of training only six frequencies carry visible weight in the paper's network: {{num:paper.we_top6}}. Five of them — {{num:paper.key_freqs}} — reappear in every later part of the network; the paper calls them the key frequencies:

{{quote:nanda-key-frequencies}}

(The sixth, k = 31, shows up in the embedding but not in the later layers; the paper notes it without naming it, and ARENA's text lists it {{cite:arena152}}.) Out of 56 possible frequencies the network uses 5 — and *which* five is arbitrary: other random seeds pick different ones (chapter 20).

::: check
**A random control.** Is this sparsity special? A [[dfn:control]] answers that: compute the same spectrum for the network at initialisation (move the slider to step 1 in the dashboard) — every frequency has similar weight. The Gini coefficient of the spectrum is {{num:paper.gini_we_init}} at initialisation and {{num:paper.gini_we}} at step 40,001.
:::

## Attention patterns are periodic

For each head, the fraction of attention “=” pays to a, for every input (a, b), forms a striped 113 × 113 picture:

{{fig:attn-grids-all}}

{{fig:attn-grid}}

The paper reports head 0's pattern to be periodic with frequency 35; our analysis gives head frequencies {{num:paper.head_freqs}} for heads 0–3 (paper Table 2: 35, 42, 52, 42). Hover the grid: the value depends on a and b through waves of one frequency.

## Neurons are periodic

The activation of a neuron over all inputs is equally striking:

{{fig:neuron-grid}}

The right panel is the neuron's 2D Fourier transform (chapter 10, with its average removed): almost all of it sits in the 3 × 3 block of one frequency — constant, linear terms and quadratic terms of that frequency. Neuron 0 of the paper's network is periodic with frequency 42 {{cite:nanda2023}} (Sec. 4.1). Here are more neurons, three from each frequency cluster:

{{fig:neuron-small-multiples}}

## Logits are periodic

Finally the paper takes the 2D Fourier transform of the logits over (a, b) for every answer c and finds only twenty significant components — the products of sines and cosines at the five key frequencies, five 2 × 2 blocks (paper Fig. 4, right). We see the same in the dashboard's geometry and output panels, and test it quantitatively in chapter 15.

::: nuance
**Periodic ≠ explained.** Periodicity shows *which* basis to look in, not *what* is computed with it. The next three chapters test the mechanism itself: neurons computing products (ch. 14), the logits reading off cos(w(a+b−c)) (ch. 15), and interventions that would break if the story were wrong (ch. 16).
:::
