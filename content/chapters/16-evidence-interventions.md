Correlations can mislead: a pattern can look like an algorithm without being what the network relies on. A [[dfn:causal-intervention]] changes something inside the network and checks whether the output changes the way the explanation predicts {{cite:geiger2021}}. The paper's fourth line of evidence is a set of such ablations {{cite:nanda2023}}; we add a counterfactual swap test.

## Remove one frequency at a time

Take the 2D Fourier transform of every logit grid (one per answer c), set all components belonging to one frequency k to zero (its 8 linear and quadratic terms), transform back, and measure the loss:

{{fig:freq-ablation-bars}}

Try it yourself: choose frequencies to remove from the logits of every input, or the only ones to keep, and see the loss, the accuracy and the confidence on every input. The run's key frequencies are outlined.

{{fig:ablation-playground}}

On the paper's network (unablated loss {{num:rc.R1}}):

- removing **any single key frequency** raises the loss by a factor of {{num:paper.key_ablation_ratio_range}};
- removing **any single other frequency** changes it by at most a factor of {{num:paper.nonkey_ablation_max_ratio}};
- removing all five key frequencies at once destroys the network (loss {{num:paper.all_key_removed}}, worse than guessing at 4.727);
- removing **everything except** the 8 × 5 = 40 key components *lowers* the loss to {{num:paper.only_key_kept}} (paper: 7.24 × 10⁻⁸).

The same test works on the neurons instead of the logits. Remove the contribution of one frequency cluster of neurons (chapter 14) and measure the loss; ARENA uses this to confirm that every cluster is needed {{cite:arena152}}:

{{fig:cluster-ablations}}

{{quote:nanda-ablate-others}}

::: check
**Random control.** Is any set of five frequencies this important? Removing five *random non-key* frequencies (100 random draws) gives losses between {{num:paper.random5_min}} and {{num:paper.random5_max}} — indistinguishable from the original. The effect belongs to the key frequencies, not to removing “some” structure.
:::

## Replace neurons by their idealised version

If the neurons really are degree-2 polynomials of one frequency (chapter 14), replacing each clustered neuron's activations by its polynomial should not hurt:

- replacing the {{num:paper.neurons_085}} clustered neurons by their 9-term polynomial gives loss {{num:rc.R10a}} (paper: 2.48 × 10⁻⁷, from 2.41 × 10⁻⁷);
- keeping *only* the cos/sin(w_k(a+b)) parts of the logits — everything else removed except each answer's average logit — gives {{num:rc.R19a}}, matching the paper's 5.54 × 10⁻⁸ (which the paper calls a 77 % *improvement*). The paper describes this ablation as restricting each *neuron* to those terms; restricting every neuron separately gives {{num:rc.R10b}} in our implementation, so we believe the paper's number was computed on the logits (see the [errata](../errata.html)).

The idealised algorithm is not just compatible with the network — it is *better* than the network, whose extra components are noise left over from memorisation.

## Swap the angle: counterfactual activation patching

[[dfn:activation-patching|Activation patching]] replaces part of an internal activation with its value from a *different* input. Our test {{cite:geiger2021}}: run the network on (a, b), but replace the part of a's embedding that lies in the **key-frequency subspace** — the 10 directions of $W_E$ that carry cos/sin(w_k a) for k ∈ 𝒦 — by the corresponding part of the embedding of a different number a′. If the network reads a *only* through those angles, it should now answer a′ + b. We measure the [[dfn:logit-difference]] $D = \Logits_{a'+b} - \Logits_{a+b}$ and report the fraction of the clean change recovered, $(D_{\text{patched}} - D(a,b)) / (D(a',b) - D(a,b))$: 1 = full switch, 0 = no effect.

{{fig:patching-result}}

Over 1,000 random triples (a, a′, b) on the paper's network:

| Patched part of a's embedding | Dimensions | Fraction recovered (95 % bootstrap CI) |
|---|---|---|
| key-frequency subspace | 10 | **{{num:paper.patch_key}}** [{{num:paper.patch_key_lo}}, {{num:paper.patch_key_hi}}] |
| random subspace of the same size (control) | 10 | {{num:paper.patch_random}} |
| everything *except* the key subspace (control) | 118 | {{num:paper.patch_complement}} |

Swapping a 10-dimensional part of the 128-dimensional embedding moves the answer almost completely; swapping the other 118 dimensions does essentially nothing. Try it on any input:

{{fig:patching-live}}

::: nuance
**What these tests do not show.** They confirm that the network's *output* depends on the key-frequency components the way the algorithm predicts. They do not pin down every internal step (e.g. the exact roles of the four heads), and a few of the paper's intervention numbers we could not reproduce exactly — listed on the [reproduce page](../reproduce.html).
:::
