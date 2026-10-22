We now know *what* the trained network computes. The paper's main contribution is to use that knowledge to look *back* at training and see progress that the loss curves hide.

## Progress measures

Barak et al. (2022) proposed looking for hidden progress {{cite:barak2022}}; Nanda et al. define a [[dfn:progress-measure]] as one of the {{cite:nanda2023}}

{{quote:nanda-hidden-progress}}

Knowing the final algorithm suggests two such measures. Both act on the logits of all 12,769 inputs, using the run's key frequencies 𝒦 (for the paper's run, those of the step-40,001 model):

- **[[dfn:restricted-loss|Restricted loss]]:** keep only the cos/sin(w_k(a+b)) directions of the logit grids (10 directions for 5 key frequencies) plus each answer's average logit, and measure the loss on all inputs. It asks: *how good is the network's Fourier-multiplication part on its own?*
- **[[dfn:excluded-loss|Excluded loss]]:** *remove* those directions and measure the loss on the **training** inputs. It asks: *how much of the training performance would survive without the Fourier circuit?* The idea, in the paper's words: {{quote:nanda-excluded-idea}}

Both definitions follow the paper's code exactly {{cite:paperrepo}}, and ARENA's reference implementations {{cite:arena152code}} reproduce to every digit we checked (see [Reproduce](../reproduce.html)).

{{fig:progress-measures}}

::: example
**Reading the plot** (paper's run): at step 5,001 the test loss is still ~21 — worse than guessing — but the restricted loss is already down to {{num:paper.restricted_5k}}: the Fourier part alone would already do far better than the whole network on unseen inputs. Meanwhile the excluded loss climbs from {{num:paper.excluded_t1}} at the end of memorisation to {{num:paper.excluded_t2}} at step {{num:paper.t2}}: by then, without its Fourier circuit, the network could no longer even fit its *training* data — the memorised answers have been replaced.
:::

## Three phases

From these curves the paper splits training into three phases {{cite:nanda2023}} — it {{quote:nanda-three-phases}}

The paper drew the boundaries by eye (1.4k, 9.4k, 14k). To compare runs fairly we defined objective rules in advance, using only the train/test loss and the weight norm (so they cannot be circular with respect to the progress measures): see [docs/operationalizations.md](https://github.com/BurnyCoder/visualizing-grokking-modular-addition/blob/main/docs/operationalizations.md). On the paper's run they give {{num:paper.t1}}, {{num:paper.t2}} and {{num:paper.t3}}.

1. **[[dfn:phase-memorization|Memorisation]]** (0 – {{num:paper.t1}}): train loss and excluded loss fall together, test and restricted loss stay high: the network memorises without using the key frequencies.
2. **[[dfn:phase-circuit-formation|Circuit formation]]** ({{num:paper.t1}} – {{num:paper.t2}}): excluded loss rises, restricted loss falls, the weights shrink, while train and test loss barely move. {{quote:nanda-circuit-before}}
3. **[[dfn:phase-cleanup|Cleanup]]** ({{num:paper.t2}} – {{num:paper.t3}}): the test loss suddenly drops, the weights shrink fast and the Fourier spectra become sparse: {{quote:nanda-cleanup}}

The test/restricted-loss ratio shows the “lag” between having the circuit and using it cleanly, which ARENA highlights {{cite:arena152}}:

{{fig:trig-loss-ratio}}

## Sparsity and weight norm

Two more measures track the same story. The [[term:gini]] coefficient of the Fourier spectra of $W_E$ and $W_L$ (chapter 11) measures how few frequencies they use:

{{fig:gini-curves}}

and the [[dfn:sum-squared-weights|sum of squared weights]] — the quantity weight decay pushes down — rises during memorisation to a peak of {{num:paper.ssw_peak}} (step {{num:paper.ssw_peak_step}}), falls steadily during circuit formation, and drops steeply during cleanup to {{num:paper.ssw_t3}} at step {{num:paper.t3}}:

{{fig:ssw-curves}}

Per frequency, the excluded loss shows each key frequency taking over part of the job:

{{fig:excluded-per-freq}}

Computing the restricted loss separately on the training and the test questions shows why it measures hidden progress: the two curves coincide at every step, so the part of the logits that uses only the key frequencies generalises from the start, while the ordinary test loss is still dominated by memorised answers.

{{fig:restricted-split-curves}}

::: nuance
**What the progress measures can and cannot do.** They reveal that grokking is gradual and identify the phases *after the fact*. They required knowing the final algorithm, and they do not predict *when* the jump will happen — the paper is explicit: {{quote:nanda-criticality}}
:::
