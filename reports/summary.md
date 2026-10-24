# Summary of all experiments

This is the one-page overview. Each experiment has its own report, with methods, numbers and learnings:

- [exp-01: paper validation](exp-01-paper-validation.md)
- [exp-02: training](exp-02-training.md)
- [exp-03: hypotheses](exp-03-hypotheses.md)
- [exp-04: training dynamics (exploratory)](exp-04-training-dynamics.md)

The paper (`paper.pdf`) compresses all of them. The website explains everything from scratch, in 22 chapters.

## Key terms in plain words

For the full glossary, with sources, see the website's glossary page (`content/glossary.yaml`).

| Term | Plain meaning |
|---|---|
| Modular addition | "Clock" arithmetic: (a + b) mod 113 is the remainder after dividing a + b by 113, so 100 + 50 → 37. |
| Grokking | A network first memorises its training examples and fails on new ones. Much later, without any change to training, it suddenly gets the new ones right too. |
| Transformer (one layer) | The network studied here. It reads the three tokens `a b =`, mixes information between them (attention), transforms it (an MLP of 512 "neurons"), and outputs one score per possible answer. |
| Logits / softmax / loss | Scores for each answer, turned into probabilities. The loss is how surprised the network is by the right answer; 0 is perfect. |
| Weight decay | Shrinking all weights a little at every training step. It favours simple solutions and is what eventually removes the memorised one. |
| Full-batch AdamW | Every training step uses all 3,830 training examples, with the Adam optimiser plus weight decay. |
| Seed | The number that fixes the random starting weights (and here also the train/test split). Different seeds give independent runs. |
| Fourier basis / frequency k | Describing a pattern over 0…112 as a sum of waves cos(2πk·n/113) and sin(2πk·n/113). The network turns each number into positions on a few such "clocks". |
| Key frequencies | The few waves (about 5) that a trained network actually uses. For the paper's network they are 14, 35, 41, 42 and 52. |
| Fourier multiplication algorithm | The network's method: angle(a) + angle(b) = angle(a + b) via the cos/sin addition formulas, then pick the answer c whose angle matches. |
| Restricted / excluded loss | The loss using *only* the key-frequency part of the output (restricted), or *everything except* it (excluded). The first tracks the general algorithm, the second the memorisation. |
| Gini (sparsity) | 0 when weight is spread evenly over all frequencies, close to 1 when a few frequencies hold all of it. |
| FVE (fraction of variance explained) | How much of a pattern a simple formula reproduces: 1 is all of it, 0 is none. |
| Ablation | Deleting one part of the computation and measuring how much worse the network gets. |
| Activation patching (counterfactual) | Replacing part of the network's internal state with the state it has for a different input, and checking whether the answer changes as predicted. |
| Phases | Memorisation (training loss falls), circuit formation (the algorithm is built while test loss stays high) and cleanup (memorisation is removed; test loss collapses). |
| Preregistration | Writing the hypotheses and pass/fail rules down before training, so they cannot be tuned to the results. |
| Sign test, Holm, p-value, CI | Statistics over seeds. The sign test asks how unlikely "k of 11 seeds agree" would be by coin flip. Holm corrects for testing 5 hypotheses. A p-value is that unlikeliness. A CI is a 95 % range for an average. |

## Networks

| Network | Runs | Architecture | Training (algorithm, objective) | Data | Curriculum | Final train / test accuracy | Domain |
|---|---|---|---|---|---|---|---|
| Paper mainline (Nanda et al. 2023) | 1 (downloaded) | 1-layer transformer: d_model 128, 4 heads × 32, MLP 512 ReLU, no LayerNorm; 226,816 parameters | Full-batch AdamW (lr 10⁻³, λ = 1, β = (0.9, 0.98)), 10-step warm-up; cross-entropy over 114 tokens in float64 (none of its 55,000 logged losses is float32-representable, although the released `helpers.py` casts to float32); 50,000 epochs (the paper's text says 40,000) | 30 % of all 12,769 pairs (3,830), fixed split (`random.seed(0)`) | none (the same full batch every step) | 100 % / 100 % (test loss 3.0×10⁻⁷ at step 40,001); key frequencies 14, 35, 41, 42, 52 | a + b mod 113 |
| `wd1_s0…10` | 11 | same | same (float64 cross-entropy, as in the paper's notebook and ARENA); 40,000 steps | 30 %, split seed = s (s = 0 is the paper's split) | none | 100 % / 100 % in all 11; grokked at steps 4,600–11,000; 3–6 key frequencies each | a + b mod 113 |
| `wd0_s0…4` | 5 | same | same, but λ = 0 | 30 %, split seed s | none | 100 % / 1.9–21.5 % (no grokking; no sparse Fourier structure) | a + b mod 113 |
| `rand_s0…2` | 3 | same | same, λ = 1 | 30 %, labels replaced by a random table | none | 100 % / 0.8–0.9 % (chance = 1/113 ≈ 0.9 %) | random function (nothing to generalise) |
| `split0_init100…102` | 3 | same | same, λ = 1 | the paper's split, new initial weights | none | 100 % / 100 % in all 3; grokked at steps 4,300–9,900; key frequencies differ from the paper's | a + b mod 113 |

## Experiments at a glance

| # | Question | Result | Confidence |
|---|---|---|---|
| 1 | Does our analysis code reproduce the paper's and ARENA's published numbers on the paper's own network? | 26 of 30 match, two of them (R4, R12) only at the edge of the published bound. The 4 misses are under-specified ablations, documented with the variants we tried. Two definitional findings: the "test loss" 2.41×10⁻⁷ is over all pairs, and the W_L residual matches "< 0.55 %" only as a squared-norm ratio (0.552 % at step 40,001, within our < 0.6 % tolerance; 0.520 % at step 49,901), not as the norm ratio the text names (7.4 %). | High: exact checkpoints, exact split, numbers to 3–4 digits. |
| 1b | Causal test on the paper's network: is a read only through its key-frequency subspace? | Swapping that 10-dimensional subspace (of 128) moves the answer to a′ + b: 0.997 of the logit difference, 95 % CI 0.994–1.000. Complement: 0.002. Random 10-dimensional subspace: 0.062. | High (1,000 triples, bootstrap CI); one network. |
| 2 | Can the paper's recipe be retrained for 22 runs, reproducibly, on one laptop GPU? | Yes: 73 min with vmap-batched training, bitwise reproducible within a process and across resume (tests); a relaunch in a new process reproduced every logged test accuracy to the printed 3 decimals. Early full-batch Adam amplifies rounding noise, so runs are compared as distributions. | High (tested). |
| 3 | Do the paper's mechanistic claims (H1–H5) hold across 11 seeds, against controls? | H1, H2, H3 supported (11/11 seeds each, Holm p = 0.002); H5 supported (11/11 vs 0/5 grokked within 40,000 steps, Holm p = 0.001); H4 not supported (6/11, Holm p = 0.5): in 2 seeds the weight-norm rule names a key frequency whose removal costs less than 100× (5.9×, 70×), and in 3 seeds the frequencies it leaves out jointly cost 1.6–2.8× (limit 1.5×). Patching recovers 0.997 [0.995, 0.998] of the logit difference in the seeds. | High for the verdicts (preregistered, 11 seeds, Holm-corrected); the explanation of H4's failure is post hoc. |
| 3b | Exploratory: what selects the key frequencies? | The run started from the paper's initial weights (`wd1_s0`, equal up to float32 rounding: max \|Δ\| = 3.0 × 10⁻⁸) diverges after ~52 steps (weight cosine 0.74 at the end) yet ends with the paper's key frequencies; 3 new initialisations on the same split choose different ones. 2 of 14 grokked runs show repeated late loss spikes. | Low–moderate: one same-init pair; not preregistered. |
| 4 | Exploratory: which forces act during the phases (exact gradients; Adam's actual step from saved moments)? | During circuit formation the train-loss gradient opposes the test-loss gradient (negative cosine at 97.9 % of checkpoints between t1 and t2). At the end, Adam's loss step and the weight-decay step are nearly equal and opposite (ratio 0.992, cosine −0.991). Weight decay outweighs the loss step most (ratio 0.906) within 1,000 steps of grokking in 10 of 11 seeds. | Moderate for the end-of-training balance (11 seeds); low for the grokking dip (post hoc, 1,000-step resolution). |

## What we learned about doing this (cross-cutting)

- **Validate before extending.** Checking 27 published numbers and claims before training (3 trajectory checks were
  added later, 30 in all) exposed two definitions the paper's text leaves implicit. Without that check, every figure on the site would have used a different loss than the paper.
- **Bind numbers, don't type them.** The measured numbers on the website and in the paper are bound to one
  `summary.json`, and the report tables are generated from the analysis outputs. Numbers in the report prose are typed
  and checked by hand against those files. When the R7 reference value was corrected, every bound copy updated together.
- **Expect chaos in early training.** Mathematically equal computations diverge within ~50 steps of full-batch Adam.
  Equivalence tests must therefore use short horizons or bitwise non-interaction checks, never long-run trajectories.
- **Preregister before training.** The hypotheses, thresholds and seed count (from a power analysis) were tagged
  before the first full run. The tests in exp-03 are exactly those.
