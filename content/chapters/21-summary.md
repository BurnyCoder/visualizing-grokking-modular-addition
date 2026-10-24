## The objects, explained

Here are the three objects from chapter 0 again — now every piece has a name and a job.

::: object
**1. What the network computes** (chapters 5–6).

$$\underbrace{x^{(0)}_i = W_E t_i + W_{pos}[i]}_{\text{tokens} \to \text{vectors (embedding)}}\qquad \underbrace{A^j \approx \big(\sigma(C_j[a]{-}C_j[b]),\ 1{-}\sigma(\cdot),\ \approx 0\big)}_{\text{each head: a two-way choice between a and b}}$$

$$\underbrace{x^{(1)} = x^{(0)}_2 + \textstyle\sum_j W_O^jW_V^j \sum_i A^j_i x^{(0)}_i}_{\text{attention gathers a and b}}\qquad \underbrace{\mathrm{MLP}(a,b) = \ReLU(W_{in}x^{(1)} + b_{in})}_{\text{512 neurons; the ReLU multiplies}}$$

$$\underbrace{\Logits(a,b) \approx W_L\,\mathrm{MLP}(a,b)}_{\text{skip connection negligible; } W_L = W_UW_{out}}\qquad \text{answer} = \argmax_c\Logits(a,b)_c$$
:::

And as a picture with the live numbers (click a stage for its formula):

{{fig:forward-flow}}

::: object
**2. What the numbers mean** (chapters 9–16).

$$\begin{aligned}&\underbrace{a \mapsto \cos(w_ka),\ \sin(w_ka)}_{\text{step 1: } W_E \text{ (sparse in Fourier, ch. 13)}}\\ \Rightarrow\;&\underbrace{\cos(w_k(a{+}b)),\ \sin(w_k(a{+}b))}_{\text{step 2: products via attention + ReLU (ch. 14)}}\\ \Rightarrow\;&\underbrace{\sum_{k\in\mathcal K}\alpha_k\cos(w_k(a{+}b{-}c))}_{\text{steps 3–4: } W_L \text{ reads off (ch. 15)}}\end{aligned}$$

For the paper's network 𝒦 = {{num:paper.key_freqs}}; the cosine formula explains {{num:paper.logit_fve}} of the logits; removing one key frequency multiplies the loss by {{num:paper.key_ablation_ratio_range}}; swapping only the 10 key-frequency directions of a's embedding moves the answer {{num:paper.patch_key}} of the way to a′ + b.
:::

The core of the algorithm on one clock:

{{widget:add-angles}}

::: object
**3. How it got there** (chapters 7, 8, 17–20).

$$\begin{aligned}&\theta \leftarrow \theta(1-\gamma\lambda) - \gamma\,\text{AdamStep}(\nabla_\theta\mathcal L):\\ &\underbrace{\text{memorisation}}_{0\,-\,{\sim}1.5\text{k}} \;\to\; \underbrace{\text{circuit formation}}_{\text{restricted loss}\downarrow,\ \text{excluded loss}\uparrow} \;\to\; \underbrace{\text{cleanup}}_{\text{weights}\downarrow,\ \text{test loss}\downarrow\downarrow}\end{aligned}$$

The generalising circuit forms during the plateau, *before* the test loss drops; weight decay then removes the memorisation.

And the whole story in one view once more:

{{anim:hero-grokking}}
:::

## What we learned, in five sentences

1. A one-layer transformer trained on 30 % of all modular sums learns to represent numbers as angles at a few frequencies and adds them with the angle-addition identities ({{cite:nanda2023}}; ch. 12).
2. Four lines of evidence — Fourier sparsity, single-frequency neurons, a rank-10 read-out, and interventions including counterfactual patching — support this description of the trained network (ch. 13–16).
3. Knowing the mechanism lets us measure hidden progress: the restricted loss falls and the excluded loss rises long before the test loss drops (ch. 17).
4. Throughout circuit formation and over most of memorisation the gradients of the training and test losses partly oppose each other (negative cosine, {{num:paper.cos_tt_cf_range}} in circuit formation), turning positive only during cleanup, after step {{num:paper.cos_tt_last_neg}}; after grokking the network sits in a balance between the loss and weight decay (ch. 19).
5. In all 11 of our retrained seeds with weight decay, the Fourier signatures appear — sparse spectra, single-frequency neurons, cosine-shaped logits, and the patching test — each seed with its own key frequencies; the strictest preregistered ablation criterion (H4) held in only {{num:hyp.h4_successes}}, because our rule for naming the key frequencies sometimes includes a weak one or leaves out a little used one. Without weight decay, none of 5 networks grokked (ch. 20).

Point 5 at a glance: the three phases of every run whose phases were detected, on one time axis, and the paper's run with its authors' own boundaries:

{{fig:seed-phase-strips}}

## Limitations and nuances

- **A toy model.** One task, one layer, and a mechanism that took significant manual effort to find. The paper itself notes {{quote:nanda-specific}}
- **Hindsight.** The progress measures needed the final algorithm and do not predict *when* grokking happens (ch. 17).
- **Approximations.** “Most” neurons are single-frequency (~85 %); the attention mechanism is described by the paper as a sketch; several components (e.g. always-firing neurons, the late attention change around step 43,000) remain partly unexplained (ch. 14, 18).
- **Reproduction.** Our code reproduces most of the paper's numbers on its own network; a few intervention numbers could not be matched without knowing the exact procedure, and we list them openly ([Reproduce](../reproduce.html), [Errata](../errata.html)).
- **Scope of our replication.** We retrained only the paper's main setting; robustness to other data fractions, depths and moduli comes from the paper's appendix.

## Related work

- **Grokking itself** was discovered by Power et al. {{cite:power2022}}; Liu et al. built an “effective theory” with a phase diagram of when it happens {{cite:liu2022}}; Thilak et al. linked loss spikes in adaptive optimisers to it {{cite:thilak2022}}.
- **Analytic solutions:** Gromov derived the periodic weights of a two-layer MLP for modular arithmetic by hand {{cite:gromov2023}}.
- **Other algorithms:** Zhong et al. found that modified architectures learn a different circuit (“pizza”) instead of this “clock” {{cite:zhong2023}}; Chughtai et al. generalised the Fourier picture to arbitrary finite groups via representation theory {{cite:chughtai2023}}.
- **Hidden progress:** Barak et al. proposed hidden progress measures for learning parities {{cite:barak2022}}.
- **Explanations elsewhere:** Neel Nanda's original write-up {{cite:nandapost2022}}, the ARENA course chapter {{cite:arena152}}, Google PAIR's interactive explorable with a smaller MLP {{cite:pair2023}}, and the three videos {{cite:video_replication}} {{cite:video_paper}} {{cite:video_welch}}.

Zhong et al.'s test can be run on this site's networks too: re-arranging the right answer's logit by the sum and the difference of the inputs gives their distance irrelevance, which separates a Clock from a Pizza.

{{fig:clock-vs-pizza}}

Every term used on this site is in the [glossary](../glossary.html); every source in [Sources](../references.html).
