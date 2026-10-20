// main.typ — the paper: "Grokking, visualized" (compiled by the pipeline phase build_paper_pdf → reports/paper.pdf).
// Local explanation: prose and structure live here; every number is read from `sys.inputs` (filled by
// src/grokviz/paper.py from site/data/summary.json and reports/tables/hypotheses.json), figures are the website's
// own figures rendered to reports/figures/ by render_report_figures.
// Global context: compressed write-up of reports/exp-*.md; the website is the long-form, interactive version.
// Sources: Typst reference (https://typst.app/docs/reference/), sys.inputs (https://typst.app/docs/reference/foundations/sys/).

#let n(key) = sys.inputs.at(key, default: "⟨" + key + "⟩")  // bound number; ⟨key⟩ marks a missing one
#let fig(id, caption, width: 100%) = figure(image("/reports/figures/" + id + ".png", width: width), caption: caption)

#set document(title: "Grokking, visualized", author: "Libor Burian")
#set page(paper: "a4", margin: (x: 2.1cm, y: 2.2cm), numbering: "1")
#set text(size: 10pt, lang: "en")
#set par(justify: true)
#set heading(numbering: "1.1")
#show link: set text(fill: rgb("#1f5fa8"))
#show figure.caption: set text(size: 9pt)

#align(center)[
  #text(16pt, weight: "bold")[Grokking, visualized: an explorable, reproducible account of the modular-addition
  transformer, with a preregistered multi-seed replication]
  #v(6pt)
  Libor Burian \
  Independent researcher \
  #link("https://burnycoder.github.io/visualizing-grokking-modular-addition/")[burnycoder.github.io/visualizing-grokking-modular-addition] ·
  #link("https://github.com/BurnyCoder/visualizing-grokking-modular-addition")[code]
]

#block(inset: (x: 1.2em, y: 0.4em))[
  *Abstract.* Nanda et al. @nanda2023 reverse-engineered a one-layer transformer that groks addition modulo 113 and
  showed that it computes $(a + b) mod 113$ with a Fourier-multiplication algorithm. We (i) re-implement their
  analysis and that of the ARENA course @arena152, and check it against 27 numbers they publish for their own network:
  23 reproduce, and we explain the 4 that do not; (ii) add a counterfactual causal test — swapping the key-frequency
  subspace of the embedding of $a$ with that of $a'$ moves the output to $a' + b$, recovering #n("paper.patch_key")
  of the logit difference (95 % CI #n("paper.patch_key_lo")–#n("paper.patch_key_hi")) versus #n("paper.patch_complement")
  for the complementary subspace and #n("paper.patch_random") for a random subspace of equal dimension; (iii) retrain the
  paper's exact recipe for 11 seeds plus no-weight-decay, random-label and fixed-split controls, deterministically and
  bit-reproducibly, and test five preregistered hypotheses: #n("hyp.summary") (iv) publish an explorable website in
  which every step of the evidence is shown on the real weights, every term is defined before use, and every number
  is bound to the analysis output.
]

= Introduction

*Grokking* is delayed generalisation: a network first fits its training data while test performance stays at chance,
and much later, without any change to the training setup, suddenly generalises @power2022. Nanda et al. @nanda2023
studied the smallest well-understood instance — a one-layer transformer trained on $a + b mod 113$ — and showed that
the trained network implements a specific, human-readable algorithm, and that measures derived from this algorithm
("progress measures") move smoothly long before test loss falls.

Their evidence comes from one training run (the "mainline"), with appendix checks on four further seeds. Two things
are missing for a reader who wants to verify the story: (a) a way to _see_ every intermediate object on the real
network, and (b) a pre-specified, statistically controlled test of the main claims across independently trained
networks. This work provides both.

*Contributions.*
+ An analysis pipeline whose outputs match 23 of 27 published numbers on the paper's own checkpoints
  (@sec-validation), with documented definitional findings (e.g., the reported "test loss" $2.41 times 10^(-7)$ is the
  loss over all 12,769 pairs).
+ A counterfactual subspace-patching test of the claim that the embedding encodes $a$ only through
  $cos(w_k a), sin(w_k a)$ at the key frequencies (@sec-patching).
+ A preregistered replication over 11 seeds with controls, trained deterministically (@sec-seeds).
+ An explorable website with 22 chapters and a dashboard of linked figures, built so that the text cannot drift from
  the data (@sec-site).

= Background <sec-background>

*Task.* All $113^2$ = 12,769 pairs $(a, b)$ are written as tokens `a b =`; the label is $c^* = (a + b) mod 113$.
A fixed random #n("task.n_train") pairs (30 %) are used for training, the remaining #n("task.n_test") for testing.

*Model* @nanda2023[Sec. 3]. One transformer layer with $d_"model" = 128$, four attention heads of size 32, an MLP with
512 ReLU neurons, no LayerNorm, untied embedding $W_E$ and unembedding $W_U$; #n("model.n_params") parameters.
It is trained full-batch with AdamW @loshchilov2019 (learning rate $10^(-3)$, weight decay $lambda = 1$,
$beta = (0.9, 0.98)$) for 40,000 steps.

*The algorithm* @nanda2023[Sec. 4]. For a few *key frequencies* $k$ (with $w_k = 2 pi k \/ 113$), the embedding maps
$a$ to $cos(w_k a), sin(w_k a)$; attention and MLP compute $cos(w_k (a+b))$ and $sin(w_k (a+b))$ using
$cos(x + y) = cos x cos y - sin x sin y$; the unembedding reads out
$ "logit"(c) approx sum_(k in K) alpha_k cos(w_k (a + b - c)), $
which is maximal exactly at $c = c^*$. All frequencies' cosines peak together there and interfere destructively
elsewhere.

*Progress measures* @nanda2023[Sec. 5]. The *restricted loss* keeps only the key-frequency components of the logits
(it measures how well the algorithm alone would do); the *excluded loss* removes them and evaluates on training data
(it measures how much the network relies on memorisation instead). Training splits into *memorisation*, *circuit
formation* and *cleanup* phases by these measures.

= Methods <sec-methods>

== The paper's run and our analysis code <sec-validation-methods>

We use the authors' published mainline run @paperckpt (500 checkpoints, pinned revision and SHA-256). Our model
re-implements the paper's parameter layout so the checkpoints load unmodified. Recomputing the training loss from the
checkpoints shows that the checkpoint labelled "epoch $e$" holds the parameters after $e + 1$ optimiser updates (the
learning-rate warm-up starts at zero) and confirms that we rebuilt the exact training split. Analysis functions are
re-implemented from the paper's code @paperrepo and ARENA @arena152, each with a source link; exact definitions are in
`docs/operationalizations.md` of the repository.

== Counterfactual subspace patching <sec-patching-methods>

Let $Q in RR^(128 times 2|K|)$ be an orthonormal basis of the embedding directions $W_E F_(2k-1), W_E F_(2k)$
($F$: the Fourier basis over $ZZ_113$). For a triple $(a, a', b)$ we replace the component of $a$'s embedding in
span($Q$) by that of $a'$ and measure the recovered fraction of the logit difference
$D = L[a'+b] - L[a+b]$: $(D_"patched" - D(a)) \/ (D(a') - D(a))$ — 1 means the network now answers $a' + b$.
Controls: the complementary subspace, and a random $2|K|$-dimensional subspace. 1,000 random triples; bootstrap
95 % intervals. This is an interchange intervention in the sense of @geiger2021.

== Training and preregistration <sec-train-methods>

We trained 22 runs with the paper's recipe: `wd1` (11 seeds; seed 0 uses the paper's split), `wd0` (5 seeds,
$lambda = 0$), `rand` (3 seeds, random label table) and `split0` (3 new initialisations on the paper's split). All runs
train together with vectorised forward/backward passes; batched and single-run updates agree, runs are bitwise
independent of their batch partners, and training is bitwise reproducible across processes and resumption. Early
full-batch Adam training amplifies rounding differences (a mathematically identical forward pass with a different
summation order diverges from $3 times 10^(-7)$ after 20 steps to $0.24$ after 200 steps in max parameter difference),
so seed-level results must be compared as distributions, not trajectories.

Before training, we fixed five hypotheses, their metrics and falsification rules (git tag `prereg-v1`):
*H1* restricted loss falls before test loss; *H2* excluded loss rises ≥ 10× during circuit formation; *H3* the final
Gini sparsity @hurley2009 of the $W_E$ and $W_L$ Fourier spectra exceeds both its initial value + 0.3 and the 99th
percentile under 100 random orthonormal bases; *H4* each single key-frequency ablation raises loss ≥ 100×, each
non-key ablation ≤ 2×, removing all non-key frequencies ≤ 1.5×; *H5* weight decay is necessary (all `wd1` and no
`wd0` runs reach 99 % test accuracy). H1–H4 use one-sided exact sign tests over the 11 `wd1` seeds, H5 a one-sided
Fisher exact test; the five p-values are Holm-corrected @holm1979 at $alpha = 0.05$. Eleven seeds is the smallest
number at which one dissenting seed still passes the strictest Holm threshold ($P(X >= 10 | 11, 1\/2) = 0.0059 < 0.01$).

== Website <sec-site>

The site is generated from Markdown with macros: `[[dfn:…]]` defines a term, `[[term:…]]` uses one, and the build fails
if any term or symbol is used before its definition in reading order, if a quote does not match its PDF page, or if a
number is not bound to `summary.json`. Every figure is one JavaScript module shared by the chapters and the dashboard;
per-input views run a JavaScript copy of the model on float16 weights, tested against PyTorch. The figures in this paper
are screenshots of the same modules.

= Results <sec-results>

== Reproducing the published numbers <sec-validation>

Table 1 lists all 27 checks. 23 match within stated tolerances. Two definitional findings: the paper's
"test loss" $2.41 times 10^(-7)$ equals the loss over _all_ pairs (test-only: #n("paper.loss40k_test")), and the
"$W_L$ residual < 0.55 %" is a squared-norm ratio (#n("paper.wl_resid_sq"); the norm ratio is #n("paper.wl_resid_norm")).
The four non-reproduced rows are projections and ablations whose exact procedure the paper does not specify; for
R10b, the paper's value equals our restricted-_logit_ loss rather than a per-neuron restriction.

#let rrows = json(bytes(sys.inputs.at("rchecks.json", default: "[]")))
#figure(
  text(size: 7.5pt, table(
    columns: (auto, 1fr, auto, auto, auto), stroke: 0.4pt, inset: 3pt,
    table.header([*ID*], [*Quantity*], [*Paper / ARENA*], [*Ours*], [*Match*]),
    ..rrows.flatten(),
  )),
  caption: [Reproduction checks on the paper's network at epoch 40k (step 40,001). Sources: paper pages and ARENA tests
  as listed in `reports/tables/exp01_rchecks.md`.],
  kind: table,
)

== The mechanism on the paper's network <sec-mechanism>

The embedding is sparse in the Fourier basis: six frequencies carry visible weight (#n("paper.we_top6")), five of which
recur everywhere in the network and are the key frequencies #n("paper.key_freqs") (@fig-we). #n("paper.neurons_085") of
512 MLP neurons have more than 85 % of their variance explained by a single frequency, forming clusters of
#n("paper.cluster_sizes") neurons. The logits are fit by $sum_k alpha_k cos(w_k(a+b-c))$ with
#n("paper.logit_fve") variance explained; replacing the logits by this fit gives loss #n("paper.logit_fit_loss").
Ablating any single key frequency from the logits multiplies the loss by #n("paper.key_ablation_ratio_range"); the
largest effect of a non-key frequency is #n("paper.nonkey_ablation_max_ratio") (@fig-ablation).

#fig("we-fourier-bars", [Norm of each Fourier component of the embedding $W_E$ at step 40,001 (paper run).]) <fig-we>
#fig("freq-ablation-bars", [Loss after removing all components of one frequency from the logits (log scale), paper
run at step 40,001. Red bars: the key frequencies; dotted line: the unablated loss.]) <fig-ablation>

== The causal test <sec-patching>

Swapping the key-frequency subspace (dimension 10 of 128) recovers #n("paper.patch_key") of the logit difference
(95 % CI #n("paper.patch_key_lo")–#n("paper.patch_key_hi")); swapping everything _except_ that subspace recovers
#n("paper.patch_complement"), and a random 10-dimensional subspace #n("paper.patch_random") — close to the
$10\/128 approx 0.078$ expected if the random subspace captured a proportional share of the relevant direction
(@fig-patch). The number $a$ therefore enters the computation essentially only through its key-frequency components.

#fig("patching-result", [Counterfactual subspace patching on the paper run: recovered fraction of the logit difference
for the key-frequency subspace and the two controls (1,000 random triples; bootstrap 95 % intervals).], width: 80%) <fig-patch>

== Training dynamics <sec-dynamics>

On the paper's run our objective phase rules put the end of memorisation at step #n("paper.t1"), the end of circuit
formation at #n("paper.t2"), and the end of cleanup at #n("paper.t3"); test accuracy reaches 99 % at
#n("paper.t_grok") (@fig-progress). The excluded loss rises from #n("paper.excluded_t1") at the end of memorisation to
#n("paper.excluded_t2") at the end of circuit formation, while the restricted loss has already fallen far below the
test loss. Exploratory gradient analysis (exact full-batch gradients at every checkpoint): the cosine between the
training-set and test-set gradients is #n("paper.cos_tt_start") at initialisation and becomes negative during
memorisation and circuit formation (#n("paper.cos_tt_mid") at step #n("paper.mid_step")); after grokking the
gradient is almost exactly anti-parallel to the weights (cosine #n("paper.cos_gw_40k") at step 40,001), the signature
of an equilibrium between the loss gradient and weight decay.

#fig("progress-measures", [Train, test, restricted and excluded loss over training of the paper run (log scale);
shading marks the three phases from our objective rules.]) <fig-progress>

== Replication across seeds and controls <sec-seeds>

#let hyp = json(bytes(sys.inputs.at("hypotheses.json", default: "{}")))
#figure(
  table(
    columns: (auto, 1fr, auto, auto), stroke: 0.4pt, inset: 4pt,
    table.header([*H*], [*Result*], [*Holm p*], [*Supported*]),
    ..hyp.pairs().map(((h, d)) => (
      h,
      if "successes" in d { str(d.successes) + " / " + str(d.n) + " seeds" } else {
        str(d.wd1_grokked) + "/" + str(d.wd1_n) + " with vs " + str(d.wd0_grokked) + "/" + str(d.wd0_n) + " without weight decay grokked" },
      str(calc.round(d.p_holm, digits: 6)),
      if d.supported { "yes" } else { "no" },
    )).flatten(),
  ),
  caption: [Preregistered hypotheses over the 11 `wd1` seeds (H5: `wd1` vs `wd0`).],
  kind: table,
) <tab-hyp>

Across the 11 weight-decay seeds, memorisation ends at step #n("seeds.t1_ci"), circuit formation at
#n("seeds.t2_ci"), and test accuracy reaches 99 % at #n("seeds.tgrok_ci") (mean, 95 % t-interval). Each run uses
#n("seeds.nkey_ci") key frequencies; single-frequency neurons make up #n("seeds.neurons085_ci") of the MLP, the cosine
fit explains #n("seeds.logitfve_ci") of the logit variance, and the patching test recovers #n("seeds.patch_key_ci")
for the key subspace versus #n("seeds.patch_random_ci") for a random one.

*Controls.* No weight decay: #n("controls.wd0"). Random labels: #n("controls.rand"). Paper split with new
initialisations: #n("controls.split0").

#fig("seed-test-loss", [Test loss of every run over training (log scale); seeds as thin lines.]) <fig-seeds>

= Related work

Power et al. @power2022 introduced grokking on algorithmic tasks. Liu et al. @liu2022 explain it through
representation learning dynamics; Thilak et al. @thilak2022 link it to adaptive-optimiser instabilities; Barak et al.
@barak2022 show "hidden progress" in parity learning. Nanda et al. @nanda2023 give the mechanistic account we test;
Chughtai et al. @chughtai2023 generalise it to group operations, Zhong et al. @zhong2023 show that architecturally
similar networks may instead learn a different ("pizza") algorithm, and Gromov @gromov2023 gives an analytic
construction for MLPs. Interactive explanations exist as the PAIR explorable @pair2023 and a Welch Labs video
@welch2025; ARENA @arena152 provides exercises. Our contribution is a verified, preregistered replication of the
mechanistic claims together with an explorable presentation of the full evidence.

= Discussion and limitations

*What the evidence supports.* On the paper's network, the reproduction and the patching test agree with the
Fourier-multiplication account to within a few percent; the patching result is the most direct causal statement we
make: $a$ is read through a 10-dimensional subspace of a 128-dimensional embedding. The multi-seed results are
summarised in @tab-hyp.

*Limitations.* (i) All runs share the paper's architecture, data fraction, modulus and optimiser; we do not test other
settings, where other algorithms are known to arise @zhong2023. (ii) Four of the paper's numbers could not be
reproduced because procedures are under-specified; we do not use them as evidence. (iii) Phase boundaries come from
rules we calibrated on the paper's run before seeing our runs; other reasonable rules shift them. (iv) Our runs last
40,000 steps; the paper's late change near step 43,000 is outside this window. (v) Training is bitwise reproducible
only on the same software and hardware stack.

= Conclusion

The modular-addition transformer is, as claimed, a Fourier-multiplication machine, and the claim survives a
counterfactual test on the original network. The website makes every step of this evidence inspectable on the real
weights; the pipeline, data and preregistration make it reproducible.

#bibliography("refs.bib", style: "ieee")

#pagebreak()
#heading(numbering: none)[Appendix A: Reproducibility]

Code, data formats and the full pipeline: `github.com/BurnyCoder/visualizing-grokking-modular-addition`
(`uv run grokviz --phases all`). Runs: Hugging Face `BurnyCoder/grokking-modular-addition-runs`; site data:
`BurnyCoder/grokking-modular-addition-site-data`. Hardware: one RTX 5070 Laptop GPU; training of all 22 runs takes
≈ 70 minutes, per-checkpoint analysis ≈ 50 minutes. Per-experiment reports with learnings are in `reports/`.
