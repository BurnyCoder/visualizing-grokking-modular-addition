# Experiment 4: What drives the phases? Gradients, Adam and weight decay (exploratory)

**Question.** The paper attributes circuit formation and cleanup to weight decay (Sec. 5.2–5.3). ARENA's bonus chapter
suggests a mechanism: memorising gradients from individual examples cancel, while pattern gradients add up. Can we see
these forces directly, on the paper's run and across our 11 weight-decay seeds?

**Status:** done; exploratory, not preregistered. The phase is `compute_gradient_dynamics` (`src/grokviz/gradients.py`),
writing `runs/<id>/analysis/gradients.npz`. The numbers below come from `site/data/summary.json` (keys `paper.cos_*` and
`seeds.*`, computed in `src/grokviz/summary.py`). Website: chapter 19 and dashboard group F.

## Method

- **Exact gradients.**
  - **Why they are exact.** Training is full-batch, so the gradient the optimiser saw at a checkpoint can be recomputed
    exactly from that checkpoint's weights.
  - **What we compute.** At every checkpoint (every 100 steps), the gradient of the 114-way float64 training loss,
    g_train, and of the test loss, g_test.
  - **What we record:**
    - cos(g_train, θ);
    - cos(g_train, g_test);
    - per-parameter gradient norms;
    - the Fourier spectrum of the embedding gradient.
- **Adam's actual step (our runs only).**
  - **Saved state.** AdamW's moments m and v were saved every 1,000 steps.
  - **The direction.** From them we rebuild Adam's direction d = m̂/(√v̂ + ε), with the bias corrections of
    `torch.optim.AdamW`. The update is −η·d − η·λ·θ.
  - **What we record:**
    - cos(d, g_train);
    - cos(d, θ);
    - ‖d‖ / (λ‖θ‖), the size of the loss step relative to the decay step. Both steps carry the same learning rate.
- **Why not the raw-gradient ratio.** We first plotted ‖g‖ / (λ‖θ‖). After memorisation it is about 10⁻⁸, but Adam
  divides each coordinate by its running RMS, so a tiny but consistent gradient still produces steps of size ~η. The
  raw ratio therefore says nothing about which force dominates, and we removed that panel. Measuring d instead requires
  the saved moments, which the paper's run does not have.

## Results

Paper run (gradients only) and the 11 `wd1` seeds; means with 95 % percentile-bootstrap intervals over seeds.

1. **Memorising steps hurt generalisation.**
   - **Paper run:** cos(g_train, g_test) is 0.80 at initialisation, −0.27 midway through circuit formation (step 5,401),
     and 0.33 at step 40,001.
   - **Across seeds:** midway between t1 and t2 it is −0.251 [−0.305, −0.199], and it is negative at 97.9 %
     [96.0, 99.6] of the checkpoints between t1 and t2.
   - **What this means:** during the whole plateau, a step that lowers the training loss raises the test loss. This
     agrees with ARENA's picture of memorising gradients that work against the general solution.
2. **Equilibrium with weight decay at the end.**
   - **Adam's direction:** at step 40,000, ‖d‖/(λ‖θ‖) = 0.992 [0.972, 1.004] and cos(d, θ) = −0.991 [−1.000, −0.976].
   - **Interpretation:** Adam's loss step and the decay step are nearly equal and opposite, a steady state in which the
     loss keeps pushing the logits up and decay pulls the weights back.
   - **Ten seeds are within 1.2 % of balance.** The exception is `wd1_s4` (ratio 0.90, cos −0.92), the seed with
     repeated late loss spikes (exp-03).
3. **The raw gradient alone is misleading.**
   - **The spread:** at the last checkpoint, cos(g_train, θ) ranges from −0.996 to −0.157 across seeds (mean −0.814).
   - **Two seeds in detail:** in `wd1_s6` and `wd1_s9`, cos(d, g) ≈ 0.01 and 0.21. The single-step gradient is noisy,
     but Adam's momentum-averaged, per-coordinate-normalised direction is still aligned with −θ (cos −0.99 and −1.00).
   - **Takeaway:** conclusions about "which force wins" need the optimiser's actual step, not the raw gradient.
4. **Weight decay wins briefly, exactly at grokking.**
   - **The dip:** the ratio ‖d‖/(λ‖θ‖) is slightly below 1 during memorisation (≈ 0.95–0.98 at steps 2k–4k). Its
     minimum, 0.906 [0.897, 0.914], falls within 1,000 steps of t_grok in 10 of 11 seeds. `wd1_s4`'s minimum is
     during its late spikes.
   - **Interpretation:** while the network groks, decay outweighs the loss step by ~10 % and the weights shrink. That
     is the paper's cleanup phase, driven by weight decay, here measured on the optimiser's actual steps.
   - **Resolution:** the moments were saved every 1,000 steps, so timing is resolved only to that grid.
5. **Update sizes** (logged every 10 steps; inspected visually for `wd1_s0` only). Every parameter's update size peaks
   together around grokking (step ≈ 10,000) and then decays slowly (dashboard panel "How much each parameter actually
   moves").

## Learnings

- **Save the optimiser state.**
  - **The constraint:** with Adam, the gradient is not the step. The paper's run cannot be analysed this way because
    its optimiser state was not saved.
  - **What saving gave us:** we saved moments every 1,000 steps (72 MB per run), which made points 2–4 measurable.
  - **What it costs:** a finer save interval would sharpen point 4, at a larger storage cost.
- **Drop panels that cannot be read correctly.**
  - **The panel:** the first version of the site had a "‖∇L‖ ÷ λ‖θ‖" panel. After memorisation its value of ~10⁻⁸
    suggested that weight decay dominates by eight orders of magnitude.
  - **Why it was wrong:** Adam's normalisation makes that comparison meaningless.
  - **What replaced it:** we removed it and wrote down why (chapter 19), before having the Adam-based replacement.
- **Bounded quantities need bounded intervals.** t-intervals for cosines and shares spilled past −1 and 100 %, so
  these use the percentile bootstrap. The preregistered secondary statistics keep the t-intervals that were registered.
- **Claims are post hoc.** Points 1–5 were not preregistered. Point 4 in particular is a pattern noticed in the data,
  reported with its exception and the time resolution, and should be tested on new seeds before being treated as
  established.
