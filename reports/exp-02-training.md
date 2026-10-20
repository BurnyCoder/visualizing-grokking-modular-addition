# Experiment 2: Training 22 runs of the paper's model, deterministically

**Question.** Can we retrain the paper's network from its released recipe, at its exact settings, for 11 seeds and
3 control groups? We want this on one laptop GPU, in a way that anyone can reproduce bit for bit.

**Status:** done. Phase `train_all_runs`, implemented in `src/grokviz/train.py`. It writes the outputs
`runs/<run_id>/{meta.json, checkpoints.safetensors, curves.npz, optim.safetensors}` and the log
`logs/20261021-000752_pipeline.log`. The run matrix and hypotheses were fixed beforehand in
[`docs/preregistration.md`](../docs/preregistration.md) (git tag `prereg-v1`).

## Setup

- **Recipe.** We use the paper's released code (`progress-measures-paper/transformers.py`) and checked every setting
  against it:
  - **Model and data.** The one-layer transformer, its initialisation, the 114-way float64 cross-entropy, and the 30 %
    training fraction.
  - **Optimiser.** Full-batch AdamW with lr 10⁻³, β = (0.9, 0.98) and weight decay λ = 1.
  - **Schedule.** `LambdaLR(min(step/10, 1))`, so the first update has learning rate 0.
- **Runs.** 22 runs: `wd1_s0…10`, `wd0_s0…4`, `rand_s0…2` and `split0_init100…102` (roles are given in the
  preregistration). Each runs for 40,000 steps.
- **Logged data:**
  - a checkpoint every 100 steps (401 per run);
  - train loss and accuracy at every step;
  - test loss and accuracy every 10 steps;
  - per-parameter weight, gradient and update norms every 10 steps;
  - Adam moments every 1,000 steps.
- **Hardware.** An RTX 5070 Laptop GPU (compute capability 12.0) with torch 2.14.1+cu130 and deterministic algorithms
  on (`CUBLAS_WORKSPACE_CONFIG=:4096:8`).

## Implementation choices and what we learned

1. **Sequential training was too slow, so all runs train at once.**
   - **The problem.** One run took ≈ 30 ms per step, so 22 runs × 40,000 steps would take ≈ 7.5 h. The model has
     only 226,816 parameters, so the GPU was mostly idle, waiting on kernel launches.
   - **The fix.** We stack the 22 parameter sets with `torch.func.stack_module_state` and run forward and backward for
     all of them in one call (`torch.func.vmap` over `functional_call`). This follows the PyTorch model-ensembling
     tutorial (<https://pytorch.org/tutorials/intermediate/ensembling.html>).
   - **Optimiser.** Each run needs its own weight decay (0 or 1), so AdamW's decay cannot be shared across runs. We set
     AdamW's own decay to 0 and apply the decoupled decay θ ← θ(1 − lr·λ_run) by hand before each Adam step. This is
     what `torch.optim.AdamW` does internally (its documented algorithm), and we checked it numerically (point 2).
   - **Cheaper forward pass.** The logits are only needed at the last position, so the forward pass slices the
     residual stream to that position before the MLP (`last_only=True`).
   - **Sparser test logging.** The test loss is logged every 10 steps rather than every step. This matches the
     paper's own logging interval.
   - **Result.** About 2,000 steps per 210 s for all 22 runs together, or ≈ 70 min in total: roughly 6× faster.
2. **A float-rounding surprise: early training amplifies tiny differences.**
   - **The observation.** Our first equivalence test failed. It compared batched against single-run training over a
     few hundred steps.
   - **Isolating the cause.** We trained the same single model twice: once with the full forward pass and once with
     `last_only`. The two are mathematically identical and differ only in floating-point summation order. The maximum
     absolute parameter difference was:

     | Steps | Max parameter difference |
     |---|---|
     | 5 | 3.1 × 10⁻⁷ |
     | 20 | 3.0 × 10⁻⁷ |
     | 50 | 1.8 × 10⁻² |
     | 200 | 0.24 |

     Adam divides by √v̂. Early in training, v̂ is tiny for some coordinates, so rounding-level differences are blown
     up to order-one changes within tens of steps. The batched code was therefore not wrong. Instead, *any* change in
     summation order produces a different (but equally valid) trajectory.
   - **The tests we use instead** (`tests/test_train.py`):
     - **Same update.** Batched and single-run updates agree after 4 steps at λ = 100, which makes the decay term
       dominate so that a wrong decay implementation would show.
     - **No cross-talk.** Perturbing one run in the batch leaves every other run bitwise unchanged.
     - **Exact resume.** A run that is resumed reproduces an uninterrupted run bitwise.
   - **What this means for results.** Seed-level results must be compared as *distributions*. A trajectory cannot be
     compared bitwise with another implementation (such as the paper's own code at the same seed).
3. **Two real bugs found by the tests.**
   - **Missing final checkpoint.** It was not saved when `num_steps < ckpt_every`. The final checkpoint is now always
     saved.
   - **Crash on short runs.** Runs shorter than the test interval crashed (`n_test = 0`). The counts are now
     ⌊(n − 1)/every⌋ + 1.
4. **An OS restart killed the first full launch.**
   - **What happened.** The first launch (`logs/20261018-180359_pipeline.log`) reached step 10,000 and then the
     machine restarted.
   - **Resumable state.** We added a resume point: stacked parameters, Adam and scheduler state, and the curves logged
     so far. It is saved atomically (write, then rename) every 2,000 steps to `runs/_resume/state.pt`, while
     checkpoints stream to memory-mapped files. Full-batch training draws no random numbers after initialisation, so
     no RNG state is needed.
   - **Relaunch.** We deleted the partial runs and relaunched from scratch three days later. The logged test
     accuracies of the second launch at steps 2,000, 4,000, …, 10,000 are identical to the first launch to every
     printed digit. For example, at step 10,000 `wd1_s0` = 0.782 and `wd1_s4` = 1.000 in both logs.
   - **Conclusion.** Training is deterministic across processes as well as within one.

## Results

RESULTS_PLACEHOLDER

## Limitations

- **Single code path.** All runs share one implementation and one GPU type. Bitwise reproducibility holds only on the
  same software and hardware stack; across stacks, expect seed-level differences (point 2).
- **Steps, not wall time.** Training length is fixed at 40,000 steps. The paper trained its mainline for 50,000 epochs,
  and its late-phase change near 43,000 steps cannot appear in our runs.
