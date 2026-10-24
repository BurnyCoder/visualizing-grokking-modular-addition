# Experiment 2: Training 22 runs of the paper's model, deterministically

**Question.** Can we retrain the paper's network from its released recipe, at its exact settings, for 11 seeds and
3 control groups? We want this on one laptop GPU, deterministically, so that a rerun on the same software and
hardware gives the same runs.

**Status:** done. Phase `train_all_runs`, implemented in `src/grokviz/train.py`. It writes the outputs
`runs/<run_id>/{meta.json, checkpoints.safetensors, curves.npz, optim.safetensors}` and the log
`logs/20261021-000752_pipeline.log`. The run matrix and hypotheses were fixed beforehand in
[`docs/preregistration.md`](../docs/preregistration.md) (git tag `prereg-v1`).

## Setup

- **Recipe.** We use the paper's settings (Sec. 3) and its released code (`progress-measures-paper/transformers.py`),
  and checked every setting against them:
  - **Model and data.** The one-layer transformer, its initialisation, the 114-way cross-entropy and the 30 % training
    fraction. We compute the loss in float64, as the paper's notebook and ARENA do; the released `helpers.py` casts it
    to float32.
  - **Optimiser.** Full-batch AdamW with lr 10⁻³, β = (0.9, 0.98) and weight decay λ = 1.
  - **Schedule.** `LambdaLR(min(step/10, 1))`, so the first update has learning rate 0.
- **Runs.** 22 runs: `wd1_s0…10`, `wd0_s0…4`, `rand_s0…2` and `split0_init100…102` (roles are given in the
  preregistration). Each runs for 40,000 steps, as the paper's text states (the released run continued to 50,000).
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
   - **Result.** About 2,000 steps per 210 s for all 22 runs together, or 73 min in total: roughly 6× faster.
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
   - **Conclusion.** The tests show bitwise reproducibility within a process and across resume. Across processes we
     have this check: the relaunch reproduced every logged test accuracy (six runs, steps 2,000 to 10,000) to the printed
     3 decimals.

## Results

All 22 runs trained together in 4,376 s (73 min), from `logs/20261021-000752_pipeline.log`. The final checkpoint is at
step 40,000. The phase boundaries follow the frozen rules in `docs/operationalizations.md`, and K is each run's key
frequencies at its final checkpoint.

| Run | Final train acc | Final test acc | Test loss at the final checkpoint (113-way) | t1 | t2 | t3 | t_grok | Key frequencies K |
|---|---|---|---|---|---|---|---|---|
| `wd1_s0` | 100.0 % | 100.0 % | 4.44e-07 | 1,561 | 9,600 | 12,900 | 10,600 | 14, 35, 41, 42, 52 |
| `wd1_s1` | 100.0 % | 100.0 % | 2.55e-07 | 1,562 | 9,600 | 13,800 | 10,700 | 2, 3, 23, 37, 39 |
| `wd1_s2` | 100.0 % | 100.0 % | 3.40e-07 | 1,597 | 3,900 | 6,500 | 4,600 | 6, 7, 15, 27, 38 |
| `wd1_s3` | 100.0 % | 100.0 % | 8.00e-07 | 1,572 | 7,400 | 13,900 | 8,300 | 9, 27, 53, 55 |
| `wd1_s4` | 100.0 % | 100.0 % | 1.58e-07 | 1,578 | 6,500 | 11,900 | 7,900 | 11, 25, 49 |
| `wd1_s5` | 100.0 % | 100.0 % | 1.86e-07 | 1,570 | 9,200 | 12,700 | 10,100 | 9, 34, 38, 40 |
| `wd1_s6` | 100.0 % | 100.0 % | 5.61e-07 | 1,563 | 9,800 | 17,300 | 10,900 | 7, 26, 35, 41 |
| `wd1_s7` | 100.0 % | 100.0 % | 5.07e-07 | 1,586 | 4,900 | 7,800 | 5,600 | 12, 18, 20, 48, 55 |
| `wd1_s8` | 100.0 % | 100.0 % | 1.16e-06 | 1,576 | 9,900 | 14,500 | 11,000 | 37, 44, 45, 46, 47, 51 |
| `wd1_s9` | 100.0 % | 100.0 % | 1.42e-07 | 1,604 | 4,200 | 7,300 | 5,100 | 19, 34, 35 |
| `wd1_s10` | 100.0 % | 100.0 % | 2.12e-07 | 1,570 | 6,900 | 10,800 | 8,200 | 19, 31, 45, 47 |
| `wd0_s0` | 100.0 % | 6.0 % | 53.9 | 11,817 | — | — | — | all 56 (no sparse set) |
| `wd0_s1` | 100.0 % | 1.9 % | 58.2 | 11,831 | — | — | — | all 56 |
| `wd0_s2` | 100.0 % | 8.1 % | 50.8 | 11,819 | — | — | — | all 56 |
| `wd0_s3` | 100.0 % | 21.5 % | 49.2 | 11,834 | — | — | — | all 56 |
| `wd0_s4` | 100.0 % | 16.2 % | 50.2 | 11,835 | — | — | — | all 56 |
| `rand_s0` | 100.0 % | 0.9 % | 30.0 | 1,586 | — | — | — | all 56 |
| `rand_s1` | 100.0 % | 0.9 % | 29.8 | 1,580 | — | — | — | all 56 |
| `rand_s2` | 100.0 % | 0.8 % | 30.4 | 1,582 | — | — | — | all 56 |
| `split0_init100` | 100.0 % | 100.0 % | 1.21e-06 | 1,590 | 8,000 | 12,000 | 9,100 | 2, 14, 27, 30, 37, 52 |
| `split0_init101` | 100.0 % | 100.0 % | 3.23e-07 | 1,573 | 8,400 | 11,700 | 9,900 | 9, 14, 15, 27 |
| `split0_init102` | 100.0 % | 100.0 % | 4.81e-07 | 1,693 | 3,700 | 6,500 | 4,300 | 7, 38, 49, 53 |
| paper (for reference) | 100 % | 100 % | 3.02e-07 (step 40,001) | 1,567 | 9,301 | 14,001 | 10,601 | 14, 35, 41, 42, 52 |

- **Grokking.** Every run with weight decay (wd1, split0) groks; t_grok ranges from 4,300 to 11,000 steps. No run without
  weight decay groks.
- **Same initialisation as the paper.** `wd1_s0` starts from the paper's initial weights, equal up to float32 rounding
  (largest difference 3.0 × 10⁻⁸), on the paper's split:
  - **Same curve at first.** Its training loss matches the paper's logged curve to 5 significant digits for the first
    52 steps.
  - **Then it diverges.** This is point 2 above, now observed against an independent implementation.
  - **Same end point.** It ends with the paper's key frequencies; its weights at step 40,000 have cosine similarity
    0.74 with the paper's step-40,001 weights. See [exp-03](exp-03-hypotheses.md), exploratory observation 1.
- **Late spikes.** Two runs (`wd1_s4`, `split0_init102`) show repeated test-loss spikes from 2,000 steps after
  t_grok on (exp-03, exploratory observation 2).

## Limitations

- **Single code path.** All runs share one implementation and one GPU type. Bitwise reproducibility is tested only
  within one process on this software and hardware stack; across stacks, expect seed-level differences (point 2).
- **Steps, not wall time.** Training length is fixed at 40,000 steps, as the paper's text states. The released mainline
  run continued to 50,000 epochs, and its late-phase change near 43,000 steps cannot appear in our runs.
