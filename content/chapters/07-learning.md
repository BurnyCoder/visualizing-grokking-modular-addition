## Training means lowering the loss

[[dfn:training|Training]] repeatedly nudges all {{num:model.n_params}} parameters so that the loss on the training set goes down. To know which way to nudge, we need slopes.

The [[dfn:derivative]] of a function tells how fast its output changes when the input changes a little: the slope of its graph. For $f(w) = (w-3)^2$ the derivative is $f'(w) = 2(w-3)$; at $w = 5$ it is 4 (going right increases f), at $w = 1$ it is −4. The [[dfn:gradient]] is the list of derivatives of the loss with respect to *every* parameter at once — a vector with one entry per parameter, pointing in the direction in which the loss increases fastest. Computer libraries compute it exactly and automatically.

[[dfn:gradient-descent|Gradient descent]] moves the parameters a small step *against* the gradient:

$$\theta \leftarrow \theta - \gamma\, \nabla_\theta \mathcal L,$$

where the step size [[symdef:gamma]] is the [[dfn:learning-rate]].

::: example
**Three steps of gradient descent on $f(w) = (w-3)^2$**, starting at $w = 5$ with $\gamma = 0.25$:

- $w = 5$: slope $2(5-3) = 4$, new $w = 5 - 0.25\cdot4 = 4$;
- $w = 4$: slope 2, new $w = 3.5$;
- $w = 3.5$: slope 1, new $w = 3.25$ — approaching the minimum at $w = 3$, halving the distance each step.
:::

Press ▶ next to “steps” to watch the path grow one step at a time.

{{widget:gd-toy}}

## Full-batch training

The paper computes each update from all {{num:task.n_train}} training examples at once — [[dfn:full-batch]] training {{cite:nanda2023}}. Then one parameter update, a [[dfn:step]], is also one pass over the whole training set (an “epoch”), and the training process has no randomness after the start: given the starting point, every step is determined. (This will let us recompute the exact gradient at any saved moment of training in chapter 19.)

## Adam, weight decay, AdamW

The rule that turns gradients into updates is the [[dfn:optimizer]]. Plain gradient descent struggles when some parameters need much bigger steps than others. [[dfn:adam|Adam]] keeps, for every parameter, a running average of its recent gradients (“momentum”, decay 0.9 per step) and of its squared gradients (decay 0.98 here), and steps by their ratio — so each parameter moves by roughly the learning rate in its consistent direction, whatever the size of its gradient {{cite:kingma2015}}.

[[dfn:weight-decay|Weight decay]] additionally shrinks every parameter a little each step. [[dfn:adamw|AdamW]] applies it directly (“decoupled”) {{cite:loshchilov2019}} {{cite:torch_adamw}}:

$$\theta \;\leftarrow\; \theta\,(1 - \gamma\lambda) \;-\; \gamma\cdot\frac{\hat m}{\sqrt{\hat v} + \epsilon},$$

with [[symdef:lam]] the weight-decay strength and $\hat m$, $\hat v$ Adam's running averages. Weight decay is a form of [[dfn:regularization]]: it prefers solutions with small weights among those that fit the data. With $\gamma = 0.001$ and $\lambda = 1$, every weight shrinks by 0.1 % per step unless the loss gradient pushes back — after 1,000 steps an unopposed weight would shrink to $0.999^{1000} \approx 37\%$ of its size.

The three rules side by side on a loss with two parameters, shaped like a long valley (its lowest point is marked ×). Press ▶ next to “steps” and compare where each one goes.

{{widget:adamw-toy}}

The same pull on the real network: each training run as a path through (size of all weights, loss). With weight decay (λ = 1) the weights first grow and then shrink while the test loss falls; without it (λ = 0) they keep growing and the test loss stays high. Chapter 19 returns to this balance.

{{anim:loss-vs-norm}}

## The settings of this experiment

Settings chosen before training are [[dfn:hyperparameter|hyperparameters]]. The parameters start at random values (their [[dfn:initialization]]); a [[dfn:seed|random seed]] fixes those values and the train/test split so a run can be repeated exactly.

| Setting | Paper's run | Our runs |
|---|---|---|
| Optimizer | AdamW, β = (0.9, 0.98) | same |
| Learning rate γ | 0.001, warm-up min(step/10, 1) | same |
| Weight decay λ | 1 | 1 (and 0 for a control) |
| Batch | full (3,830 examples) | same |
| Steps | 40,000 in the paper's text (its released code and run continue to 50,000) | 40,000 |
| Loss | cross-entropy over all 114 outputs | same, computed in float64 |
| Initialization | Gaussian, scale 1/√128 (W_U: 1/√114); no seed call in the released code, yet equal to our seed-0 draw up to rounding (chapter 20) | same rule, seeded |
| Split | python random.shuffle, seed 0 | seed = run's seed (seed 0 = paper's split) |

Sources: paper Sec. 3 {{cite:nanda2023}} and the paper's training code {{cite:paperrepo}}. Two details matter for exact reproduction: the warm-up multiplies the learning rate by 0 at the very first step (so the first update leaves the weights unchanged, though Adam's running averages are updated), and the paper's checkpoint labelled “epoch e” holds the weights after e + 1 updates. We verified both by recomputing the paper's logged losses from its checkpoints (see [Reproduce](../reproduce.html)).

## Overfitting

[[dfn:overfitting]] is doing much better on the training set than on unseen data. A network with 226,816 parameters and only 3,830 training examples can overfit easily: it has more than enough capacity to memorise. The next chapter shows what happens next.

::: nuance
**Chaos in early training.** While retraining, we found that early full-batch Adam training is extremely sensitive to tiny numerical differences: the largest parameter difference between two mathematically identical implementations is 3 × 10⁻⁷ after 20 steps but ~0.02 after 50 and ~0.2 after 200 steps (tiny rounding differences are amplified by orders of magnitude within tens of steps). Runs are still exactly repeatable on the same hardware and software (we use PyTorch's deterministic mode), but they are not reproducible bit-for-bit across different GPU kernels — one reason to report results over many seeds, as we do in chapter 20.
:::
