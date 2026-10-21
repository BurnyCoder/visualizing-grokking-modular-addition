# Reproduce

Everything on this site is computed by an open pipeline: [github.com/BurnyCoder/visualizing-grokking-modular-addition](https://github.com/BurnyCoder/visualizing-grokking-modular-addition).

```
git clone --recurse-submodules https://github.com/BurnyCoder/visualizing-grokking-modular-addition
cd visualizing-grokking-modular-addition
uv sync --extra cu130        # or --extra cpu
uv run grokviz --phases all  # download, validate, train, analyse, export, build
```

## Does our code reproduce the paper?

Before trusting any figure we checked that our re-implementation of the paper's analysis gives the paper's own numbers on the paper's own network (its published checkpoint at epoch 40,000). Rows marked **no** are listed honestly, with our best explanation; see also the [errata](errata.html).

<!-- RCHECK_TABLE -->

## Validating the checkpoints

Recomputing the training loss of the paper's checkpoints on our reconstruction of its train/test split reproduces the paper's logged losses at every checked step (e.g. step 101: 2.211129 vs 2.211129). This confirms both the data split and the step labelling (checkpoint “epoch e” = weights after e + 1 updates).

## Our own runs

22 networks were trained with the paper's settings (chapter 20); their checkpoints, curves and optimiser states are published on Hugging Face, and the hypotheses were frozen before training (git tag `prereg-v1`).
