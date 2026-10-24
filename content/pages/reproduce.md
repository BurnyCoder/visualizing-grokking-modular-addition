# Reproduce

Everything on this site is computed by an open pipeline: [github.com/BurnyCoder/visualizing-grokking-modular-addition](https://github.com/BurnyCoder/visualizing-grokking-modular-addition).

```
git clone --recurse-submodules https://github.com/BurnyCoder/visualizing-grokking-modular-addition
cd visualizing-grokking-modular-addition
uv sync --extra cu130        # or --extra cpu
uv run grokviz --phases all --skip publish_runs_to_hub,publish_site_data  # download, validate, train, analyse, export, build (no uploads)
```

## Does our code reproduce the paper?

Before trusting any figure we checked that our re-implementation of the paper's analysis gives the paper's own numbers on the paper's own network (its published checkpoint at epoch 40,000). Rows marked **no** are listed honestly, with our best explanation; see also the [errata](errata.html).

<!-- RCHECK_TABLE -->

## Validating the checkpoints

Recomputing the training loss of the paper's checkpoints on our reconstruction of its train/test split reproduces the paper's logged losses at every checked step (e.g. step {{num:paper.val_step}}: {{num:paper.val_loss_recomputed}} recomputed vs {{num:paper.val_loss_logged}} logged). This confirms both the data split and the step labelling (checkpoint “epoch e” = weights after e + 1 updates).

## Our own runs

22 networks were trained with the paper's settings (chapter 20); their checkpoints, curves and analysis results are published on Hugging Face (Adam's saved states are not), and the hypotheses were frozen before training (git tag `prereg-v1`).
