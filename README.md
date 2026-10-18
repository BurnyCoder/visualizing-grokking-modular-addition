# Grokking, Visualized — modular addition, step by step

An explorable website (plus the reproducible pipeline behind it) that visualizes the one-layer
transformer which *groks* addition modulo 113, as reverse-engineered by Nanda et al. (2023),
[*Progress measures for grokking via mechanistic interpretability*](https://arxiv.org/abs/2301.05217).

> **Status: under construction.** This README grows with each milestone; the plan lives in
> `docs/` as it is written. Sections marked *(planned)* are not implemented yet.

## Methodology (summary)

- **Models.** The paper's own mainline run (Hugging Face
  [`callummcdougall/grokking_full_run_data`](https://huggingface.co/callummcdougall/grokking_full_run_data),
  checkpoints every 100 epochs) is the default subject. We also retrain the same architecture from
  scratch over several seeds with dense logging (gradients, optimizer state), plus controls without
  weight decay and with random labels. *(planned)*
- **Analysis.** Paper-faithful re-implementations of the Fourier analysis, restricted/excluded loss,
  ablations and progress measures, ported from the paper's notebook, the ARENA 1.5.2 grokking
  chapter and this repo's submodule notebooks, each with source links. *(planned)*
- **Website.** Step-by-step chapters for readers with minimal prerequisites, plus one big dashboard
  of linked visualizations scrubbable across training. *(planned)*

## Repository layout

- `ai-mechanistic-interpretability-transformer-modular-addition-grokking/` — git submodule with the
  original notebooks (reference only).
- `src/grokviz/` — Python pipeline package.
- `.env.example` — every configurable setting; copy to `.env` to override.

## Setup

```bash
git clone --recurse-submodules https://github.com/BurnyCoder/visualizing-grokking-modular-addition
cd visualizing-grokking-modular-addition
uv sync
```

## Third-party material

See [`NOTICE.md`](NOTICE.md) for the terms of the paper, checkpoints, ARENA material and libraries.
