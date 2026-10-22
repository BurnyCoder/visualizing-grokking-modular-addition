# Architecture

## Data flow

The phase-by-phase diagram is in the README ([What happens when you run it](../README.md#what-happens-when-you-run-it)).

## Layers

| Layer | Files | Responsibility |
|---|---|---|
| Wrapper | `pipeline.py`, `__main__.py` | ordered registry of phases; CLI (`--phases`, `--runs`, `--force`, `--skip`) |
| Settings & logs | `config.py` (+ `.env.example`), `logs.py`, `context.py`, `envcheck.py` | typed config from .env files; timestamped file+terminal logging; provenance |
| Core math | `task.py`, `model.py`, `losses.py`, `fourier.py` | dataset/split, paper-layout transformer with hooks, float64 losses, Fourier basis, Gini |
| Runs | `runstore.py`, `sources.py`, `train.py` | on-disk run format; paper checkpoints; resumable vmap-batched training |
| Analysis | `metrics.py`, `interventions.py`, `patching.py`, `checkpoint_metrics.py`, `gradients.py`, `geometry.py`, `phases.py`, `rchecks.py`, `hypotheses.py`, `stats.py` | every measurement on the site and in the reports |
| Output | `export.py`, `summary.py`, `formatting.py`, `site_build.py`, `quotes.py`, `hf_publish.py`, `serve.py`, `figures.py`, `paper.py` | binary site data, prose numbers and their formatting, HTML build with content/link/forward-use checks, quote verification, Hub uploads, preview server, report figures rendered from the site, Typst paper |
| Website | `content/`, `templates/`, `web/` | chapters/glossary/symbols/quotes/figures metadata; Jinja templates; ES-module figures, widgets, in-browser model |

## Design choices (and why)

- **No TransformerLens.** TransformerLens 4.x removed `HookedTransformer`; a ~150-line model with the paper's exact
  parameter layout loads the paper's checkpoints without conversion and is mirrored line by line in JavaScript.
- **Batched training.** The model is tiny, so one run leaves the GPU idle; all 22 runs train together with
  `torch.func.vmap` (runs provably independent, see `tests/test_train.py`), ~70 minutes instead of ~7.5 hours.
- **Static site, one implementation per figure.** Figures are ES modules registered by id; chapters and the
  dashboard place the same implementation, so the step-by-step text and the dashboard cannot disagree.
- **Numbers are bound, not typed.** Every number in the prose comes from `site/data/summary.json` via `{{num:key}}`.
- **One render pipeline for every figure.** A state change goes store → flush → visibility gate → latest-wins
  runner → dedupe → figure (`web/js/core/store.js`, `core/visibility.js`, `figures/common.js` `watch`):
  - `set()` updates the state at once; listeners are notified once per animation frame with the union of the changed keys;
  - a figure that is off screen only marks itself dirty and draws once when it scrolls back;
  - at most one draw per figure runs at a time, always with the latest state, so frames never land out of order;
  - a figure whose stored frame (checkpoint, weight snapshot, geometry frame) did not change skips the draw;
  - the step cursor (`viz/overlay.js`) and heatmap marks are overlays moved with CSS, not chart redraws;
  - an element inside `[data-scope]` gets its own step (`core/state.js` `storeOf`), so an animated figure can play
    without moving the rest of the page.
- **Heavy compute in Python, per-input compute in the browser.** Curves, spectra and geometry are precomputed; any
  (a, b) / neuron / head view is computed from float16 weights by `web/js/model/grokmodel.js` (tested against torch).
