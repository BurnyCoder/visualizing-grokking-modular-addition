# Architecture

## Data flow

```mermaid
flowchart LR
  subgraph Sources
    HF[(HF: paper checkpoints<br/>callummcdougall/grokking_full_run_data)]
    ARX[(arXiv PDFs<br/>quote checking only)]
  end
  subgraph Pipeline["src/grokviz — uv run grokviz --phases …"]
    ENV[check_environment] --> ACQ[acquire_paper_sources] --> CONV[convert_paper_run] --> VAL[validate_paper_run]
    VAL --> RC[validate_against_paper<br/>R-checks]
    VAL --> TR[train_all_runs<br/>22 runs, vmap-batched]
    TR --> CM[compute_checkpoint_metrics]
    CM --> GR[compute_gradient_dynamics]
    GR --> GEO[compute_geometry]
    GEO --> PH[detect_phases]
    PH --> FIN[analyse_final_models]
    FIN --> HYP[test_hypotheses]
    HYP --> PUB[publish_runs_to_hub]
    PUB --> EXP[export_site_data<br/>+ summary.json]
    EXP --> PSD[publish_site_data] --> BUILD[build_site]
  end
  HF --> ACQ
  ARX --> ACQ
  RUNS[(runs/&lt;id&gt;/<br/>checkpoints, curves, analysis)]
  CONV --> RUNS
  TR --> RUNS
  RUNS --> CM
  BUILD --> SITE[(site/<br/>static HTML + data)]
  SITE --> PAGES[GitHub Pages]
  PSD --> HFD[(HF dataset: site data)]
  PUB --> HFM[(HF model repo: our runs)]
```

## Layers

| Layer | Files | Responsibility |
|---|---|---|
| Wrapper | `pipeline.py`, `__main__.py` | ordered registry of phases; CLI (`--phases`, `--runs`, `--force`, `--skip`) |
| Settings & logs | `config.py` (+ `.env.example`), `logs.py`, `context.py`, `envcheck.py` | typed config from .env files; timestamped file+terminal logging; provenance |
| Core math | `task.py`, `model.py`, `losses.py`, `fourier.py` | dataset/split, paper-layout transformer with hooks, float64 losses, Fourier basis, Gini |
| Runs | `runstore.py`, `sources.py`, `train.py` | on-disk run format; paper checkpoints; resumable vmap-batched training |
| Analysis | `metrics.py`, `interventions.py`, `patching.py`, `checkpoint_metrics.py`, `gradients.py`, `geometry.py`, `phases.py`, `rchecks.py`, `hypotheses.py`, `stats.py` | every measurement on the site and in the reports |
| Output | `export.py`, `summary.py`, `site_build.py`, `quotes.py`, `hf_publish.py`, `serve.py` | binary site data, prose numbers, HTML build with content checks, quote verification, Hub uploads, preview server |
| Website | `content/`, `templates/`, `web/` | chapters/glossary/symbols/quotes/figures metadata; Jinja templates; ES-module figures, widgets, in-browser model |

## Design choices (and why)

- **No TransformerLens.** TransformerLens 4.x removed `HookedTransformer`; a ~150-line model with the paper's exact
  parameter layout loads the paper's checkpoints without conversion and is mirrored line by line in JavaScript.
- **Batched training.** The model is tiny, so one run leaves the GPU idle; all 22 runs train together with
  `torch.func.vmap` (runs provably independent, see `tests/test_train.py`), ~70 minutes instead of ~7.5 hours.
- **Static site, one implementation per figure.** Figures are ES modules registered by id; chapters and the
  dashboard place the same implementation, so the step-by-step text and the dashboard cannot disagree.
- **Numbers are bound, not typed.** Every number in the prose comes from `site/data/summary.json` via `{{num:key}}`.
- **Heavy compute in Python, per-input compute in the browser.** Curves, spectra and geometry are precomputed; any
  (a, b) / neuron / head view is computed from float16 weights by `web/js/model/grokmodel.js` (tested against torch).
