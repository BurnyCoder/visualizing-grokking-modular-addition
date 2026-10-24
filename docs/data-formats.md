# Data formats

## Run folder (`runs/<run_id>/`, written by `runstore.write_run`)

| File | Content |
|---|---|
| `meta.json` | `kind` (paper/trained), `group`, seeds, task, hyper-parameters, `ckpt_steps` (optimizer steps of each checkpoint), `model_shape`, provenance (`environment`, source revision/sha) |
| `checkpoints.safetensors` | one tensor per state_dict key (paper names: `embed.W_E`, `blocks.0.attn.W_Q`, …), stacked over checkpoints: shape `[T, ...]` |
| `curves.npz` | `train_loss` (per step), `train_acc`, `test_loss`/`test_acc` (every `test_every`), `*_steps`, `lr`, per-parameter `grad_norm`/`weight_norm`/`update_norm` (every `norms_every`) |
| `optim.safetensors` | (our runs) Adam `exp_avg.<param>` / `exp_avg_sq.<param>` stacked every `optim_every` steps |
| `analysis/` | `ckpt_metrics.npz`, `gradients.npz`, `geometry.npz`, `phases.json`, `final.json`, `key_freqs.json`, (paper) `validation.json`, `rchecks.json` |

Step semantics: checkpoint t holds the parameters after `ckpt_steps[t]` optimizer updates. For the paper's file,
“epoch e” = step e + 1 (verified, see reports/exp-01).

## Site data (`site/data/`, written by `export.py`)

- `manifest.json`: `{schema_version, p, basis_names, train_masks, scalar_columns, runs: [...]}`; each run has `id`, `title`,
  `kind`, `group`, `split_seed`, `init_seed`, `weight_decay`, `fn_name`, `label_seed`, `key_freqs`, `ckpt_steps`,
  `geometry_steps`, `spectra_steps`, `weight_steps`, `param_layout` (name + shape in flat order), `phases`, `final`
  (final-checkpoint analysis) and `arrays: {name: {file, dtype, shape, desc, bytes, sha256, [min, max], [columns],
  [step0, step_every]}}`. Per-step arrays (`curve_*`, `update_norm`) carry `step0`/`step_every` instead of checkpoint indices.
  `sha256` holds the first 16 hex digits of the file's SHA-256. Every run has `spectra_steps`, the frames of the 2D-Fourier
  maps `spec_*`: many frames for the showcase runs (`paper`, `wd1_s0`), only the final checkpoint for the others.
  `adam_scalars` (with `steps`) and `update_norm` exist for our retrained runs only.
- Array files `runs/<id>/<name>.bin`: little-endian; `dtype` ∈ `f32`, `f16`, `i16`, `u8q` (uint8, value = min + q/255·(max − min)).
  Shapes are row-major; e.g. `we_fourier` is `[T, 113]` (Const, cos 1, sin 1, …, sin 56); `attn_C` is `[T, 4, 113]`, each head's
  attention score lookup C_j[t] (paper App. C.1.3), float16.
  `grad_norm` and `weight_norm` are `[T, 11]`, one column per parameter in the same order; only `grad_norm` lists the
  parameter names in `columns` (the `weight-norms` panel reads them from there).
- Weights `w_<step>.bin`: float16 flat vector of the state_dict in `param_layout` order (in-browser model). Showcase runs
  store every 2nd (`paper`) or 5th (`wd1_s0`) checkpoint. Every other run stores the checkpoints chosen by `export.snapshot_steps`:
  every `fine` steps up to t3 + 2,000, then every `coarse` steps (only the coarse grid when no t3 was detected), plus the
  final weights. Spacing comes from `GROKVIZ_SITE_WEIGHT_SNAPSHOTS` (default `1000,4000`).
- `train_mask_seed<s>.bin`: 12,769 bytes, 1 = training pair (row a·113 + b).
- `summary.json`: `{key: {value, fmt}}`, every measured number quoted in the prose (`src/grokviz/summary.py`). Entries with
  `fmt: "json"` are structured data for the figures (`hypotheses.verdict`, `hypotheses.per_run`, `hypotheses.controls`
  from `reports/tables/hypotheses.json`); the site build rejects them in `{{num:}}`. `rchecks.rows` (the R-check table
  from `runs/paper/analysis/rchecks.json`) is a list of row objects stored with `fmt: "str"`.
