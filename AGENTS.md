# AGENTS.md — instructions for coding agents working in this repo

## What this repo is
A reproducible pipeline (`src/grokviz/`, Python, uv) plus a static website. Both concern the modular-addition grokking
transformer of Nanda et al. 2023 (arXiv:2301.05217). The README covers methodology, setup and data flow.

The original notebooks live in the git submodule `ai-mechanistic-interpretability-transformer-modular-addition-grokking/`.
Treat it as read-only reference: port code from it with `Source:` links instead of editing it.

## Environment
- **Python.** Use the local venv managed by uv: `uv sync --extra cu130` (or `--extra cpu`), then `uv run <cmd>`. No
  global installs. Python 3.12 (`.python-version`).
- **Config.** Configuration comes from `.env`; `.env.example` holds every key with its default. When adding a setting,
  add a field to `config.Config` *and* a commented line to `.env.example`.
- **Windows:**
  - `PYTHONUTF8=1` (set in `.env.example`) keeps logs with math symbols from crashing.
  - While a pipeline process is running, `uv sync` cannot replace `grokviz.exe`. Use `.venv/Scripts/python.exe -m grokviz …`
    and `uv pip install` instead of re-syncing.
- **Determinism.** `CUBLAS_WORKSPACE_CONFIG=:4096:8` is required. Training is bitwise reproducible across processes and
  across resume (`tests/test_train.py`).

## Pipeline conventions
- **One wrapper.** `pipeline.PHASES` is the ordered registry. A new phase is a function `phase(ctx: PipelineContext)` in
  its own module, registered there and added to the README diagram.
- **Idempotent phases.** Skip runs whose outputs exist unless `ctx.force`; respect `ctx.wants(run_id)` for `--runs`.
- **Logging.** Log through `logging.getLogger("grokviz.<module>")`. `logs.setup_logging` writes timestamped lines to
  the terminal and to `logs/<ts>_pipeline.log`; never use `print` in pipeline code.
- **Step semantics.** A checkpoint at step t holds the parameters after t optimizer updates. The paper's "epoch e" is
  step e + 1.
- **Fourier basis order.** Follows the paper's code: index 0 = constant, `2k-1` = cos k, `2k` = sin k.
- **Frozen definitions.** Metric definitions are frozen in `docs/operationalizations.md` and the hypotheses in
  `docs/preregistration.md` (tag `prereg-v1`). Any change after the tag goes under *Deviations* there, with date and reason.
- **Large artifacts.** `runs/`, `data/`, `site/` and `logs/` are gitignored. Never commit `.env`, tokens or `paper.pdf`.

## Website conventions
- **Content.** Content lives in `content/`; `src/grokviz/site_build.py` turns it into `site/`. The checks it enforces
  are listed in `docs/content-style.md`, and the build exits non-zero on any violation.
- **Adding a figure.** Implement `render(body, dataset)` in a module under `web/js/figures/` and register its id in
  `web/js/figures/index.js`. Add its metadata to `content/figures.yaml`, then place it with `{{fig:id}}` or in
  `content/dashboard.yaml`. A figure that needs new data gets an array in `export.py`, documented in `docs/data-formats.md`.
- **Figure subscriptions.** Subscribe with `watch(body, keys, draw, {dedupe, theme})` from `web/js/figures/common.js`,
  never with `subscribe` directly (a build test rejects the old `watch(keys, fn)`). Pass `dedupe: frameKey(ds, field, ...keys)`
  when the figure shows a stored frame, and `theme: false` when its colours do not depend on the theme. Move cursors
  and marks with `Plot.cursor`, `Heatmap.marks` and `follow(body, keys, fn)`; never call `Plotly.relayout` per frame.
  Bind Plotly events with `Plot.on`.
- **Prose numbers.** Every number in the prose comes from `summary.py`, referenced as `{{num:key}}`; never type one by hand.
- **Preview.** Preview with `uv run python -m grokviz.serve --dir site --port 8766` (no-cache headers, so edited ES
  modules reload). This server is also the "site" entry in `.claude/launch.json`.
- **Edits to `web/`.** After editing `web/`, rebuild (`python -m grokviz.site_build`) or copy the file into `site/assets/`.

## Tests
- **TDD.** Write tests first (pytest; tests live in `tests/`).
- **Markers.** `gpu`, `slow`, `network` and `browser` gate the heavy tests. CI runs
  `-m "not gpu and not network and not slow and not browser"`.
- **JS parity.** The JavaScript model must match torch: `tests/test_js_model.py` runs `tests/js/grokmodel.test.mjs`.
- **JS unit tests.** Pure front-end modules (no DOM at import time) get Node tests in `tests/js/unit/*.test.mjs`,
  run by `tests/test_js_unit.py`.

## Code style
- **File docstrings.** Every module starts with a docstring holding *Local explanation*, *Global context* and *Sources*.
  Functions get a docstring; non-obvious lines get a short comment.
- **Source links.** Logic taken from the paper, its code, ARENA, the submodule notebooks or library docs carries a
  `Source:` link. Pin URLs to commits where possible.
- **Quotes.** Quotes from papers are short and exact (≤ 25 words) with a page number; `tests/test_quotes.py` verifies
  them against the PDFs. See `NOTICE.md` for the license terms behind this rule.
