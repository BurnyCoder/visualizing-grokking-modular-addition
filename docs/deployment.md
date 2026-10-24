# Deployment

- **Website:** `.github/workflows/pages.yml` runs on every push to `main`: installs the project without torch,
  downloads the site data pinned in `site-data.lock` from the Hugging Face dataset, runs
  `python -m grokviz.site_build`, and publishes `site/` to GitHub Pages (source: GitHub Actions).
- **Site data:** too large for git. It is about 643 MB for the paper run and 22 runs:
  - float16 weight snapshots, so every run can be animated in the browser (314 MB; spacing set by
    `GROKVIZ_SITE_WEIGHT_SNAPSHOTS`, see `docs/data-formats.md`);
  - per-input and key-plane geometry (223 MB);
  - per-checkpoint metrics (72 MB);
  - 2D Fourier maps (26 MB);
  - attention lookup tables (8 MB).

  `tests/test_site_data.py` keeps the total below the project's 0.8 GB budget (the
  [GitHub Pages limit](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits) is 1 GB)
  and every file below 16 MB. Every array is fetched lazily, so a page only loads what its figures need. `uv run grokviz --phases export_site_data,publish_site_data` uploads
  `site/data/` and a dataset card (`DATASET_CARD` in `hf_publish.py`) to `GROKVIZ_HF_SITE_DATA_REPO` and rewrites
  `site-data.lock` with the new commit; commit the lock file to deploy the new data.
- **Runs:** `uv run grokviz --phases publish_runs_to_hub` uploads our runs (without optimizer states) to
  `GROKVIZ_HF_RUNS_REPO` and (re)builds the collection `GROKVIZ_HF_COLLECTION_TITLE`.
- **Local preview:** `uv run python -m grokviz.serve --dir site --port 8766` (no-cache headers), or `.claude/launch.json`.
- **CI:** `.github/workflows/ci.yml` runs the fast test suite on CPU (GPU, network, slow and browser tests run locally).
- **Credentials:** Hugging Face uses the cached `hf auth login` (or `HF_TOKEN` in a private `.env`); nothing secret is
  stored in the repository.
