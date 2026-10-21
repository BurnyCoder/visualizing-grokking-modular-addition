# Deployment

- **Website:** `.github/workflows/pages.yml` runs on every push to `main`: installs the project without torch,
  downloads the site data pinned in `site-data.lock` from the Hugging Face dataset, runs
  `python -m grokviz.site_build`, and publishes `site/` to GitHub Pages (source: GitHub Actions).
- **Site data:** too large for git. It is 453 MB for the paper run and 22 runs, above the 300 MB planned at the
  start. Most of it is float16 weights (181 MB) and per-input/key-plane geometry (213 MB). Every array is fetched lazily,
  so a page only loads what its figures need. The GitHub Pages limit is 1 GB. `uv run grokviz --phases export_site_data,publish_site_data` uploads
  `site/data/` to `GROKVIZ_HF_SITE_DATA_REPO` and rewrites `site-data.lock` with the new commit; commit the lock file to
  deploy the new data.
- **Runs:** `uv run grokviz --phases publish_runs_to_hub` uploads our runs (without optimizer states) to
  `GROKVIZ_HF_RUNS_REPO` and (re)builds the collection `GROKVIZ_HF_COLLECTION_TITLE`.
- **Local preview:** `uv run python -m grokviz.serve --dir site --port 8766` (no-cache headers), or `.claude/launch.json`.
- **CI:** `.github/workflows/ci.yml` runs the fast test suite on CPU (GPU, network, slow and browser tests run locally).
- **Credentials:** Hugging Face uses the cached `hf auth login` (or `HF_TOKEN` in a private `.env`); nothing secret is
  stored in the repository.
