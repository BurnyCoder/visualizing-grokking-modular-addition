"""Phases `publish_runs_to_hub` and `publish_site_data`: share our trained runs and the site data on Hugging Face.

Local explanation:
- `publish_runs_to_hub` uploads every run we trained (checkpoints, curves, metadata, analysis) to one model
  repository with a model card, and gathers it — together with the site-data dataset, this project's older
  checkpoint repo and the paper's original checkpoints — in a Hugging Face collection.
- `publish_site_data` uploads site/data (the website's arrays and weights) to a dataset repository and records the
  resulting commit in site-data.lock, which the GitHub Pages workflow downloads at deploy time (the data is too
  large to keep in git).
The paper's own run is never re-uploaded as checkpoints (it stays at its original repository); the website data
contains float16 copies of its weights, with attribution (decision recorded in NOTICE.md).
Optimiser states (optim.safetensors) are not uploaded: they are large and exactly regenerable by re-running the
deterministic training.
Global context: makes every number on the site re-checkable without retraining.
Sources: huggingface_hub upload_folder for large folders (https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-large-folder),
collections API (https://huggingface.co/docs/huggingface_hub/guides/collections).
"""

from __future__ import annotations  # postponed annotations

import json  # lock file
import logging  # module logger

from huggingface_hub import HfApi  # Hub client (uses the cached login or HF_TOKEN)

from grokviz.config import REPO_ROOT  # lock file location
from grokviz.context import PipelineContext  # phase argument

log = logging.getLogger("grokviz.hf_publish")
GITHUB_URL = "https://github.com/BurnyCoder/visualizing-grokking-modular-addition"

MODEL_CARD = """---
license: mit
tags: [grokking, mechanistic-interpretability, modular-arithmetic, transformer]
---
# Grokking modular addition — retrained runs

{n} one-layer transformers trained on (a + b) mod 113 with the settings of Nanda et al. 2023,
*Progress measures for grokking via mechanistic interpretability* (arXiv:2301.05217): 30 % of all pairs for
training, full-batch AdamW (lr 1e-3, betas (0.9, 0.98), weight decay 1 or 0), 10-step warm-up, 40,000 steps,
deterministic PyTorch. Runs (seeds and controls, see `docs/preregistration.md` in the code repository):

{runs}

Each folder holds `checkpoints.safetensors` (every parameter stacked over checkpoints: tensor `<param>` has shape
`[T, ...]`, checkpoint t taken after `meta.json["ckpt_steps"][t]` optimiser steps; parameter names follow the
paper's code, so `GrokTransformer` in the code repository loads them with `strict=True`), `curves.npz` (per-step
train loss/accuracy, test loss/accuracy every 10 steps, per-parameter weight/gradient/update norms) and
`analysis/` (all measurements shown on the website).

Code, website and reports: {github}. Author: Libor Burian.
"""


def publish_runs_to_hub(ctx: PipelineContext) -> None:
    """Phase entry point: upload our runs (not the paper's) + model card; build the collection."""
    api = HfApi()
    repo = ctx.cfg.hf_runs_repo
    runs = sorted(d.name for d in ctx.cfg.runs_dir.iterdir() if (d / "meta.json").exists() and d.name != "paper")
    lines = []
    for r in runs:
        meta = json.loads((ctx.cfg.runs_dir / r / "meta.json").read_text(encoding="utf-8"))
        lines.append(f"- `{r}` — {meta.get('title', r)}")
    card = ctx.cfg.runs_dir / "README.md"
    card.write_text(MODEL_CARD.format(n=len(runs), runs="\n".join(lines), github=GITHUB_URL), encoding="utf-8")
    api.create_repo(repo, repo_type="model", exist_ok=True)
    log.info("uploading %d runs to %s (resumable)", len(runs), repo)
    # huggingface_hub ≥ 2.0 removed upload_large_folder; upload_folder now streams large folders in several commits
    # and resumes when re-run (https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-large-folder)
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(ctx.cfg.runs_dir), commit_message="Upload runs",
                      allow_patterns=["README.md"] + [f"{r}/*" for r in runs] + [f"{r}/analysis/*" for r in runs],
                      ignore_patterns=["*/optim.safetensors", "*/_tmp/*", "_resume/*", "paper/*"])
    build_collection(ctx, api)


def build_collection(ctx: PipelineContext, api: HfApi) -> None:
    """Create (or reuse) the collection and add every related repository."""
    coll = api.create_collection(title=ctx.cfg.hf_collection_title, namespace=api.whoami()["name"], exists_ok=True,
                                 # the Hub rejects collection descriptions longer than 150 characters
                                 description="Retrained runs, website data and original checkpoints of the transformer that groks modular addition.")
    items = [(ctx.cfg.hf_runs_repo, "model", "Our 22 retrained runs (seeds and controls)"),
             (ctx.cfg.hf_site_data_repo, "dataset", "Arrays and float16 weights used by the website"),
             ("BurnyCoder/grokking-modular-addition-transformer", "model", "Earlier TransformerLens replication (seed 999)"),
             (ctx.cfg.paper_hf_repo, "model", "The paper's own mainline run (original uploader)")]
    present = {item.item_id for item in coll.items}
    for item_id, kind, note in items:
        if item_id in present:
            continue
        try:
            api.add_collection_item(coll.slug, item_id=item_id, item_type=kind, note=note, exists_ok=True)
        except Exception as exc:  # e.g. the dataset repo does not exist yet on a first run
            log.warning("could not add %s to the collection: %s", item_id, exc)
    log.info("collection: https://huggingface.co/collections/%s", coll.slug)


def publish_site_data(ctx: PipelineContext) -> None:
    """Phase entry point: upload site/data to the dataset repo and pin its commit in site-data.lock."""
    api = HfApi()
    repo = ctx.cfg.hf_site_data_repo
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    data_dir = ctx.cfg.site_dir / "data"
    # delete_patterns="*": remote files under data/ that are not in this upload are deleted, so the dataset mirrors the
    # local export (no stale arrays shipped to Pages; https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-folder)
    info = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(data_dir), path_in_repo="data",
                             commit_message="Update website data", delete_patterns="*")
    lock = {"repo": repo, "revision": info.oid, "path": "data"}
    (REPO_ROOT / "site-data.lock").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    log.info("site data uploaded to %s @ %s (site-data.lock updated)", repo, info.oid)
    build_collection(ctx, api)
