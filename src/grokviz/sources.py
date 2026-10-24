"""Phases that acquire, convert and validate the paper's own mainline training run.

Local explanation:
- `acquire_paper_sources` downloads the paper's checkpoint file from Hugging Face at a pinned revision,
  verifies its sha256, and downloads the two arXiv PDFs (used only locally to verify website quotes).
- `convert_paper_run` turns the 456 MB torch file (500 state_dicts + loss lists; a pickle-based zip archive, read
  with torch.load(weights_only=True), which unpickles only tensors and plain containers) into our run format
  `runs/paper/` (stacked safetensors + curves + meta), recording the step of every checkpoint.
- `validate_paper_run` proves the conversion is faithful: recomputing the training loss from each
  checkpoint on our reconstruction of the paper's train/test split must reproduce the logged losses.
Global context: the paper run is the default subject of the website, so every later number depends on
(1) loading it correctly and (2) knowing exactly which optimizer step each checkpoint and loss belongs to.
Empirical alignment (checked on the real file, see reports/exp-01): checkpoint labelled "epoch e" holds the
parameters after e+1 optimizer steps (its train loss equals train_losses[e+1]); train_losses[j] is the loss
after j steps; test_losses[j] is the test loss after 10·j steps. The first optimizer step has learning rate 0
(the paper's LambdaLR warm-up min(step/10, 1) starts at 0), which is why train_losses[0] == train_losses[1].
Sources: checkpoint https://huggingface.co/callummcdougall/grokking_full_run_data (loaded with
weights_only=True as in ARENA 1.5.2: https://learn.arena.education/chapter1_transformer_interp/32_grokking/1-periodicity-fourier-basis/);
training loop and warm-up: https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/23d2c64/transformers.py;
hf_hub_download (https://huggingface.co/docs/huggingface_hub/guides/download); torch.save's zip-of-pickle format
(https://pytorch.org/docs/stable/notes/serialization.html) and torch.load's weights_only
(https://pytorch.org/docs/stable/generated/torch.load.html).
"""

from __future__ import annotations  # postponed annotations

import hashlib  # sha256 of downloaded files
import json  # validation report
import logging  # module logger
import math  # isclose
import urllib.request  # download arXiv PDFs
from pathlib import Path  # paths

import numpy as np  # curves
import torch  # load checkpoint file, tensors
from huggingface_hub import hf_hub_download  # pinned, cached download

from grokviz import losses, task  # loss functions and dataset/split
from grokviz.context import PipelineContext  # phase argument
from grokviz.envcheck import resolve_device  # GPU/CPU choice
from grokviz.model import ModelShape  # architecture recorded in meta
from grokviz.runstore import RunStore, write_run  # run format

log = logging.getLogger("grokviz.sources")

PAPER_RUN_ID = "paper"  # folder name runs/paper/
PAPER_ANALYSIS_INDEX = 400  # state_dicts[400] (epoch 40,000) is the model analysed in the paper and ARENA


def sha256_of(path: Path) -> str:
    """Stream a file through SHA-256 in 1 MiB blocks (constant memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def paper_checkpoint_path(ctx: PipelineContext) -> Path:
    """Local path of the downloaded checkpoint file."""
    return ctx.cfg.data_dir / "external" / "paper_run" / ctx.cfg.paper_hf_file


def paper_pdf_paths(ctx: PipelineContext) -> dict[str, Path]:
    """Local paths of the reference PDFs (keyed by citation key used in content/quotes.yaml)."""
    folder = ctx.cfg.data_dir / "external" / "papers"
    return {"nanda2023": folder / "2301.05217v3.pdf", "power2022": folder / "2201.02177v1.pdf"}


def acquire_paper_sources(ctx: PipelineContext) -> None:
    """Phase: download (if missing) and hash-check the paper checkpoint; download reference PDFs."""
    cfg = ctx.cfg
    target = paper_checkpoint_path(ctx)
    if ctx.force or not target.exists():
        log.info("downloading %s/%s @ %s", cfg.paper_hf_repo, cfg.paper_hf_file, cfg.paper_hf_revision)
        hf_hub_download(cfg.paper_hf_repo, cfg.paper_hf_file, revision=cfg.paper_hf_revision, local_dir=target.parent)
    digest = sha256_of(target)  # integrity + exact-version check
    if digest != cfg.paper_sha256:
        raise RuntimeError(f"sha256 mismatch for {target}: {digest} != {cfg.paper_sha256}")
    log.info("checkpoint ok: %s (%.1f MB, sha256 %s…)", target, target.stat().st_size / 1e6, digest[:12])
    urls = {"nanda2023": cfg.paper_pdf_url, "power2022": cfg.power_pdf_url}
    for key, path in paper_pdf_paths(ctx).items():
        if ctx.force or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            log.info("downloading %s → %s", urls[key], path)
            request = urllib.request.Request(urls[key], headers={"User-Agent": "grokviz (research; quote check)"})
            with urllib.request.urlopen(request, timeout=120) as resp:  # arXiv asks for an identifying UA
                path.write_bytes(resp.read())
        log.info("pdf %s: %s (%.1f MB)", key, path, path.stat().st_size / 1e6)


def convert_paper_run(ctx: PipelineContext) -> None:
    """Phase: convert full_run_data.pth into runs/paper/ (stacked safetensors, curves.npz, meta.json)."""
    cfg = ctx.cfg
    run_dir = cfg.runs_dir / PAPER_RUN_ID
    if (run_dir / "meta.json").exists() and not ctx.force:
        log.info("runs/%s already converted (use --force to redo)", PAPER_RUN_ID)
        return
    data = torch.load(paper_checkpoint_path(ctx), map_location="cpu", weights_only=True)  # plain tensors only
    state_dicts = data["state_dicts"]  # 500 OrderedDicts, epochs 0, 100, ..., 49,900
    stacked = {k: torch.stack([sd[k] for sd in state_dicts]) for k in state_dicts[0]}  # [500, ...] per parameter
    train = np.asarray(data["train_losses"], dtype=np.float64)  # 50,000 values: loss after j steps
    test = np.asarray(data["test_losses"], dtype=np.float64)  # 5,000 values: test loss after 10·j steps
    curves = {
        "train_loss": train,
        "train_loss_steps": np.arange(len(train)),
        "test_loss": test,
        "test_loss_steps": np.arange(len(test)) * 10,
    }
    meta = {
        "run_id": PAPER_RUN_ID,
        "kind": "paper",
        "title": "Paper mainline run (Nanda et al. 2023)",
        "fn_name": "add",
        "split_seed": int(data["config"]["seed"]),  # python random seed of the train/test shuffle
        "init_seed": None,  # the paper's code did not seed the initialisation
        "frac_train": float(data["config"]["frac_train"]),
        "lr": float(data["config"]["lr"]),
        "weight_decay": float(data["config"]["weight_decay"]),
        "betas": [0.9, 0.98],
        "warmup_steps": 10,
        "num_steps": int(data["config"]["num_epochs"]),
        "ckpt_steps": [int(e) + 1 for e in data["epochs"]],  # saved after the step of epoch e → e+1 steps
        "paper_analysis_index": PAPER_ANALYSIS_INDEX,
        "model_shape": ModelShape().__dict__,
        "source": {
            "hf_repo": cfg.paper_hf_repo,
            "hf_file": cfg.paper_hf_file,
            "hf_revision": cfg.paper_hf_revision,
            "sha256": cfg.paper_sha256,
            "raw_config": data["config"],
            "note": "The file's separate 'model' entry (final weights) is not used; analyses use state_dicts.",
        },
    }
    write_run(run_dir, meta, stacked, curves)
    log.info("wrote %s: %d checkpoints, steps %d..%d", run_dir, len(state_dicts), meta["ckpt_steps"][0], meta["ckpt_steps"][-1])


def validate_paper_run(ctx: PipelineContext) -> None:
    """Phase: recompute losses from checkpoints and compare with the logged curves; write analysis/validation.json."""
    cfg = ctx.cfg
    store = RunStore(cfg.runs_dir / PAPER_RUN_ID)
    device = resolve_device(cfg.device)
    tokens = task.all_pairs(cfg.p).to(device)
    labels = task.make_labels(cfg.p).to(device)
    split = task.paper_split(cfg.p, cfg.frac_train, store.meta["split_seed"])
    is_train, is_test = split.is_train.to(device), split.is_test.to(device)
    curves = store.curves()
    checks = []
    for t in [1, 5, 14, 20, 40, 60, 80, 100, 400, 499]:  # spans memorisation, plateau, grokking, late training
        step = store.ckpt_steps[t]
        with torch.no_grad():
            logits = store.load_model(t, device)(tokens)[:, -1]  # [p², 114] at the '=' position
        train_loss = losses.grid_loss(logits, labels, is_train).item()  # 114-way: the training objective
        logged = float(curves["train_loss"][step])
        ok = math.isclose(train_loss, logged, rel_tol=1e-3)  # float32 vs float64 rounding only
        checks.append({"index": t, "step": step, "train_loss_recomputed": train_loss, "train_loss_logged": logged, "ok": ok})
        log.info("ckpt %3d (step %5d): train loss recomputed %.6e vs logged %.6e → %s", t, step, train_loss, logged, "ok" if ok else "MISMATCH")
    with torch.no_grad():
        logits = store.load_model(PAPER_ANALYSIS_INDEX, device)(tokens)[:, -1, :-1]  # 113-way: '=' output dropped
    all_loss = losses.grid_loss(logits, labels).item()  # the paper's "2.41e-7" is this all-pairs loss
    test_loss = losses.grid_loss(logits, labels, is_test).item()
    report = {
        "checks": checks,
        "paper_reported_loss": 2.41e-7,
        "epoch40k_loss_all_pairs_113": all_loss,
        "epoch40k_loss_test_113": test_loss,
        "epoch40k_matches_paper": math.isclose(all_loss, 2.41e-7, rel_tol=0.01),
        "step_semantics": "ckpt 'epoch e' = params after e+1 steps; train_losses[j] after j steps; test_losses[j] after 10j steps",
    }
    out = store.dir / "analysis" / "validation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("epoch-40k loss: all pairs %.4e (paper 2.41e-7), test only %.4e", all_loss, test_loss)
    if not all(c["ok"] for c in checks) or not report["epoch40k_matches_paper"]:
        raise RuntimeError(f"paper run validation failed; see {out}")
