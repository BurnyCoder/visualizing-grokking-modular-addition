"""Configuration: one frozen `Config` object built from `.env.example` ← `.env` ← real environment.

Local explanation: `load_config` reads key/value pairs from `.env.example` (documented defaults),
overlays the user's private `.env`, then overlays real environment variables, and parses each value
into a typed dataclass field.
Global context: every pipeline phase receives this object, so all tunable numbers (modulus, model
sizes, optimiser settings, seed lists, pinned download revisions, output folders) live in exactly one
place — `.env.example` — and are documented there. This module deliberately does not import torch so
lightweight phases (site building) stay fast.
Sources: python-dotenv `dotenv_values` (https://pypi.org/project/python-dotenv/#other-use-cases);
dataclasses (https://docs.python.org/3/library/dataclasses.html).
"""

from __future__ import annotations  # allow forward references in type hints (PEP 563 semantics)

import os  # read real environment variables, which take precedence over files
from dataclasses import dataclass, field, fields  # typed, immutable configuration container
from pathlib import Path  # OS-independent path handling
from typing import Callable  # type of the per-field parser functions

from dotenv import dotenv_values  # parses KEY=VALUE files without mutating os.environ

# Repository root: this file is src/grokviz/config.py, so the root is two directories above `src`.
REPO_ROOT = Path(__file__).resolve().parents[2]
# File holding the documented defaults (tracked in git); the single source of default values.
DEFAULTS_FILE = REPO_ROOT / ".env.example"
# File holding the user's private overrides (gitignored).
USER_ENV_FILE = REPO_ROOT / ".env"


def parse_seed_list(spec: str) -> list[int]:
    """Parse "0-2,5" into [0, 1, 2, 5]: comma-separated items, each an int or an inclusive range."""
    seeds: list[int] = []  # accumulate seeds in the order written
    for item in (part.strip() for part in spec.split(",")):  # split on commas, trim spaces
        if not item:  # empty spec or trailing comma contributes nothing
            continue
        if "-" in item:  # "a-b" means every integer from a to b inclusive
            start, end = (int(x) for x in item.split("-", 1))
            seeds.extend(range(start, end + 1))
        else:  # a single integer seed
            seeds.append(int(item))
    return seeds


def _path(value: str) -> Path:
    """Resolve a directory setting: relative paths are taken relative to the repository root."""
    path = Path(value)  # may be absolute (user override) or relative (default)
    return path if path.is_absolute() else REPO_ROOT / path


def _setting(env_key: str, parse: Callable[[str], object]) -> object:
    """Declare a dataclass field bound to an environment key and a string→value parser."""
    # The metadata travels with the field so load_config can find the key and parser generically.
    return field(metadata={"env": env_key, "parse": parse})


@dataclass(frozen=True)
class Config:
    """All pipeline settings; one field per GROKVIZ_* key in `.env.example` (see comments there for meaning).
    PYTHONUTF8, CUBLAS_WORKSPACE_CONFIG and HF_TOKEN are process environment, not fields."""

    # --- runtime ---
    device: str = _setting("GROKVIZ_DEVICE", str)  # "auto" | "cuda" | "cpu"
    # --- output folders (gitignored) ---
    runs_dir: Path = _setting("GROKVIZ_RUNS_DIR", _path)  # one sub-folder per training run
    data_dir: Path = _setting("GROKVIZ_DATA_DIR", _path)  # downloads (paper checkpoint, PDF)
    site_dir: Path = _setting("GROKVIZ_SITE_DIR", _path)  # built static website
    logs_dir: Path = _setting("GROKVIZ_LOGS_DIR", _path)  # timestamped log files
    reports_dir: Path = _setting("GROKVIZ_REPORTS_DIR", _path)  # experiment reports (tracked)
    # --- task and model (paper Sec. 3 / App. A) ---
    p: int = _setting("GROKVIZ_P", int)  # prime modulus P
    frac_train: float = _setting("GROKVIZ_FRAC_TRAIN", float)  # fraction of the P*P pairs used for training
    d_model: int = _setting("GROKVIZ_D_MODEL", int)  # residual-stream width d
    d_mlp: int = _setting("GROKVIZ_D_MLP", int)  # number of MLP neurons n
    n_heads: int = _setting("GROKVIZ_N_HEADS", int)  # attention heads
    d_head: int = _setting("GROKVIZ_D_HEAD", int)  # query/key/value size per head
    n_ctx: int = _setting("GROKVIZ_N_CTX", int)  # sequence length ("a b =")
    # --- optimisation ---
    lr: float = _setting("GROKVIZ_LR", float)  # AdamW learning rate γ
    weight_decay: float = _setting("GROKVIZ_WEIGHT_DECAY", float)  # AdamW decoupled weight decay λ
    beta1: float = _setting("GROKVIZ_BETA1", float)  # Adam first-moment decay
    beta2: float = _setting("GROKVIZ_BETA2", float)  # Adam second-moment decay
    warmup_steps: int = _setting("GROKVIZ_WARMUP_STEPS", int)  # linear LR warm-up length
    num_steps: int = _setting("GROKVIZ_NUM_STEPS", int)  # full-batch optimiser steps (= epochs)
    ckpt_every: int = _setting("GROKVIZ_CKPT_EVERY", int)  # parameter snapshot interval
    optim_every: int = _setting("GROKVIZ_OPTIM_EVERY", int)  # Adam-moment snapshot interval
    norms_every: int = _setting("GROKVIZ_NORMS_EVERY", int)  # per-parameter norm logging interval
    test_every: int = _setting("GROKVIZ_TEST_EVERY", int)  # test loss/accuracy logging interval
    # --- run matrix ---
    wd1_seeds: list = _setting("GROKVIZ_WD1_SEEDS", parse_seed_list)  # confirmatory runs (λ = 1)
    wd0_seeds: list = _setting("GROKVIZ_WD0_SEEDS", parse_seed_list)  # control: no weight decay
    rand_seeds: list = _setting("GROKVIZ_RAND_SEEDS", parse_seed_list)  # control: random labels
    split0_init_seeds: list = _setting("GROKVIZ_SPLIT0_INIT_SEEDS", parse_seed_list)  # paper split, new init
    # --- paper sources (pinned) ---
    paper_hf_repo: str = _setting("GROKVIZ_PAPER_HF_REPO", str)
    paper_hf_file: str = _setting("GROKVIZ_PAPER_HF_FILE", str)
    paper_hf_revision: str = _setting("GROKVIZ_PAPER_HF_REVISION", str)
    paper_sha256: str = _setting("GROKVIZ_PAPER_SHA256", str)
    paper_pdf_url: str = _setting("GROKVIZ_PAPER_PDF_URL", str)  # Nanda et al. 2023 (quote checking only)
    power_pdf_url: str = _setting("GROKVIZ_POWER_PDF_URL", str)  # Power et al. 2022 (quote checking only)
    # --- Hugging Face publishing ---
    hf_runs_repo: str = _setting("GROKVIZ_HF_RUNS_REPO", str)
    hf_site_data_repo: str = _setting("GROKVIZ_HF_SITE_DATA_REPO", str)
    hf_collection_title: str = _setting("GROKVIZ_HF_COLLECTION_TITLE", str)
    # --- website data ---
    site_weight_snapshots: list = _setting("GROKVIZ_SITE_WEIGHT_SNAPSHOTS", parse_seed_list)  # [fine, coarse] spacing (steps) of exported weights


def export_process_env(env_file: Path | None = None) -> None:
    """Copy non-GROKVIZ_ keys (PYTHONUTF8, CUBLAS_WORKSPACE_CONFIG, HF_TOKEN, ...) into os.environ.

    Libraries such as torch/cuBLAS and huggingface_hub read these from the process environment, so
    they must be exported before those libraries initialise. Existing shell values are never replaced,
    and empty values (e.g. a blank HF_TOKEN) are skipped so they cannot mask a cached login.
    """
    values: dict[str, str | None] = dict(dotenv_values(DEFAULTS_FILE))  # documented defaults
    user_file = USER_ENV_FILE if env_file is None else env_file  # private overrides
    if user_file.exists():
        values.update(dotenv_values(user_file))
    for key, value in values.items():
        if not key.startswith("GROKVIZ_") and value:  # GROKVIZ_ keys are read by load_config instead
            os.environ.setdefault(key, value)  # setdefault: the real shell environment wins


def load_config(env_file: Path | None = None) -> Config:
    """Build a `Config` from defaults (.env.example), then `env_file` (default: .env), then os.environ."""
    values: dict[str, str | None] = dict(dotenv_values(DEFAULTS_FILE))  # documented defaults
    user_file = USER_ENV_FILE if env_file is None else env_file  # private overrides, if any
    if user_file.exists():  # a missing .env is normal (fresh clone, CI)
        values.update(dotenv_values(user_file))
    values.update({k: v for k, v in os.environ.items() if k.startswith("GROKVIZ_")})  # shell wins
    kwargs = {}  # parsed constructor arguments, keyed by dataclass field name
    for f in fields(Config):  # every declared setting must be present somewhere
        key, parse = f.metadata["env"], f.metadata["parse"]
        if values.get(key) is None:  # catches a key deleted from .env.example by mistake
            raise KeyError(f"missing setting {key}; add it to .env.example")
        kwargs[f.name] = parse(values[key])  # string → typed value
    return Config(**kwargs)
