"""Phase `check_environment`: record software/hardware versions and verify the compute device works.

Local explanation: logs Python/torch/CUDA versions, the GPU name and compute capability, git commits
of this repo and of the notebooks submodule, runs a small matrix multiply on the chosen device, and
writes everything to `<logs_dir>/environment.json`.
Global context: first phase of the pipeline; its output is the provenance record that reports and
run metadata cite, and it fails fast (before hours of training) if CUDA is misconfigured.
Sources: torch.cuda API (https://pytorch.org/docs/stable/cuda.html); reproducibility notes
(https://pytorch.org/docs/stable/notes/randomness.html).
"""

from __future__ import annotations  # postponed annotations

import json  # write the environment record
import logging  # module logger
import os  # read CUBLAS_WORKSPACE_CONFIG
import platform  # OS and Python implementation details
import subprocess  # ask git for commit hashes
import sys  # Python version

import torch  # deep-learning framework whose build/device we verify

from grokviz.config import REPO_ROOT  # repository root for git queries
from grokviz.context import PipelineContext  # uniform phase argument

log = logging.getLogger("grokviz.envcheck")  # logger named after this module


def resolve_device(setting: str) -> torch.device:
    """Map the GROKVIZ_DEVICE setting to a torch.device ("auto" → CUDA if available, else CPU)."""
    if setting == "auto":  # default: use the GPU when the CUDA build and driver work
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(setting)  # explicit "cuda", "cuda:1" or "cpu"


def git_commit(path: os.PathLike | str) -> str:
    """Return the HEAD commit hash of the git checkout at `path` ("unknown" if git is unavailable)."""
    try:  # git may be missing in some minimal environments; never crash provenance logging on that
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_environment(device: torch.device) -> dict:
    """Gather versions, device facts and git commits into a JSON-serialisable dict."""
    info = {
        "python": sys.version.split()[0],  # e.g. "3.12.13"
        "platform": platform.platform(),  # OS name and version
        "torch": torch.__version__,  # e.g. "2.14.1+cu130"
        "cuda_build": torch.version.cuda,  # CUDA version torch was compiled against (None on CPU builds)
        "device": str(device),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),  # needed for determinism
        "repo_commit": git_commit(REPO_ROOT),
        "submodule_commit": git_commit(REPO_ROOT / "ai-mechanistic-interpretability-transformer-modular-addition-grokking"),
    }
    if device.type == "cuda":  # GPU-specific facts
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["gpu_capability"] = list(torch.cuda.get_device_capability(device))  # (12, 0) = Blackwell sm_120
    return info


def check_environment(ctx: PipelineContext) -> None:
    """Phase entry point: log the environment, smoke-test the device, save environment.json."""
    device = resolve_device(ctx.cfg.device)  # honour GROKVIZ_DEVICE
    info = collect_environment(device)
    for key, value in info.items():  # one log line per fact, easy to grep
        log.info("%s = %s", key, value)
    x = torch.randn(64, 64, device=device)  # tiny random matrix on the target device
    info["matmul_ok"] = bool(torch.isfinite(x @ x).all().item())  # kernel runs and returns finite numbers
    if not info["matmul_ok"]:  # a broken CUDA install can silently produce NaNs
        raise RuntimeError(f"matrix multiply on {device} produced non-finite values")
    if info["cublas_workspace_config"] is None:  # determinism needs it; warn rather than fail
        log.warning("CUBLAS_WORKSPACE_CONFIG is not set; GPU training will not be bitwise deterministic")
    out_path = ctx.cfg.logs_dir / "environment.json"  # provenance record next to the logs
    out_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.info("matmul on %s ok; wrote %s", device, out_path)
