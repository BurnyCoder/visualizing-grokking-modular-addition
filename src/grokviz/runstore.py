"""On-disk format for a training run (the paper's downloaded run and every run we train ourselves).

Local explanation: a run lives in `runs/<run_id>/` as
  meta.json                  — what the run is (kind, seeds, hyper-parameters, provenance, checkpoint steps),
  checkpoints.safetensors    — every parameter stacked over checkpoints: tensor "<param>" has shape [T, ...],
  curves.npz                 — per-epoch scalars (losses, accuracies, norms) as numpy arrays,
  optim.safetensors          — (our runs only) Adam moment estimates at a coarser interval,
  analysis/                  — outputs of later phases.
`RunStore` reads checkpoint t lazily (one slice per tensor) so a 500-checkpoint run never has to sit in RAM.
Global context: every analysis phase iterates over `RunStore.iter_models()`, so the paper's run and our own
runs are processed by exactly the same code.
Sources: safetensors format and `safe_open` slicing (https://huggingface.co/docs/safetensors/index);
numpy npz (https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html).
"""

from __future__ import annotations  # postponed annotations

import json  # meta.json
from pathlib import Path  # paths
from typing import Iterator  # generator return types

import numpy as np  # curves
import torch  # tensors
from safetensors import safe_open  # lazy, sliceable reads
from safetensors.torch import save_file  # write stacked tensors

from grokviz.model import GrokTransformer, ModelShape  # model to load checkpoints into


def write_run(run_dir: Path, meta: dict, stacked: dict[str, torch.Tensor], curves: dict[str, np.ndarray]) -> None:
    """Create/overwrite a run folder from metadata, stacked checkpoints [T, ...] and per-epoch curves."""
    run_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in stacked.items()}, str(run_dir / "checkpoints.safetensors"))
    np.savez_compressed(run_dir / "curves.npz", **curves)  # compressed: curves are long but smooth
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def save_analysis(path: Path, **arrays) -> None:
    """Write an analysis .npz atomically (temporary file, then rename) and create `analysis/` if needed.

    Phases skip a run whose output already exists, so a file truncated by an interrupt must never appear under
    its final name. Source: os.replace is atomic on one filesystem (https://docs.python.org/3/library/os.html#os.replace).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")  # np.savez keeps names that already end in .npz
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


class RunStore:
    """Read-only access to one run folder."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.meta: dict = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))  # run description

    @property
    def run_id(self) -> str:
        """Folder name doubles as the run identifier (e.g. "paper", "wd1_s3")."""
        return self.dir.name

    @property
    def ckpt_steps(self) -> list[int]:
        """Optimizer-step count of each stored checkpoint (index t ↔ ckpt_steps[t])."""
        return list(self.meta["ckpt_steps"])

    @property
    def shape(self) -> ModelShape:
        """Architecture recorded in meta.json."""
        return ModelShape(**self.meta["model_shape"])

    def curves(self) -> dict[str, np.ndarray]:
        """Per-epoch scalar arrays (loaded fully; they are small)."""
        with np.load(self.dir / "curves.npz") as data:
            return {k: data[k] for k in data.files}

    def state_dict(self, t: int) -> dict[str, torch.Tensor]:
        """Parameters of checkpoint index t (negative indices count from the end)."""
        t = t % len(self.ckpt_steps)  # allow -1 for "final"
        with safe_open(str(self.dir / "checkpoints.safetensors"), framework="pt") as f:
            return {name: f.get_slice(name)[t] for name in f.keys()}  # slice reads only checkpoint t from disk

    def load_model(self, t: int, device: torch.device | str = "cpu") -> GrokTransformer:
        """Instantiate the model and load checkpoint t (strict: every key must match the paper layout)."""
        model = GrokTransformer(self.shape)
        model.load_state_dict(self.state_dict(t), strict=True)
        return model.to(device).eval()

    def iter_models(self, device: torch.device | str = "cpu", every: int = 1) -> Iterator[tuple[int, int, GrokTransformer]]:
        """Yield (index, step, model) for every `every`-th checkpoint, reusing one model object."""
        model = GrokTransformer(self.shape).to(device).eval()
        with safe_open(str(self.dir / "checkpoints.safetensors"), framework="pt", device=str(device)) as f:
            names = list(f.keys())
            for t in range(0, len(self.ckpt_steps), every):
                model.load_state_dict({n: f.get_slice(n)[t] for n in names}, strict=True)
                yield t, self.ckpt_steps[t], model
