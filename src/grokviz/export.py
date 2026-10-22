"""Phase `export_site_data`: turn analysis outputs into compact binary arrays + a manifest for the website.

Local explanation: every array the site plots is written as a little-endian binary file under
`site/data/runs/<run_id>/<name>.bin` with an entry in `site/data/manifest.json` describing its dtype, shape,
optional linear quantisation (uint8 with min/max) and a one-line description; numbers that appear in the
prose go to `site/data/summary.json`. Weights for the in-browser model are exported as float16 vectors in the
model's state_dict order (layout in the manifest): densely for the two showcase runs (paper, seed 0) and, for every
other run, at the checkpoints chosen by `snapshot_steps` (dense through cleanup, sparse afterwards), so every run can
be animated in the browser.
Global context: the website is static, so all heavy computation happens here once; the browser only fetches
and draws (or runs the tiny model for per-input views). The data folder is published to a Hugging Face dataset
and downloaded by the Pages deploy (docs/deployment.md).
Sources: numpy tofile (https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html);
IEEE-754 half precision (https://en.wikipedia.org/wiki/Half-precision_floating-point_format).
"""

from __future__ import annotations  # postponed annotations

import bisect  # nearest checkpoint to a target step
import hashlib  # file digests
import json  # manifest
import logging  # module logger
import shutil  # clean output folder
from pathlib import Path  # paths

import numpy as np  # arrays
import torch  # weights

from grokviz import fourier, task  # basis names, split masks
from grokviz.checkpoint_metrics import run_dirs  # selected runs
from grokviz.context import PipelineContext  # phase argument
from grokviz.geometry import HERO_RUNS  # runs exported in full detail
from grokviz.runstore import RunStore  # run access
from grokviz.summary import build_summary  # prose numbers

log = logging.getLogger("grokviz.export")
SCHEMA_VERSION = 1

SCALAR_COLUMNS = [  # per-checkpoint scalars packed into one [T, S] array (order = manifest column names)
    "loss_train", "loss_test", "loss_all", "acc_train", "acc_test", "restricted_loss", "restricted_loss_train",
    "restricted_loss_test", "restricted_acc_test", "excluded_loss", "excluded_loss_max", "logit_fve", "gini_we",
    "gini_wl", "wl_resid_sq_frac", "ssw_total", "frac_neurons_085", "trig_frac_pre", "trig_frac_post", "trig_frac_logit",
]


class Writer:
    """Accumulates array files + manifest entries for one run."""

    def __init__(self, root: Path, run_id: str):
        self.dir = root / "runs" / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.arrays: dict[str, dict] = {}

    def add(self, name: str, data, desc: str, dtype: str = "f32", **extra) -> None:
        """Write `data` as <name>.bin in dtype f32 | f16 | i16 | u8q (uint8 linearly quantised between min and max)."""
        a = np.asarray(data, dtype=np.float64)
        entry = {"file": f"runs/{self.run_id}/{name}.bin", "dtype": dtype, "shape": list(a.shape), "desc": desc, **extra}
        if dtype == "f32":
            raw = a.astype("<f4").tobytes()
        elif dtype == "f16":
            raw = a.astype("<f2").tobytes()
        elif dtype == "i16":
            raw = a.astype("<i2").tobytes()
        elif dtype == "u8q":  # value ≈ min + q/255·(max − min); 0.4 % resolution is plenty for colour maps
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
            q = np.round((a - lo) / ((hi - lo) or 1.0) * 255).clip(0, 255).astype(np.uint8)
            raw = q.tobytes()
            entry.update({"min": lo, "max": hi})
        else:
            raise ValueError(dtype)
        (self.dir / f"{name}.bin").write_bytes(raw)
        entry["bytes"] = len(raw)
        entry["sha256"] = hashlib.sha256(raw).hexdigest()[:16]
        self.arrays[name] = entry


def weight_vector(state: dict[str, torch.Tensor], names: list[str]) -> np.ndarray:
    """Concatenate parameters in `names` order into one flat float vector."""
    return np.concatenate([state[n].detach().cpu().numpy().reshape(-1) for n in names])


def snapshot_steps(ckpt_steps: list[int], t3: int | None, fine: int, coarse: int, margin: int = 2000) -> list[int]:
    """Indices of the checkpoints whose weights are exported for a non-showcase run: one every `fine` steps up to
    t3 + `margin` (memorisation, circuit formation and cleanup, where the network changes most), then one every `coarse`
    steps; only the coarse grid when no t3 was detected (runs that never grok); the final checkpoint always, and only it
    when both spacings are 0."""
    last = ckpt_steps[-1]
    targets: set[int] = set()
    if fine and t3 is not None:
        targets |= set(range(0, min(last, t3 + margin) + 1, fine))
    if coarse:
        targets |= set(range(0, last + 1, coarse))

    def nearest(s: int) -> int:  # index of the stored checkpoint closest to step s (ckpt_steps is sorted)
        i = bisect.bisect_left(ckpt_steps, s)
        return min((j for j in (i - 1, i) if 0 <= j < len(ckpt_steps)), key=lambda j: abs(ckpt_steps[j] - s))

    return sorted({nearest(s) for s in targets} | {len(ckpt_steps) - 1})


def export_run(root: Path, store: RunStore, p: int, weight_every: int | None, snapshots: tuple[int, int] = (0, 0)) -> dict:
    """Export one run; return its manifest entry. Weights: every `weight_every`-th checkpoint (showcase runs), else
    the checkpoints `snapshot_steps` picks with the (fine, coarse) spacing `snapshots`."""
    w = Writer(root, store.run_id)
    a = store.dir / "analysis"
    m = np.load(a / "ckpt_metrics.npz")
    ph = json.loads((a / "phases.json").read_text(encoding="utf-8")) if (a / "phases.json").exists() else {}
    fin = json.loads((a / "final.json").read_text(encoding="utf-8")) if (a / "final.json").exists() else None
    c = store.curves()
    # Per-step curves (subsampled to every 10 steps to keep files small).
    tr_steps = c.get("train_loss_steps", np.arange(len(c["train_loss"])))
    sel = tr_steps % 10 == 0
    w.add("curve_train_loss", c["train_loss"][sel], "train loss (training objective) every 10 steps", step0=0, step_every=10)
    te_steps = c.get("test_loss_steps", np.arange(len(c["test_loss"])) * 10)  # paper run: logged every 10 epochs
    every = max(10, int(te_steps[1] - te_steps[0])) if len(te_steps) > 1 else 10  # coarser if GROKVIZ_TEST_EVERY > 10
    te_sel = te_steps % every == 0
    w.add("curve_test_loss", c["test_loss"][te_sel], f"test loss every {every} steps", step0=0, step_every=every)
    # Per-checkpoint metrics.
    w.add("ckpt_scalars", np.stack([m[k] for k in SCALAR_COLUMNS], 1), "per-checkpoint scalar metrics", columns=SCALAR_COLUMNS)
    w.add("excl_per_freq", m["excl_per_freq"], "excluded loss (train) removing one frequency k=1..56")
    w.add("cos_coeffs", m["cos_coeffs"], "coefficient of cos(w_k(a+b-c)) in the logits, k=1..56")
    w.add("we_fourier", m["we_fourier"], "norm of each Fourier component of W_E (113: Const, cos1, sin1, ...)")
    w.add("wl_fourier", m["wl_fourier"], "norm of each Fourier component of W_L")
    w.add("we_svals", m["we_svals"], "singular values of W_E (d x p)")
    w.add("ssw", m["ssw"], "sum of squared weights per parameter", columns=[str(x) for x in m["param_names"]])
    w.add("avg_attn", m["avg_attn"], "mean attention from '=' to (a, b, =) per head")
    w.add("attn_C", m["attn_C"], "attention score lookup C_j[t] per head (paper App. C.1.3)", dtype="f16")
    w.add("attn_C_fourier", m["attn_C_fourier"], "attention lookup C_j in the Fourier basis", dtype="f16")
    w.add("ov_fourier", m["ov_fourier"], "Fourier norms of each head's OV output", dtype="f16")
    w.add("neuron_freq", m["neuron_freq"], "best frequency of each neuron", dtype="i16")
    for k in ("neuron_fve", "neuron_quad_post", "neuron_quad_pre", "frac_active"):
        w.add(k, m[k], f"{k} per neuron (0..1)", dtype="u8q")
    # Gradient dynamics.
    if (a / "gradients.npz").exists():
        g = np.load(a / "gradients.npz")
        w.add("grad_norm", g["grad_norm"], "norm of the train-loss gradient per parameter", columns=[str(x) for x in g["param_names"]])
        w.add("weight_norm", g["weight_norm"], "norm of each parameter")
        w.add("grad_scalars", np.stack([g["cos_grad_weight"], g["cos_train_test"], g["grad_ratio_decay"]], 1),
              "cos(g,θ), cos(g_train,g_test), |g|/(λ|θ|)", columns=["cos_grad_weight", "cos_train_test", "grad_over_decay"])
        w.add("grad_we_fourier", g["grad_we_fourier"], "Fourier norms of the W_E gradient", dtype="f16")
        if "adam_steps" in g.files:  # our runs only: Adam's direction from the saved moments
            w.add("adam_scalars", np.stack([g["cos_adam_grad"], g["cos_adam_weight"], g["adam_over_decay"]], 1),
                  "cos(Adam direction, g), cos(Adam direction, θ), |Adam direction|/(λ|θ|)",
                  columns=["cos_adam_grad", "cos_adam_weight", "adam_over_decay"], steps=g["adam_steps"].tolist())
    if "update_norm" in c:  # our runs only: size of each parameter's actual AdamW update, logged during training
        w.add("update_norm", c["update_norm"], "norm of each parameter's AdamW update", columns=[str(x) for x in store.meta["param_names"]],
              step0=0, step_every=int(c["norms_steps"][1] - c["norms_steps"][0]))
    # Geometry frames.
    geo_steps = spectra_steps = None
    if (a / "geometry.npz").exists():
        geo = np.load(a / "geometry.npz")
        geo_steps = geo["steps"].tolist()
        for k in ("we_planes", "we_pca", "mlp_planes", "logit_pca", "prob_pca"):
            w.add(f"geo_{k}", geo[k], f"geometry: {k}", dtype="f16")
        for k in ("we_pca_var", "logit_pca_var", "prob_pca_var"):
            w.add(f"geo_{k}", geo[k], f"explained variance of {k[:-4]}")
        w.add("geo_p_correct", geo["p_correct"], "probability of the correct answer per input", dtype="u8q")
        w.add("geo_entropy", geo["entropy"], "normalised entropy of the prediction per input", dtype="u8q")
        if "spectra_steps" in geo.files and len(geo["spectra_steps"]):  # 2D Fourier maps (log10 of the share of the max)
            spectra_steps = geo["spectra_steps"].tolist()
            for k, desc in (("logit_fourier2d", "norm of each 2D Fourier component of the logits"),
                            ("neuron_b2", "mean squared 2D Fourier coefficient over neurons")):
                x = geo[k] / geo[k].max(axis=(1, 2), keepdims=True)  # per frame, relative to its largest component
                w.add(f"spec_{k}", np.log10(x + 1e-6), f"{desc} (log10, relative to the frame's maximum)", dtype="u8q")
            w.add("spec_neuron_freq", geo["neuron_freq_energy"], "share of each neuron's variance per frequency k=1..56", dtype="u8q")
    # Weights for the in-browser model.
    names = list(store.state_dict(0).keys())
    shapes = {k: list(v.shape) for k, v in store.state_dict(0).items()}
    weight_steps = []
    n = len(store.ckpt_steps)
    idx = sorted(set(range(0, n, weight_every)) | {n - 1}) if weight_every else snapshot_steps(store.ckpt_steps, ph.get("t3"), *snapshots)
    for t in idx:
        w.add(f"w_{store.ckpt_steps[t]}", weight_vector(store.state_dict(t), names), f"weights at step {store.ckpt_steps[t]}", dtype="f16")
        weight_steps.append(store.ckpt_steps[t])
    return {
        "id": store.run_id, "title": store.meta.get("title", store.run_id), "kind": store.meta["kind"],
        "group": store.meta.get("group", "paper"), "split_seed": store.meta["split_seed"], "init_seed": store.meta.get("init_seed"),
        "weight_decay": store.meta["weight_decay"], "fn_name": store.meta.get("fn_name", "add"), "label_seed": store.meta.get("label_seed"),
        "key_freqs": [int(k) for k in m["key_freqs"]], "ckpt_steps": [int(s) for s in m["steps"]], "geometry_steps": geo_steps, "spectra_steps": spectra_steps,
        "weight_steps": weight_steps, "param_layout": [{"name": n, "shape": shapes[n]} for n in names],
        "phases": ph, "final": fin, "arrays": w.arrays,
    }


def export_site_data(ctx: PipelineContext) -> None:
    """Phase entry point: write site/data (manifest, summary, arrays) for every analysed run."""
    root = ctx.cfg.site_dir / "data"
    if root.exists() and ctx.force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    for run_dir in run_dirs(ctx, all_runs=True):  # the manifest lists every run, whatever --runs says
        if not (run_dir / "analysis" / "ckpt_metrics.npz").exists():
            continue
        store = RunStore(run_dir)
        weight_every = (2 if store.run_id == "paper" else 5) if store.run_id in HERO_RUNS else None  # showcase runs: dense
        snapshots = tuple((ctx.cfg.site_weight_snapshots + [0, 0])[:2])  # (fine, coarse) spacing for the other runs
        runs.append(export_run(root, store, ctx.cfg.p, weight_every, snapshots))
        log.info("exported %s (%d arrays)", store.run_id, len(runs[-1]["arrays"]))
    _, names = fourier.fourier_basis(ctx.cfg.p)
    masks = {}
    for seed in sorted({r["split_seed"] for r in runs}):  # train/test mask per split seed, 1 byte per (a, b)
        mask = task.paper_split(ctx.cfg.p, ctx.cfg.frac_train, seed).is_train.numpy().astype(np.uint8)
        (root / f"train_mask_seed{seed}.bin").write_bytes(mask.tobytes())
        masks[str(seed)] = f"train_mask_seed{seed}.bin"
    manifest = {"schema_version": SCHEMA_VERSION, "p": ctx.cfg.p, "basis_names": names, "train_masks": masks,
                "scalar_columns": SCALAR_COLUMNS, "runs": runs}
    (root / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":"), default=float), encoding="utf-8")
    summary = build_summary(ctx.cfg.runs_dir, ctx.cfg.reports_dir, ctx.cfg.p, ctx.cfg.frac_train)  # numbers quoted in the prose
    (root / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    log.info("site data: %d runs, %.1f MB in %s", len(runs), total / 1e6, root)
