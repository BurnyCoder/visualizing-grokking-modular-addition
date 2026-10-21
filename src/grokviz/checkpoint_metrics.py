"""Phase `compute_checkpoint_metrics`: every paper/ARENA measurement at every checkpoint of every run.

Local explanation: for each run folder, fixes the run's key frequencies K (paper run: from its epoch-40k
model, as in the paper; our runs: from their final checkpoint), then loops over all checkpoints, evaluates
the model on the full input grid once, and records losses/accuracies, restricted/excluded loss (+ per
frequency), cosine coefficients, Fourier norms and Gini of W_E/W_L, singular values of W_E, per-parameter
sum of squared weights, neuron frequencies/FVE/quadratic shares, trig-variance fractions, average attention,
attention lookup tables and OV spectra. Output: `runs/<id>/analysis/ckpt_metrics.npz` (+ key_freqs.json).
Global context: these arrays are the "as it is trained" curves of the website (chapters 17–18, dashboard
groups A–C) and the inputs to phase detection and the hypothesis tests.
Sources: per-metric sources in grokviz.metrics; ARENA `get_metrics` loop pattern
(https://learn.arena.education/chapter1_transformer_interp/32_grokking/3-analysis-during-training/).
"""

from __future__ import annotations  # postponed annotations

import json  # key_freqs.json
import logging  # module logger
import time  # progress timing

import numpy as np  # output arrays
import torch  # tensors

from grokviz import fourier, losses, metrics  # measurements
from grokviz.context import PipelineContext  # phase argument
from grokviz.envcheck import resolve_device  # device
from grokviz.runstore import RunStore, save_analysis  # run access; atomic analysis writes

log = logging.getLogger("grokviz.checkpoint_metrics")


def run_dirs(ctx: PipelineContext, all_runs: bool = False) -> list:
    """Finished run folders (with meta.json) selected by --runs; `all_runs` ignores --runs for phases whose output
    aggregates every run (hypothesis tests, site export), so a partial selection can never overwrite them."""
    if not ctx.cfg.runs_dir.exists():
        return []
    if all_runs and ctx.runs:
        log.warning("--runs %s ignored: this phase always aggregates every run", ",".join(ctx.runs))
    return [d for d in sorted(ctx.cfg.runs_dir.iterdir()) if (d / "meta.json").exists() and (all_runs or ctx.wants(d.name))]


def grid_for(store: RunStore, p: int, device) -> metrics.Grid:
    """Evaluation grid matching the run's split seed and task (random labels for the 'rand' control)."""
    m = store.meta
    return metrics.Grid.build(p, m["frac_train"], m["split_seed"], device, m.get("fn_name", "add"), m.get("label_seed"))


def reference_index(store: RunStore) -> int:
    """Checkpoint whose W_L defines K: the paper's analysis checkpoint for the paper run, else the final one."""
    return store.meta.get("paper_analysis_index", len(store.ckpt_steps) - 1)


@torch.no_grad()
def measure_checkpoint(model, grid: metrics.Grid, K: list[int]) -> dict[str, np.ndarray]:
    """All per-checkpoint quantities for one model (scalars and small vectors)."""
    acts = metrics.evaluate(model, grid)
    L, y, F = acts.logits, grid.labels, grid.F
    we = metrics.we_fourier_norms(model, F)
    wl = metrics.wl_fourier_norms(model, F)
    mats = metrics.paper_matrices(model, grid.p)
    W_E, W_L = mats["W_E"], mats["W_L"]
    excl = metrics.excluded_loss_per_freq(L, grid)
    nf_post = metrics.neuron_frequencies(metrics.neuron_fourier(acts.post, F))
    nf_pre = metrics.neuron_frequencies(metrics.neuron_fourier(acts.pre, F))
    _, logit_fve, _ = metrics.logit_cos_fit(L, grid, K)
    ssw = metrics.sum_sq_weights(model)
    C = metrics.attention_lookup(model, grid.p)
    wl_resid = metrics.wl_residual_sq_frac(W_L, fourier.key_rows(F, K))
    out = {
        "loss_train": losses.grid_loss(L, y, grid.is_train).item(),
        "loss_test": losses.grid_loss(L, y, grid.is_test).item(),
        "loss_all": losses.grid_loss(L, y).item(),
        "acc_train": losses.accuracy(L, y, grid.is_train),
        "acc_test": losses.accuracy(L, y, grid.is_test),
        "restricted_loss": metrics.restricted_loss(L, grid, K),
        "restricted_loss_train": metrics.restricted_loss(L, grid, K, grid.is_train),
        "restricted_loss_test": metrics.restricted_loss(L, grid, K, grid.is_test),
        "restricted_acc_test": losses.accuracy(metrics.restricted_logits(L, F, K), y, grid.is_test),
        "excluded_loss": metrics.excluded_loss(L, grid, K),
        "excluded_loss_max": float(excl.max()),
        "excl_per_freq": excl.numpy(),
        "cos_coeffs": metrics.cos_coefficients(L, grid).cpu().numpy(),
        "logit_fve": logit_fve,
        "we_fourier": we.cpu().numpy(),
        "wl_fourier": wl.cpu().numpy(),
        "we_svals": torch.linalg.svdvals(W_E).cpu().numpy(),
        "gini_we": fourier.gini(we),
        "gini_wl": fourier.gini(wl),
        "wl_resid_sq_frac": wl_resid,
        "ssw": np.array(list(ssw.values())),
        "ssw_total": float(sum(ssw.values())),
        "neuron_freq": nf_post[0].cpu().numpy().astype(np.int16),
        "neuron_fve": nf_post[1].cpu().numpy(),
        "neuron_quad_post": nf_post[2].cpu().numpy(),
        "neuron_quad_pre": nf_pre[2].cpu().numpy(),
        "neuron_fve_pre": nf_pre[1].cpu().numpy(),
        "frac_active": (acts.pre > 0).double().mean(0).cpu().numpy(),
        "frac_neurons_085": float((nf_post[1] > metrics.NEURON_FVE_THRESHOLD).double().mean()),
        "trig_frac_pre": float(metrics.trig_fraction(acts.pre, F, K).mean()),
        "trig_frac_post": float(metrics.trig_fraction(acts.post, F, K).mean()),
        "trig_frac_logit": float(metrics.trig_fraction(L, F, K).mean()),
        "avg_attn": acts.pattern.mean(0).cpu().numpy(),  # [heads, 3]
        "attn_C": C.cpu().numpy(),  # [heads, p]
        "attn_C_fourier": (C @ F.T).cpu().numpy(),
        "ov_fourier": metrics.ov_fourier_norms(model, F).cpu().numpy(),
    }
    return out


def compute_checkpoint_metrics(ctx: PipelineContext) -> None:
    """Phase entry point: loop over runs and checkpoints; write analysis/ckpt_metrics.npz per run."""
    device = resolve_device(ctx.cfg.device)
    for run_dir in run_dirs(ctx):
        out_path = run_dir / "analysis" / "ckpt_metrics.npz"
        if out_path.exists() and not ctx.force:
            log.info("%s: ckpt_metrics.npz exists, skipping", run_dir.name)
            continue
        store = RunStore(run_dir)
        grid = grid_for(store, ctx.cfg.p, device)
        K = metrics.key_frequencies(store.load_model(reference_index(store), device), grid.F)
        log.info("%s: key frequencies %s (from checkpoint %d)", store.run_id, K, reference_index(store))
        rows: dict[str, list] = {}
        start = time.perf_counter()
        for t, step, model in store.iter_models(device):
            for key, value in measure_checkpoint(model, grid, K).items():
                rows.setdefault(key, []).append(value)
            if t % 50 == 0:
                log.info("%s: checkpoint %d/%d (step %d) test loss %.3e restricted %.3e excluded %.3e (%.0f s)", store.run_id, t,
                         len(store.ckpt_steps), step, rows["loss_test"][-1], rows["restricted_loss"][-1], rows["excluded_loss"][-1],
                         time.perf_counter() - start)
        arrays = {k: np.stack([np.asarray(v) for v in vals]) for k, vals in rows.items()}
        arrays["steps"] = np.asarray(store.ckpt_steps)
        arrays["key_freqs"] = np.asarray(K)
        arrays["param_names"] = np.asarray(list(dict(store.load_model(0).named_parameters()).keys()))
        save_analysis(out_path, **arrays)
        (run_dir / "analysis" / "key_freqs.json").write_text(json.dumps({"key_freqs": K, "reference_index": reference_index(store)}), encoding="utf-8")
        log.info("%s: wrote %s (%d checkpoints, %.0f s)", store.run_id, out_path, len(store.ckpt_steps), time.perf_counter() - start)
