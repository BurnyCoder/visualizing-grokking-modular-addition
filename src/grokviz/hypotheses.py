"""Phases `analyse_final_models` and `test_hypotheses`: the preregistered tests H1–H5 plus secondary statistics.

Local explanation:
- `analyse_final_models` runs, for every run's final checkpoint, the logit-frequency ablations (+ random
  controls), the Gini values in the Fourier basis vs 100 random orthonormal bases, neuron clustering, the
  logit cosine fit, the W_L residual and counterfactual patching → runs/<id>/analysis/final.json.
- `test_hypotheses` combines those with the per-checkpoint curves and phase boundaries and evaluates exactly the
  tests written in docs/preregistration.md (sign tests / Fisher test, Holm correction), plus descriptive
  secondary statistics with 95 % intervals → reports/tables/hypotheses.json and .md.
Global context: the robustness claims on the website ("this happens in every seed", error bars, controls)
and the report's statistics come from here. Nothing in this module may be changed after tag prereg-v1
without a "Deviations" entry in docs/preregistration.md.
Sources: docs/preregistration.md; Holm (1979); Fisher exact test (scipy.stats.fisher_exact).
"""

from __future__ import annotations  # postponed annotations

import json  # outputs
import logging  # module logger

import numpy as np  # arrays
import torch  # tensors

from grokviz import fourier, interventions, metrics, stats  # measurements, tests
from grokviz.checkpoint_metrics import grid_for, reference_index, run_dirs  # shared helpers
from grokviz.context import PipelineContext  # phase argument
from grokviz.envcheck import resolve_device  # device
from grokviz.patching import patch_recovery  # counterfactual swaps
from grokviz.runstore import RunStore  # run access

log = logging.getLogger("grokviz.hypotheses")


def random_basis_gini(model, F: torch.Tensor, n: int = 100, seed: int = 0) -> dict[str, list[float]]:
    """Gini of W_E / W_L component norms in n random orthonormal bases (null distribution for H3)."""
    p = F.shape[0]
    mats = metrics.paper_matrices(model, p)
    gen = torch.Generator().manual_seed(seed)
    out = {"we": [], "wl": []}
    for _ in range(n):
        Q, _ = torch.linalg.qr(torch.randn(p, p, generator=gen, dtype=torch.float64))  # Haar-like random rotation
        Q = Q.to(F.device)
        out["we"].append(fourier.gini((mats["W_E"] @ Q).norm(dim=0)))
        out["wl"].append(fourier.gini((Q.T @ mats["W_L"]).norm(dim=1)))
    return out


@torch.no_grad()
def analyse_final(store: RunStore, p: int, device) -> dict:
    """Final-checkpoint analysis used by H3/H4 and the secondary statistics (paper run: its epoch-40k model)."""
    ref = reference_index(store)  # last checkpoint, or the paper's analysis checkpoint for the paper run
    model = store.load_model(ref, device)
    init = store.load_model(0, device)
    grid = grid_for(store, p, device)
    acts = metrics.evaluate(model, grid)
    K = metrics.key_frequencies(model, grid.F)
    abl = interventions.freq_ablation_table(acts, grid, K, n_random=100)
    freqs, fve, _ = metrics.neuron_frequencies(metrics.neuron_fourier(acts.post, grid.F))
    _, logit_fve, _ = metrics.logit_cos_fit(acts.logits, grid, K)
    wlp = interventions.wl_direction_projection(model, acts, grid, K)
    null = random_basis_gini(model, grid.F)
    return {
        "key_freqs": K,
        "final_step": store.ckpt_steps[ref],
        "ablation": abl,
        "gini_we": fourier.gini(metrics.we_fourier_norms(model, grid.F)),
        "gini_wl": fourier.gini(metrics.wl_fourier_norms(model, grid.F)),
        "gini_we_init": fourier.gini(metrics.we_fourier_norms(init, grid.F)),
        "gini_wl_init": fourier.gini(metrics.wl_fourier_norms(init, grid.F)),
        "gini_null_we_p99": float(np.quantile(null["we"], 0.99)),
        "gini_null_wl_p99": float(np.quantile(null["wl"], 0.99)),
        "frac_neurons_085": float((fve > metrics.NEURON_FVE_THRESHOLD).double().mean()),
        "neuron_cluster_sizes": {str(k): int(((freqs == k) & (fve > metrics.NEURON_FVE_THRESHOLD)).sum()) for k in K},
        "logit_fve": logit_fve,
        "wl_resid_sq_frac": wlp["wl_residual_sq_frac"],
        "patching": patch_recovery(model, grid, K, n=1000, seed=0),
    }


def analyse_final_models(ctx: PipelineContext) -> None:
    """Phase entry point: write analysis/final.json for every selected run."""
    device = resolve_device(ctx.cfg.device)
    for run_dir in run_dirs(ctx):
        out = run_dir / "analysis" / "final.json"
        if out.exists() and not ctx.force:
            continue
        if not (run_dir / "analysis" / "ckpt_metrics.npz").exists() and run_dir.name != "paper":
            log.warning("%s: not analysed yet, skipping", run_dir.name)
            continue
        store = RunStore(run_dir)
        result = analyse_final(store, ctx.cfg.p, device)
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log.info("%s: K=%s gini WE %.3f WL %.3f, neurons>0.85 %.2f, patch key %.3f", store.run_id, result["key_freqs"],
                 result["gini_we"], result["gini_wl"], result["frac_neurons_085"], result["patching"]["key"]["mean"])


def _load(run_dir) -> dict:
    """Everything H1–H5 need for one run."""
    a = run_dir / "analysis"
    return {
        "id": run_dir.name,
        "meta": json.loads((run_dir / "meta.json").read_text(encoding="utf-8")),
        "m": np.load(a / "ckpt_metrics.npz"),
        "phases": json.loads((a / "phases.json").read_text(encoding="utf-8")),
        "final": json.loads((a / "final.json").read_text(encoding="utf-8")),
    }


def _drop_time(steps: np.ndarray, values: np.ndarray, t1: int, factor: float = 0.1) -> int | None:
    """First checkpoint after t1 where `values` ≤ factor × its value at t1 (None if never)."""
    i1 = int(np.searchsorted(steps, t1))
    hit = np.nonzero((values <= factor * values[i1]) & (np.arange(len(steps)) > i1))[0]
    return int(steps[hit[0]]) if len(hit) else None


def evaluate_hypotheses(runs: list[dict]) -> dict:
    """The preregistered tests (docs/preregistration.md) on loaded runs."""
    wd1 = [r for r in runs if r["meta"].get("group") == "wd1"]
    wd0 = [r for r in runs if r["meta"].get("group") == "wd0"]
    n = len(wd1)
    per_run = {}
    for r in wd1:
        m, ph, fin = r["m"], r["phases"], r["final"]
        t1, t2 = ph["t1"], ph["t2"]
        t_test = _drop_time(m["steps"], m["loss_test"], t1) if t1 else None
        t_restr = _drop_time(m["steps"], m["restricted_loss"], t1) if t1 else None
        # Δ = t_test − t_restr; if the test loss never drops 10× but the restricted loss does, Δ is +∞ (H1 holds),
        # stored as None so the JSON stays standard (H1_ok below keeps the +∞ meaning)
        delta = None if (t_restr is None or t_test is None) else t_test - t_restr
        i1 = int(np.searchsorted(m["steps"], t1)) if t1 else None
        i2 = int(np.searchsorted(m["steps"], t2)) if t2 else None
        excl_ratio = float(m["excluded_loss"][i2] / m["excluded_loss"][i1]) if (i1 is not None and i2 is not None) else None
        abl = fin["ablation"]
        keys = fin["key_freqs"]
        single = np.asarray(abl["single_freq"]) / abl["original"]
        key_min = float(min(single[k - 1] for k in keys))
        nonkey_max = float(max(single[k - 1] for k in range(1, len(single) + 1) if k not in keys))
        keep_ratio = abl["only_key_kept"] / abl["original"]
        per_run[r["id"]] = {
            "t1": t1, "t2": t2, "t_test_drop": t_test, "t_restricted_drop": t_restr, "H1_delta": delta, "H1_ok": t_restr is not None and (t_test is None or delta > 0),
            "H2_ratio": excl_ratio, "H2_ok": excl_ratio is not None and excl_ratio > 10,
            "H3_ok": (fin["gini_we"] > max(fin["gini_we_init"] + 0.3, fin["gini_null_we_p99"]))
                     and (fin["gini_wl"] > max(fin["gini_wl_init"] + 0.3, fin["gini_null_wl_p99"])),
            "H4_key_min_ratio": key_min, "H4_nonkey_max_ratio": nonkey_max, "H4_keep_ratio": keep_ratio,
            "H4_ok": key_min >= 100 and nonkey_max <= 2 and keep_ratio <= 1.5,
            "grokked": bool(m["acc_test"][-1] >= 0.99),
        }
    pvals = {f"H{i}": stats.sign_test(sum(v[f"H{i}_ok"] for v in per_run.values()), n) for i in (1, 2, 3, 4)}
    wd1_grok = sum(v["grokked"] for v in per_run.values())
    wd0_grok = sum(bool(r["m"]["acc_test"][-1] >= 0.99) for r in wd0)
    pvals["H5"] = stats.fisher_one_sided(wd1_grok, n, wd0_grok, len(wd0))
    holm = stats.holm(pvals)
    verdict = {}
    for h in ("H1", "H2", "H3", "H4"):
        fails = sum(not v[f"{h}_ok"] for v in per_run.values())
        verdict[h] = {"successes": n - fails, "n": n, "p": pvals[h], **holm[h], "supported": fails < 2 and holm[h]["p_holm"] < 0.05}
    verdict["H5"] = {"wd1_grokked": wd1_grok, "wd1_n": n, "wd0_grokked": wd0_grok, "wd0_n": len(wd0), "p": pvals["H5"], **holm["H5"],
                     "supported": wd1_grok == n and wd0_grok == 0 and holm["H5"]["p_holm"] < 0.05}
    secondary = {
        "S1_frac_neurons_085": stats.t_interval([r["final"]["frac_neurons_085"] for r in wd1]),
        "S2_logit_fve": stats.t_interval([r["final"]["logit_fve"] for r in wd1]),
        "S3_wl_resid_sq_frac": stats.t_interval([r["final"]["wl_resid_sq_frac"] for r in wd1]),
        "S4_patch_key": stats.bootstrap_interval([r["final"]["patching"]["key"]["mean"] for r in wd1]),
        "S4_patch_complement": stats.bootstrap_interval([r["final"]["patching"]["complement"]["mean"] for r in wd1]),
        "S4_patch_random": stats.bootstrap_interval([r["final"]["patching"]["random"]["mean"] for r in wd1]),
        "S5_key_freqs": {r["id"]: r["final"]["key_freqs"] for r in wd1},
        "S5_n_key_freqs": stats.t_interval([len(r["final"]["key_freqs"]) for r in wd1]),
        **{f"S6_{t}": stats.t_interval([r["phases"][t] for r in wd1 if r["phases"][t] is not None]) for t in ("t1", "t2", "t3", "t_grok")},
    }
    controls = {
        r["id"]: {
            "group": r["meta"]["group"],
            "final_train_acc": float(r["m"]["acc_train"][-1]),
            "final_test_acc": float(r["m"]["acc_test"][-1]),
            "max_excluded_loss_K_free_peak": float(np.max(r["m"]["excluded_loss_max"])),
            "gini_we": r["final"]["gini_we"], "gini_wl": r["final"]["gini_wl"],
            "gini_we_init": r["final"]["gini_we_init"], "gini_wl_init": r["final"]["gini_wl_init"],
            "key_freqs": r["final"]["key_freqs"],
        }
        for r in runs if r["meta"].get("group") in ("wd0", "rand", "split0")
    }
    return {"verdict": verdict, "per_run": per_run, "secondary": secondary, "controls": controls}


def write_markdown(result: dict, path) -> None:
    """Human-readable summary table (included by reports/exp-03)."""
    lines = ["| Hypothesis | Successes / n | p (one-sided) | Holm-adjusted p | Supported |", "|---|---|---|---|---|"]
    for h, v in result["verdict"].items():
        succ = f"{v['successes']}/{v['n']}" if "successes" in v else f"wd1 {v['wd1_grokked']}/{v['wd1_n']} vs wd0 {v['wd0_grokked']}/{v['wd0_n']}"
        lines.append(f"| {h} | {succ} | {v['p']:.3g} | {v['p_holm']:.3g} | {'yes' if v['supported'] else '**no**'} |")
    lines += ["", "| Secondary | mean | 95 % interval |", "|---|---|---|"]
    for k, v in result["secondary"].items():
        if isinstance(v, (list, tuple)) and len(v) == 3:
            lines.append(f"| {k} | {v[0]:.4g} | [{v[1]:.4g}, {v[2]:.4g}] |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_hypotheses(ctx: PipelineContext) -> None:
    """Phase entry point: evaluate H1–H5 on all trained runs and write reports/tables/hypotheses.{json,md}."""
    runs = [_load(d) for d in run_dirs(ctx, all_runs=True) if (d / "analysis" / "final.json").exists() and json.loads((d / "meta.json").read_text(encoding="utf-8")).get("kind") == "trained"]
    if not any(r["meta"].get("group") == "wd1" for r in runs):
        log.warning("no analysed wd1 runs; skipping the hypothesis tests")
        return
    result = evaluate_hypotheses(runs)
    out = ctx.cfg.reports_dir / "tables" / "hypotheses.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")  # write-then-rename: an interrupted run never leaves a truncated table
    tmp.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    tmp.replace(out)
    write_markdown(result, ctx.cfg.reports_dir / "tables" / "hypotheses.md")
    for h, v in result["verdict"].items():
        log.info("%s: supported=%s p=%.3g p_holm=%.3g", h, v["supported"], v["p"], v["p_holm"])
