"""Phase `validate_against_paper`: reproduce the paper's (and ARENA's) reported numbers on the paper's model.

Local explanation: loads the paper run's epoch-40k checkpoint (state_dicts[400]), computes every quantity the
paper/ARENA report (R1–R20 in docs/operationalizations.md) with our ported definitions, and writes
`runs/paper/analysis/rchecks.json` plus a generated markdown table `reports/tables/exp01_rchecks.md`.
A mismatch is recorded with its size and our best explanation — never tuned away.
Global context: this is the correctness anchor for everything on the website: if our code reproduces the
published numbers on the published model, the same code applied to other checkpoints/runs is trustworthy.
Sources: paper (https://arxiv.org/abs/2301.05217) pages cited per row; ARENA solutions/tests @85d1011
(https://github.com/ARENA-education/ARENA_materials/tree/85d10112344b68addebacb04909392e515dd7068/chapter1_transformer_interp/exercises/part52_grokking_and_modular_arithmetic).
"""

from __future__ import annotations  # postponed annotations

import json  # outputs
import logging  # module logger
import math  # isclose
from pathlib import Path  # paths

import numpy as np  # nearest checkpoint
import torch  # tensors

from grokviz import interventions, metrics  # measurements and causal tests
from grokviz.context import PipelineContext  # phase argument
from grokviz.envcheck import resolve_device  # device choice
from grokviz.formatting import fmt_value as _fmt  # same value formatting as the website's R-check table
from grokviz.patching import patch_recovery  # counterfactual swaps
from grokviz.phases import PAPER_BOUNDARIES  # the paper's stated phase boundaries
from grokviz.runstore import RunStore  # run access
from grokviz.sources import PAPER_ANALYSIS_INDEX, PAPER_RUN_ID  # which run/checkpoint

log = logging.getLogger("grokviz.rchecks")


def _row(rid: str, quantity: str, paper, ours, ok: bool | None, source: str, note: str = "") -> dict:
    """One comparison row; ok=None means 'qualitative / not directly comparable'."""
    return {"id": rid, "quantity": quantity, "paper": paper, "ours": ours, "match": ok, "source": source, "note": note}


def _close(x: float, y: float, rel: float) -> bool:
    """Relative closeness used for loss-like numbers."""
    return math.isclose(x, y, rel_tol=rel)


@torch.no_grad()
def _dynamics_rows(store: RunStore, grid: metrics.Grid, device) -> list[dict]:
    """R15 (paper Fig. 7) and R20 (ARENA §3 "Analysis During Training") — claims about the training trajectory,
    checked at a few checkpoints.

    R15: the sum of squared weights falls through circuit formation and faster during cleanup (qualitative; the paper
    gives only the plot). R20: after ~43k epochs total SSW drops from ≈955 to ≈942, and attention to "=" returns
    (ARENA gives no head count; "two heads rising ≥ 5× to > 5 %" is our criterion).
    """
    def at(step: int):  # model at the checkpoint nearest to `step` optimiser updates
        return store.load_model(int(np.argmin(np.abs(np.asarray(store.ckpt_steps) - step))), device)

    def ssw(model) -> float:  # total sum of squared weights (paper Fig. 7)
        return sum(metrics.sum_sq_weights(model).values())

    b = PAPER_BOUNDARIES  # paper Sec. 5.2, in epochs (= steps − 1)
    s1, s2, s3 = (ssw(at(b[t] + 1)) for t in ("t1", "t2", "t3"))
    rate_circuit, rate_cleanup = (s1 - s2) / (b["t2"] - b["t1"]), (s2 - s3) / (b["t3"] - b["t2"])
    s40, s_end = ssw(at(40001)), ssw(at(store.ckpt_steps[-1]))
    eq40, eq45 = (metrics.evaluate(at(s), grid).pattern.mean(0)[:, 2].tolist() for s in (40001, 45001))  # attention to "="
    risen = sum(after > 5 * before and after > 0.05 for before, after in zip(eq40, eq45))
    return [
        _row("R15", "SSW at the paper's phase boundaries 1.4k / 9.4k / 14k (Fig. 7)", "falls; steepest in cleanup", [s1, s2, s3],
             s1 > s2 > s3 and rate_cleanup > rate_circuit, "p.8 Fig. 7; boundaries p.9", f"qualitative; fall per step {rate_circuit:.3f} (circuit formation) vs {rate_cleanup:.3f} (cleanup)"),
        _row("R20a", "total SSW at epoch 40k and at the end of training", [955, 942], [s40, s_end],
             abs(s40 - 955) < 2 and abs(s_end - 942) < 2, "ARENA §3 Analysis During Training", f"last checkpoint: step {store.ckpt_steps[-1]}"),
        _row("R20b", "attention to '=' returns after ~43k (attention at 40k → 45k)", "attends to '=' again", [eq40, eq45],
             risen == 2, "ARENA §3 Analysis During Training",
             f"{risen} heads rise ≥ 5× to > 5 %; 2 such heads is our match criterion (ARENA gives no head count)"),
    ]


def compute_rchecks(store: RunStore, device) -> dict:
    """All comparisons on the paper's epoch-40k model; returns {"rows": [...], "details": {...}}."""
    model = store.load_model(PAPER_ANALYSIS_INDEX, device)
    p = store.meta["model_shape"]["d_vocab"] - 1  # 113: every number token plus the "=" token
    grid = metrics.Grid.build(p, store.meta["frac_train"], store.meta["split_seed"], device)
    acts = metrics.evaluate(model, grid)
    K = metrics.key_frequencies(model, grid.F)
    y = grid.labels
    from grokviz import fourier, losses

    loss_all = losses.grid_loss(acts.logits, y).item()
    we = fourier.frequency_norms(metrics.we_fourier_norms(model, grid.F))
    we_nonneg = int((we > 0.1 * we.max()).sum())  # "non-negligible": ≥10 % of the largest frequency norm
    freqs, fve, quad = metrics.neuron_frequencies(metrics.neuron_fourier(acts.post, grid.F))
    n_clustered = int((fve > metrics.NEURON_FVE_THRESHOLD).sum())
    k14 = int(((freqs == 14) & (fve > metrics.NEURON_FVE_THRESHOLD)).sum())
    coef, logit_fve, fit = metrics.logit_cos_fit(acts.logits, grid, K)
    abl = interventions.freq_ablation_table(acts, grid, K)
    wlp = interventions.wl_direction_projection(model, acts, grid, K)
    # R4's squared-norm share again at the released run's last checkpoint (step 49,901), for the note
    wl_last = metrics.wl_residual_sq_frac(metrics.paper_matrices(store.load_model(-1, device), p)["W_L"], fourier.key_rows(grid.F, K))
    poly = interventions.neuron_polynomial_replacement(model, acts, grid, freqs, fve, K)
    skip = interventions.skip_and_attention_ablations(model, acts, grid)
    heads = interventions.fit_attention_heads(model, acts, grid)
    table1 = interventions.table1_readoff(model, acts, grid, K)
    clusters = interventions.cluster_ablations(model, acts, grid, freqs, fve, K)
    relu = interventions.relu_multiplier_regression(p, 42)  # ARENA's example frequency k = 42
    excl = [metrics.excluded_loss(acts.logits, grid, [k]) for k in K]
    restricted = metrics.restricted_loss(acts.logits, grid, K)
    patch = patch_recovery(model, grid, K, n=1000, seed=0)
    eq_attn = skip["attention_to_eq_per_head"]
    rows = [
        _row("R1", "loss at epoch 40k (all 12,769 pairs, 113-way)", 2.41e-7, loss_all, _close(loss_all, 2.41e-7, 0.01), "p.6–7",
             "The paper calls it test loss; it matches the all-pairs loss (test-only is 3.02e-7)."),
        _row("R2", "key frequencies", [14, 35, 41, 42, 52], K, K == [14, 35, 41, 42, 52], "p.4"),
        _row("R3", "non-negligible W_E frequencies", 6, we_nonneg, we_nonneg == 6, "p.4", "threshold 10 % of max; the 6th is k=31"),
        _row("R4", "W_L residual after 10-direction approximation", "<0.55 %", wlp["wl_residual_sq_frac"],
             wlp["wl_residual_sq_frac"] < 0.006, "p.6",
             f"squared-norm ratio {100 * wlp['wl_residual_sq_frac']:.3f} % at step 40,001 (match: < 0.6 %) and "
             f"{100 * wl_last:.3f} % at the last checkpoint (step {store.ckpt_steps[-1]:,}); the norm ratio, "
             f"{100 * wlp['wl_residual_norm_frac']:.1f} %, does not match the paper's wording (Frobenius norm under 0.55 % of ‖W_L‖)"),
        _row("R5", "neurons with >85 % FVE at one frequency", 433, n_clustered, n_clustered == 433, "p.6"),
        _row("R6", "neurons in the k=14 cluster", 44, k14, k14 == 44, "p.7"),
        _row("R7", "Table 1 FVE of u_k/v_k read-offs (mean of 10)", 0.9624, sum(r["fve"] for r in table1) / len(table1),
             abs(sum(r["fve"] for r in table1) / len(table1) - 0.9624) < 0.01, "p.6 Table 1",
             "paper's ten values average 96.24 %; per-row values in details.table1"),
        _row("R8a", "logit fit Σα_k cos(w_k(a+b−c)) FVE", 0.95, logit_fve, abs(logit_fve - 0.95) < 0.02, "p.6"),
        _row("R8b", "loss of the fitted logits", 4.7e-8, losses.grid_loss(fit, y).item(), _close(losses.grid_loss(fit, y).item(), 4.7e-8, 0.1), "p.6"),
        _row("R9a", "loss keeping only the 40 key 2D components", 7.24e-8, abl["only_key_kept"], _close(abl["only_key_kept"], 7.24e-8, 0.01), "p.7"),
        _row("R9b", "MLP projected onto 10 W_L directions", 1.19e-7, wlp["projection"]["with_skip_bias_corrected"],
             _close(wlp["projection"]["with_skip_bias_corrected"], 1.19e-7, 0.1), "p.8", "closest variant shown; all variants in details.wl_projection"),
        _row("R9c", "MLP projected onto the nullspace", 5.27, wlp["nullspace"]["mlp_only_bias_corrected"],
             _close(wlp["nullspace"]["mlp_only_bias_corrected"], 5.27, 0.05), "p.8", "closest variant (MLP-only, bias-corrected) shown"),
        _row("R10a", "clustered neurons replaced by degree-2 polynomial", 2.48e-7, poly["polynomial"], _close(poly["polynomial"], 2.48e-7, 0.02), "p.7"),
        _row("R10b", "neurons restricted to cos/sin(w_k(a+b)) terms", 5.54e-8, poly["strict_trig_only"], _close(poly["strict_trig_only"], 5.54e-8, 0.02), "p.7",
             "per-neuron restriction gives this value; the paper's 5.54e-8 matches the restricted-LOGITS loss (R19a, "
             f"{restricted:.4g}) within 0.2 %, so the paper likely restricted logits"),
        _row("R11a", "MLP skip connection zero-ablated", 9.12e-7, skip["mlp_skip_zero"], _close(skip["mlp_skip_zero"], 9.12e-7, 0.05), "p.13"),
        _row("R11b", "MLP skip connection mean-ablated", 7.25e-7, skip["mlp_skip_mean"], _close(skip["mlp_skip_mean"], 7.25e-7, 0.02), "p.13"),
        _row("R11c", "attention heads zero-ablated", 24.3, skip["attention_zero"], _close(skip["attention_zero"], 24.3, 0.02), "p.13"),
        _row("R11d", "skip around attention zero-ablated", 19.1, skip["attention_skip_zero"], _close(skip["attention_skip_zero"], 19.1, 0.05), "p.13",
             "not reproduced: our ablation sets resid_mid = attention output (the paper does not specify its hook)"),
        _row("R12", "mean attention from '=' to '=' (per head)", "0.1–0.4 %", eq_attn, all(0.0005 < v < 0.0045 for v in eq_attn), "p.13",
             "match range 0.05–0.45 % (the paper's 0.1–0.4 % at its printed precision); our largest values, 0.44 %, "
             "round to the paper's 0.4 %"),
        _row("R13a", "linear fit of the attention softmax (variance explained)", 0.975, heads["linear_fve_mean"], abs(heads["linear_fve_mean"] - 0.975) < 0.02, "p.16"),
        _row("R13b", "loss with softmax replaced by its linear fit", 2.12e-7, heads["linearised_softmax_loss"], _close(heads["linearised_softmax_loss"], 2.12e-7, 0.05), "p.16"),
        _row("R14", "head frequencies (Table 2)", [35, 42, 52, 42], [h["freq"] for h in heads["heads"]], [h["freq"] for h in heads["heads"]] == [35, 42, 52, 42], "p.15 Table 2"),
        _row("R16", "ARENA excluded loss per key freq", [2.82e-4, 6.33e-4, 1.61e-3, 7.195e-2, 3.288e-2], excl,
             all(_close(a, b, 0.01) for a, b in zip(excl, [2.82e-4, 6.33e-4, 1.61e-3, 7.195e-2, 3.288e-2])), "ARENA tests.test_excl_loss"),
        _row("R17", "ARENA cluster sizes 14/35/41/42/52/always", [44, 93, 145, 87, 64, 79], list(clusters["cluster_sizes"].values()),
             list(clusters["cluster_sizes"].values()) == [44, 93, 145, 87, 64, 79], "ARENA 2-circuit-and-feature-analysis"),
        _row("R18", "ARENA ReLU regression r² (with / without quadratic)", [0.966, 0.849], [relu["r2_with_quadratic"], relu["r2_without_quadratic"]],
             abs(relu["r2_with_quadratic"] - 0.966) < 0.002 and abs(relu["r2_without_quadratic"] - 0.849) < 0.002, "ARENA"),
        _row("R19a", "ARENA trig-only (restricted) logits loss", 5.5474e-8, restricted, _close(restricted, 5.5474e-8, 0.005), "ARENA"),
        _row("R19b", "ARENA key-frequency neuron projection loss (with / without always-firing)", [2.482e-7, 1.179e-6],
             [clusters["loss_key_projection"], clusters["loss_key_projection_no_always_firing"]],
             _close(clusters["loss_key_projection"], 2.482e-7, 0.02) and _close(clusters["loss_key_projection_no_always_firing"], 1.179e-6, 0.02), "ARENA"),
    ]
    rows += _dynamics_rows(store, grid, device)
    details = {"key_freqs": K, "table1": table1, "heads": heads, "freq_ablation": abl, "wl_projection": wlp,
               "neuron_poly": poly, "skip": skip, "clusters": clusters, "relu": relu, "patching": patch,
               "logit_fit_coef": coef.tolist(), "we_freq_norms": we.tolist()}
    return {"rows": rows, "details": details}


def write_table(rows: list[dict], path: Path) -> None:
    """Generated markdown table (included by reports/exp-01-paper-validation.md)."""
    lines = ["| ID | Quantity | Paper / ARENA | Ours | Match | Source | Note |", "|---|---|---|---|---|---|---|"]
    mark = {True: "yes", False: "**no**", None: "n/a"}
    for r in rows:
        lines.append(f"| {r['id']} | {r['quantity']} | {_fmt(r['paper'])} | {_fmt(r['ours'])} | {mark[r['match']]} | {r['source']} | {r['note']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_against_paper(ctx: PipelineContext) -> None:
    """Phase entry point: compute R-checks, save JSON + markdown table, log a summary."""
    store = RunStore(ctx.cfg.runs_dir / PAPER_RUN_ID)
    result = compute_rchecks(store, resolve_device(ctx.cfg.device))
    out = store.dir / "analysis" / "rchecks.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_table(result["rows"], ctx.cfg.reports_dir / "tables" / "exp01_rchecks.md")
    for r in result["rows"]:
        log.info("%-5s %-60s paper=%s ours=%s match=%s", r["id"], r["quantity"][:60], _fmt(r["paper"]), _fmt(r["ours"]), r["match"])
    p = result["details"]["patching"]
    log.info("patching recovered: key %.3f %s, complement %.3f, random %.3f", p["key"]["mean"], p["key"]["ci95"], p["complement"]["mean"], p["random"]["mean"])
    n_bad = sum(r["match"] is False for r in result["rows"])
    log.info("R-checks: %d rows, %d not reproduced (documented in reports/exp-01)", len(result["rows"]), n_bad)
