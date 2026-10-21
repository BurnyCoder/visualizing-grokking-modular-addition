"""Numbers quoted in the website's prose, computed from the analysis outputs (site/data/summary.json).

Local explanation: every {{num:key}} in content/ is looked up here, so no number in the text is typed by hand.
Each entry is {"value": ..., "fmt": ...} where fmt is "int", "pct", "sci", a Python format spec like ".3f",
or "str" for pre-formatted text; lists are printed comma-separated. Keys are grouped:
  task.*, model.* — dataset/model sizes; paper.* — the paper's run (validation, R-checks, curves, gradients);
  rc.<id> — our value of each reproduction check; seeds.*, controls.*, hyp.* — our 22 runs (after training).
Global context: called by `export_site_data`; the site build fails if any referenced key is missing.
Sources: definitions in docs/operationalizations.md; tests in docs/preregistration.md.
"""

from __future__ import annotations  # postponed annotations

import json  # inputs/outputs
from pathlib import Path  # paths

import numpy as np  # arrays
import torch  # weights

from grokviz import task  # split sizes
from grokviz.runstore import RunStore  # paper run
from grokviz.sources import PAPER_ANALYSIS_INDEX  # index of the paper's epoch-40k checkpoint


def _v(value, fmt="g") -> dict:
    """One summary entry."""
    return {"value": value, "fmt": fmt}


def _load(path: Path):
    """JSON file or None."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _at(m: dict, key: str, step: int) -> float:
    """Value of a per-checkpoint array at the checkpoint nearest to `step`."""
    i = int(np.argmin(np.abs(m["steps"] - step)))
    return float(m[key][i])


def _ci(t: list | None, fmt: str) -> dict:
    """Format a (mean, low, high) interval as 'mean [low, high]'."""
    if not t:
        return _v("—", "str")
    mean, lo, hi = t
    f = (lambda x: f"{100 * x:.1f} %") if fmt == "pct" else (lambda x: f"{x:,.0f}") if fmt == "int" else (lambda x: f"{x:.3f}")
    return _v(f"{f(mean)} [{f(lo)}, {f(hi)}]", "str")


def paper_numbers(runs_dir: Path, p: int, frac: float) -> dict:
    """task/model/paper/rc keys from the paper run's analysis outputs."""
    out: dict = {}
    split = task.paper_split(p, frac, 0)
    out["task.n_train"] = _v(len(split.train_idx), "int")
    out["task.n_test"] = _v(len(split.test_idx), "int")
    pr = runs_dir / "paper"
    a = pr / "analysis"
    rc = _load(a / "rchecks.json")
    m = dict(np.load(a / "ckpt_metrics.npz"))
    ph = _load(a / "phases.json")
    val = _load(a / "validation.json")
    g = dict(np.load(a / "gradients.npz"))
    store = RunStore(pr)
    model = store.load_model(PAPER_ANALYSIS_INDEX)  # the epoch-40k model the paper analyses
    out["model.n_params"] = _v(sum(q.numel() for q in model.parameters()), "int")
    out["paper.key_freqs"] = _v(rc["details"]["key_freqs"])
    for t in ("t1", "t2", "t3", "t_grok"):
        out[f"paper.{t}"] = _v(ph[t], "int")
    out["paper.loss40k_test"] = _v(val["epoch40k_loss_test_113"], "sci")
    for row in rc["rows"]:  # rc.R* = our value of each reproduction check
        ours = row["ours"]
        out[f"rc.{row['id']}"] = _v(ours, "sci") if isinstance(ours, float) else _v(ours)
    out["rchecks.rows"] = _v(rc["rows"], "str")
    det = rc["details"]
    eq = det["skip"]["attention_to_eq_per_head"]
    out["paper.attn_eq_pct"] = _v(", ".join(f"{100 * x:.2f} %" for x in eq), "str")
    avg45 = m["avg_attn"][int(np.argmin(np.abs(m["steps"] - 45001)))][:, 2]
    out["paper.attn_eq_45k"] = _v(", ".join(f"{100 * x:.1f} %" for x in avg45), "str")
    W = model.pos_embed.W_pos.detach().double()
    WE = model.embed.W_E[:, :p].detach().double()
    for i in range(3):
        out[f"paper.pos_norm{i}"] = _v(float(W[i].norm()), ".2f")
    out["paper.we_norm_mean"] = _v(float(WE.norm(dim=0).mean()), ".2f")
    out["paper.pos_cos01"] = _v(float(torch.nn.functional.cosine_similarity(W[0], W[1], dim=0)), ".2f")
    we = np.asarray(det["we_freq_norms"])
    out["paper.we_top6"] = _v(sorted(int(k) + 1 for k in np.argsort(-we)[:6]))
    i40 = int(np.argmin(np.abs(m["steps"] - 40001)))
    out["paper.gini_we"] = _v(float(m["gini_we"][i40]), ".2f")
    out["paper.gini_we_init"] = _v(float(m["gini_we"][0]), ".2f")
    sizes = det["clusters"]["cluster_sizes"]
    out["paper.cluster_sizes"] = _v(", ".join(f"k = {k}: {n}" for k, n in sizes.items() if k != "-1"), "str")
    out["paper.always_firing"] = _v(sizes.get("-1", 0), "int")
    out["paper.neurons_085"] = _v(int(sum(n for k, n in sizes.items() if k != "-1")), "int")
    t1 = [r["fve"] for r in det["table1"]]
    out["paper.table1_fve_mean"] = _v(float(np.mean(t1)), "pct")
    out["paper.table1_fve_range"] = _v(f"{100 * min(t1):.1f}–{100 * max(t1):.1f} %", "str")
    out["paper.head_freqs"] = _v([h["freq"] for h in det["heads"]["heads"]])
    out["paper.linear_fve"] = _v(det["heads"]["linear_fve_mean"], "pct")
    out["paper.linearised_loss"] = _v(det["heads"]["linearised_softmax_loss"], "sci")
    rows = {r["id"]: r for r in rc["rows"]}
    out["paper.logit_fve"] = _v(rows["R8a"]["ours"], "pct")
    out["paper.logit_fit_loss"] = _v(rows["R8b"]["ours"], "sci")
    out["paper.logit_alpha"] = _v([round(x, 2) for x in det["logit_fit_coef"][:-1]])
    out["paper.wl_resid_sq"] = _v(det["wl_projection"]["wl_residual_sq_frac"], "pct")
    out["paper.wl_resid_norm"] = _v(det["wl_projection"]["wl_residual_norm_frac"], "pct")
    ab = det["freq_ablation"]
    single = np.asarray(ab["single_freq"]) / ab["original"]
    keys = det["key_freqs"]
    key_r = [single[k - 1] for k in keys]
    non_r = [single[k - 1] for k in range(1, len(single) + 1) if k not in keys]
    out["paper.key_ablation_ratio_range"] = _v(f"{min(key_r):,.0f}× to {max(key_r):,.0f}×", "str")
    out["paper.nonkey_ablation_max_ratio"] = _v(f"{max(non_r):.2f}×", "str")
    out["paper.only_key_kept"] = _v(ab["only_key_kept"], "sci")
    out["paper.all_key_removed"] = _v(ab["all_key_removed"], ".1f")
    out["paper.random5_min"] = _v(min(ab["random_nonkey_removed"]), "sci")
    out["paper.random5_max"] = _v(max(ab["random_nonkey_removed"]), "sci")
    pt = det["patching"]
    out["paper.patch_key"] = _v(pt["key"]["mean"], ".3f")
    out["paper.patch_key_lo"] = _v(pt["key"]["ci95"][0], ".3f")
    out["paper.patch_key_hi"] = _v(pt["key"]["ci95"][1], ".3f")
    out["paper.patch_random"] = _v(pt["random"]["mean"], ".3f")
    out["paper.patch_complement"] = _v(pt["complement"]["mean"], ".3f")
    out["paper.relu_r2_full"] = _v(det["relu"]["r2_with_quadratic"], ".3f")
    out["paper.relu_r2_lin"] = _v(det["relu"]["r2_without_quadratic"], ".3f")
    out["paper.restricted_5k"] = _v(_at(m, "restricted_loss", 5001), ".2f")
    out["paper.excluded_t1"] = _v(_at(m, "excluded_loss", ph["t1"]), "sci")
    out["paper.excluded_t2"] = _v(_at(m, "excluded_loss", ph["t2"]), ".2f")
    ssw = m["ssw_total"]
    out["paper.ssw_peak"] = _v(float(ssw.max()), ",.0f")
    out["paper.ssw_peak_step"] = _v(int(m["steps"][int(ssw.argmax())]), "int")
    out["paper.ssw_t3"] = _v(_at(m, "ssw_total", ph["t3"]), ",.0f")
    mid = int(m["steps"][int(np.argmin(np.abs(m["steps"] - (ph["t1"] + ph["t2"]) / 2)))])  # midpoint of circuit formation
    gi = lambda s: int(np.argmin(np.abs(g["steps"] - s)))  # noqa: E731 — nearest gradient checkpoint
    out["paper.mid_step"] = _v(mid, "int")
    out["paper.cos_tt_start"] = _v(float(g["cos_train_test"][0]), ".2f")
    out["paper.cos_tt_mid"] = _v(float(g["cos_train_test"][gi(mid)]), ".2f")
    out["paper.cos_tt_40k"] = _v(float(g["cos_train_test"][gi(40001)]), ".2f")
    out["paper.cos_gw_40k"] = _v(float(g["cos_grad_weight"][gi(40001)]), ".3f")
    return out


def run_numbers(reports_dir: Path) -> dict:
    """seeds.*, controls.*, hyp.* keys from reports/tables/hypotheses.json (empty before training)."""
    hyp = _load(reports_dir / "tables" / "hypotheses.json")
    if hyp is None:
        return {}
    out: dict = {"hypotheses.verdict": _v(hyp["verdict"], "str")}
    sec = hyp["secondary"]
    out["seeds.t1_ci"] = _ci(sec.get("S6_t1"), "int")
    out["seeds.t2_ci"] = _ci(sec.get("S6_t2"), "int")
    out["seeds.tgrok_ci"] = _ci(sec.get("S6_t_grok"), "int")
    out["seeds.nkey_ci"] = _ci(sec.get("S5_n_key_freqs"), "f")
    out["seeds.neurons085_ci"] = _ci(sec.get("S1_frac_neurons_085"), "pct")
    out["seeds.logitfve_ci"] = _ci(sec.get("S2_logit_fve"), "pct")
    out["seeds.patch_key_ci"] = _ci(sec.get("S4_patch_key"), "f")
    out["seeds.patch_random_ci"] = _ci(sec.get("S4_patch_random"), "f")
    v = hyp["verdict"]
    parts = []
    for h, d in v.items():
        res = f"{d['successes']}/{d['n']} seeds" if "successes" in d else f"{d['wd1_grokked']}/{d['wd1_n']} weight-decay runs vs {d['wd0_grokked']}/{d['wd0_n']} without grokked"
        parts.append(f"**{h}** {'supported' if d['supported'] else 'not supported'} ({res}; Holm-adjusted p = {d['p_holm']:.2g})")
    out["hyp.summary"] = _v("; ".join(parts) + ".", "str")
    # post-hoc decomposition of H4 into its three preregistered sub-conditions (counts only, no new test)
    pr = hyp["per_run"].values()
    out["hyp.h4_key_ok"] = _v(f"{sum(r['H4_key_min_ratio'] >= 100 for r in pr)}/{len(pr)}", "str")
    out["hyp.h4_nonkey_ok"] = _v(f"{sum(r['H4_nonkey_max_ratio'] <= 2 for r in pr)}/{len(pr)}", "str")
    out["hyp.h4_keep_ok"] = _v(f"{sum(r['H4_keep_ratio'] <= 1.5 for r in pr)}/{len(pr)}", "str")
    out["hyp.h4_key_min"] = _v(min(r["H4_key_min_ratio"] for r in pr), ".1f")
    out["hyp.h4_nonkey_max"] = _v(max(r["H4_nonkey_max_ratio"] for r in pr), ".2f")
    out["hyp.h4_keep_max"] = _v(max(r["H4_keep_ratio"] for r in pr), ".2f")
    ctl = hyp["controls"]
    for group, text in (("wd0", "no weight decay"), ("rand", "random labels"), ("split0", "paper split")):
        rows = [(k, c) for k, c in ctl.items() if c["group"] == group]
        if not rows:
            continue
        acc = ", ".join(f"{100 * c['final_test_acc']:.1f} %" for _, c in rows)
        tr = ", ".join(f"{100 * c['final_train_acc']:.0f} %" for _, c in rows)
        if group == "split0":
            ks = "; ".join(f"{k}: {c['key_freqs']}" for k, c in rows)
            out[f"controls.{group}"] = _v(f"final test accuracy {acc}; key frequencies {ks} (the paper's run: {{14, 35, 41, 42, 52}})", "str")
        else:
            gi = ", ".join(f"{c['gini_we']:.2f}" for _, c in rows)
            out[f"controls.{group}"] = _v(f"final train accuracy {tr}, final test accuracy {acc}; Gini of the embedding spectrum {gi} (initially ≈ {np.mean([c['gini_we_init'] for _, c in rows]):.2f})", "str")
    return out


def same_init_numbers(runs_dir: Path) -> dict:
    """seeds.same_init.*: our wd1_s0 starts from the paper's exact initial weights and split (exploratory comparison).

    The paper's first checkpoint is taken after one update with learning rate 0 (warm-up), i.e. it is the
    initialisation; comparing it with our step-0 checkpoint shows whether the two runs start identically.
    """
    p_dir, o_dir = runs_dir / "paper", runs_dir / "wd1_s0"
    if not (o_dir / "analysis" / "key_freqs.json").exists() or not (p_dir / "meta.json").exists():
        return {}
    paper, ours = RunStore(p_dir), RunStore(o_dir)
    flat = lambda d: torch.cat([d[k].double().flatten() for k in sorted(d) if k != "blocks.0.attn.mask"])  # noqa: E731
    init_diff = float((flat(paper.state_dict(0)) - flat(ours.state_dict(0))).abs().max())
    a, b = flat(paper.state_dict(PAPER_ANALYSIS_INDEX)), flat(ours.state_dict(-1))  # paper step 40,001 vs ours 40,000
    k_paper = _load(p_dir / "analysis" / "key_freqs.json")["key_freqs"]
    k_ours = _load(o_dir / "analysis" / "key_freqs.json")["key_freqs"]
    c_paper, c_ours = paper.curves(), ours.curves()
    same_upto = int(np.nonzero(~np.isclose(c_paper["train_loss"][:1000], c_ours["train_loss"][:1000], rtol=1e-5))[0][0])
    return {"seeds.same_init_diff": _v(init_diff, "sci"), "seeds.same_init_cos40k": _v(float(a @ b / a.norm() / b.norm()), ".2f"),
            "seeds.same_init_keys": _v(k_ours), "seeds.same_init_keys_equal": _v("the same" if k_ours == k_paper else "different", "str"),
            "seeds.same_init_loss_steps": _v(same_upto, "int")}


def late_spike_numbers(runs_dir: Path, factor: float = 100.0, gap: int = 2000) -> dict:
    """seeds.spikes: runs whose test loss, ≥ `gap` steps after grokking, jumps above `factor` × its running minimum
    (exploratory description of the loss spikes visible in the seed figure; cf. the "slingshot" of Thilak et al. 2022,
    https://arxiv.org/abs/2206.04817). An event = a spike more than 500 steps after the previous one."""
    found = []
    for d in sorted(runs_dir.iterdir()):
        ph = _load(d / "analysis" / "phases.json")
        if d.name == "paper" or ph is None or ph.get("t_grok") is None:
            continue
        c = RunStore(d).curves()
        sel = c["test_loss_steps"] >= ph["t_grok"] + gap
        loss, steps, acc = c["test_loss"][sel], c["test_loss_steps"][sel], c["test_acc"][sel]
        hits = steps[loss > factor * np.minimum.accumulate(loss)]
        events = [s for i, s in enumerate(hits) if i == 0 or s - hits[i - 1] > 500]
        if events:
            found.append(f"{d.name} ({len(events)} spikes; test loss up to {loss.max():.2f}, test accuracy down to {100 * acc.min():.0f} %)")
    return {"seeds.spikes": _v("; ".join(found) if found else "none", "str"), "seeds.spike_n": _v(len(found), "int")}


def build_summary(runs_dir: Path, reports_dir: Path, p: int, frac: float) -> dict:
    """All prose numbers."""
    return {**paper_numbers(runs_dir, p, frac), **run_numbers(reports_dir), **same_init_numbers(runs_dir), **late_spike_numbers(runs_dir)}
