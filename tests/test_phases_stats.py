"""Tests for phase detection rules and statistics helpers (milestone M4).

Global context: phase boundaries shade every training plot and define the time points of the preregistered
hypotheses; the statistics helpers decide whether those hypotheses pass. Both are tested on synthetic data
with known answers.
"""

import numpy as np  # synthetic curves
import pytest  # approx

from grokviz import phases, stats  # modules under test


def test_phase_rules_on_synthetic_grokking_curves():
    """Train loss plateaus at ~1000, test loss drops at ~5000, SSW settles at ~8000 → boundaries near those steps."""
    steps = np.arange(20_001)
    train = np.where(steps < 1000, np.exp(-steps / 100.0), np.exp(-10.0))  # flat after 1000
    ck = np.arange(0, 20_001, 100)
    test = np.where(ck < 5000, 20.0, 1e-3)  # plateau, then sudden drop
    acc = np.where(ck < 5200, 0.01, 1.0)
    ssw = np.where(ck < 8000, 3000.0 - 0.1 * ck, 1000.0)  # declining, then flat
    out = phases.detect(steps, train, ck, test, acc, ssw)
    assert 900 <= out["t1"] <= 1100
    assert out["t2"] == 5000
    assert 7000 <= out["t3"] <= 8000
    assert out["t_grok"] == 5200


def test_no_grokking_gives_none():
    """A run whose test loss never drops has no t2/t3/t_grok."""
    steps = np.arange(5001)
    train = np.exp(-steps / 100.0) + 1e-6
    ck = np.arange(0, 5001, 100)
    out = phases.detect(steps, train, ck, np.full(len(ck), 20.0), np.zeros(len(ck)), np.full(len(ck), 3000.0))
    assert out["t2"] is None and out["t3"] is None and out["t_grok"] is None


def test_drop_time_is_the_first_tenfold_fall_after_t1():
    """drop_time (H1's rule, also quoted in the prose): the first checkpoint after t1 at or below a tenth of the value
    at t1; earlier falls do not count, and a curve that never gets there gives None."""
    steps = np.arange(0, 1000, 100)
    v = np.array([5.0, 0.1, 2.0, 1.5, 0.5, 0.21, 0.2, 0.1, 0.05, 0.01])  # value at t1 = 200 is 2.0
    assert phases.drop_time(steps, v, 200) == 600  # 0.2 ≤ 0.1 × 2.0; the 0.1 at step 100 is before t1
    assert phases.drop_time(steps, np.full(10, 3.0), 200) is None


def test_t_interval_and_bootstrap_contain_mean():
    """Intervals bracket the sample mean and the t-interval matches the textbook formula."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, low, high = stats.t_interval(x)
    assert mean == 3.0 and low < 3.0 < high
    assert high - mean == pytest.approx(2.776445 * np.std(x, ddof=1) / np.sqrt(5), rel=1e-5)
    bmean, blow, bhigh = stats.bootstrap_interval(x)
    assert blow <= bmean <= bhigh


def test_sign_test_and_holm():
    """11/11 agreeing seeds → p = 2^-11; Holm rejects step-down in order."""
    assert stats.sign_test(11, 11) == pytest.approx(0.5**11)
    res = stats.holm({"a": 0.001, "b": 0.03, "c": 0.04})
    assert res["a"]["reject"]  # 0.001 ≤ 0.05/3
    assert not res["b"]["reject"]  # 0.03 > 0.05/2 → stop
    assert not res["c"]["reject"]  # step-down: nothing after the first failure is rejected
    assert res["b"]["p_holm"] == pytest.approx(0.06)  # adjusted p = 2 × 0.03


def test_power_table_numbers_for_eleven_seeds():
    """With 11 seeds and Holm over 5 tests, one disagreeing seed still passes the strictest threshold."""
    tab = stats.power_table(11, 0.05, 5)
    assert tab["p_one_disagrees"] == pytest.approx(12 / 2048)
    assert tab["p_one_disagrees"] < tab["holm_strictest_alpha"]


def _synthetic_run(run_id: str, group: str, excl_t2: float = 1.0, grokked: bool = True) -> dict:
    """A run shaped like hypotheses._load's output, built so that H1–H4 hold unless told otherwise.

    Checkpoints every 100 steps; t1 = 200, t2 = 600. Restricted loss drops 10× at step 400, test loss at 600 (Δ = 200);
    excluded loss goes 1e-4 → excl_t2; key ablations ×10⁴, non-key ×1.1, keeping only key frequencies ×1.2.
    """
    steps = np.arange(0, 1100, 100)
    drop = lambda at: np.where(steps >= at, 0.01, 5.0)  # noqa: E731 — tiny step-function helper
    excl = np.where(steps >= 600, excl_t2, 1e-4)
    acc = np.full(len(steps), 1.0 if grokked else 0.05)
    keys = [14, 35, 41]
    single = [1e-3 if k in keys else 1.1e-7 for k in range(1, 57)]
    final = {"key_freqs": keys, "ablation": {"single_freq": single, "original": 1e-7, "only_key_kept": 1.2e-7},
             "gini_we": 0.8, "gini_we_init": 0.05, "gini_null_we_p99": 0.3, "gini_wl": 0.8, "gini_wl_init": 0.05,
             "gini_null_wl_p99": 0.3, "frac_neurons_085": 0.85, "logit_fve": 0.95, "wl_resid_sq_frac": 0.005,
             "patching": {n: {"mean": v} for n, v in (("key", 0.99), ("complement", 0.01), ("random", 0.07))}}
    m = {"steps": steps, "loss_test": drop(600), "restricted_loss": drop(400), "excluded_loss": excl, "acc_test": acc,
         "acc_train": np.ones(len(steps)), "excluded_loss_max": excl}
    return {"id": run_id, "meta": {"group": group}, "m": m, "final": final,
            "phases": {"t1": 200, "t2": 600, "t3": 800, "t_grok": 600 if grokked else None}}


def test_preregistered_hypotheses_on_synthetic_runs():
    """11/11 seeds agree → every H supported; H5 Fisher p = 1/C(16,5) = 2.3e-4 (as stated in the preregistration);
    two seeds with a too-small excluded-loss rise make H2 fail by the "≥ 2 failures" rule."""
    from grokviz.hypotheses import evaluate_hypotheses

    wd1 = [_synthetic_run(f"wd1_s{s}", "wd1") for s in range(11)]
    wd0 = [_synthetic_run(f"wd0_s{s}", "wd0", grokked=False) for s in range(5)]
    v = evaluate_hypotheses(wd1 + wd0)["verdict"]
    assert all(v[h]["supported"] for h in ("H1", "H2", "H3", "H4", "H5"))
    assert v["H1"]["successes"] == 11 and v["H1"]["p"] == pytest.approx(0.5 ** 11)
    assert v["H5"]["p"] == pytest.approx(1 / 4368)
    wd1[0], wd1[1] = _synthetic_run("wd1_s0", "wd1", excl_t2=5e-4), _synthetic_run("wd1_s1", "wd1", excl_t2=5e-4)  # ratio 5 < 10
    v = evaluate_hypotheses(wd1 + wd0)["verdict"]
    assert v["H2"]["successes"] == 9 and not v["H2"]["supported"] and v["H1"]["supported"]


def test_paper_curve_numbers_read_the_curves():
    """The caption numbers (summary.paper_curve_numbers) come straight off the curves: accuracy at a step, the
    test ÷ restricted loss range inside [t1, t2], the last negative train/test gradient cosine, and norm ratios."""
    from grokviz.summary import paper_curve_numbers

    s = np.arange(1, 1002, 100)  # checkpoints 1, 101, …, 1001
    m = {"steps": s, "acc_test": s / 1001, "acc_train": (s >= 201).astype(float), "loss_test": np.full(len(s), 20.0),
         "restricted_loss": np.full(len(s), 2.0), "ssw_total": np.linspace(100, 50, len(s)), "we_svals": np.tile(np.arange(14.0, 0, -1), (len(s), 1))}
    m["restricted_loss"][s >= 701] = 1e-3  # ratio 2e4 from step 701 (after t2)
    ph = {"t1": 201, "t2": 601, "t3": 901, "t_grok": 801}
    m["wl_resid_sq_frac"] = np.linspace(0.9, 0.0, len(s))  # 90 % outside K at the start, 0 % at the end
    lookup = np.zeros((len(s), 1, 113))  # one head's attention lookup C_j in the Fourier basis
    lookup[:, 0, 1:9] = 1.0  # plateau: 4 equal frequencies (cos/sin of k = 1…4) → 25 % each
    lookup[s >= 801, 0, 3:9] = 0.0  # from step 801 only k = 1 remains → 100 %
    m["attn_C_fourier"] = lookup
    m["avg_attn"] = np.tile(np.array([[0.3, 0.3, 0.4]]), (len(s), 1, 1))
    m["avg_attn"][s >= 101, :, 2] = 0.005  # attention to '=' falls below 1 % from step 101
    g = {"steps": s, "cos_train_test": np.where(s <= 501, -0.3, 0.2), "param_names": np.array(["embed.W_E", "blocks.0.mlp.b_in", "unembed.W_U"]),
         "weight_norm": np.array([[1.0, 0.0, 2.0]] * (len(s) - 1) + [[0.5, 0.1, 3.0]]),
         "grad_ratio_decay": np.where(s < 201, 1.0, np.where(s < 601, 2e-8, 1e-8))}
    gi = lambda step: int(np.argmin(np.abs(s - step)))  # noqa: E731
    out = paper_curve_numbers(m, ph, g, gi)
    assert out["paper.acc_test_t2"]["value"] == pytest.approx(601 / 1001)
    assert out["paper.train_acc_full_step"]["value"] == 201
    assert out["paper.ratio_cf_range"]["value"] == "10–10" and out["paper.ratio_1e4_steps"]["value"] == "701–1,001"
    assert out["paper.cos_tt_last_neg"]["value"] == 501
    assert out["paper.norm_shrunk"]["value"] == "W_E 0.50×" and out["paper.norm_grown"]["value"] == "W_U 1.50×"  # biases skipped
    assert out["paper.we_svals_next2"]["value"] == "4.00 and 3.00"
    assert out["paper.wl_outside_init"]["value"] == pytest.approx(0.9) and out["paper.grad_decay_ratio"]["value"] == "1.00×10⁻⁸ to 2.00×10⁻⁸"
    assert out["paper.head_lookup_plateau_max"]["value"] == pytest.approx(0.25) and out["paper.head_lookup_share_40k"]["value"] == "100.0–100.0 %"
    assert out["paper.attn_eq_drop_step"]["value"] == 101


def test_logged_loss_precision_counts_float32_values():
    """summary.logged_precision_numbers: a loss computed in float32 and read with .item() is exactly representable in
    float32; a float64 loss almost never is (the evidence that the released paper run used a float64 loss)."""
    from grokviz.summary import logged_precision_numbers

    f32 = np.float32(np.random.default_rng(0).random(5)).astype(np.float64)
    out = logged_precision_numbers({"train_loss": f32, "test_loss": np.array([1 / 3, 2 / 3])})
    assert out["paper.logged_losses_n"]["value"] == 7 and out["paper.logged_losses_f32"]["value"] == 5


def test_seed_spread_numbers_count_grokked_runs_and_gini(tmp_path):
    """summary.seed_spread_numbers: grokked trained runs, and the final Gini range of runs with / without a sparse
    Fourier solution (figure gini-vs-null)."""
    import json

    from grokviz.summary import seed_spread_numbers

    for name, t_grok, gw in (("wd1_s0", 900, 0.7), ("split0_init100", 800, 0.8), ("wd0_s0", None, 0.07), ("rand_s0", None, 0.09)):
        d = tmp_path / name / "analysis"
        d.mkdir(parents=True)
        (d / "phases.json").write_text(json.dumps({"t_grok": t_grok}))
        (d / "final.json").write_text(json.dumps({"gini_we": gw, "gini_wl": gw + 0.05}))
    out = seed_spread_numbers(tmp_path)
    assert out["seeds.n_grokked"]["value"] == 2
    assert out["controls.gini_sparse_range"]["value"] == "0.70–0.85" and out["controls.gini_flat_max"]["value"] == pytest.approx(0.14)


def test_h4_detail_names_the_failing_seeds(tmp_path):
    """summary.h4_detail_numbers lists the seed whose weakest key frequency costs < 100× (with its W_L norm share)
    and the seed where keeping only the key frequencies costs > 1.5×."""
    import json

    from grokviz.summary import h4_detail_numbers

    for name, weak, keep in (("wd1_s0", 1e3, 1.0), ("wd1_s1", 5.0, 2.0)):
        d = tmp_path / name / "analysis"
        d.mkdir(parents=True)
        single = [1.0] * 56
        single[1], single[2] = weak, 1e5  # k = 2 (the weak one), k = 3
        (d / "final.json").write_text(json.dumps({"key_freqs": [2, 3], "ablation": {"single_freq": single, "original": 1.0, "only_key_kept": keep}}))
        wl = np.zeros(113)
        wl[3], wl[5] = 1.0, 4.0  # cos 2 → norm 1; cos 3 → norm 4 (the largest)
        np.savez(d / "ckpt_metrics.npz", wl_fourier=wl[None])
    out = h4_detail_numbers(tmp_path)
    assert out["hyp.h4_key_fails"]["value"] == "wd1_s1: k = 2 costs 5.0× (25 % of the largest W_L norm; its other key frequencies ≥ 100,000×)"
    assert out["hyp.h4_keep_fails"]["value"] == "wd1_s1 (2 key frequencies, 2.00×)" and out["hyp.n_few_keys"]["value"] == 2
    assert out["hyp.h4_keep_fail_n"]["value"] == 1
