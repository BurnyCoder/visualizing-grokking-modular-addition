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
