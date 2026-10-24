"""Tests for the training loop (milestone M5): paper warm-up semantics, equivalence of batched training to an
unbatched torch AdamW reference, determinism, run matrix and saved format.

Global context: 22 runs × 40k steps are trained once (together, vectorised over runs) and analysed afterwards;
these short runs prove that batching does not couple runs, that the first step has learning rate 0 as in the
paper's code, and that training is bitwise reproducible.
"""

import numpy as np  # curves
import pytest  # fixtures
import torch  # tensors

from grokviz import losses, task  # reference loop pieces
from grokviz.config import load_config  # settings
from grokviz.envcheck import resolve_device  # device
from grokviz.model import GrokTransformer  # reference model
from grokviz.runstore import RunStore  # reload written runs
from grokviz.train import RunSpec, run_matrix, train_runs  # module under test


@pytest.fixture(scope="module")
def cfg():
    """Default configuration (paper settings)."""
    return load_config()


@pytest.fixture(scope="module")
def device(cfg):
    """GPU when available."""
    return resolve_device(cfg.device)


def test_run_matrix_is_the_preregistered_one(cfg):
    """11 + 5 + 3 + 3 = 22 runs; seed 0 of wd1 uses the paper's split; controls have the intended settings."""
    specs = run_matrix(cfg)
    ids = [s.run_id for s in specs]
    assert len(specs) == 22 and ids[0] == "wd1_s0" and "rand_s2" in ids and "split0_init102" in ids
    by_id = {s.run_id: s for s in specs}
    assert by_id["wd1_s0"].split_seed == 0 and by_id["wd0_s3"].weight_decay == 0.0
    assert by_id["rand_s1"].fn_name == "rand" and by_id["split0_init101"].split_seed == 0


def reference_train(cfg, spec: RunSpec, device, n_steps: int) -> dict[str, torch.Tensor]:
    """Plain single-run loop with torch.optim.AdamW(weight_decay=λ) — the paper's released training code, unbatched,
    except for float64 cross-entropy (as in the paper's notebook and ARENA; the released helpers.py casts to float32)."""
    torch.use_deterministic_algorithms(True)
    model = GrokTransformer(seed=spec.init_seed).to(device)
    split = task.paper_split(cfg.p, cfg.frac_train, spec.split_seed)
    x = task.all_pairs(cfg.p)[split.train_idx].to(device)
    y = task.make_labels(cfg.p)[split.train_idx].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=spec.weight_decay, betas=(cfg.beta1, cfg.beta2))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / cfg.warmup_steps, 1))
    for _ in range(n_steps):
        losses.cross_entropy_f64(model(x)[:, -1], y).backward()
        opt.step()
        sched.step()
        opt.zero_grad()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


@pytest.fixture(scope="module")
def batched_runs(cfg, device, tmp_path_factory):
    """Three runs trained together for 200 steps (one with λ = 0) — written to a temp runs dir."""
    out = tmp_path_factory.mktemp("runs")
    specs = [RunSpec("a", "wd1", 0, 0, 1.0), RunSpec("b", "wd0", 1, 1, 0.0), RunSpec("c", "wd1", 2, 0, 1.0)]
    train_runs(cfg, specs, device, out, num_steps=200)
    return out, specs


@pytest.mark.parametrize("spec", [RunSpec("a", "wd1", 0, 0, 100.0), RunSpec("b", "wd0", 1, 1, 0.0)])
def test_batched_training_matches_unbatched_adamw(cfg, device, tmp_path, spec):
    """A batched run (R = 1) matches an independent torch.optim.AdamW run, with strong and with zero weight decay.

    Compared after 4 steps only: early full-batch Adam training is chaotic — rounding differences between
    mathematically identical kernels are amplified by orders of magnitude within tens of steps (measured max parameter
    difference: 3e-7 after 20 steps → 2e-2 after 50 → 0.24 after 200, see reports/exp-02-training.md), so long
    horizons would test floating-point noise, not correctness. λ = 100 makes a
    wrong decay visible: each step shrinks weights by lr·λ = up to 3 %, far above rounding noise.
    """
    train_runs(cfg, [spec], device, tmp_path, num_steps=4)
    ref = reference_train(cfg, spec, device, 4)
    got = RunStore(tmp_path / spec.run_id).state_dict(-1)
    for k in ref:
        assert torch.allclose(got[k], ref[k], atol=1e-6, rtol=1e-5), (spec.run_id, k, (got[k] - ref[k]).abs().max())


def test_batched_runs_do_not_interact(cfg, device, tmp_path):
    """Run "a" ends bitwise identical whichever other run shares its batch (λ = 0 partner or λ = 1 partner)."""
    a = RunSpec("a", "wd1", 0, 0, 1.0)
    train_runs(cfg, [a, RunSpec("b", "wd0", 1, 1, 0.0)], device, tmp_path / "x", num_steps=40)
    train_runs(cfg, [a, RunSpec("c", "wd1", 5, 5, 1.0)], device, tmp_path / "y", num_steps=40)
    x, y = RunStore(tmp_path / "x" / "a").state_dict(-1), RunStore(tmp_path / "y" / "a").state_dict(-1)
    assert all(torch.equal(x[k], y[k]) for k in x)


def test_first_step_has_zero_learning_rate_and_format(batched_runs):
    """Paper warm-up starts at lr 0 → train loss identical before/after step 1; checkpoints every 100 steps."""
    out, specs = batched_runs
    store = RunStore(out / "a")
    c = store.curves()
    assert c["lr"][0] == 0.0 and c["lr"][1] == pytest.approx(1e-4) and c["lr"][20] == pytest.approx(1e-3)
    assert c["train_loss"][0] == c["train_loss"][1]
    assert c["train_loss"][-1] < 0.5 * c["train_loss"][0]
    assert store.ckpt_steps == [0, 100, 200]
    assert np.isfinite(c["grad_norm"]).all() and c["test_loss"].shape == (20,)
    assert not (out / "a" / "_tmp").exists() and (out / "a" / "optim.safetensors").exists()


def test_resumed_training_is_bitwise_identical(cfg, device, tmp_path):
    """Crash after 40 steps (resume state saved every 20), resume, finish 60 → identical to an uninterrupted run."""
    specs = [RunSpec("r1", "wd1", 2, 2, 1.0), RunSpec("r2", "wd0", 3, 3, 0.0)]
    train_runs(cfg, specs, device, tmp_path / "straight", num_steps=60, resume_every=20)
    train_runs(cfg, specs, device, tmp_path / "crashy", num_steps=60, resume_every=20, stop_after=40)
    assert not (tmp_path / "crashy" / "r1" / "meta.json").exists()  # the "crash" left an unfinished run
    train_runs(cfg, specs, device, tmp_path / "crashy", num_steps=60, resume_every=20)  # resumes at step 40
    for rid in ("r1", "r2"):
        a, b = RunStore(tmp_path / "straight" / rid), RunStore(tmp_path / "crashy" / rid)
        assert all(torch.equal(a.state_dict(t)[k], b.state_dict(t)[k]) for t in range(len(a.ckpt_steps)) for k in a.state_dict(0))
        assert np.array_equal(a.curves()["train_loss"], b.curves()["train_loss"])
    assert not (tmp_path / "crashy" / "_resume").exists()


def test_training_is_bitwise_deterministic(cfg, device, tmp_path):
    """Two identical batched trainings give identical weights."""
    specs = [RunSpec("d", "wd1", 3, 3, 1.0), RunSpec("e", "wd1", 4, 4, 1.0)]
    train_runs(cfg, specs, device, tmp_path / "x", num_steps=60)
    train_runs(cfg, specs, device, tmp_path / "y", num_steps=60)
    a, b = RunStore(tmp_path / "x" / "d").state_dict(-1), RunStore(tmp_path / "y" / "d").state_dict(-1)
    assert all(torch.equal(a[k], b[k]) for k in a)
