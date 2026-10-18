"""Phase `train_all_runs`: retrain the paper's architecture from scratch for every run in the run matrix.

Local explanation: one run = (initialisation seed, data-split seed, weight decay, task). Training replicates
the paper's released code: full-batch AdamW (lr 1e-3, betas (0.9, 0.98)), LambdaLR warm-up min(step/10, 1)
— so the very first step has learning rate 0 —, cross-entropy over all 114 outputs, trainable MLP biases,
checkpoints every `ckpt_every` steps. Unlike the paper we log densely: per-step train loss/accuracy, test
loss/accuracy every `test_every` steps, per-parameter weight/gradient/update norms every `norms_every`
steps, and Adam's moment estimates every `optim_every` steps (they cannot be recomputed afterwards).
Speed: the model is tiny, so one run leaves the GPU mostly idle. All runs are therefore trained *together*:
their parameters are stacked along a leading run axis and the same forward pass is vectorised over runs with
torch.func.vmap. Each run still has its own data, labels and weight decay, and every operation (forward,
loss, Adam's element-wise moments, decoupled weight decay) acts per run, so the runs are mathematically
independent — tested against an unbatched torch.optim.AdamW reference in tests/test_train.py.
Global context: these runs provide error bars across seeds, the controls (no weight decay; random labels;
same split, new initialisation) and optimiser-state data. The run matrix and hypotheses are fixed in
docs/preregistration.md before training.
Sources: paper code Trainer/train_model (https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/transformers.py);
AdamW decoupled decay (https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html);
model ensembling with vmap (https://pytorch.org/tutorials/intermediate/ensembling.html);
determinism (https://pytorch.org/docs/stable/notes/randomness.html).
"""

from __future__ import annotations  # postponed annotations

import gc  # release memmaps before deleting their files
import logging  # module logger
import shutil  # remove temporary memmaps
import time  # timing / ETA
from dataclasses import dataclass  # run specification
from pathlib import Path  # paths

import numpy as np  # memmaps and curves
import torch  # training
from safetensors.torch import save_file  # Adam moments
from torch.func import functional_call, stack_module_state, vmap  # vectorised ensembles

from grokviz import task  # data
from grokviz.config import Config  # settings
from grokviz.context import PipelineContext  # phase argument
from grokviz.envcheck import collect_environment, resolve_device  # provenance, device
from grokviz.model import GrokTransformer, ModelShape  # network
from grokviz.runstore import write_run  # on-disk format

log = logging.getLogger("grokviz.train")


@dataclass(frozen=True)
class RunSpec:
    """Everything that distinguishes one training run."""

    run_id: str  # folder name, e.g. "wd1_s3"
    group: str  # "wd1" | "wd0" | "rand" | "split0"
    init_seed: int  # seed of the weight initialisation
    split_seed: int  # python-random seed of the train/test shuffle (0 = the paper's split)
    weight_decay: float  # AdamW λ
    fn_name: str = "add"  # task: "add" or "rand" (random labels)
    label_seed: int | None = None  # seed of the random-label table


def run_matrix(cfg: Config) -> list[RunSpec]:
    """The preregistered run matrix (docs/preregistration.md), built from the seed lists in .env."""
    specs = [RunSpec(f"wd1_s{s}", "wd1", s, s, cfg.weight_decay) for s in cfg.wd1_seeds]
    specs += [RunSpec(f"wd0_s{s}", "wd0", s, s, 0.0) for s in cfg.wd0_seeds]
    specs += [RunSpec(f"rand_s{s}", "rand", s, s, cfg.weight_decay, "rand", s) for s in cfg.rand_seeds]
    specs += [RunSpec(f"split0_init{s}", "split0", s, 0, cfg.weight_decay) for s in cfg.split0_init_seeds]
    return specs


def _data(cfg: Config, specs: list[RunSpec], device) -> tuple[torch.Tensor, ...]:
    """Stacked train/test inputs and labels: [R, n_train, 3], [R, n_train], [R, n_test, 3], [R, n_test]."""
    tokens = task.all_pairs(cfg.p)
    xs, ys, xt, yt = [], [], [], []
    for s in specs:
        split = task.paper_split(cfg.p, cfg.frac_train, s.split_seed)
        labels = task.make_labels(cfg.p, s.fn_name, s.label_seed)
        xs.append(tokens[split.train_idx])  # training inputs in the paper's shuffled order
        ys.append(labels[split.train_idx])
        xt.append(tokens[split.test_idx])
        yt.append(labels[split.test_idx])
    return tuple(torch.stack(v).to(device) for v in (xs, ys, xt, yt))


def _ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-run float64 cross-entropy: logits [R, B, C], labels [R, B] → [R] (mean over B)."""
    logprobs = torch.log_softmax(logits.double(), dim=-1)
    return -logprobs.gather(-1, labels[..., None])[..., 0].mean(dim=1)


def _per_run_norm(t: torch.Tensor) -> torch.Tensor:
    """ℓ2 norm of each run's slice of a stacked tensor [R, ...] → [R] (float64)."""
    return t.detach().double().flatten(1).norm(dim=1)


def train_runs(cfg: Config, specs: list[RunSpec], device, runs_dir: Path, num_steps: int | None = None) -> None:
    """Train all `specs` together (vmap over the run axis) and write each run to runs_dir/<run_id>/."""
    torch.use_deterministic_algorithms(True)  # bitwise-reproducible kernels (needs CUBLAS_WORKSPACE_CONFIG)
    n_steps = num_steps or cfg.num_steps
    R = len(specs)
    shape = ModelShape.from_config(cfg)
    models = [GrokTransformer(shape, seed=s.init_seed).to(device) for s in specs]  # same init as unbatched code
    params, buffers = stack_module_state(models)  # {name: [R, ...]} leaf tensors with requires_grad
    names = list(params)
    base = GrokTransformer(shape).to("meta")  # stateless template for functional_call

    def one(p, b, x):
        """Logits at the '=' position for one run's parameters p, buffers b and inputs x."""
        return functional_call(base, (p, b), (x,), {"last_only": True})[:, -1]

    batched = vmap(one)  # maps over the leading run axis of p, b and x
    xs, ys, xt, yt = _data(cfg, specs, device)
    lam = torch.tensor([s.weight_decay for s in specs], device=device)  # per-run decoupled weight decay λ_r
    opt = torch.optim.AdamW(params.values(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(step / cfg.warmup_steps, 1))
    ckpt_steps = list(range(0, n_steps + 1, cfg.ckpt_every))  # 0, 100, 200, ... (step 0 = initialisation)
    if ckpt_steps[-1] != n_steps:  # always keep the final state, even when n_steps is not a multiple
        ckpt_steps.append(n_steps)
    ckpt_slot = {step: i for i, step in enumerate(ckpt_steps)}  # optimizer step → checkpoint index
    n_ckpt = len(ckpt_steps)
    n_opt = n_steps // cfg.optim_every
    n_test = (n_steps - 1) // cfg.test_every + 1  # steps 0, test_every, ... below n_steps
    n_norm = (n_steps - 1) // cfg.norms_every + 1  # steps 0, norms_every, ... below n_steps
    tmp = {s.run_id: runs_dir / s.run_id / "_tmp" for s in specs}
    for d in tmp.values():
        d.mkdir(parents=True, exist_ok=True)
    full = {**params, **buffers}  # buffers (the causal mask) are part of the state_dict
    state_names = list(full)

    def mm(run_id: str, key: str, n: int, item_shape) -> np.memmap:
        """Disk-backed [n, ...] array for one run (keeps RAM flat whatever the number of checkpoints)."""
        return np.lib.format.open_memmap(tmp[run_id] / f"{key}.npy", mode="w+", dtype=np.float32, shape=(n, *item_shape))

    ck = {s.run_id: {k: mm(s.run_id, k, n_ckpt, full[k].shape[1:]) for k in state_names} for s in specs}
    om = {s.run_id: {f"{m}.{k}": mm(s.run_id, f"{m}.{k}", n_opt, params[k].shape[1:]) for k in names for m in ("exp_avg", "exp_avg_sq")}
          for s in specs}
    g = {k: torch.zeros(R, n, dtype=torch.float64, device=device) for k, n in
         (("train_loss", n_steps), ("train_acc", n_steps), ("test_loss", n_test), ("test_acc", n_test))}
    gn = {k: torch.zeros(R, n_norm, len(names), dtype=torch.float64, device=device) for k in ("grad_norm", "weight_norm", "update_norm")}
    lrs = np.zeros(n_steps)

    def save_ckpt(i: int) -> None:
        """Copy every run's current state into checkpoint slot i."""
        for k in state_names:
            values = full[k].detach().cpu().numpy()
            for r, s in enumerate(specs):
                ck[s.run_id][k][i] = values[r]

    save_ckpt(0)
    start = time.perf_counter()
    for step in range(n_steps):  # `step` = optimizer steps already applied
        logits = batched(params, buffers, xs)  # [R, n_train, 114]
        per_run = _ce(logits, ys)  # [R] training losses
        g["train_loss"][:, step] = per_run.detach()
        g["train_acc"][:, step] = (logits.argmax(-1) == ys).double().mean(1)
        if step % cfg.test_every == 0 and step // cfg.test_every < n_test:
            with torch.no_grad():
                tl = batched(params, buffers, xt)
                g["test_loss"][:, step // cfg.test_every] = _ce(tl, yt)
                g["test_acc"][:, step // cfg.test_every] = (tl.argmax(-1) == yt).double().mean(1)
        lr = opt.param_groups[0]["lr"]
        lrs[step] = lr
        per_run.sum().backward()  # runs are independent, so the sum's gradient is each run's own gradient
        track = step % cfg.norms_every == 0 and step // cfg.norms_every < n_norm
        if track:
            j = step // cfg.norms_every
            before = {k: v.detach().clone() for k, v in params.items()}
            gn["grad_norm"][:, j] = torch.stack([_per_run_norm(params[k].grad) for k in names], 1)
            gn["weight_norm"][:, j] = torch.stack([_per_run_norm(params[k]) for k in names], 1)
        with torch.no_grad():  # decoupled weight decay θ ← θ(1 − lr·λ_r), applied before Adam like torch's AdamW
            for v in params.values():
                v.mul_((1 - lr * lam).view(R, *[1] * (v.dim() - 1)).to(v.dtype))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if track:
            gn["update_norm"][:, j] = torch.stack([_per_run_norm(params[k] - before[k]) for k in names], 1)
        done = step + 1
        if done in ckpt_slot:
            save_ckpt(ckpt_slot[done])
        if done % cfg.optim_every == 0:
            i = done // cfg.optim_every - 1
            for k in names:
                st = opt.state[params[k]]
                for m in ("exp_avg", "exp_avg_sq"):
                    values = st[m].detach().cpu().numpy()
                    for r, s in enumerate(specs):
                        om[s.run_id][f"{m}.{k}"][i] = values[r]
        if done % 2000 == 0 or done == n_steps:
            elapsed = time.perf_counter() - start
            ti = min(step // cfg.test_every, n_test - 1)
            summary = ", ".join(f"{s.run_id}: {g['test_acc'][r, ti]:.3f}" for r, s in enumerate(specs[:6]))
            log.info("step %6d/%d (%.0f s, ETA %.0f s) test-acc %s%s", done, n_steps, elapsed, elapsed / done * (n_steps - done),
                     summary, " ..." if R > 6 else "")
    seconds = time.perf_counter() - start
    env = collect_environment(device)
    curves_all = {k: v.cpu().numpy() for k, v in {**g, **gn}.items()}
    for r, s in enumerate(specs):
        stacked = {k: torch.from_numpy(np.array(ck[s.run_id][k])) for k in state_names}  # copy out of the memmap
        curves = {k: v[r] for k, v in curves_all.items()}
        curves.update({"lr": lrs, "train_loss_steps": np.arange(n_steps), "test_loss_steps": np.arange(n_test) * cfg.test_every,
                       "norms_steps": np.arange(n_norm) * cfg.norms_every})
        meta = {
            "run_id": s.run_id, "kind": "trained", "group": s.group,
            "title": f"{s.group}: init seed {s.init_seed}, split seed {s.split_seed}, weight decay {s.weight_decay}, task {s.fn_name}",
            "fn_name": s.fn_name, "label_seed": s.label_seed, "split_seed": s.split_seed, "init_seed": s.init_seed,
            "frac_train": cfg.frac_train, "lr": cfg.lr, "weight_decay": s.weight_decay, "betas": [cfg.beta1, cfg.beta2],
            "warmup_steps": cfg.warmup_steps, "num_steps": n_steps,
            "ckpt_steps": ckpt_steps,
            "optim_steps": [(i + 1) * cfg.optim_every for i in range(n_opt)],
            "param_names": names, "model_shape": shape.__dict__, "deterministic": True,
            "trained_with": [x.run_id for x in specs], "train_seconds_all_runs": seconds, "environment": env,
        }
        write_run(runs_dir / s.run_id, meta, stacked, curves)
        save_file({k: torch.from_numpy(np.array(v)).contiguous() for k, v in om[s.run_id].items()}, str(runs_dir / s.run_id / "optim.safetensors"))
        del stacked
    ck.clear()  # drop memmap references so Windows releases the files
    om.clear()
    gc.collect()
    for d in tmp.values():
        shutil.rmtree(d, ignore_errors=True)


def train_all_runs(ctx: PipelineContext) -> None:
    """Phase entry point: train (together) every selected run that does not exist yet (all with --force)."""
    device = resolve_device(ctx.cfg.device)
    todo = [s for s in run_matrix(ctx.cfg) if ctx.wants(s.run_id) and (ctx.force or not (ctx.cfg.runs_dir / s.run_id / "meta.json").exists())]
    if not todo:
        log.info("all selected runs already exist")
        return
    log.info("training %d runs together: %s", len(todo), ", ".join(s.run_id for s in todo))
    train_runs(ctx.cfg, todo, device, ctx.cfg.runs_dir)
    log.info("finished %d runs", len(todo))
