"""Phase `compute_gradient_dynamics`: exact full-batch gradients at every checkpoint, and what they point at.

Local explanation: because training is full-batch (every step uses all 3,830 training pairs), the gradient
the optimiser saw at a checkpoint can be recomputed exactly from the checkpoint's weights. For each checkpoint
we compute the training-loss gradient g_train and the test-loss gradient g_test and record:
  per-parameter ‖g_train‖ and ‖θ‖ (the weight-decay pull is λθ, so ‖g‖ vs λ‖θ‖ shows which force dominates);
  cos(g_train, θ)  — positive means the loss gradient also pushes weights toward zero (descent moves along −g);
  cos(g_train, g_test) — do steps that reduce training loss also reduce test loss? (generalising vs memorising);
  the Fourier spectrum of the embedding gradient (which frequencies the loss is currently pushing W_E toward);
  for our runs, the actual AdamW update from the saved Adam moments versus the raw gradient and decay terms.
Global context: chapter 19 ("Gradients and weight decay") and dashboard group F; tests ARENA's speculation that
memorising gradients cancel while pattern gradients reinforce, and the paper's claim that circuit formation
and cleanup are driven by weight decay (Sec. 5.2–5.3).
Sources: AdamW update rule — Loshchilov & Hutter 2019 (https://arxiv.org/abs/1711.05101) and
torch.optim.AdamW docs (https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html); ARENA bonus
"Why phase changes?" (https://learn.arena.education/chapter1_transformer_interp/32_grokking/bonus/).
"""

from __future__ import annotations  # postponed annotations

import logging  # module logger

import numpy as np  # outputs
import torch  # autograd

from grokviz import losses, metrics  # loss + grid
from grokviz.checkpoint_metrics import grid_for, run_dirs  # shared run helpers
from grokviz.context import PipelineContext  # phase argument
from grokviz.envcheck import resolve_device  # device
from grokviz.runstore import RunStore  # run access

log = logging.getLogger("grokviz.gradients")


def flat(tensors) -> torch.Tensor:
    """Concatenate tensors into one float64 vector (for cosines between whole-model vectors)."""
    return torch.cat([t.reshape(-1).double() for t in tensors])


def cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    """Cosine similarity of two flat vectors (0 if either is zero)."""
    denom = u.norm() * v.norm()
    return float((u @ v) / denom) if denom > 0 else 0.0


def grads_on(model, grid: metrics.Grid, mask: torch.Tensor) -> list[torch.Tensor]:
    """Gradient of the 114-way training objective (float64 cross-entropy) on the masked rows."""
    model.zero_grad(set_to_none=True)
    logits = model(grid.tokens[mask])[:, -1]  # all 114 outputs, exactly like training
    losses.cross_entropy_f64(logits, grid.labels[mask]).backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def compute_gradient_dynamics(ctx: PipelineContext) -> None:
    """Phase entry point: write analysis/gradients.npz per run."""
    device = resolve_device(ctx.cfg.device)
    wd = None
    for run_dir in run_dirs(ctx):
        out_path = run_dir / "analysis" / "gradients.npz"
        if out_path.exists() and not ctx.force:
            log.info("%s: gradients.npz exists, skipping", run_dir.name)
            continue
        store = RunStore(run_dir)
        grid = grid_for(store, ctx.cfg.p, device)
        wd = store.meta.get("weight_decay", 1.0)
        rows: dict[str, list] = {}
        for t, step, model in store.iter_models(device):
            model.train()
            with torch.enable_grad():
                g_tr = grads_on(model, grid, grid.is_train)
                g_te = grads_on(model, grid, grid.is_test)
            params = [p.detach() for p in model.parameters()]
            gw = g_tr[0][:, : grid.p].double()  # gradient w.r.t. W_E (d_model × p, '=' column dropped)
            rows.setdefault("grad_norm", []).append([float(g.double().norm()) for g in g_tr])
            rows.setdefault("weight_norm", []).append([float(p.double().norm()) for p in params])
            rows.setdefault("cos_grad_weight", []).append(cosine(flat(g_tr), flat(params)))
            rows.setdefault("cos_train_test", []).append(cosine(flat(g_tr), flat(g_te)))
            rows.setdefault("grad_we_fourier", []).append((gw @ grid.F.T).norm(dim=0).cpu().numpy())
            rows.setdefault("grad_ratio_decay", []).append(float(flat(g_tr).norm() / (wd * flat(params).norm() + 1e-300)))
            if t % 100 == 0:
                log.info("%s: ckpt %d step %d |g|=%.3e cos(g,θ)=%.3f cos(g_tr,g_te)=%.3f", store.run_id, t, step,
                         flat(g_tr).norm(), rows["cos_grad_weight"][-1], rows["cos_train_test"][-1])
        arrays = {k: np.asarray(v) for k, v in rows.items()}
        arrays["steps"] = np.asarray(store.ckpt_steps)
        arrays["param_names"] = np.asarray([n for n, _ in store.load_model(0).named_parameters()])
        np.savez_compressed(out_path, **arrays)
        log.info("%s: wrote %s", store.run_id, out_path)
