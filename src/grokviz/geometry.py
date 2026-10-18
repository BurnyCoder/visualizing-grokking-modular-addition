"""Phase `compute_geometry`: activation geometry and output-distribution geometry over training.

Local explanation: for a subset of checkpoints ("hero" frames) we compute low-dimensional pictures:
- embedding circles: each token's embedding projected onto the 2D plane carrying cos/sin(w_k a) for each key
  frequency k — a perfect circle means the embedding encodes "angle w_k·a" (the clock picture);
- embedding PCA: top-3 principal components of the 113 token embeddings (+ explained variance);
- MLP class circles: average neuron activation over all inputs with the same answer (a+b mod p), projected on
  the read-off directions u_k, v_k of W_L — the circle on which the network represents a+b;
- logit and softmax-output PCA: class means of the logit vectors / probability vectors (output geometry);
- output statistics per input: probability of the correct answer and normalised entropy of the prediction.
Global context: dashboard group D and chapter 18 animate these frames across training (circles emerging out of
noise during circuit formation); output_stats grids show where on the (a, b) grid the model is confident.
Sources: PCA via SVD (https://en.wikipedia.org/wiki/Principal_component_analysis#Singular_value_decomposition);
clock/circle picture — paper Fig. 1 and Sec. 3.1 (https://arxiv.org/abs/2301.05217); Welch Labs linear-probe
circles (https://www.youtube.com/watch?v=D8GOeCFFby4, chapter "Linear Probes").
"""

from __future__ import annotations  # postponed annotations

import logging  # module logger

import numpy as np  # outputs
import torch  # tensors

from grokviz import metrics  # grid, evaluate, matrices
from grokviz.checkpoint_metrics import grid_for, reference_index, run_dirs  # shared helpers
from grokviz.context import PipelineContext  # phase argument
from grokviz.envcheck import resolve_device  # device
from grokviz.runstore import RunStore  # run access

log = logging.getLogger("grokviz.geometry")


def pca(x: torch.Tensor, k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k principal-component coordinates of the rows of x and their explained-variance fractions."""
    centred = x - x.mean(0, keepdim=True)
    U, S, _ = torch.linalg.svd(centred, full_matrices=False)  # centred = U S Vᵀ
    var = S**2
    return U[:, :k] * S[:k], var[:k] / var.sum()


def key_plane_coords(W_E: torch.Tensor, F: torch.Tensor, k: int) -> torch.Tensor:
    """Coordinates [p, 2] of each token's embedding in the plane spanned by W_E's cos k and sin k components."""
    plane = torch.stack([W_E @ F[2 * k - 1], W_E @ F[2 * k]], 1)  # [d, 2] residual-space directions of cos/sin(w_k a)
    Q, _ = torch.linalg.qr(plane)  # orthonormal basis of that plane
    return W_E.T @ Q  # [p, 2]


@torch.no_grad()
def frame(model, grid: metrics.Grid, K: list[int]) -> dict[str, np.ndarray]:
    """All geometry arrays for one checkpoint."""
    p = grid.p
    acts = metrics.evaluate(model, grid)
    mats = metrics.paper_matrices(model, p)
    W_E, W_L = mats["W_E"], mats["W_L"]
    answer = grid.labels  # class of each input
    onehot = torch.nn.functional.one_hot(answer, p).double()  # [p², p]
    counts = onehot.sum(0)[:, None]
    mlp_means = (onehot.T @ acts.post) / counts  # [p, n] mean activation per answer
    logit_means = (onehot.T @ acts.logits) / counts  # [p, p]
    probs = torch.softmax(acts.logits, dim=-1)
    prob_means = (onehot.T @ probs) / counts  # [p, p] mean predicted distribution per answer
    mlp_planes = []
    for k in K:
        u = W_L.T @ grid.F[2 * k - 1]  # neuron direction read off as cos(w_k c)
        v = W_L.T @ grid.F[2 * k]  # ... and as sin(w_k c)
        Q, _ = torch.linalg.qr(torch.stack([u, v], 1))
        mlp_planes.append(mlp_means @ Q)  # [p, 2]
    p_correct = probs.gather(1, answer[:, None])[:, 0]
    entropy = -(probs * torch.log(probs.clamp_min(1e-300))).sum(1) / np.log(p)  # 0 = certain, 1 = uniform
    we_pca, we_var = pca(W_E.T)
    logit_pca, logit_var = pca(logit_means)
    prob_pca, prob_var = pca(prob_means)
    return {
        "we_planes": torch.stack([key_plane_coords(W_E, grid.F, k) for k in K]).cpu().numpy(),  # [K, p, 2]
        "we_pca": we_pca.cpu().numpy(), "we_pca_var": we_var.cpu().numpy(),
        "mlp_planes": torch.stack(mlp_planes).cpu().numpy(),  # [K, p, 2]
        "logit_pca": logit_pca.cpu().numpy(), "logit_pca_var": logit_var.cpu().numpy(),
        "prob_pca": prob_pca.cpu().numpy(), "prob_pca_var": prob_var.cpu().numpy(),
        "p_correct": p_correct.float().cpu().numpy(), "entropy": entropy.float().cpu().numpy(),
    }


def compute_geometry(ctx: PipelineContext, every: int = 2) -> None:
    """Phase entry point: geometry frames for every `every`-th checkpoint → analysis/geometry.npz."""
    device = resolve_device(ctx.cfg.device)
    for run_dir in run_dirs(ctx):
        out_path = run_dir / "analysis" / "geometry.npz"
        if out_path.exists() and not ctx.force:
            log.info("%s: geometry.npz exists, skipping", run_dir.name)
            continue
        store = RunStore(run_dir)
        grid = grid_for(store, ctx.cfg.p, device)
        K = metrics.key_frequencies(store.load_model(reference_index(store), device), grid.F)
        rows: dict[str, list] = {}
        steps = []
        for t, step, model in store.iter_models(device, every=every):
            for key, value in frame(model, grid, K).items():
                rows.setdefault(key, []).append(value)
            steps.append(step)
        arrays = {k: np.stack(v) for k, v in rows.items()}
        arrays["steps"] = np.asarray(steps)
        arrays["key_freqs"] = np.asarray(K)
        np.savez_compressed(out_path, **arrays)
        log.info("%s: wrote %s (%d frames)", store.run_id, out_path, len(steps))
