"""Causal tests on a trained model: ablations and replacements that test whether the Fourier story is faithful.

Local explanation: each function changes one part of the computation (logit Fourier components, MLP
directions, skip connections, neurons, attention softmax) and reports the resulting loss, so the effect can
be compared with the unmodified model and with random controls.
Global context: these are the paper's "fourth line of evidence" (Sec. 4.4 + App. A.1, C.1.3) and ARENA's
cluster validations, computed for any run; results feed reports/exp-01 (paper comparison), the
interventions chapter and dashboard group G. Definitions follow docs/operationalizations.md.
Sources: paper Sec. 4.4, App. A.1, App. C.1.3 (https://arxiv.org/abs/2301.05217); ARENA 1.5.2 "Exercise -
validate neuron clusters" and "ReLU as multiplier" regression
(https://learn.arena.education/chapter1_transformer_interp/32_grokking/2-circuit-and-feature-analysis/).
"""

from __future__ import annotations  # postponed annotations

import math  # sqrt, pi

import torch  # tensors

from grokviz import fourier, losses  # Fourier tools, losses
from grokviz.metrics import Acts, Grid, paper_matrices, wl_residual_sq_frac  # shared containers and helpers
from grokviz.model import GrokTransformer  # network


def freq_block_indices(p: int, k: int) -> list[int]:
    """Flattened 2D-Fourier indices (i*p+j) of the 8 terms involving frequency k (linear + quadratic, no Const⊗Const)."""
    idx = (0, 2 * k - 1, 2 * k)
    return [i * p + j for i in idx for j in idx if (i, j) != (0, 0)]


def ablate_logit_freqs(logits: torch.Tensor, grid: Grid, zero: list[int] | None = None, keep_only: list[int] | None = None) -> torch.Tensor:
    """Zero chosen 2D-Fourier components of every output's logit grid (paper Sec. 4.4 "Logit frequencies").

    zero=[k,...]: remove the 8 components of each listed frequency. keep_only=[k,...]: keep only those
    components (5 key freqs → 40 components; everything else, including Const⊗Const, is zeroed).
    """
    p = grid.p
    coeffs = fourier.fft2d(logits, grid.F)  # [p², p] in the 2D Fourier basis
    mask = torch.ones(p * p, dtype=torch.bool, device=logits.device)
    if keep_only is not None:
        mask[:] = False
        mask[[i for k in keep_only for i in freq_block_indices(p, k)]] = True
    if zero is not None:
        mask[[i for k in zero for i in freq_block_indices(p, k)]] = False
    return fourier.ifft2d(coeffs * mask[:, None], grid.F)


def freq_ablation_table(acts: Acts, grid: Grid, key_freqs: list[int], n_random: int = 100, seed: int = 0) -> dict:
    """Loss (all pairs) after ablating each single frequency, all non-key frequencies, and random controls.

    Random control: ablate len(key_freqs) frequencies drawn uniformly from the non-key ones, `n_random`
    times, giving the distribution expected if the key frequencies were not special.
    """
    L, y = acts.logits, grid.labels
    single = [losses.grid_loss(ablate_logit_freqs(L, grid, zero=[k]), y).item() for k in range(1, grid.p // 2 + 1)]
    gen = torch.Generator().manual_seed(seed)
    others = [k for k in range(1, grid.p // 2 + 1) if k not in key_freqs]
    random_losses = []
    for _ in range(n_random):
        pick = [others[i] for i in torch.randperm(len(others), generator=gen)[: len(key_freqs)].tolist()]
        random_losses.append(losses.grid_loss(ablate_logit_freqs(L, grid, zero=pick), y).item())
    return {
        "original": losses.grid_loss(L, y).item(),
        "single_freq": single,  # index k-1 ↔ frequency k
        "all_key_removed": losses.grid_loss(ablate_logit_freqs(L, grid, zero=key_freqs), y).item(),
        "only_key_kept": losses.grid_loss(ablate_logit_freqs(L, grid, keep_only=key_freqs), y).item(),
        "random_nonkey_removed": random_losses,
    }


def logits_from_post(model: GrokTransformer, acts: Acts, post: torch.Tensor, resid: torch.Tensor | None = None) -> torch.Tensor:
    """Recompute '=' logits [p², p] from (possibly modified) neuron activations; `resid` replaces the skip input."""
    m = model.blocks[0].mlp
    W_U = model.unembed.W_U.double()
    base = acts.resid_mid if resid is None else resid
    return ((base + post @ m.W_out.double().T + m.b_out.double()) @ W_U)[:, : acts.logits.shape[1]]


def wl_direction_projection(model: GrokTransformer, acts: Acts, grid: Grid, key_freqs: list[int]) -> dict:
    """Project MLP activations onto the 10-dim span of W_L's key cos/sin rows, or onto its orthogonal complement.

    Paper Sec. 4.4 "Directions in W_L" reports 1.19e-7 (projection) and 5.27 (nullspace); we report the
    variants we could compute (with/without the skip connection) because the paper does not say which.
    """
    W_L = paper_matrices(model, grid.p)["W_L"]  # [p, n]
    rows = fourier.key_rows(grid.F, key_freqs).to(W_L.dtype)  # [10, p]
    Q, _ = torch.linalg.qr((rows @ W_L).T)  # orthonormal basis [n, 10] of the read-off directions u_k, v_k
    proj = acts.post @ Q @ Q.T
    null = acts.post - proj
    y = grid.labels
    out = {}
    for name, post in (("projection", proj), ("nullspace", null)):
        with_skip = logits_from_post(model, acts, post)
        mlp_only = post @ W_L.T
        out[name] = {
            "with_skip": losses.grid_loss(with_skip, y).item(),
            "with_skip_bias_corrected": losses.grid_loss(losses.bias_correct(with_skip, acts.logits), y).item(),
            "mlp_only": losses.grid_loss(mlp_only, y).item(),
            "mlp_only_bias_corrected": losses.grid_loss(losses.bias_correct(mlp_only, acts.logits), y).item(),
        }
    out["wl_residual_sq_frac"] = wl_residual_sq_frac(W_L, rows)  # fraction of squared norm (≈ paper's 0.55 %)
    out["wl_residual_norm_frac"] = out["wl_residual_sq_frac"] ** 0.5  # Frobenius ratio
    return out


@torch.no_grad()
def skip_and_attention_ablations(model: GrokTransformer, acts: Acts, grid: Grid) -> dict:
    """Zero/mean-ablate the skip around the MLP, zero the attention output, zero the skip around attention (paper App. A.1)."""
    y, p = grid.labels, grid.p
    W_U = model.unembed.W_U.double()
    mlp_out = acts.post @ model.blocks[0].mlp.W_out.double().T + model.blocks[0].mlp.b_out.double()
    zero_mlp_skip = (mlp_out @ W_U)[:, :p]  # resid_post = MLP output only
    mean_mlp_skip = ((acts.resid_mid.mean(0, keepdim=True) + mlp_out) @ W_U)[:, :p]  # skip replaced by its mean
    store: dict[str, torch.Tensor] = {}

    def keep_attn(x):  # remember the attention output so the next hook can use it
        store["attn_out"] = x

    def drop_skip(x):  # resid_mid = attention output only (skip around attention removed)
        return store["attn_out"]

    zero_attn = model(grid.tokens, hooks={"blocks.0.attn_out": lambda x: torch.zeros_like(x)})[:, -1, :p]
    zero_attn_skip = model(grid.tokens, hooks={"blocks.0.attn_out": keep_attn, "blocks.0.resid_mid": drop_skip})[:, -1, :p]
    return {
        "mlp_skip_zero": losses.grid_loss(zero_mlp_skip, y).item(),
        "mlp_skip_mean": losses.grid_loss(mean_mlp_skip, y).item(),
        "attention_zero": losses.grid_loss(zero_attn.double(), y).item(),
        "attention_skip_zero": losses.grid_loss(zero_attn_skip.double(), y).item(),
        "attention_to_eq_per_head": acts.pattern[:, :, 2].mean(0).tolist(),
    }


def neuron_polynomial_replacement(model: GrokTransformer, acts: Acts, grid: Grid, freqs: torch.Tensor, fve: torch.Tensor,
                                  key_freqs: list[int], threshold: float = 0.85) -> dict:
    """Replace clustered neurons by their degree-2 polynomial (3×3 block of their frequency); strict version keeps only cos/sin(w_k(a+b)).

    Paper Sec. 4.4 "MLP neurons": 2.41e-7 → 2.48e-7 (polynomial), → 5.54e-8 (strict, all neurons restricted
    to the key cos/sin(a+b) terms). ARENA's cluster projection (9 terms incl. Const) is the same as the first.
    """
    p = grid.p
    post = acts.post.clone()
    clustered = torch.nonzero(fve > threshold).flatten().tolist()
    for n in clustered:  # project each clustered neuron onto the 9 2D terms (Const/cos k/sin k)² of its frequency
        k = int(freqs[n])
        dirs = torch.stack([fourier.basis_term_2d(grid.F, i, j) for i in (0, 2 * k - 1, 2 * k) for j in (0, 2 * k - 1, 2 * k)])
        post[:, n] = fourier.project(acts.post[:, n : n + 1], dirs.to(post.dtype))[:, 0]
    strict = fourier.project(acts.post, fourier.trig_dirs(grid.F, key_freqs))  # only cos/sin(w_k(a+b)) per neuron
    y = grid.labels
    W_L = paper_matrices(model, p)["W_L"]

    def mlp_only(pp):
        """Paper simplification Logits ≈ W_L·MLP (skip ignored, App. A.1), bias-corrected as in ARENA."""
        return losses.grid_loss(losses.bias_correct(pp @ W_L.T, acts.logits), y).item()

    return {
        "n_replaced": len(clustered),
        "polynomial": mlp_only(post),  # primary: reproduces paper 2.48e-7 / ARENA 2.482e-7
        "strict_trig_only": mlp_only(strict),  # primary: paper 5.54e-8
        "polynomial_with_skip": losses.grid_loss(logits_from_post(model, acts, post), y).item(),
        "strict_trig_only_with_skip": losses.grid_loss(
            losses.bias_correct(logits_from_post(model, acts, strict + acts.post.mean(0, keepdim=True)), acts.logits), y).item(),
    }


def cluster_ablations(model: GrokTransformer, acts: Acts, grid: Grid, freqs: torch.Tensor, fve: torch.Tensor,
                      key_freqs: list[int], threshold: float = 0.85) -> dict:
    """ARENA cluster validation: logits from each cluster's in-frequency projection; loss when each cluster is excluded."""
    p, y = grid.p, grid.labels
    W_L = paper_matrices(model, p)["W_L"]
    labels = torch.where(fve > threshold, freqs, torch.full_like(freqs, -1))  # -1 = always-firing / unclustered
    parts = {}
    for k in key_freqs + [-1]:
        sel = labels == k
        acts_k = acts.post[:, sel]
        if k != -1:  # keep only Const/linear/quadratic terms of k (ARENA project_onto_frequency)
            dirs = torch.stack([fourier.basis_term_2d(grid.F, i, j) for i in (0, 2 * k - 1, 2 * k) for j in (0, 2 * k - 1, 2 * k)])
            acts_k = fourier.project(acts_k, dirs.to(acts_k.dtype))
        parts[k] = acts_k @ W_L[:, sel].T  # this cluster's contribution to the logits
    total = sum(parts.values())
    excl = {str(k): losses.grid_loss(losses.bias_correct(total - parts[k], acts.logits), y).item() for k in parts}
    no_always = total - parts[-1]
    return {
        "cluster_sizes": {str(k): int((labels == k).sum()) for k in parts},
        "loss_key_projection": losses.grid_loss(losses.bias_correct(total, acts.logits), y).item(),
        "loss_key_projection_no_always_firing": losses.grid_loss(losses.bias_correct(no_always, acts.logits), y).item(),
        "loss_excluding_cluster": excl,
    }


def fit_attention_heads(model: GrokTransformer, acts: Acts, grid: Grid) -> dict:
    """Per head: best single-frequency fit of C_j, of the attention pattern (paper Table 2), and the linear-softmax replacement.

    Table 2 form: A_0 ≈ 0.5 + α(cos w a − cos w b) + β(sin w a − sin w b). Linearisation (App. C.1.3):
    attention to a = β0 + E·(score_a − score_b), to b = 1 − that, to '=' = 0, with β0, E fitted by least squares.
    """
    from grokviz.metrics import attention_lookup

    p, F = grid.p, grid.F
    C = attention_lookup(model, p)  # [heads, p] pre-softmax score lookup tables
    a = grid.tokens[:, 0]
    b = grid.tokens[:, 1]
    heads = []
    hooks_patterns = []
    for h in range(C.shape[0]):
        coeff = C[h] @ F.T  # Fourier coefficients of C_j
        k = int(fourier.frequency_norms(coeff).argmax()) + 1  # dominant frequency
        X = torch.stack([torch.ones(p, device=C.device, dtype=C.dtype), F[2 * k - 1], F[2 * k]], 1)
        fit = X @ torch.linalg.lstsq(X, C[h][:, None]).solution
        fve_C = float(1 - ((C[h] - fit[:, 0]) ** 2).sum() / ((C[h] - C[h].mean()) ** 2).sum())
        w = 2 * math.pi * k / p
        A0 = acts.pattern[:, h, 0]  # attention from '=' to a
        Xa = torch.stack([torch.cos(w * a) - torch.cos(w * b), torch.sin(w * a) - torch.sin(w * b)], 1).double()
        ab = torch.linalg.lstsq(Xa, (A0 - 0.5)[:, None]).solution[:, 0]  # α, β with intercept fixed at 0.5
        fve_A = float(1 - ((A0 - 0.5 - Xa @ ab) ** 2).sum() / ((A0 - A0.mean()) ** 2).sum())
        z = (C[h][a] - C[h][b]).double()  # score difference a vs b
        Xl = torch.stack([torch.ones_like(z), z], 1)
        lin = torch.linalg.lstsq(Xl, A0[:, None]).solution[:, 0]  # β0, E
        fve_lin = float(1 - ((A0 - Xl @ lin) ** 2).sum() / ((A0 - A0.mean()) ** 2).sum())
        heads.append({"head": h, "freq": k, "C_fve": fve_C, "alpha": float(ab[0]), "beta": float(ab[1]), "pattern_fve": fve_A,
                      "linear_beta0": float(lin[0]), "linear_E": float(lin[1]), "linear_fve": fve_lin})
        hooks_patterns.append((Xl @ lin).float())

    def linear_softmax(x):  # replace each head's '=' row by the fitted linear function of the score difference
        x = x.clone()
        for h, a_att in enumerate(hooks_patterns):
            x[:, h, 2, 0] = a_att
            x[:, h, 2, 1] = 1 - a_att
            x[:, h, 2, 2] = 0.0
        return x

    with torch.no_grad():
        lin_logits = model(grid.tokens, hooks={"blocks.0.attn.pattern": linear_softmax})[:, -1, :p].double()
    return {"heads": heads, "linearised_softmax_loss": losses.grid_loss(lin_logits, grid.labels).item(),
            "linear_fve_mean": float(sum(hh["linear_fve"] for hh in heads) / len(heads))}


def table1_readoff(model: GrokTransformer, acts: Acts, grid: Grid, key_freqs: list[int]) -> list[dict]:
    """Paper Table 1: project MLP activations on u_k, v_k (W_L's cos/sin rows) and measure the FVE of one cos/sin(w_k(a+b)) term."""
    W_L = paper_matrices(model, grid.p)["W_L"]
    centred = acts.post - acts.post.mean(0, keepdim=True)
    rows = []
    for k in key_freqs:
        for kind, idx, direction in (("cos", 2 * k - 1, fourier.cos_apb_dir(grid.F, k)), ("sin", 2 * k, fourier.sin_apb_dir(grid.F, k))):
            u = W_L.T @ grid.F[idx]  # neuron-space direction that writes cos(w_k c) (or sin) into the logits
            readoff = centred @ u  # [p²]
            coef = float(direction.to(readoff.dtype) @ readoff)  # coefficient on the unit cos/sin(w_k(a+b)) grid
            fve = coef**2 / float(readoff.pow(2).sum())
            rows.append({"freq": k, "component": kind, "coef_unit_dir": coef, "fve": fve})
    return rows


def relu_multiplier_regression(p: int, k: int = 42) -> dict:
    """ARENA: ReLU(A + cos w a + cos w b) ≈ α + β cos w a + β cos w b + γ cos w a cos w b; r² with/without γ.

    Uses unit-normalised basis vectors and A = 0.5/√p exactly as ARENA's code (its printed label says 0.5).
    """
    F, _ = fourier.fourier_basis(p)
    v = F[2 * k - 1]  # unit-norm cos(w_k x)
    y = torch.relu(0.5 * p**-0.5 + v[None, :] + v[:, None]).flatten()  # [a, b] grid of the ReLU output
    terms = [fourier.basis_term_2d(F, i, j) for i, j in ((0, 0), (2 * k - 1, 0), (0, 2 * k - 1), (2 * k - 1, 2 * k - 1))]
    X = torch.stack(terms, 1)

    def r2(design):
        sol = torch.linalg.lstsq(design, y[:, None]).solution
        resid = y - (design @ sol)[:, 0]
        return float(1 - resid.pow(2).sum() / (y - y.mean()).pow(2).sum()), sol[:, 0].tolist()

    r2_full, coefs = r2(X)
    r2_lin, _ = r2(X[:, :3])
    return {"k": k, "coefs": coefs, "r2_with_quadratic": r2_full, "r2_without_quadratic": r2_lin}
