"""Paper-faithful measurements of one model on the full input grid (definitions in docs/operationalizations.md).

Local explanation: `Grid` holds the p² inputs, labels, train/test masks and the Fourier basis on the compute
device; `evaluate` runs the model once on all inputs and returns the activations every metric needs; the
remaining functions turn those activations/weights into the quantities the paper reports: key frequencies,
Fourier norms of W_E and W_L, Gini sparsity, restricted/excluded loss, neuron frequency clusters, logit
cosine fits, ARENA's trig-variance fractions, attention lookup tables C_j and the sum of squared weights.
Global context: `checkpoint_metrics.py` applies these to every checkpoint of every run (the curves over
training) and `rchecks.py` applies them to the paper's epoch-40k model to compare with the paper's numbers.
Each definition was checked against the paper's code and numbers before being fixed here:
- restricted loss = paper code `calculate_trig_loss` / ARENA `trig_loss` (10 cos/sin(w_k(a+b)) directions per
  output, bias-corrected, all pairs) — reproduces ARENA's 5.5474e-8 on the paper model;
- excluded loss (subtract those directions, no bias correction, train pairs): all of K at once as in the paper's
  notebook `calculate_excluded_loss_2D_full` (paper Sec. 5.1); one frequency at a time as in paper code
  `calculate_excluded_loss` / ARENA `excl_loss`, which reproduces ARENA's per-frequency test values exactly;
- neuron frequency/FVE = paper code `calculate_key_freqs` / ARENA `find_neuron_freqs` — gives the paper's 433
  neurons above 0.85 and ARENA's cluster sizes 44/93/145/87/64.
Sources: https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/23d2c64/transformers.py;
ARENA solutions.py @85d1011 (https://github.com/ARENA-education/ARENA_materials/blob/85d10112344b68addebacb04909392e515dd7068/chapter1_transformer_interp/exercises/part52_grokking_and_modular_arithmetic/solutions.py);
paper Sec. 4–5 (https://arxiv.org/abs/2301.05217).
"""

from __future__ import annotations  # postponed annotations

import math  # pi
from dataclasses import dataclass  # containers

import torch  # tensors

from grokviz import fourier, losses, task  # Fourier tools, losses, dataset
from grokviz.model import GrokTransformer  # the network

KEY_FREQ_TAU = 0.25  # key frequency ⇔ its W_L Fourier norm ≥ τ·max (paper model: every τ in (0.060, 0.826] gives the same 5 keys)
NEURON_FVE_THRESHOLD = 0.85  # paper Sec. 4.3 / ARENA: neuron belongs to a frequency cluster if FVE > 0.85


@dataclass
class Grid:
    """All p² inputs with labels, train/test masks and the Fourier basis, on one device (float64 analysis)."""

    p: int
    tokens: torch.Tensor  # [p², 3] (a, b, '=')
    labels: torch.Tensor  # [p²] correct answers
    is_train: torch.Tensor  # [p²] bool
    is_test: torch.Tensor  # [p²] bool
    F: torch.Tensor  # [p, p] orthonormal Fourier basis (float64)
    names: list[str]  # basis row names ("Const", "cos 1", ...)

    @classmethod
    def build(cls, p: int, frac_train: float, split_seed: int, device, fn_name: str = "add", label_seed: int | None = None) -> "Grid":
        """Construct the grid for a run (its split seed and task function)."""
        split = task.paper_split(p, frac_train, split_seed)
        F, names = fourier.fourier_basis(p)
        return cls(
            p=p,
            tokens=task.all_pairs(p).to(device),
            labels=task.make_labels(p, fn_name, label_seed).to(device),
            is_train=split.is_train.to(device),
            is_test=split.is_test.to(device),
            F=F.to(device),
            names=names,
        )


@dataclass
class Acts:
    """Activations of one model on the whole grid, at the '=' position (float64 for analysis)."""

    logits: torch.Tensor  # [p², p] logits over the p real answers ('=' output dropped, paper `[:, -1, :-1]`)
    pre: torch.Tensor  # [p², d_mlp] neuron pre-activations
    post: torch.Tensor  # [p², d_mlp] neuron activations (after ReLU)
    resid_mid: torch.Tensor  # [p², d_model] residual stream after attention
    pattern: torch.Tensor  # [p², heads, 3] attention from '=' to (a, b, '=')


@torch.no_grad()
def evaluate(model: GrokTransformer, grid: Grid) -> Acts:
    """Run the model once on every input and keep the '='-position activations."""
    logits, cache = model.run_with_cache(grid.tokens)
    return Acts(
        logits=logits[:, -1, : grid.p].double(),
        pre=cache["blocks.0.mlp.pre"][:, -1].double(),
        post=cache["blocks.0.mlp.post"][:, -1].double(),
        resid_mid=cache["blocks.0.resid_mid"][:, -1].double(),
        pattern=cache["blocks.0.attn.pattern"][:, :, -1, :].double(),
    )


# ------------------------------------------------------------------------------------------------
# Weights in the paper's (column-vector) notation
# ------------------------------------------------------------------------------------------------
def paper_matrices(model: GrokTransformer, p: int) -> dict[str, torch.Tensor]:
    """W_E (d×p), W_U (p×d), W_out (d×n), W_L = W_U W_out (p×n) as in paper Sec. 3, '=' row/column dropped."""
    W_E = model.embed.W_E[:, :p].double()  # our layout is already [d_model, vocab]
    W_U = model.unembed.W_U[:, :p].T.double()  # ours is [d_model, vocab] → transpose to p×d
    W_out = model.blocks[0].mlp.W_out.double()  # [d_model, d_mlp]
    return {"W_E": W_E, "W_U": W_U, "W_out": W_out, "W_L": W_U @ W_out}


def wl_residual_sq_frac(W_L: torch.Tensor, rows: torch.Tensor) -> float:
    """‖W_L − R^T R W_L‖² / ‖W_L‖² for orthonormal key rows R [2|K|, p]: the share of W_L's squared norm outside the
    key cos/sin directions. Paper Sec. 4.2 (p. 6) says the residual's Frobenius norm is under 0.55 % of ‖W_L‖; on the
    paper model this squared-norm share is 0.552 % at step 40,001 (0.520 % at the released run's last checkpoint,
    49,901), while the norm ratio itself is 7.4 % (summary keys paper.wl_resid_sq, _sq_last, _norm)."""
    return float(((W_L - rows.T @ (rows @ W_L)).norm() / W_L.norm()) ** 2)


def we_fourier_norms(model: GrokTransformer, F: torch.Tensor) -> torch.Tensor:
    """ℓ2 norm (over d_model) of each Fourier component of the embedding: [p] (paper Fig. 3 left)."""
    W_E = paper_matrices(model, F.shape[0])["W_E"]
    return (W_E @ F.T).norm(dim=0)  # Fourier transform along the input (token) axis


def wl_fourier_norms(model: GrokTransformer, F: torch.Tensor) -> torch.Tensor:
    """ℓ2 norm (over neurons) of each Fourier component of the neuron-logit map W_L: [p] (paper Fig. 3 right)."""
    W_L = paper_matrices(model, F.shape[0])["W_L"]
    return (F @ W_L).norm(dim=1)  # Fourier transform along the logit (output) axis


def key_frequencies(model: GrokTransformer, F: torch.Tensor, tau: float = KEY_FREQ_TAU) -> list[int]:
    """Frequencies whose combined cos/sin norm in W_L is ≥ τ·max (paper p.18: "DFT on W_L ... nontrivial coefficients")."""
    norms = fourier.frequency_norms(wl_fourier_norms(model, F))  # [(p-1)/2], index k-1 ↔ frequency k
    return [k + 1 for k in torch.nonzero(norms >= tau * norms.max()).flatten().tolist()]


def sum_sq_weights(model: GrokTransformer) -> dict[str, float]:
    """Σ w² per named parameter (paper code take_metrics.sum_sq_weights; total = paper Fig. 7 bottom right)."""
    return {name: float(param.detach().double().pow(2).sum()) for name, param in model.named_parameters()}


# ------------------------------------------------------------------------------------------------
# Progress measures on logits
# ------------------------------------------------------------------------------------------------
def restricted_logits(logits: torch.Tensor, F: torch.Tensor, freqs: list[int]) -> torch.Tensor:
    """Keep only the cos/sin(w_k(a+b)) components of every output's logit grid, then bias-correct."""
    kept = fourier.project(logits, fourier.trig_dirs(F, freqs))
    return losses.bias_correct(kept, logits)


def restricted_loss(logits: torch.Tensor, grid: Grid, freqs: list[int], mask: torch.Tensor | None = None) -> float:
    """Restricted (trig) loss: loss of `restricted_logits` on the rows in `mask` (None = all pairs, as in Fig. 7)."""
    return losses.grid_loss(restricted_logits(logits, grid.F, freqs), grid.labels, mask).item()


def excluded_loss(logits: torch.Tensor, grid: Grid, freqs: list[int]) -> float:
    """Excluded loss: remove the cos/sin(w_k(a+b)) components of the given frequencies; train loss, no bias correction."""
    removed = logits - fourier.project(logits, fourier.trig_dirs(grid.F, freqs))
    return losses.grid_loss(removed, grid.labels, grid.is_train).item()


def excluded_loss_per_freq(logits: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Excluded loss removing one frequency at a time, for k = 1..(p-1)/2 → [(p-1)/2] (paper footnote 3, Fig. 15)."""
    return torch.tensor([excluded_loss(logits, grid, [k]) for k in range(1, grid.p // 2 + 1)])


def cos_coefficients(logits: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Coefficient of each unit-normalised cube cos(w_k(a+b−c)) in the logits, k = 1..(p-1)/2 (paper code calculate_coefficients, Fig. 14)."""
    p = grid.p
    dev = logits.device
    ab = (torch.arange(p, device=dev)[:, None] + torch.arange(p, device=dev)[None, :]).flatten()  # a+b per row
    c = torch.arange(p, device=dev)
    out = []
    for k in range(1, p // 2 + 1):  # one 113³ cube at a time keeps memory small
        cube = torch.cos(2 * math.pi * k / p * (ab[:, None] - c[None, :]).double())  # [p², p]
        out.append((cube * logits).sum() / cube.norm())
    return torch.stack(out)


def logit_cos_fit(logits: torch.Tensor, grid: Grid, freqs: list[int]) -> tuple[torch.Tensor, float, torch.Tensor]:
    """OLS fit logits ≈ Σ_k α_k cos(w_k(a+b−c)) + β → (α incl. β last, centred FVE, fitted logits) (paper Sec. 4.2)."""
    p = grid.p
    dev = logits.device
    ab = (torch.arange(p, device=dev)[:, None] + torch.arange(p, device=dev)[None, :]).flatten()
    diff = (ab[:, None] - torch.arange(p, device=dev)[None, :]).double()  # a+b−c for every row/output
    X = torch.stack([torch.cos(2 * math.pi * k / p * diff).flatten() for k in freqs] + [torch.ones(p * p * p, device=dev, dtype=torch.float64)], 1)
    y = logits.flatten()
    coef = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]  # least squares coefficients
    fit = X @ coef
    fve = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()  # centred fraction of variance explained
    return coef, float(fve), fit.reshape(p * p, p)


def trig_fraction(values: torch.Tensor, F: torch.Tensor, freqs: list[int]) -> torch.Tensor:
    """Per column: fraction of centred variance lying in the cos/sin(w_k(a+b)) directions (ARENA tensor_trig_ratio)."""
    centred = values - values.mean(0, keepdim=True)
    total = centred.pow(2).sum(0)
    D = fourier.trig_dirs(F, freqs).to(centred.dtype)  # orthonormal rows
    explained = (D @ centred).pow(2).sum(0)  # squared length of the projection
    return explained / total.clamp_min(1e-300)


# ------------------------------------------------------------------------------------------------
# Neurons and attention
# ------------------------------------------------------------------------------------------------
def neuron_fourier(post: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """2D Fourier coefficients of centred neuron activations: [p, p, d_mlp]."""
    p = F.shape[0]
    centred = post - post.mean(0, keepdim=True)  # remove each neuron's constant (Const⊗Const) term
    return fourier.fft2d(centred, F).reshape(p, p, -1)


def neuron_frequencies(coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Best frequency per neuron, its fraction of variance (3×3 block energy share) and the quadratic-term share.

    Returns (freq [n] int, fve [n], quad_fve [n]). The 3×3 block of k holds the constant, linear and quadratic
    terms of k (paper extract_freq_2d / ARENA arrange_by_2d_freqs, which however stops at k = p//2 − 1).
    """
    blocks, total = _freq_blocks(coeffs)
    best, arg = blocks.pow(2).sum((1, 2)).max(0)  # energy of the strongest 3×3 block per neuron
    quad = blocks[arg, 1:, 1:, torch.arange(coeffs.shape[-1], device=coeffs.device)].pow(2).sum((1, 2))  # quadratic 2×2 part
    return arg + 1, best / total, quad / total


def _freq_blocks(coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """3×3 blocks of every frequency k = 1..p//2 → [K, 3, 3, n], and each neuron's total energy [n] (≥ 1e-300)."""
    p = coeffs.shape[0]
    total = coeffs.pow(2).sum((0, 1)).clamp_min(1e-300)  # [n] total energy (= variance × p² after centring)
    return torch.stack([fourier.extract_freq_2d(coeffs, k) for k in range(1, p // 2 + 1)]), total


def neuron_freq_energy(coeffs: torch.Tensor) -> torch.Tensor:
    """Share of each neuron's variance in each frequency's 3×3 block → [n, K] (rows sum to ≤ 1; the rest is outside
    every block, e.g. cos(w_k a)·cos(w_j b) cross terms). The full map behind `neuron_frequencies`' arg-max."""
    blocks, total = _freq_blocks(coeffs)
    return (blocks.pow(2).sum((1, 2)) / total).T


def attention_lookup(model: GrokTransformer, p: int) -> torch.Tensor:
    """C_j[a] = W_E^T W_K^jT W_Q^j x2: pre-softmax score from '=' to token a, per head → [heads, p] (paper App. C.1.3).

    x2 is the '=' residual at the start (embedding of '=' plus position-2 embedding); the score is scaled by
    1/√d_head as in the forward pass (ARENA W_attn convention).
    """
    W_E = model.embed.W_E.double()  # [d_model, vocab]
    x2 = W_E[:, p] + model.pos_embed.W_pos[2].double()  # '=' token at position 2
    q = model.blocks[0].attn.W_Q.double() @ x2  # [heads, d_head]
    k = model.blocks[0].attn.W_K.double() @ W_E[:, :p]  # [heads, d_head, p]
    return torch.einsum("ih,ihp->ip", q, k) / math.sqrt(model.shape.d_head)


def ov_fourier_norms(model: GrokTransformer, F: torch.Tensor) -> torch.Tensor:
    """Fourier-component norms of each head's OV circuit applied to token embeddings, [heads, p] (paper Fig. 11)."""
    p = F.shape[0]
    a = model.blocks[0].attn
    W_E = model.embed.W_E[:, :p].double()  # [d, p]
    d_head = model.shape.d_head
    outs = []
    for h in range(model.shape.n_heads):
        W_O_h = a.W_O[:, h * d_head : (h + 1) * d_head].double()  # this head's slice of the output matrix
        ov = W_O_h @ a.W_V[h].double() @ W_E  # [d_model, p]: what the head writes when attending to token t
        outs.append((ov @ F.T).norm(dim=0))
    return torch.stack(outs)
