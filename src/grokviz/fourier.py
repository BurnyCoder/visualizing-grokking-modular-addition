"""Fourier tools on Z_p: the real orthonormal Fourier basis, 1D/2D transforms, the cos/sin(w(a+b)) directions,
per-frequency norms and the Gini sparsity index.

Local explanation: a function of one input x ∈ {0..p-1} is a vector in R^p; the Fourier basis F is an
orthonormal p×p matrix whose rows are a constant and cos/sin waves of frequency k = 1..(p-1)/2, so
`x @ F.T` re-expresses the vector as "how much of each wave it contains". For functions of two inputs
(a, b) we use outer products of rows (the 2D basis).
Global context: the paper's central claim is that the trained network is *sparse* in this basis — it uses
only a handful of "key frequencies" — so almost every figure on the site is computed with these helpers.
Convention (paper code): row 0 = "Const", row 2k-1 = "cos k", row 2k = "sin k" (ARENA uses the same in
code). The older TransformerLens notebook in the submodule (cell 74) puts sin first; we do not.
Sources: paper notebook cells 30–31 (submodule grokking-original.ipynb @38776db); paper code
`make_fourier_basis` and helpers.py `fft2d`, `get_component_cos_xpy`, `extract_freq_2d`
(https://github.com/mechanistic-interpretability-grokking/progress-measures-paper);
ARENA 1.5.2 "Fourier Transforms" (https://learn.arena.education/chapter1_transformer_interp/32_grokking/1-periodicity-fourier-basis/);
DFT as orthonormal change of basis (https://en.wikipedia.org/wiki/Discrete_Fourier_transform#The_unitary_DFT);
Gini index of sparsity: Hurley & Rickard 2009, "Comparing Measures of Sparsity" (https://arxiv.org/abs/0811.4706).
"""

from __future__ import annotations  # postponed annotations

import math  # pi

import torch  # tensors


def fourier_basis(p: int, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, list[str]]:
    """Return (F, names): F is [p, p] with orthonormal rows Const, cos 1, sin 1, ..., cos (p-1)/2, sin (p-1)/2."""
    x = torch.arange(p, dtype=dtype)  # input positions 0..p-1
    rows = [torch.ones(p, dtype=dtype)]  # constant wave (frequency 0)
    names = ["Const"]
    for k in range(1, p // 2 + 1):  # frequencies 1..56 for p = 113 (p odd, so no Nyquist row)
        rows.append(torch.cos(2 * math.pi * k * x / p))  # cos(w_k x) with w_k = 2πk/p
        rows.append(torch.sin(2 * math.pi * k * x / p))  # sin(w_k x)
        names += [f"cos {k}", f"sin {k}"]
    F = torch.stack(rows)  # [p, p], one wave per row
    F = F / F.norm(dim=1, keepdim=True)  # normalise each row: const → 1/√p, waves → √(2/p)
    return F, names


def fft1d(x: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Fourier coefficients along the LAST dimension (size p): x @ F.T (projection onto each row of F)."""
    return x @ F.T


def fft2d(mat: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """2D Fourier coefficients over the first axis/axes indexing (a, b); trailing axes are batch. Same shape out.

    Accepts [p*p, ...] (rows a*p+b, as produced by running the model on `all_pairs`) or [p, p, ...].
    Coefficient [i, j] is the dot product with the 2D basis term F[i] ⊗ F[j] (outer product).
    """
    p = F.shape[0]
    shape = mat.shape  # remember the input layout to restore it
    grid = mat.reshape(p, p, -1)  # [a, b, rest]
    coeffs = torch.einsum("abz,ia,jb->ijz", grid, F.to(grid.dtype), F.to(grid.dtype))  # change basis on both axes
    return coeffs.reshape(shape)


def ifft2d(coeffs: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Inverse of fft2d (F is orthonormal, so the inverse uses F transposed)."""
    p = F.shape[0]
    shape = coeffs.shape
    grid = coeffs.reshape(p, p, -1)
    values = torch.einsum("ijz,ia,jb->abz", grid, F.to(grid.dtype), F.to(grid.dtype))
    return values.reshape(shape)


def basis_term_2d(F: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """The flattened [p*p] 2D basis function F[i](a)·F[j](b) (row a*p+b)."""
    return (F[i][:, None] * F[j][None, :]).flatten()


def cos_apb_dir(F: torch.Tensor, k: int) -> torch.Tensor:
    """Unit vector over the (a, b) grid proportional to cos(w_k(a+b)).

    Uses cos(w(a+b)) = cos(wa)cos(wb) − sin(wa)sin(wb): the difference of two orthonormal 2D terms,
    divided by √2 to keep unit length (paper helpers.get_component_cos_xpy).
    """
    return (basis_term_2d(F, 2 * k - 1, 2 * k - 1) - basis_term_2d(F, 2 * k, 2 * k)) / math.sqrt(2)


def sin_apb_dir(F: torch.Tensor, k: int) -> torch.Tensor:
    """Unit vector proportional to sin(w_k(a+b)) = sin(wa)cos(wb) + cos(wa)sin(wb) (helpers.get_component_sin_xpy)."""
    return (basis_term_2d(F, 2 * k, 2 * k - 1) + basis_term_2d(F, 2 * k - 1, 2 * k)) / math.sqrt(2)


def trig_dirs(F: torch.Tensor, freqs: list[int]) -> torch.Tensor:
    """Stack the orthonormal directions [cos_apb(k), sin_apb(k) for k in freqs] → [2·len(freqs), p*p]."""
    return torch.stack([d for k in freqs for d in (cos_apb_dir(F, k), sin_apb_dir(F, k))])


def project(values: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection of each column of values [p*p, ...] onto the span of orthonormal rows `dirs` [m, p*p]."""
    flat = values.reshape(values.shape[0], -1)  # [p*p, rest]
    d = dirs.to(flat.dtype)
    return (d.T @ (d @ flat)).reshape(values.shape)  # Σ_i d_i (d_i · v): sum of single-direction projections


def extract_freq_2d(coeffs_2d: torch.Tensor, k: int) -> torch.Tensor:
    """3×3 block of 2D coefficients for frequency k: rows/cols (Const, cos k, sin k). Input [p, p, ...].

    The block holds the constant, the 4 linear terms (cos/sin in a or b) and the 4 quadratic terms of k
    (paper helpers.extract_freq_2d). Works for every k = 1..(p-1)/2, including the last one.
    """
    idx = torch.tensor([0, 2 * k - 1, 2 * k])  # basis rows for Const, cos k, sin k
    return coeffs_2d[idx][:, idx]  # select rows then columns


def frequency_norms(coeffs: torch.Tensor) -> torch.Tensor:
    """Combine cos/sin coefficients per frequency along the last axis: [..., p] → [..., (p-1)/2] with √(c_cos² + c_sin²)."""
    return torch.sqrt(coeffs[..., 1::2] ** 2 + coeffs[..., 2::2] ** 2)  # odd indices = cos k, even (≥2) = sin k


def gini(values: torch.Tensor) -> float:
    """Gini sparsity index of a non-negative vector (Hurley & Rickard 2009, eq. for G(c)).

    G = 1 − 2 Σ_{k=1..N} (c_(k)/‖c‖₁)·(N − k + ½)/N with c sorted ascending. 0 = perfectly spread
    (all equal), 1 − 1/N = all mass on one entry. Scale-invariant, so it compares sparsity across training.
    """
    c = values.detach().abs().flatten().double().sort().values  # ascending order statistics c_(1) ≤ ... ≤ c_(N)
    n = c.numel()
    total = c.sum()
    if total == 0:  # all-zero vector: define as perfectly spread
        return 0.0
    k = torch.arange(1, n + 1, dtype=torch.float64)  # ranks 1..N
    return float(1 - 2 * torch.sum((c / total) * (n - k + 0.5) / n))
