"""Tests for the core math: task/dataset, Fourier basis, losses and the transformer (milestone M2).

Global context: every later analysis (key frequencies, restricted/excluded loss, ablations) and the
website's in-browser model rest on these four modules, so they are pinned against first principles
(orthonormality, trigonometric identities, known loss values) and against the paper's own code
(parameter names, shapes, initialisation scales, train/test split procedure).
"""

import math  # exact reference values (log, cos, sin)
import random  # reference implementation of the paper's split procedure

import pytest  # fixtures, parametrisation, approx
import torch  # tensors

from grokviz import fourier, losses, model, task  # modules under test

P = 113  # paper's modulus


# ----------------------------------------------------------------------------------------------
# task.py — the dataset of all (a, b) pairs and the paper's train/test split
# ----------------------------------------------------------------------------------------------
def test_all_pairs_layout():
    """Row a*p+b holds tokens (a, b, '=') with '=' encoded as p (paper Sec. 3 input format)."""
    tokens = task.all_pairs(P)
    assert tokens.shape == (P * P, 3)
    assert tokens[5 * P + 7].tolist() == [5, 7, P]
    assert tokens[:, 2].eq(P).all()


def test_labels_are_sum_mod_p():
    """Label of (a, b) is (a + b) mod p."""
    labels = task.make_labels(P, fn_name="add")
    assert labels[100 * P + 50].item() == (100 + 50) % P
    assert labels.shape == (P * P,)


def test_random_labels_are_seeded_and_in_range():
    """The random-label control ('rand') is reproducible given its seed."""
    a = task.make_labels(P, fn_name="rand", seed=3)
    b = task.make_labels(P, fn_name="rand", seed=3)
    assert torch.equal(a, b) and int(a.min()) >= 0 and int(a.max()) < P


def test_paper_split_matches_reference_procedure():
    """Split = python random.seed(seed); random.shuffle(all pairs); first int(0.3*p^2) are train."""
    split = task.paper_split(P, frac_train=0.3, seed=0)
    pairs = [(i, j, P) for i in range(P) for j in range(P)]  # reference: paper gen_train_test
    random.seed(0)
    random.shuffle(pairs)
    reference_train = [i * P + j for i, j, _ in pairs[: int(0.3 * len(pairs))]]
    assert split.train_idx.tolist() == reference_train  # same members in the same (shuffled) order
    assert len(split.train_idx) == 3830 and len(split.test_idx) == 8939
    assert split.is_train.sum().item() == 3830
    assert not set(split.train_idx.tolist()) & set(split.test_idx.tolist())


# ----------------------------------------------------------------------------------------------
# fourier.py — orthonormal Fourier basis (paper convention: 0 = const, 2k-1 = cos k, 2k = sin k)
# ----------------------------------------------------------------------------------------------
def test_basis_is_orthonormal_with_paper_ordering():
    """F @ F.T = I, and rows follow the paper's naming/ordering."""
    F, names = fourier.fourier_basis(P)
    assert F.shape == (P, P)
    assert torch.allclose(F @ F.T, torch.eye(P, dtype=F.dtype), atol=1e-10)
    assert names[:5] == ["Const", "cos 1", "sin 1", "cos 2", "sin 2"] and names[-1] == "sin 56"
    x = torch.arange(P, dtype=F.dtype)
    cos14 = torch.cos(2 * math.pi * 14 * x / P)
    assert torch.allclose(F[2 * 14 - 1], cos14 / cos14.norm(), atol=1e-12)


def test_fft2d_of_a_basis_term_is_one_hot():
    """Transforming a single 2D basis term returns a single 1 at its (i, j) coordinate."""
    F, _ = fourier.fourier_basis(P)
    term = fourier.basis_term_2d(F, 27, 84)  # flattened p*p vector
    coeffs = fourier.fft2d(term, F).reshape(P, P)
    expected = torch.zeros(P, P, dtype=F.dtype)
    expected[27, 84] = 1.0
    assert torch.allclose(coeffs, expected, atol=1e-10)


@pytest.mark.parametrize("k", [1, 14, 35, 56])
def test_cos_and_sin_of_a_plus_b_directions(k):
    """d_cos,k equals the normalised grid cos(w_k(a+b)) — i.e. the angle-addition identity holds."""
    F, _ = fourier.fourier_basis(P)
    a = torch.arange(P, dtype=F.dtype)[:, None]
    b = torch.arange(P, dtype=F.dtype)[None, :]
    w = 2 * math.pi * k / P
    cos_grid = torch.cos(w * (a + b)).flatten()
    sin_grid = torch.sin(w * (a + b)).flatten()
    assert torch.allclose(fourier.cos_apb_dir(F, k), cos_grid / cos_grid.norm(), atol=1e-10)
    assert torch.allclose(fourier.sin_apb_dir(F, k), sin_grid / sin_grid.norm(), atol=1e-10)


def test_angle_addition_identities_numerically():
    """cos(x+y) = cos x cos y − sin x sin y and sin(x+y) = sin x cos y + cos x sin y on random angles."""
    gen = torch.Generator().manual_seed(0)
    x, y = torch.rand(2, 1000, generator=gen, dtype=torch.float64) * 20 - 10
    assert torch.allclose(torch.cos(x + y), torch.cos(x) * torch.cos(y) - torch.sin(x) * torch.sin(y))
    assert torch.allclose(torch.sin(x + y), torch.sin(x) * torch.cos(y) + torch.cos(x) * torch.sin(y))


def test_extract_freq_2d_covers_k56():
    """The 3x3 block of the last frequency (56) is reachable (ARENA's arrange_by_2d_freqs drops it)."""
    F, _ = fourier.fourier_basis(P)
    block = fourier.extract_freq_2d(torch.arange(P * P, dtype=F.dtype).reshape(P, P), 56)
    idx = [0, 111, 112]
    assert block.tolist() == [[i * P + j for j in idx] for i in idx]


def test_gini_known_values():
    """Hurley & Rickard Gini: one-hot → 1 − 1/N, uniform → 0, scale-invariant."""
    one_hot = torch.zeros(10, dtype=torch.float64)
    one_hot[3] = 5.0
    assert fourier.gini(one_hot) == pytest.approx(1 - 1 / 10)
    assert fourier.gini(torch.ones(10, dtype=torch.float64)) == pytest.approx(0.0)
    v = torch.rand(50, dtype=torch.float64)
    assert fourier.gini(v) == pytest.approx(fourier.gini(7 * v))


def test_frequency_norms_combine_cos_and_sin():
    """Per-frequency norm k = sqrt(c_{2k-1}^2 + c_{2k}^2); constant term is reported separately."""
    coeffs = torch.zeros(P, dtype=torch.float64)
    coeffs[2 * 14 - 1], coeffs[2 * 14] = 3.0, 4.0
    norms = fourier.frequency_norms(coeffs)
    assert norms.shape == (56,) and norms[13].item() == pytest.approx(5.0)


# ----------------------------------------------------------------------------------------------
# losses.py — float64 cross-entropy (paper notebook cell 26) and split-aware evaluation
# ----------------------------------------------------------------------------------------------
def test_uniform_logits_give_log_classes():
    """All-equal logits over C classes give loss ln C (ln 113 ≈ 4.727 = 'chance')."""
    logits = torch.zeros(4, P)
    assert losses.cross_entropy_f64(logits, torch.tensor([0, 1, 2, 3])).item() == pytest.approx(math.log(P))


def test_cross_entropy_matches_torch_reference():
    """Same value as torch.nn.functional.cross_entropy evaluated in float64."""
    gen = torch.Generator().manual_seed(1)
    logits = torch.randn(32, 114, generator=gen)
    labels = torch.randint(0, 113, (32,), generator=gen)
    ref = torch.nn.functional.cross_entropy(logits.double(), labels)
    assert losses.cross_entropy_f64(logits, labels).item() == pytest.approx(ref.item(), rel=1e-12)
    assert losses.cross_entropy_f64(logits, labels).dtype == torch.float64


def test_bias_correction_restores_per_class_mean():
    """Bias correction adds back the mean (over inputs) of what an ablation removed."""
    gen = torch.Generator().manual_seed(2)
    original = torch.randn(10, 5, generator=gen, dtype=torch.float64)
    ablated = original - original.mean(0, keepdim=True)  # remove the per-class mean
    assert torch.allclose(losses.bias_correct(ablated, original), original)


# ----------------------------------------------------------------------------------------------
# model.py — GrokTransformer with the paper's parameter layout
# ----------------------------------------------------------------------------------------------
PAPER_KEYS = {  # state_dict keys/shapes of the paper's Transformer (progress-measures-paper/transformers.py)
    "embed.W_E": (128, 114),
    "pos_embed.W_pos": (3, 128),
    "blocks.0.attn.W_K": (4, 32, 128),
    "blocks.0.attn.W_Q": (4, 32, 128),
    "blocks.0.attn.W_V": (4, 32, 128),
    "blocks.0.attn.W_O": (128, 128),
    "blocks.0.attn.mask": (3, 3),
    "blocks.0.mlp.W_in": (512, 128),
    "blocks.0.mlp.b_in": (512,),
    "blocks.0.mlp.W_out": (128, 512),
    "blocks.0.mlp.b_out": (128,),
    "unembed.W_U": (128, 114),
}


@pytest.fixture(scope="module")
def net():
    """A freshly initialised paper-architecture model (seed 0)."""
    return model.GrokTransformer(model.ModelShape(), seed=0)


def test_state_dict_matches_paper_layout(net):
    """Same keys and shapes as the paper's checkpoints, so they load with strict=True."""
    assert {k: tuple(v.shape) for k, v in net.state_dict().items()} == PAPER_KEYS


def test_parameter_count(net):
    """226,816 trainable numbers (the older TransformerLens repo's HF card said ~2.5M until it was corrected to 227,313,
    which includes TransformerLens's zero biases)."""
    assert sum(p.numel() for p in net.parameters()) == 226_816


def test_initialisation_scales(net):
    """randn/sqrt(d_model) everywhere except W_U (randn/sqrt(d_vocab)); biases start at zero."""
    assert net.embed.W_E.std().item() == pytest.approx(1 / math.sqrt(128), rel=0.05)
    assert net.unembed.W_U.std().item() == pytest.approx(1 / math.sqrt(114), rel=0.05)
    assert net.blocks[0].mlp.b_in.abs().sum().item() == 0.0


def test_seeded_initialisation_is_reproducible():
    """Same seed → identical weights; different seed → different weights."""
    a = model.GrokTransformer(model.ModelShape(), seed=5).state_dict()
    b = model.GrokTransformer(model.ModelShape(), seed=5).state_dict()
    c = model.GrokTransformer(model.ModelShape(), seed=6).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(a["embed.W_E"], c["embed.W_E"])


def test_forward_shapes_and_cache(net):
    """Logits are [batch, 3, 114]; the cache exposes every named intermediate activation."""
    tokens = task.all_pairs(P)[:7]
    logits, cache = net.run_with_cache(tokens)
    assert logits.shape == (7, 3, 114)
    assert cache["blocks.0.attn.pattern"].shape == (7, 4, 3, 3)  # [batch, head, query, key]
    assert cache["blocks.0.mlp.post"].shape == (7, 3, 512)
    assert cache["blocks.0.resid_mid"].shape == (7, 3, 128)
    assert torch.allclose(net(tokens), logits)


def test_attention_is_causal(net):
    """Position 0 cannot see later tokens: changing b leaves position-0 outputs unchanged."""
    t1 = torch.tensor([[3, 4, P]])
    t2 = torch.tensor([[3, 99, P]])
    assert torch.allclose(net(t1)[:, 0], net(t2)[:, 0])


def test_intervention_hook_replaces_activation(net):
    """A hook that zeros the MLP output makes logits equal to the attention-only path."""
    tokens = task.all_pairs(P)[:5]
    zeroed = net(tokens, hooks={"blocks.0.mlp_out": lambda x: torch.zeros_like(x)})
    _, cache = net.run_with_cache(tokens)
    attn_only = cache["blocks.0.resid_mid"] @ net.unembed.W_U
    assert torch.allclose(zeroed, attn_only, atol=1e-6)
