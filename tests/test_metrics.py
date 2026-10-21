"""Tests for the paper metrics on synthetic inputs with known answers (milestone M4).

Global context: a model that implements the Fourier multiplication algorithm *exactly* has logits
Σ_k α_k cos(w_k(a+b−c)); for such logits the restricted loss must equal the original loss, the excluded loss
must be chance (ln p), the cosine fit must explain 100 % of the variance, and a neuron computing
cos(w a)cos(w b) must be assigned frequency k with FVE 1. These first-principles checks pin the definitions
independently of any trained network; the paper-number comparisons live in rchecks (slow tests).
"""

import math  # constants

import pytest  # approx, markers
import torch  # tensors

from grokviz import metrics  # module under test

P = 13  # small prime keeps the synthetic tests fast; the math is identical for 113
K = [2, 5]  # "key frequencies" of the synthetic perfect model
ALPHA = [3.0, 2.0]  # their weights


@pytest.fixture(scope="module")
def grid():
    """Full grid for p = 13 with a 30 % split."""
    return metrics.Grid.build(P, 0.3, split_seed=0, device="cpu")


@pytest.fixture(scope="module")
def perfect_logits():
    """Logits[a*p+b, c] = Σ_k α_k cos(w_k(a+b−c)): the paper's algorithm, implemented exactly."""
    ab = (torch.arange(P)[:, None] + torch.arange(P)[None, :]).flatten()
    diff = (ab[:, None] - torch.arange(P)[None, :]).double()
    return sum(a * torch.cos(2 * math.pi * k / P * diff) for a, k in zip(ALPHA, K))


def test_perfect_logits_are_correct(grid, perfect_logits):
    """Constructive interference: the largest logit is always the true answer (a+b) mod p."""
    assert torch.equal(perfect_logits.argmax(-1), grid.labels)


def test_restricted_loss_keeps_everything(grid, perfect_logits):
    """Restricting to the key cos/sin(w_k(a+b)) directions changes nothing for a perfect Fourier model."""
    from grokviz import losses

    original = losses.grid_loss(perfect_logits, grid.labels).item()
    assert metrics.restricted_loss(perfect_logits, grid, K) == pytest.approx(original, rel=1e-9)


def test_excluded_loss_removes_everything(grid, perfect_logits):
    """Removing the key directions leaves zero logits → uniform prediction → loss ln p on the train set."""
    assert metrics.excluded_loss(perfect_logits, grid, K) == pytest.approx(math.log(P), rel=1e-9)
    per_freq = metrics.excluded_loss_per_freq(perfect_logits, grid)
    assert per_freq.shape == (P // 2,)
    assert per_freq[0].item() < per_freq[1].item()  # removing a non-key frequency (1) hurts less than key 2


def test_logit_cos_fit_recovers_alphas(grid, perfect_logits):
    """OLS on cos(w_k(a+b−c)) recovers α exactly with FVE 1 and zero intercept."""
    coef, fve, fit = metrics.logit_cos_fit(perfect_logits, grid, K)
    assert coef[:2].tolist() == pytest.approx(ALPHA, abs=1e-9) and abs(coef[2].item()) < 1e-9
    assert fve == pytest.approx(1.0, abs=1e-12)


def test_cos_coefficients_peak_at_key_frequencies(grid, perfect_logits):
    """Projection onto each normalised cos(w_k(a+b−c)) cube is non-zero only at the key frequencies."""
    coeffs = metrics.cos_coefficients(perfect_logits, grid)
    nonzero = {k + 1 for k in torch.nonzero(coeffs.abs() > 1e-6).flatten().tolist()}
    assert nonzero == set(K)


def test_trig_fraction_of_a_pure_trig_column(grid):
    """A column equal to cos(w_k(a+b)) has trig fraction 1 for {k} and 0 for another frequency."""
    ab = (torch.arange(P)[:, None] + torch.arange(P)[None, :]).flatten().double()
    col = torch.cos(2 * math.pi * 5 / P * ab)[:, None]
    assert metrics.trig_fraction(col, grid.F, [5]).item() == pytest.approx(1.0)
    assert metrics.trig_fraction(col, grid.F, [3]).item() == pytest.approx(0.0, abs=1e-12)


def test_neuron_frequency_of_a_quadratic_neuron(grid):
    """A neuron computing cos(w a)·cos(w b) + 0.5 cos(w a) is assigned frequency 5 with FVE 1."""
    a = torch.arange(P).double()[:, None]
    b = torch.arange(P).double()[None, :]
    w = 2 * math.pi * 5 / P
    act = (torch.cos(w * a) * torch.cos(w * b) + 0.5 * torch.cos(w * a)).flatten()[:, None]
    freq, fve, quad = metrics.neuron_frequencies(metrics.neuron_fourier(act, grid.F))
    assert freq.item() == 5 and fve.item() == pytest.approx(1.0)
    assert 0.0 < quad.item() < 1.0  # the linear 0.5·cos(w a) part is not quadratic


def test_spectra_of_the_perfect_algorithm(grid, perfect_logits):
    """Perfect logits live only on the cos/sin(w_k(a+b)) cells of the 2D Fourier map, i.e. the 2×2 block of each key
    frequency (no linear terms); a cos(w a)cos(w b) + cos(w a) neuron puts all its variance into frequency 5's block."""
    from grokviz.geometry import spectra

    a = torch.arange(P).double()[:, None]
    b = torch.arange(P).double()[None, :]
    w = 2 * math.pi * 5 / P
    post = (torch.cos(w * a) * torch.cos(w * b) + torch.cos(w * a)).flatten()[:, None]
    acts = metrics.Acts(logits=perfect_logits, pre=post, post=post, resid_mid=None, pattern=None)
    s = spectra(acts, grid)
    energy = torch.as_tensor(s["logit_fourier2d"]).double() ** 2
    quad_cells = [(i, j) for k in K for i in (2 * k - 1, 2 * k) for j in (2 * k - 1, 2 * k)]
    assert sum(energy[i, j] for i, j in quad_cells) / energy.sum() == pytest.approx(1.0, abs=1e-9)
    assert energy[0, :].sum() + energy[:, 0].sum() == pytest.approx(0.0, abs=1e-9)  # linear terms cancel in the logits
    assert s["neuron_freq_energy"].shape == (1, P // 2) and s["neuron_freq_energy"][0, 4] == pytest.approx(1.0, abs=1e-6)
    assert s["neuron_b2"][2 * 5 - 1, 0] > 0  # the neuron does carry the linear cos(w a) term
