"""Losses: float64 cross-entropy, split-aware evaluation on the full p² grid, bias correction, accuracy.

Local explanation: cross-entropy = average of −log(softmax probability of the correct answer); computing it
in float64 avoids float32 underflow once the model is very confident (the paper's notebook comment: float32
"can only return multiples of 1.2e-7", causing loss spikes and bad gradients).
Global context: used for training (114 output classes, as in the paper's code) and for every analysis
metric (113 classes: the '=' output is dropped, as in the paper's `[:, -1, :-1]`), including restricted and
excluded loss, which evaluate *modified* logits for the whole input grid and then select train/test rows.
Sources: paper notebook cell 26 `cross_entropy_high_precision`, `test_logits` (submodule grokking-original.ipynb);
note that the published helpers.py casts to float32 despite its float64 comment — we follow the notebook/ARENA
(float64): https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/23d2c64/helpers.py;
torch log_softmax (https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html).
"""

from __future__ import annotations  # postponed annotations

import torch  # tensors
import torch.nn.functional as F  # log_softmax


def cross_entropy_f64(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean −log p(correct) over the batch, computed in float64. logits [batch, C], labels [batch]."""
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)  # log of softmax probabilities, high precision
    correct = logprobs.gather(dim=-1, index=labels[:, None])[:, 0]  # pick the log-probability of each true label
    return -correct.mean()


def bias_correct(new_logits: torch.Tensor, original_logits: torch.Tensor) -> torch.Tensor:
    """Add back the input-independent part an ablation removed: new + mean_over_inputs(original − new).

    A projection onto a few Fourier directions also deletes each output's constant offset; adding the
    per-class average difference restores exactly that offset (= keeping the 2D "Const⊗Const" term) and
    nothing that depends on (a, b). Paper notebook `test_logits(bias_correction=True)`.
    """
    return new_logits + (original_logits - new_logits).mean(dim=0, keepdim=True)


def grid_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy of full-grid logits [p*p, C] restricted to the rows selected by boolean `mask` (None = all)."""
    if mask is None:
        return cross_entropy_f64(logits, labels)
    return cross_entropy_f64(logits[mask], labels[mask])


def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Fraction of rows whose highest logit is the correct label (argmax prediction)."""
    if mask is not None:
        logits, labels = logits[mask], labels[mask]
    return (logits.argmax(dim=-1) == labels).double().mean().item()
