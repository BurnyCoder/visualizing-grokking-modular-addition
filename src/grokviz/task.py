"""The task: every input "a b =" for a, b in {0..p-1}, its label, and the paper's train/test split.

Local explanation: `all_pairs` lists all p² token triples in a fixed order (row a·p + b), `make_labels`
gives the correct answer for each row, and `paper_split` reproduces the paper's random 30 % / 70 %
division of those rows into training and test sets.
Global context: the whole input "universe" has only p² = 12,769 points, so every analysis runs the model
on all of them at once and indexes train/test with boolean masks — the trick that makes the paper's
Fourier analysis over the input grid possible (ARENA 1.5.2 calls this out in "Helper variables").
Sources: paper Sec. 3 (https://arxiv.org/abs/2301.05217); paper code `gen_train_test`
(https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/transformers.py);
paper notebook cell 34 (submodule grokking-original.ipynb @38776db); ARENA utils.gen_train_test
(https://github.com/ARENA-education/ARENA_materials/blob/main/chapter1_transformer_interp/exercises/part52_grokking_and_modular_arithmetic/utils.py).
"""

from __future__ import annotations  # postponed annotations

import random  # Python's Mersenne-Twister RNG — the paper's split uses random.seed + random.shuffle
from dataclasses import dataclass  # container for the split

import numpy as np  # seeded RNG for the random-label control
import torch  # integer tensors


def all_pairs(p: int) -> torch.Tensor:
    """Return a [p*p, 3] LongTensor whose row a*p+b is (a, b, p); token p stands for '='."""
    a = torch.arange(p).repeat_interleave(p)  # 0,0,...,0,1,1,...: a changes slowest
    b = torch.arange(p).repeat(p)  # 0,1,...,p-1,0,1,...: b changes fastest
    eq = torch.full((p * p,), p)  # the '=' token gets the extra vocabulary index p (vocab size p+1)
    return torch.stack([a, b, eq], dim=1)


def make_labels(p: int, fn_name: str = "add", seed: int | None = None) -> torch.Tensor:
    """Return the [p*p] correct answers: (a+b) mod p for "add", or a fixed random table for "rand".

    "rand" is the memorisation-only control (paper code fns_dict['rand']): every pair gets an arbitrary
    answer, so no algorithm can generalise; we seed it so the control is reproducible.
    """
    tokens = all_pairs(p)  # same row order as the inputs
    if fn_name == "add":
        return (tokens[:, 0] + tokens[:, 1]) % p
    if fn_name == "rand":
        table = np.random.default_rng(seed).integers(0, p, size=(p, p))  # answer for each (a, b)
        return torch.from_numpy(table.reshape(-1)).long()
    raise ValueError(f"unknown task function {fn_name!r}")


@dataclass(frozen=True)
class Split:
    """Which rows of `all_pairs(p)` are used for training (in the paper's shuffled order) and testing."""

    train_idx: torch.Tensor  # [n_train] row indices, in shuffled order (training batch order)
    test_idx: torch.Tensor  # [n_test] row indices, in shuffled order
    is_train: torch.Tensor  # [p*p] boolean mask over rows

    @property
    def is_test(self) -> torch.Tensor:
        """Boolean mask of test rows (complement of is_train)."""
        return ~self.is_train


def paper_split(p: int, frac_train: float, seed: int) -> Split:
    """Reproduce the paper's split: shuffle all (i, j, p) with random.seed(seed); first int(frac·p²) train."""
    pairs = [(i, j, p) for i in range(p) for j in range(p)]  # same list construction as the paper
    rng = random.Random(seed)  # private RNG instance: same algorithm as random.seed(seed) but no global state
    rng.shuffle(pairs)  # in-place Fisher-Yates shuffle (identical sequence to random.shuffle after seed)
    cut = int(frac_train * len(pairs))  # 0.3 * 12769 = 3830.7 → 3830 training pairs
    order = torch.tensor([i * p + j for i, j, _ in pairs])  # convert (i, j) to row indices a*p+b
    is_train = torch.zeros(p * p, dtype=torch.bool)  # start with everything in test
    is_train[order[:cut]] = True  # mark the training rows
    return Split(train_idx=order[:cut], test_idx=order[cut:], is_train=is_train)
