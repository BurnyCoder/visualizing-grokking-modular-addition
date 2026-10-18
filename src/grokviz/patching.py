"""Counterfactual activation patching: swap the key-frequency part of the embedding of a with that of a′.

Local explanation: if the network reads the number a only through cos(w_k a) and sin(w_k a) for its key
frequencies k, then replacing those components of a's embedding with the ones of a different number a′
(leaving every other component untouched) should make the model answer a′ + b instead of a + b. Swapping the
complementary components, or a random subspace of the same size, should do almost nothing.
We measure the fraction of the logit difference L[a′+b] − L[a+b] that the patch recovers:
recovered = (D_patched − D_clean(a)) / (D_clean(a′) − D_clean(a)); 1 = full switch, 0 = no effect.
Global context: a causal (interventional, counterfactual) test of the paper's claim that the embedding maps
inputs onto circles at the key frequencies; preregistered secondary hypothesis (docs/preregistration.md).
Sources: activation patching / interchange interventions — Vig et al. 2020 (https://arxiv.org/abs/2004.12265),
Geiger et al. 2021 (https://arxiv.org/abs/2106.02997); subspace patching caveats — Makelov et al. 2023
(https://arxiv.org/abs/2311.17030).
"""

from __future__ import annotations  # postponed annotations

import torch  # tensors

from grokviz.metrics import Grid  # inputs, Fourier basis
from grokviz.model import GrokTransformer  # network
from grokviz.stats import bootstrap_interval  # CI over sampled triples


def key_embedding_subspace(model: GrokTransformer, grid: Grid, key_freqs: list[int]) -> torch.Tensor:
    """Orthonormal basis [d_model, 2|K|] of embedding directions that carry cos/sin(w_k a) for the key k."""
    W_E = model.embed.W_E[:, : grid.p].double()  # [d, p]
    rows = torch.stack([grid.F[i] for k in key_freqs for i in (2 * k - 1, 2 * k)]).to(W_E.dtype)  # [2|K|, p]
    Q, _ = torch.linalg.qr(W_E @ rows.T)  # columns: residual-space images of the key Fourier components
    return Q


@torch.no_grad()
def patch_recovery(model: GrokTransformer, grid: Grid, key_freqs: list[int], n: int = 1000, seed: int = 0) -> dict:
    """Recovered logit-difference fraction for key, complement and random subspace patches over n random (a, a′, b)."""
    p, d = grid.p, model.shape.d_model
    dev = grid.tokens.device
    gen = torch.Generator().manual_seed(seed)
    a = torch.randint(0, p, (n,), generator=gen)
    a2 = (a + torch.randint(1, p, (n,), generator=gen)) % p  # a′ ≠ a
    b = torch.randint(0, p, (n,), generator=gen)
    tokens = torch.stack([a, b, torch.full_like(a, p)], 1).to(dev)
    tokens2 = torch.stack([a2, b, torch.full_like(a, p)], 1).to(dev)
    right, wrong = ((a2 + b) % p).to(dev), ((a + b) % p).to(dev)  # counterfactual answer, original answer
    W_E = model.embed.W_E.double()

    def logit_diff(logits):
        """L[a′+b] − L[a+b] at the '=' position for every triple."""
        last = logits[:, -1].double()
        return last.gather(1, right[:, None])[:, 0] - last.gather(1, wrong[:, None])[:, 0]

    clean_a, clean_a2 = logit_diff(model(tokens)), logit_diff(model(tokens2))
    Q_key = key_embedding_subspace(model, grid, key_freqs)
    Q_rand, _ = torch.linalg.qr(torch.randn(d, Q_key.shape[1], generator=gen, dtype=torch.float64).to(dev))
    results = {}
    for name, Q in (("key", Q_key), ("random", Q_rand), ("complement", None)):

        def patch(x, Q=Q):
            """Swap the chosen subspace component of position 0 (token a) with that of token a′."""
            x = x.clone()
            e, e2 = x[:, 0].double(), W_E[:, a2.to(dev)].T  # current embedding of a, embedding of a′
            if Q is None:  # complement of the key subspace: swap everything except the key part
                P = torch.eye(d, device=dev, dtype=torch.float64) - Q_key @ Q_key.T
            else:
                P = Q @ Q.T  # orthogonal projector onto the subspace
            x[:, 0] = (e - e @ P + e2 @ P).to(x.dtype)
            return x

        patched = logit_diff(model(tokens, hooks={"hook_embed": patch}))
        recovered = ((patched - clean_a) / (clean_a2 - clean_a)).cpu().numpy()
        mean, low, high = bootstrap_interval(recovered, seed=seed)
        results[name] = {"mean": mean, "ci95": [low, high], "dims": int(Q_key.shape[1]) if Q is not None else d - int(Q_key.shape[1])}
    results["n_triples"] = n
    return results
