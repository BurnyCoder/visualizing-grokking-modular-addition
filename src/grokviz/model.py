"""GrokTransformer: the paper's one-layer transformer, with the paper's exact parameter names and layouts.

Local explanation: tokens (a, b, =) → embedding W_E plus positional embedding W_pos → residual stream →
4-head causal attention (W_Q, W_K, W_V, W_O) added back → ReLU MLP (W_in, b_in, W_out, b_out) added back →
unembedding W_U → 114 logits per position. No LayerNorm, untied embed/unembed, biases only in the MLP.
Every intermediate tensor passes through a named hook so analyses can record it (`run_with_cache`) or
replace it (interventions/ablations/patching) without changing the model code.
Global context: the paper's released checkpoints (Hugging Face callummcdougall/grokking_full_run_data) use
this exact state_dict layout (W_E is [d_model, d_vocab], W_O is [d_model, heads*d_head], ...), so they load
with strict=True and no conversion; our own runs use the same class, and the website's JavaScript forward
pass mirrors it line by line.
Sources: paper code `Embed`, `PosEmbed`, `Attention`, `MLP`, `TransformerBlock`, `Transformer`
(https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/transformers.py,
commit 23d2c64); paper App. A equations (https://arxiv.org/abs/2301.05217); TransformerLens hook naming
idea (https://github.com/TransformerLensOrg/TransformerLens).
"""

from __future__ import annotations  # postponed annotations

import math  # sqrt for initialisation and attention scaling
from dataclasses import dataclass  # model hyper-parameters
from typing import Callable  # hook function type

import torch  # tensors
import torch.nn as nn  # Module / Parameter containers
import torch.nn.functional as F  # softmax, relu

Hook = Callable[[torch.Tensor], "torch.Tensor | None"]  # receives an activation, may return a replacement


@dataclass(frozen=True)
class ModelShape:
    """Architecture sizes (paper Sec. 3 / App. A defaults)."""

    d_vocab: int = 114  # 113 numbers + '=' token
    d_model: int = 128  # residual stream width d
    n_ctx: int = 3  # sequence length: a, b, =
    d_mlp: int = 512  # MLP neurons n
    n_heads: int = 4  # attention heads
    d_head: int = 32  # per-head query/key/value size (= d_model / n_heads)
    n_layers: int = 1  # transformer blocks

    @classmethod
    def from_config(cls, cfg) -> "ModelShape":
        """Build from a grokviz Config (vocab = p + 1 for the '=' token)."""
        return cls(cfg.p + 1, cfg.d_model, cfg.n_ctx, cfg.d_mlp, cfg.n_heads, cfg.d_head)


class _Embed(nn.Module):
    """Token embedding table W_E [d_model, d_vocab]: column t is the vector for token t (paper `Embed`)."""

    def __init__(self, s: ModelShape, g: torch.Generator):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(s.d_model, s.d_vocab, generator=g) / math.sqrt(s.d_model))


class _PosEmbed(nn.Module):
    """Learned positional embeddings W_pos [n_ctx, d_model]: row i is added at position i (paper `PosEmbed`)."""

    def __init__(self, s: ModelShape, g: torch.Generator):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(s.n_ctx, s.d_model, generator=g) / math.sqrt(s.d_model))


class _Attention(nn.Module):
    """Multi-head attention weights (paper `Attention`): W_K/W_Q/W_V [heads, d_head, d_model], W_O [d_model, heads*d_head]."""

    def __init__(self, s: ModelShape, g: torch.Generator):
        super().__init__()
        scale = math.sqrt(s.d_model)  # paper divides every randn init by sqrt(d_model)
        self.W_K = nn.Parameter(torch.randn(s.n_heads, s.d_head, s.d_model, generator=g) / scale)
        self.W_Q = nn.Parameter(torch.randn(s.n_heads, s.d_head, s.d_model, generator=g) / scale)
        self.W_V = nn.Parameter(torch.randn(s.n_heads, s.d_head, s.d_model, generator=g) / scale)
        self.W_O = nn.Parameter(torch.randn(s.d_model, s.d_head * s.n_heads, generator=g) / scale)
        self.register_buffer("mask", torch.tril(torch.ones(s.n_ctx, s.n_ctx)))  # causal: query q sees keys ≤ q


class _MLP(nn.Module):
    """ReLU MLP weights (paper `MLP`): W_in [d_mlp, d_model], b_in [d_mlp], W_out [d_model, d_mlp], b_out [d_model]."""

    def __init__(self, s: ModelShape, g: torch.Generator):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(s.d_mlp, s.d_model, generator=g) / math.sqrt(s.d_model))
        self.b_in = nn.Parameter(torch.zeros(s.d_mlp))
        self.W_out = nn.Parameter(torch.randn(s.d_model, s.d_mlp, generator=g) / math.sqrt(s.d_model))
        self.b_out = nn.Parameter(torch.zeros(s.d_model))


class _Block(nn.Module):
    """One transformer block = attention + MLP (state_dict prefix blocks.<i>.attn / blocks.<i>.mlp)."""

    def __init__(self, s: ModelShape, g: torch.Generator):
        super().__init__()
        self.attn = _Attention(s, g)
        self.mlp = _MLP(s, g)


class _Unembed(nn.Module):
    """Unembedding W_U [d_model, d_vocab]: logits = residual @ W_U (paper `Unembed`, init /sqrt(d_vocab))."""

    def __init__(self, s: ModelShape, g: torch.Generator):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(s.d_model, s.d_vocab, generator=g) / math.sqrt(s.d_vocab))


class GrokTransformer(nn.Module):
    """The paper's 1-layer, 4-head, ReLU transformer with hookable forward pass."""

    def __init__(self, shape: ModelShape = ModelShape(), seed: int = 0):
        super().__init__()
        self.shape = shape
        g = torch.Generator().manual_seed(seed)  # private CPU RNG: init is identical on every device
        # Construction order = paper's random-draw order: W_E, W_pos, W_K, W_Q, W_V, W_O, W_in, W_out, W_U.
        self.embed = _Embed(shape, g)
        self.pos_embed = _PosEmbed(shape, g)
        self.blocks = nn.ModuleList([_Block(shape, g) for _ in range(shape.n_layers)])
        self.unembed = _Unembed(shape, g)

    def forward(
        self,
        tokens: torch.Tensor,
        hooks: dict[str, Hook] | None = None,
        cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Map tokens [batch, n_ctx] to logits [batch, n_ctx, d_vocab], applying/recording named hooks."""

        def h(name: str, x: torch.Tensor) -> torch.Tensor:
            """Hook point: optionally replace `x` (intervention), optionally record it (cache)."""
            if hooks is not None and name in hooks:
                out = hooks[name](x)
                x = x if out is None else out  # None means "observe only"
            if cache is not None:
                cache[name] = x.detach()
            return x

        x = h("hook_embed", self.embed.W_E[:, tokens].permute(1, 2, 0))  # column lookup → [batch, pos, d_model]
        x = h("blocks.0.resid_pre", x + h("hook_pos_embed", self.pos_embed.W_pos[: tokens.shape[1]]))
        for i, block in enumerate(self.blocks):
            pre = f"blocks.{i}"
            if i > 0:
                x = h(f"{pre}.resid_pre", x)
            a = block.attn
            # Per-head keys/queries/values: [batch, head, pos, d_head] (paper einsum 'ihd,bpd->biph').
            k = h(f"{pre}.attn.k", torch.einsum("ihd,bpd->biph", a.W_K, x))
            q = h(f"{pre}.attn.q", torch.einsum("ihd,bpd->biph", a.W_Q, x))
            v = h(f"{pre}.attn.v", torch.einsum("ihd,bpd->biph", a.W_V, x))
            n = x.shape[1]  # sequence length actually present
            scores = torch.einsum("biph,biqh->biqp", k, q)  # [batch, head, query, key] dot products
            scores = torch.tril(scores) - 1e10 * (1 - a.mask[:n, :n])  # causal mask exactly as in the paper
            scores = h(f"{pre}.attn.scores", scores / math.sqrt(self.shape.d_head))  # scaled pre-softmax scores
            pattern = h(f"{pre}.attn.pattern", F.softmax(scores, dim=-1))  # attention weights, rows sum to 1
            z = h(f"{pre}.attn.z", torch.einsum("biph,biqp->biqh", v, pattern))  # weighted values per head
            z_flat = z.permute(0, 2, 1, 3).reshape(z.shape[0], n, -1)  # [batch, pos, head*d_head] ('b q (i h)')
            attn_out = h(f"{pre}.attn_out", torch.einsum("df,bqf->bqd", a.W_O, z_flat))  # write back to d_model
            x = h(f"{pre}.resid_mid", x + attn_out)  # residual connection around attention
            m = block.mlp
            pre_act = h(f"{pre}.mlp.pre", torch.einsum("md,bpd->bpm", m.W_in, x) + m.b_in)  # neuron inputs
            post = h(f"{pre}.mlp.post", F.relu(pre_act))  # neuron activations (ReLU)
            mlp_out = h(f"{pre}.mlp_out", torch.einsum("dm,bpm->bpd", m.W_out, post) + m.b_out)
            x = h(f"{pre}.resid_post", x + mlp_out)  # residual connection around the MLP
        return h("logits", x @ self.unembed.W_U)  # [batch, pos, d_vocab]

    def run_with_cache(self, tokens: torch.Tensor, hooks: dict[str, Hook] | None = None):
        """Forward pass that also returns every named activation: (logits, cache dict)."""
        cache: dict[str, torch.Tensor] = {}
        logits = self.forward(tokens, hooks=hooks, cache=cache)
        return logits, cache
