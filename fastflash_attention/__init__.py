"""fastflash_attention — a drop-in exact bf16 flash-attention for CUDA (sm_120-tuned).

A single fused Triton kernel for exact softmax attention: fast forward across all
sequence lengths and a *deterministic* (non-atomic) backward. Public surface
mirrors ``torch.nn.functional.scaled_dot_product_attention`` so integration is a
textual swap at any SDPA call site::

    from fastflash_attention import fast_attention
    out = fast_attention(q, k, v, is_causal=True)   # q,k,v: [B,H,S,D] bf16, CUDA

Policy is **strict**: the kernel runs when the input matches its supported
contract, and raises :class:`UnsupportedConfig` (never a hidden slow path)
otherwise. Use :func:`is_eligible` to branch to your own fallback::

    fn = fast_attention if is_eligible(q, k, v, is_causal=True) else F.scaled_dot_product_attention
    out = fn(q, k, v, is_causal=True)

Supported contract:
  * dtype ``bfloat16``, all of q/k/v on the same CUDA device
  * shape ``[B, H, S, D]`` identical for q, k and v (no GQA/MQA, no differing
    key/value length)
  * ``head_dim`` a power of two ``<= 128``
  * ``attn_mask is None`` and ``dropout_p == 0`` (causal masking via ``is_causal``)
  * ``scale`` optional (defaults to ``1/sqrt(head_dim)``)

Anything else raises. See :func:`is_eligible` for a non-raising check.
"""
from __future__ import annotations

import torch
from torch import nn

from ._kernel import fastflash_attn, fastflash_attn_train, FastFlashAttn, LOG2E

__all__ = [
    "fast_attention",
    "FastFlashAttention",
    "is_eligible",
    "UnsupportedConfig",
    "__version__",
]

__version__ = "0.1.0"

_POW2_LE_128 = frozenset((1, 2, 4, 8, 16, 32, 64, 128))


class UnsupportedConfig(ValueError):
    """Raised by :func:`fast_attention` when the input violates the supported contract.

    The message names the exact violated constraint. Call :func:`is_eligible`
    first to test the same contract without raising.
    """


def _reason(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa):
    """Return a human-readable reason string if unsupported, else ``None``."""
    if attn_mask is not None:
        return "attn_mask is not supported (strict policy); pass attn_mask=None and use is_causal"
    if dropout_p:
        return f"dropout_p={dropout_p} is not supported (strict policy); pass dropout_p=0"
    if enable_gqa:
        return "GQA/MQA (enable_gqa=True) is not supported (strict policy)"
    for name, t in (("query", query), ("key", key), ("value", value)):
        if not isinstance(t, torch.Tensor):
            return f"{name} is not a torch.Tensor"
        if not t.is_cuda:
            return f"{name} is on device '{t.device}'; a CUDA tensor is required"
        if t.dtype is not torch.bfloat16:
            return f"{name} dtype is {t.dtype}; only bfloat16 is supported"
        if t.dim() != 4:
            return f"{name} has {t.dim()} dims; expected 4 ([B, H, S, D])"
    qs, ks, vs = tuple(query.shape), tuple(key.shape), tuple(value.shape)
    if qs != ks or qs != vs:
        return (f"shapes differ (query {qs}, key {ks}, value {vs}); GQA/MQA and "
                "differing key/value length are not supported (strict policy)")
    d = qs[-1]
    if d not in _POW2_LE_128:
        return f"head_dim={d}; must be a power of two <= 128"
    return None


def is_eligible(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, enable_gqa=False) -> bool:
    """Return True iff :func:`fast_attention` can run this input (never raises).

    Accepts the same arguments as :func:`fast_attention`; pairs with the strict
    policy so a model can branch to its own fallback when False.
    """
    return _reason(query, key, value, attn_mask, dropout_p,
                   is_causal, scale, enable_gqa) is None


def fast_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, enable_gqa=False):
    """Exact bf16 flash-attention, drop-in for ``F.scaled_dot_product_attention``.

    q, k, v are ``[B, H, S, D]`` bfloat16 CUDA tensors. Returns ``[B, H, S, D]``
    bfloat16. Differentiable: when grad is required, routes through the
    deterministic two-kernel backward; otherwise takes the lean forward-only
    path (no LSE write). Raises :class:`UnsupportedConfig` on any input outside
    the supported contract — see :func:`is_eligible` for a non-raising check.
    """
    reason = _reason(query, key, value, attn_mask, dropout_p,
                     is_causal, scale, enable_gqa)
    if reason is not None:
        raise UnsupportedConfig(reason)
    if torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    ):
        return fastflash_attn_train(query, key, value, is_causal, scale)
    return fastflash_attn(query, key, value, is_causal, scale=scale)


class FastFlashAttention(nn.Module):
    """``nn.Module`` wrapper around :func:`fast_attention`.

    Stateless apart from optional defaults for ``is_causal`` / ``scale`` set at
    construction; ``forward`` accepts the same arguments as :func:`fast_attention`
    and per-call arguments override the constructor defaults.

        attn = FastFlashAttention(is_causal=True)
        out = attn(q, k, v)                 # q,k,v: [B,H,S,D] bf16 CUDA
    """

    def __init__(self, is_causal: bool = False, scale=None):
        super().__init__()
        self.is_causal = is_causal
        self.scale = scale

    def forward(self, query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=None, scale=None, enable_gqa=False):
        return fast_attention(
            query, key, value,
            attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=self.is_causal if is_causal is None else is_causal,
            scale=self.scale if scale is None else scale,
            enable_gqa=enable_gqa,
        )

    def extra_repr(self) -> str:
        return f"is_causal={self.is_causal}, scale={self.scale}"
