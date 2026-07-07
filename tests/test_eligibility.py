"""Strict-policy contract: is_eligible mirrors fast_attention's raise decision."""
import pytest
import torch

from fastflash_attention import is_eligible, fast_attention, UnsupportedConfig, FastFlashAttention

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _bf16(B, H, S, D, device="cuda"):
    return torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)


@cuda
def test_eligible_supported():
    q = _bf16(2, 4, 128, 64)
    assert is_eligible(q, q, q, is_causal=True)
    # and it runs without raising
    fast_attention(q, q, q, is_causal=True)


@cuda
@pytest.mark.parametrize("kwargs, needle", [
    ({"attn_mask": "MASK"}, "attn_mask"),
    ({"dropout_p": 0.1}, "dropout_p"),
    ({"enable_gqa": True}, "GQA"),
])
def test_unsupported_args_raise(kwargs, needle):
    q = _bf16(2, 4, 128, 64)
    # attn_mask sentinel -> use a real tensor
    if kwargs.get("attn_mask") == "MASK":
        kwargs["attn_mask"] = torch.zeros(128, 128, device="cuda", dtype=torch.bfloat16)
    assert not is_eligible(q, q, q, **kwargs)
    with pytest.raises(UnsupportedConfig, match=needle):
        fast_attention(q, q, q, **kwargs)


@cuda
def test_fp16_and_fp32_raise():
    for dt in (torch.float16, torch.float32):
        q = torch.randn(2, 4, 128, 64, device="cuda", dtype=dt)
        assert not is_eligible(q, q, q)
        with pytest.raises(UnsupportedConfig, match="bfloat16"):
            fast_attention(q, q, q)


@cuda
def test_gqa_shape_mismatch_raises():
    q = _bf16(2, 8, 128, 64)
    kv = _bf16(2, 2, 128, 64)  # fewer KV heads
    assert not is_eligible(q, kv, kv)
    with pytest.raises(UnsupportedConfig, match="GQA|shapes differ"):
        fast_attention(q, kv, kv)


@cuda
def test_bad_head_dim_raises():
    q = _bf16(2, 4, 128, 96)  # not a power of two
    assert not is_eligible(q, q, q)
    with pytest.raises(UnsupportedConfig, match="head_dim"):
        fast_attention(q, q, q)


def test_cpu_raises():
    q = torch.randn(2, 4, 16, 64, dtype=torch.bfloat16)  # CPU
    assert not is_eligible(q, q, q)
    with pytest.raises(UnsupportedConfig, match="CUDA"):
        fast_attention(q, q, q)


@cuda
def test_module_forwards_defaults():
    q = _bf16(2, 4, 128, 64)
    attn = FastFlashAttention(is_causal=True)
    out = attn(q, q, q)
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    # per-call override wins
    with pytest.raises(UnsupportedConfig, match="dropout_p"):
        attn(q, q, q, dropout_p=0.5)
