"""Forward + backward numerical parity of fastflash_attention vs torch SDPA (fp32 truth).

Reference is ``F.scaled_dot_product_attention`` with inputs upcast to fp32, so
"truth" is the exact math independent of any other attention kernel (no
flash_attn dependency). fastflash_attention runs in bf16; tolerances are the bf16 floor.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from fastflash_attention import fast_attention

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# (B, H, S, D) — includes a ragged S (non-multiple of block sizes) and D in {64,128}.
SHAPES = [
    (2, 4, 128, 64),
    (1, 8, 512, 128),
    (2, 2, 200, 64),   # ragged S
    (1, 3, 1024, 128),
]


def _rel_l2(a, b):
    a = a.float()
    b = b.float()
    return (a - b).norm() / b.norm().clamp_min(1e-12)


def _ref(q, k, v, is_causal, scale):
    return F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=is_causal, scale=scale
    )


@cuda
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("is_causal", [False, True])
def test_forward_parity(shape, is_causal):
    torch.manual_seed(0)
    B, H, S, D = shape
    q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = fast_attention(q, k, v, is_causal=is_causal)
    ref = _ref(q, k, v, is_causal, None)
    assert _rel_l2(out, ref) < 5e-3


@cuda
def test_forward_custom_scale():
    torch.manual_seed(1)
    B, H, S, D = 2, 4, 256, 64
    q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    scale = 0.05  # deliberately != 1/sqrt(D)
    out = fast_attention(q, k, v, is_causal=True, scale=scale)
    ref = _ref(q, k, v, True, scale)
    assert _rel_l2(out, ref) < 5e-3
    # and it must actually differ from the default-scale result
    out_default = fast_attention(q, k, v, is_causal=True)
    assert _rel_l2(out, out_default) > 1e-2


@cuda
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("is_causal", [False, True])
def test_backward_parity(shape, is_causal):
    torch.manual_seed(2)
    B, H, S, D = shape
    q, k, v = (
        torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    )
    qr, kr, vr = (t.detach().float().requires_grad_(True) for t in (q, k, v))

    out = fast_attention(q, k, v, is_causal=is_causal)
    ref = _ref(qr, kr, vr, is_causal, None)

    torch.manual_seed(3)
    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g.float())

    for name, got, want in (("dq", q.grad, qr.grad), ("dk", k.grad, kr.grad), ("dv", v.grad, vr.grad)):
        assert _rel_l2(got, want) < 1e-2, f"{name} rel-L2 too high for shape={shape} causal={is_causal}"
