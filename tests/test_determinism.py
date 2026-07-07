"""The backward is deterministic: identical inputs -> bitwise-identical grads.

Guards the non-atomic two-kernel split (disjoint writes, no global atomics),
which is the property that distinguishes this backward from FA2-default.
"""
import pytest
import torch

from fastflash_attention import fast_attention

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _grads(q, k, v, g, is_causal):
    for t in (q, k, v):
        t.grad = None
    out = fast_attention(q, k, v, is_causal=is_causal)
    out.backward(g)
    return q.grad.clone(), k.grad.clone(), v.grad.clone()


@cuda
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("shape", [(2, 4, 512, 64), (1, 2, 1024, 128)])
def test_backward_bitwise_deterministic(shape, is_causal):
    torch.manual_seed(0)
    B, H, S, D = shape
    q, k, v = (
        torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    )
    g = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

    dq1, dk1, dv1 = _grads(q, k, v, g, is_causal)
    dq2, dk2, dv2 = _grads(q, k, v, g, is_causal)

    assert torch.equal(dq1, dq2)
    assert torch.equal(dk1, dk2)
    assert torch.equal(dv1, dv2)
