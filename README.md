# FastFlashAttention

**Drop-in exact bf16 flash-attention for CUDA — one fused Triton kernel with a fast forward across all sequence lengths and a *deterministic* (non-atomic) backward.** Tuned on an RTX 5090 (Blackwell / sm_120), where its forward beats FlashAttention-2 and its deterministic backward beats FA2's deterministic backward.

The public surface mirrors `torch.nn.functional.scaled_dot_product_attention`, so adoption is a textual swap at any SDPA call site.

## Status

An optimized **exact** attention kernel (not an approximation): fp32-faithful softmax at the bf16 floor (~0.2% rel-L2 vs fp32), forward + backward in a single `@triton.jit` kernel family.

- **Forward:** faster than FA2-default across the whole measured range — **1.07–1.30×** at D=128 causal (up to **1.78×** at short D=64), reaching **96%** of the bf16 matmul roofline at long context.
- **Backward:** bitwise-**deterministic** by construction (disjoint writes, no global atomics). Beats FA2's *deterministic* backward by **1.2–2.3×** (D=128), and reaches **~0.80–0.96×** of FA2's *default* (non-deterministic, atomic) backward.
- **Full training step (fwd+bwd):** beats FA2-deterministic by **1.4–2.1×**, and is roughly par with FA2-default (**0.86–1.18×**, faster at short context).
- **Scope:** exact attention only, with a strict input contract (below) and no hidden slow path.

## Install

`torch` and `triton` must already be installed with a CUDA build matching your GPU (developed on torch 2.12.1+cu130 / triton 3.7.1, CUDA 13.0). Then:

```bash
pip install -e .
```

## Use

```python
import torch
from fastflash_attention import fast_attention, FastFlashAttention, is_eligible

q = torch.randn(2, 8, 4096, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)

out = fast_attention(q, k, v, is_causal=True)            # [B, H, S, D] bf16

# differentiable: grads flow to q/k/v through the deterministic backward
q.requires_grad_(); out = fast_attention(q, k, v, is_causal=True); out.sum().backward()

attn = FastFlashAttention(is_causal=True)                # nn.Module
out = attn(q, k, v)
```

Strict policy + fallback — `fast_attention` runs when the input matches the supported contract and raises `UnsupportedConfig` otherwise (never a hidden slow path). Branch with the non-raising `is_eligible`:

```python
import torch.nn.functional as F
fn = fast_attention if is_eligible(q, k, v, is_causal=causal) else F.scaled_dot_product_attention
out = fn(q, k, v, is_causal=causal)
```

## Supported contract

| Requirement | Value |
|---|---|
| dtype | `bfloat16` |
| device | CUDA (q, k, v same device) |
| layout / shape | `[B, H, S, D]`, identical for q, k, v |
| head_dim `D` | power of two, `≤ 128` |
| masking | `is_causal` (bool) only |
| `scale` | optional, defaults to `1/√D` |
| `attn_mask` / `dropout_p` | must be `None` / `0` |

Anything else raises `UnsupportedConfig` (use `is_eligible` for a non-raising check). Not supported: GQA/MQA, fp16/fp32, additive bias/mask, dropout, differing key/value length.

## Benchmarks

Measured on **NVIDIA GeForce RTX 5090** (sm_120), torch 2.12.1+cu130, CUDA 13.0, flash_attn 2.8.4; **B=4, H=16**. CUDA-event timing, median over 30 iters (≥15 warmup excluded). Ratios are **FA2 / FastFlashAttention wall time — >1 means FastFlashAttention is faster.** bf16 matmul roofline ≈ **234 TF/s** (achieved, used as the %-roofline denominator).

Reproduce:
```bash
pip install -e ".[bench]"
python -m bench.benchmark        # full grid; add --quick for a smoke test
```

### Forward (causal, D=128)

| N | FastFlashAttention (ms) | FA2 (ms) | ratio | % roofline |
|---:|---:|---:|---:|---:|
| 512 | 0.076 | 0.094 | **1.24×** | 24.1 |
| 1024 | 0.152 | 0.197 | **1.30×** | 48.4 |
| 2048 | 0.421 | 0.522 | **1.24×** | 69.7 |
| 4096 | 1.364 | 1.590 | **1.17×** | 86.1 |
| 8192 | 5.073 | 5.481 | **1.08×** | 92.6 |
| 16384 | 19.476 | 20.825 | **1.07×** | 96.5 |

### Backward (causal, D=128)

Both sides deterministic on the `-det` columns. `ratio_det` is the apples-to-apples deterministic comparison.

| N | FastFlashAttention (ms) | FA2-det (ms) | ratio_det | FA2-default (ms) | ratio_def |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.188 | 0.229 | **1.22×** | 0.174 | 0.93× |
| 1024 | 0.433 | 0.662 | **1.53×** | 0.374 | 0.86× |
| 2048 | 1.085 | 2.305 | **2.12×** | 1.037 | 0.96× |
| 4096 | 3.732 | 8.522 | **2.28×** | 3.512 | 0.94× |
| 8192 | 15.309 | 33.384 | **2.18×** | 13.104 | 0.86× |
| 16384 | 62.642 | 132.860 | **2.12×** | 49.935 | 0.80× |

### Full training step, fwd+bwd (causal, D=128)

| N | FastFlashAttention (ms) | FA2-det (ms) | ratio_det | FA2-default (ms) | ratio_def |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.269 | 0.367 | **1.37×** | 0.318 | **1.18×** |
| 1024 | 0.536 | 0.793 | **1.48×** | 0.499 | 0.93× |
| 2048 | 1.420 | 2.725 | **1.92×** | 1.476 | **1.04×** |
| 4096 | 5.026 | 9.908 | **1.97×** | 4.994 | 0.99× |
| 8192 | 20.331 | 38.764 | **1.91×** | 18.497 | 0.91× |
| 16384 | 82.334 | 170.402 | **2.07×** | 70.931 | 0.86× |

### Full grid (all D and causal settings)

Raw output of `python -m bench.benchmark` — `r_*` columns are FA2/FastFlashAttention (>1 = faster); `_def` = vs FA2-default, `_det` = vs FA2-deterministic.

```
# regime=fwd/bwd/step  causal=True  D=64  B=4 H=16
     N |   ff_fwd  fa2_fwd  r_fwd |   ff_bwd  fa2_bwd  r_def fa2d_bwd  r_det |  ff_step fa2_step rs_def fa2d_step rs_det
   512 |   0.0484   0.0858  1.775 |   0.2687   0.1656  0.616   0.1936  0.721 |   0.2452   0.3963  1.616    0.3545  1.446
  1024 |   0.0836   0.1234  1.476 |   0.2618    0.249  0.951   0.3259  1.245 |   0.3348   0.3854  1.151    0.4701  1.404
  2048 |   0.2076   0.2749  1.324 |   0.6769   0.5898  0.871   0.9006   1.33 |   0.8376   0.7923  0.946    1.1167  1.333
  4096 |   0.6956   0.8066   1.16 |   2.2154   1.8825   0.85   3.0978  1.398 |   2.8038   2.5641  0.915    3.8168  1.361
  8192 |   2.5345    2.721  1.074 |   8.9859   6.6687  0.742  11.7782  1.311 |  11.5982   9.5629  0.825     14.33  1.236
 16384 |   9.9066  10.3255  1.042 |  35.8458  25.5077  0.712  46.6762  1.302 |  45.4103  35.4411   0.78   56.5716  1.246

# regime=fwd/bwd/step  causal=True  D=128  B=4 H=16
     N |   ff_fwd  fa2_fwd  r_fwd |   ff_bwd  fa2_bwd  r_def fa2d_bwd  r_det |  ff_step fa2_step rs_def fa2d_step rs_det
   512 |    0.076   0.0939  1.236 |   0.1882   0.1744  0.927    0.229  1.217 |   0.2687   0.3176  1.182    0.3671  1.366
  1024 |   0.1516   0.1965  1.296 |   0.4326   0.3735  0.863   0.6621  1.531 |   0.5361   0.4987   0.93    0.7932   1.48
  2048 |    0.421   0.5218   1.24 |   1.0852   1.0373  0.956   2.3048  2.124 |   1.4199   1.4762   1.04    2.7253  1.919
  4096 |   1.3641   1.5896  1.165 |   3.7315   3.5122  0.941   8.5222  2.284 |   5.0257    4.994  0.994    9.9081  1.971
  8192 |   5.0728   5.4809   1.08 |  15.3092  13.1042  0.856  33.3843  2.181 |  20.3306  18.4974   0.91   38.7636  1.907
 16384 |  19.4756  20.8246  1.069 |  62.6423  49.9351  0.797   132.86  2.121 |  82.3344  70.9306  0.861  170.4016   2.07

# regime=fwd/bwd/step  causal=False  D=64  B=4 H=16
     N |   ff_fwd  fa2_fwd  r_fwd |   ff_bwd  fa2_bwd  r_def fa2d_bwd  r_det |  ff_step fa2_step rs_def fa2d_step rs_det
   512 |   0.0542   0.0799  1.475 |   0.1523    0.155  1.018   0.1965   1.29 |   0.2207   0.3124  1.416    0.3492  1.582
  1024 |   0.1122   0.1465  1.306 |   0.3583   0.3334  0.931   0.4294  1.198 |   0.4784   0.4901  1.024    0.5649  1.181
  2048 |   0.3388   0.4032   1.19 |   1.1888   0.9977  0.839   1.5703  1.321 |   1.4221    1.298  0.913    1.8625   1.31
  4096 |   1.2306   1.3384  1.088 |   3.9973   3.3316  0.833   5.4019  1.351 |   5.1665   4.5629  0.883    6.6129   1.28
  8192 |   4.7573   4.8856  1.027 |  15.9051  12.2538   0.77  20.9283  1.316 |  20.6395  17.0469  0.826   25.6948  1.245
 16384 |  18.6905  18.7004  1.001 |  65.8158  47.1465  0.716  89.1114  1.354 |  84.5147  65.7692  0.778  106.1956  1.257

# regime=fwd/bwd/step  causal=False  D=128  B=4 H=16
     N |   ff_fwd  fa2_fwd  r_fwd |   ff_bwd  fa2_bwd  r_def fa2d_bwd  r_det |  ff_step fa2_step rs_def fa2d_step rs_det
   512 |   0.0988   0.1088  1.101 |   0.2563   0.2097  0.818   0.2886  1.126 |   0.3577   0.3775  1.055    0.4297  1.201
  1024 |   0.2204   0.2783  1.263 |   0.7521   0.5915  0.786   1.1663  1.551 |   0.8652   0.7563  0.874    1.3376  1.546
  2048 |   0.6968   0.8172  1.173 |   1.8674   1.8319  0.981    4.193  2.245 |    2.461   2.5367  1.031    4.8474   1.97
  4096 |   2.4955   2.7578  1.105 |   6.9941   6.4915  0.928  15.6969  2.244 |   9.4322    9.171  0.972   18.2489  1.935
  8192 |   9.4854  10.1479   1.07 |   28.779  24.6036  0.855  64.2439  2.232 |   38.187  34.6436  0.907   75.2939  1.972
 16384 |  37.4032  38.7926  1.037 | 119.3157  95.9693  0.804 282.0241  2.364 | 156.9293  134.626  0.858  311.1204  1.983
```

Reading the grid: forward is faster than FA2 in every cell. The deterministic backward beats FA2-deterministic everywhere except the smallest D=64 case (N=512), and closes most of the gap to FA2's faster non-deterministic default. At the full-step level FastFlashAttention wins outright against FA2-deterministic and is competitive with FA2-default, pulling ahead at short context.

## Determinism

The backward is bitwise-identical across runs — disjoint writes, no global atomics — verified by `tests/test_determinism.py`. This is the property FA2 only provides via its slower `deterministic=True` path; FastFlashAttention is deterministic by construction, at a fraction of that path's cost.

## Tests

```bash
pytest tests/     # forward+backward parity vs fp32 SDPA truth, backward determinism, eligibility contract
```

Parity reference is `F.scaled_dot_product_attention` upcast to fp32, so the suite has no `flash_attn` dependency (that is benchmark-only).

## License

MIT — see [LICENSE](LICENSE).
