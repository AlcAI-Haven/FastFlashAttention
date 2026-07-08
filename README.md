# FastFlashAttention

[![PyPI](https://img.shields.io/pypi/v/fastflash-attention.svg)](https://pypi.org/project/fastflash-attention/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![CUDA: sm_120 · RTX 5090](https://img.shields.io/badge/CUDA-sm__120%20%C2%B7%20RTX%205090-76B900.svg)](#status)
[![Kernel: Triton](https://img.shields.io/badge/kernel-Triton-EE4C2C.svg)](https://github.com/triton-lang/triton)
[![Backward: deterministic](https://img.shields.io/badge/backward-deterministic-2a78d6.svg)](#determinism)

**Drop-in exact bf16 flash-attention for CUDA — one fused Triton kernel with a fast forward across all sequence lengths and a *deterministic* (non-atomic) backward.** Tuned on a **consumer GeForce RTX 5090** (Blackwell GB202, **sm_120**), where its forward beats FlashAttention-2 and its deterministic backward beats FA2's deterministic backward.

The public surface mirrors `torch.nn.functional.scaled_dot_product_attention`, so adoption is a textual swap at any SDPA call site.

## Status

An optimized **exact** attention kernel (not an approximation): fp32-faithful softmax at the bf16 floor (~0.2% rel-L2 vs fp32), forward + backward in a single `@triton.jit` kernel family.

- **Forward:** faster than FA2-default across the whole measured range — **1.06–1.34×** at D=128 causal (up to **1.70×** at short D=64), reaching **~97%** of the bf16 matmul roofline at long context.
- **Backward:** bitwise-**deterministic** by construction (disjoint writes, no global atomics). Beats FA2's *deterministic* backward by **1.1–2.1×** (D=128), and reaches **~0.79–0.89×** of FA2's *default* (non-deterministic, atomic) backward.
- **Full training step (fwd+bwd):** beats FA2-deterministic by **1.2–1.9×**, and is roughly par with FA2-default (**0.86–1.20×**, faster at short context).
- **Scope:** exact attention only, with a strict input contract (below) and no hidden slow path.
- **Hardware:** tuned for the **consumer GeForce RTX 5090 (GB202, sm_120)** — *not* datacenter Blackwell (GB100/GB200, sm_100). It uses the standard sm_120 tensor-core MMA that Triton emits, and does **not** rely on datacenter-only 5th-gen tensor-core features (`tcgen05` MMA / tensor-memory, the `sm_100a` path). It runs on other CUDA GPUs, but the autotuned block/warp choices are picked for sm_120 and may be suboptimal elsewhere.

## Install

`torch` and `triton` must already be installed with a CUDA build matching your GPU (developed on torch 2.12.1+cu130 / triton 3.7.1, CUDA 13.0). Then:

```bash
pip install fastflash-attention
```

Or from source (add `.[bench]` for the benchmark/plot dependencies):

```bash
git clone https://github.com/AlcAI-Haven/FastFlashAttention && cd FastFlashAttention
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

Measured on **NVIDIA GeForce RTX 5090** (sm_120), torch 2.12.1+cu130, CUDA 13.0, flash_attn 2.8.4; **B=4, H=16**. CUDA-event timing, median over 30 iters (≥15 warmup excluded). Ratios are **FA2 / FastFlashAttention wall time — >1 means FastFlashAttention is faster.** bf16 matmul roofline ≈ **238 TF/s** (achieved, used as the %-roofline denominator).

Reproduce:
```bash
pip install -e ".[bench]"
python -m bench.benchmark        # full grid; add --quick for a smoke test
```

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/speedup_dark.png">
    <img alt="FastFlashAttention speedup over FlashAttention-2 across sequence length — forward, backward, and full training step (causal, D=128). Above the parity line means FastFlashAttention is faster." src="assets/speedup_light.png" width="100%">
  </picture>
</p>

<p align="center"><sub>Speedup = FA2 / FastFlashAttention wall time (>1 = FastFlashAttention faster). Regenerate with <code>python -m bench.plot</code>.</sub></p>

### Forward (causal, D=128)

| N | FastFlashAttention (ms) | FA2 (ms) | ratio | % roofline |
|---:|---:|---:|---:|---:|
| 512 | 0.073 | 0.095 | **1.29×** | 24.6 |
| 1024 | 0.149 | 0.200 | **1.34×** | 48.5 |
| 2048 | 0.414 | 0.518 | **1.25×** | 69.8 |
| 4096 | 1.359 | 1.590 | **1.17×** | 85.1 |
| 8192 | 4.966 | 5.462 | **1.10×** | 93.1 |
| 16384 | 18.987 | 20.163 | **1.06×** | 97.4 |

### Backward (causal, D=128)

Both sides deterministic on the `-det` columns. `ratio_det` is the apples-to-apples deterministic comparison.

| N | FastFlashAttention (ms) | FA2-det (ms) | ratio_det | FA2-default (ms) | ratio_def |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.189 | 0.212 | **1.12×** | 0.158 | 0.84× |
| 1024 | 0.444 | 0.688 | **1.55×** | 0.353 | 0.80× |
| 2048 | 1.203 | 2.306 | **1.92×** | 1.075 | 0.89× |
| 4096 | 4.176 | 8.304 | **1.99×** | 3.495 | 0.84× |
| 8192 | 15.906 | 31.794 | **2.00×** | 12.615 | 0.79× |
| 16384 | 60.736 | 129.294 | **2.13×** | 48.243 | 0.79× |

### Full training step, fwd+bwd (causal, D=128)

| N | FastFlashAttention (ms) | FA2-det (ms) | ratio_det | FA2-default (ms) | ratio_def |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.252 | 0.312 | **1.24×** | 0.303 | **1.20×** |
| 1024 | 0.522 | 0.816 | **1.56×** | 0.489 | 0.94× |
| 2048 | 1.480 | 2.685 | **1.81×** | 1.478 | **1.00×** |
| 4096 | 5.461 | 9.610 | **1.76×** | 4.976 | 0.91× |
| 8192 | 20.799 | 37.522 | **1.80×** | 17.991 | 0.86× |
| 16384 | 79.905 | 150.409 | **1.88×** | 68.605 | 0.86× |

<details>
<summary><b>All configurations</b> — speedup ranges across N = 512…16384 (head_dim ∈ {64, 128} × causal / non-causal)</summary>

Min–max of the FA2 / FastFlashAttention ratio over the six sequence lengths (>1 = FastFlashAttention faster).

| Config | Forward | Backward vs FA2-det | Backward vs FA2-default | Step vs FA2-det | Step vs FA2-default |
|---|---|---|---|---|---|
| causal, D=128 | 1.06–1.34× | 1.12–2.13× | 0.79–0.89× | 1.24–1.88× | 0.86–1.20× |
| causal, D=64 | 1.04–1.70× | 1.21–1.31× | 0.71–0.97× | 1.21–1.43× | 0.78–1.18× |
| non-causal, D=128 | 1.04–1.27× | 1.16–2.19× | 0.78–0.93× | 1.44–2.09× | 0.86–1.25× |
| non-causal, D=64 | 1.00–1.46× | 1.18–1.33× | 0.70–0.89× | 1.20–1.66× | 0.76–1.40× |

Forward wins in every cell. The deterministic backward now beats FA2-deterministic in every measured cell (1.12–2.19×), and stays within ~0.70–0.97× of FA2's faster non-deterministic default. Full per-N numbers: run `python -m bench.benchmark` (writes `results/benchmark.jsonl`).

</details>

## Memory

FastFlashAttention runs natively in `[B, H, S, D]` (the SDPA layout) with output-only scratch, so **at inference it uses ~29–43% less peak VRAM than FlashAttention-2** at the same N. This is intrinsic, not a layout artifact — FA2 fed already-seq-major inputs measures the same. The training step is the deliberate trade in the other direction: the **deterministic backward stores a `dS` tile** (to avoid recomputation and global atomics — the source of its speed *and* bit-exactness), so its peak is a **flat ~19% higher at N ≥ 2048** — a shape- and head-dim-aware budget on that internal buffer keeps the overhead flat instead of growing with N.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/memory_dark.png">
    <img alt="Peak GPU memory vs FlashAttention-2 — forward (inference) uses less, full training step uses more at long N (causal, D=128)." src="assets/memory_light.png" width="90%">
  </picture>
</p>

Peak allocated VRAM (MB), causal D=128, B=4 H=16. `Δ vs FA2` is negative when FastFlashAttention uses **less**. Reproduce with `python -m bench.mem` (each point measured in a fresh process).

| N | Fwd (MB) | FA2 fwd (MB) | Δ vs FA2 | Train (MB) | FA2 train (MB) | Δ vs FA2 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 34 | 59 | **−43%** | 93 | 135 | −31% |
| 1024 | 67 | 118 | **−43%** | 185 | 269 | −31% |
| 2048 | 134 | 235 | **−43%** | 639 | 538 | +19% |
| 4096 | 336 | 471 | **−29%** | 1277 | 1076 | +19% |
| 8192 | 671 | 942 | **−29%** | 2554 | 2152 | +19% |
| 16384 | 1342 | 1883 | **−29%** | 5109 | 4303 | +19% |

If inference / KV-cache memory is your constraint, FastFlashAttention is a clear win; if training-step peak memory is the binding constraint at long context, that extra `dS` storage is the price of the deterministic, faster backward.

**Advanced, opt-in:** if even the flat ~19% overhead above is too much and you can tolerate a non-deterministic `dQ`, an alternate single-kernel backward fuses dK/dV and dQ so the `dS` tile never leaves the chip (no `dS` buffer at any `N`; `dK`/`dV` stay deterministic). Enable it globally with `UNIFLASH_BWD_FUSED_ATOMIC=1`, or call `fastflash_attention._kernel.fastflash_attn_train_fused_atomic` directly. It is not wired into `fast_attention`/`is_eligible` by default because it trades away backward determinism.

## Determinism

The backward is bitwise-identical across runs — disjoint writes, no global atomics — verified by `tests/test_determinism.py`. This is the property FA2 only provides via its slower `deterministic=True` path; FastFlashAttention is deterministic by construction, at a fraction of that path's cost.

## Tests

```bash
pytest tests/     # forward+backward parity vs fp32 SDPA truth, backward determinism, eligibility contract
```

Parity reference is `F.scaled_dot_product_attention` upcast to fp32, so the suite has no `flash_attn` dependency (that is benchmark-only).

## Changelog

**0.2.0**
- Fixed a training-memory regression in the deterministic backward's internal `dS` buffer: its size cap is now shape- and head-dim-aware instead of a flat constant, which flattened a mid-sequence-length memory spike (previously up to **+168%** vs FA2-default at N=4096) down to a **flat ~19%** overhead at every N ≥ 2048.
- Added an opt-in, non-deterministic single-kernel fused backward (`UNIFLASH_BWD_FUSED_ATOMIC=1`) for workloads that want the lowest possible training memory and can tolerate a non-deterministic `dQ`.
- Fixed a `gc.collect()` gap in the memory benchmark harness that could overstate peak memory on a kernel's first (autotuning) invocation; all benchmark numbers above were re-measured with the fix.

**0.1.0** — Initial public release.

## License

MIT — see [LICENSE](LICENSE).
