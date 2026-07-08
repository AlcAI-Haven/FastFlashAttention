"""Unified exact bf16 flash-attention forward + deterministic backward.

Merges the two wins that split the field between our lean ``bf16_flash`` (fastest
at short N) and the tuned ``triton_fa2`` (fastest at long N) into a SINGLE
``@triton.jit`` kernel, so there is no runtime kernel choice:

  * lean base (few registers, small-tile-friendly autotune) -> wins short N,
  * ``exp2`` hardware path with log2(e) folded into the score scale,
  * masked-region loop split: the causal *bulk* below the diagonal runs with NO
    per-tile ``tl.where`` (plain loads); only the diagonal/boundary blocks mask.

The split is *free* at short N: a query block with no full sub-diagonal region
simply runs zero bulk iterations and falls straight into the masked loop -- so
the short-N leanness is preserved while long N gets the unmasked-bulk speedup.

Exact softmax attention at the bf16 floor (~0.2% rel-L2 vs fp32). Native
[B,H,S,D] layout, output-only scratch (no transposes, no global workspace).

    fastflash_attn(q, k, v, causal) : q,k,v [B,H,S,D] bf16 -> out [B,H,S,D] bf16

``fastflash_attn(q, k, v, causal, return_lse=True)`` additionally returns
``(o, lse)``, where ``lse`` is the per-row base-2 logsumexp.
``fastflash_attn_train(q, k, v, causal)`` is the differentiable entry point
(autograd Function ``FastFlashAttn``), providing the exact bf16 backward
(dQ/dK/dV) via a non-atomic, deterministic two-kernel split.

An optional ``scale`` argument overrides the default ``1/sqrt(head_dim)`` softmax
scale (matching ``F.scaled_dot_product_attention``); it is threaded through the
forward and both backward kernels.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

LOG2E = 1.4426950408889634  # 1/ln(2); folds exp(x) -> exp2(x*log2e)

# dS-storage backward path (7->5 matmul cut): dkdv writes the unscaled ds tiles
# to an HBM buffer [BH,S,S] bf16 and dq reads them (dq drops 3 matmuls -> 1).
# MEASURED on sm_120 (RTX 5090): wins for S*D >= 262144 (+1-8% at D=64, +15-27%
# at D=128) and loses only at 2048/64 (short-N, launch-bound) which the S*D gate
# already routes to the short-N path. Cap the DS buffer (O(S^2)) so huge S can't
# OOM; beyond the cap fall back to the long-N dual-stream path. Env-tunable for
# A/B measurement (set UNIFLASH_DS_MIN_SD huge to disable -> baseline paths).
_DS_MIN_SD = int(os.environ.get("UNIFLASH_DS_MIN_SD", "262144"))
_DS_MAX_BYTES = int(os.environ.get("UNIFLASH_DS_MAX_BYTES", str(4 * 1024 ** 3)))
# Head-group tiling budget for the dS buffer. Heads (the BH leading dim) are
# fully independent in the backward, so the dS path processes them in groups of
# G = budget // (S*S*2) heads, reusing ONE bounded [G,S,S] buffer. This keeps
# the O(S^2) buffer bounded (not O(BH*S^2)) so the 7->5 dS win applies for ANY
# batch size instead of falling back to the slower dual-stream at large BH.
# Env-tunable for A/B (huge value -> single group == ungrouped; small -> many).
# This is a hard OUTER ceiling (never exceeded regardless of shape); the actual
# per-call budget is further capped below by _DS_BUDGET_FRACTION so it also
# scales DOWN at moderate S instead of always sitting at this flat value.
_DS_BUF_BYTES = int(os.environ.get("UNIFLASH_DS_BUF_BYTES", str(2 * 1024 ** 3)))
# Shape-aware scaling for the same budget: measured peak TRAINING memory
# (bench/mem.py) showed the flat _DS_BUF_BYTES ceiling above made fastflash
# WORSE than FA2-deterministic at moderate S (worst +114.8% at S=4096,
# B=4,H=16,D=128,causal=True) purely because a ~2GiB buffer is huge relative to
# the still-small O(S) baseline (Q/K/V/O/dQ/dK/dV/dO) tensors at that S -- even
# though the SAME flat buffer is proportionally tiny (net -5% vs FA2-det) once
# baseline memory has grown large at S=16384. Fix: cap the per-call budget at
# K * baseline_bytes, where baseline_bytes = B*H*S*D*2 is the byte-size of ONE
# already-resident bf16 [B,H,S,D] tensor -- a quantity that scales with S like
# the rest of the training footprint does, instead of a constant. At large S,
# K*baseline_bytes grows past _DS_BUF_BYTES and this cap stops binding, so
# large-S behavior (already near parity) is essentially unchanged; at moderate
# S it shrinks the DS buffer (more, smaller head-groups) and flattens the hump.
# K=8 was picked empirically: K in {1,2,4} (tried first, per the original
# hypothesis) shrinks the head-group G enough at the bench_bwd guardrail's
# SMALL-BH=16 shapes (S=2048: G collapses to 1 at K<=4) that fastflash's
# backward becomes SLOWER than FA2-deterministic (ratio_det < 1.0) -- a hard
# gate violation -- because G, not just the DS-buffer byte size, scales
# directly with BH (G ~= K*BH*D/S), so a fixed K that is safe at large BH
# (bench/mem.py uses BH=64) can starve small-BH callers of head-group
# parallelism. K=8 keeps ratio_det >= ~1.5x across the guardrail's D=128,
# causal=True block (S in {2048,4096,8192} at BH=16) while cutting the
# bench/mem.py BH=64 S=4096 DS buffer 4x (2048MiB -> 512MiB), enough to flatten
# the whole hump (measured: worst pct_over_det +114.8% -> -5.0%).
# FIXED GAP (was a real regression, not a pre-existing one): at the thinner
# D=64,causal=True margin (baseline ratio_det there is already only ~1.25-
# 1.4x, vs ~2.0-2.2x at D=128), K=8's shrunk G alone pushed ratio_det to
# ~0.68-0.73 (fastflash SLOWER than FA2-det, at BH=16,S in {4096,8192}) -- a hard-
# gate violation, since D=64,S=4096 (S*D==262144, right at the dS threshold)
# was ratio_det=1.25 on unmodified trunk. Bumping K globally (tried K=16) does
# NOT cleanly fix this (best case ~0.99, a wash) while giving back most of the
# D=128 memory win (worst pct_over_det regresses -5.0% -> +35.0%), because the
# achievable G at fixed K scales with BH*D/S, not with D alone -- a single
# global K cannot serve both D=64 and D=128. Fixed instead by a D-AWARE FLOOR
# on G directly (see _DS_MIN_GROUP_DXG below), which leaves D=128 untouched
# (its byte-budget G already exceeds the floor everywhere tested) while
# protecting D=64 (small head_dim overhead-hides worse per head-group, so it
# needs a bigger G than the byte math alone would give it).
_DS_BUDGET_FRACTION = float(os.environ.get("UNIFLASH_DS_BUDGET_FRACTION", "8.0"))
# D-aware minimum group size: dS-storage overhead-hides less well at small
# head_dim (fewer FLOPs per head-group per launch for the SAME G, since the
# matmuls are D-wide), so a byte-budget-only G (which scales D symmetrically
# via baseline_bytes ~ D) can starve small-D shapes even though the pure
# memory argument says a smaller G "fits". Enforced as G*D >= _DS_MIN_GROUP_DXG,
# i.e. G_min(D) = max(1, _DS_MIN_GROUP_DXG // D) -- smaller D gets a
# proportionally bigger group floor. MEASURED (bench_bwd guardrail, BH=16,
# causal=True, isolating pure G via UNIFLASH_DS_BUF_BYTES with K neutralized):
# D=64,S=4096 needs G>=8 for ratio_det >= ~1.0 (G=4 is a 1.005 wash; G=2 is
# already 0.677); S=8192 is more forgiving (G=2 already gives 1.089). The
# anchor DXG=1024 = 16*64 gives G_min(64)=16 (== BH=16 for the guardrail --
# i.e. fully restores the pre-change ungrouped behavior at D=64, the safest
# choice since it costs nothing at D=128: G_min(128)=8 never exceeds the
# byte-budget G at any bench/mem.py BH=64 shape, verified across the whole
# sweep, so the D=128 memory win is unchanged). Also hard-capped by the OUTER
# _DS_BUF_BYTES ceiling (not the K-scaled per-call budget) below, so the floor
# can override the budget to protect small-D speed without ever exceeding the
# real memory ceiling.
_DS_MIN_GROUP_DXG = int(os.environ.get("UNIFLASH_DS_MIN_GROUP_DXG", "1024"))

# dS-storage buffer ELEMENT DTYPE (node 3.1): fp8 (1 byte/elem) instead of bf16
# (2 bytes/elem) HALVES the DS buffer's footprint at every S, and since G
# (heads per group, above) is computed as budget_bytes // (S*S*_DS_ELEM_BYTES),
# halving the per-element size doubles the achievable G at the SAME byte
# budget. That memory-side hypothesis measured out as expected (see
# bench/mem.py results in the node 3.1 report) -- but fp8 is NOT the default
# because it FAILS the backward parity test's 2e-2 rel-L2 gate at every
# shape where dS-storage activates (S*D >= _DS_MIN_SD): measured dQ rel-L2
# 3.2e-2 to 2.8e-1 (up to ~14x over gate), growing with S and worse for
# causal=False than causal=True. dK/dV stay bit-exact regardless of DS's
# dtype (they accumulate from the full-precision in-register ``ds``, never
# the quantized HBM round-trip) -- only dQ (``dq = ds @ k``, READ_DS path in
# ``_uni_bwd_dq``) is affected, and severely so, for a structural reason, not
# just "fp8 is coarser": for fixed row m, ``sum_n ds[m,n] == 0`` EXACTLY
# (``sum_n p[m,n]*dp[m,n] = delta[m]`` by definition of delta, and
# ``sum_n p[m,n] = 1``, so ``sum_n p[m,n]*(dp[m,n]-delta[m]) = delta[m] -
# delta[m] = 0``) -- i.e. dQ is a cancellation-heavy contrast, not a plain
# weighted sum. Independent per-element quantization noise does NOT respect
# that zero-sum identity, so it survives (accumulating with S) while the true
# signal has mostly cancelled -- this is also why dK/dV (no analogous
# zero-sum identity along their OWN reduction axis, over m instead of n) were
# never at risk. bf16's ~2^-7 relative step keeps this survived-noise-to-
# cancelled-signal ratio under the gate (measured ~2.3-2.5e-3, ~8x margin);
# fp8 e4m3's ~2^-3 step (~16x coarser) blows well past it, disproportionately
# so at large S where the true signal cancels harder. See the node 3.1 report
# for the full derivation and measurements.
# ``torch.float8_e4m3fn`` is CUDA's native e4m3 layout on sm_120, and it is
# what Triton's ``DS.dtype.element_ty`` resolves to for such a tensor -- so
# the kernel itself (``_uni_bwd_dkdv``/``_uni_bwd_dq``) is written generically
# against whatever dtype this buffer is allocated with; no hardcoded
# ``tl.float8e4nv`` appears anywhere below.
# UPCAST-then-matmul, not a native fp8 GEMM: ``_uni_bwd_dq`` casts the loaded
# ds tile to k's dtype (bf16) immediately after the HBM read, before ``ds @
# k`` -- Triton 3.7.1 has no vetted native fp8xbf16 ``tl.dot`` path on this
# stack, and the quantization already happened at the STORE (in dkdv), so
# upcast-then-matmul is exactly as accurate as a native fp8 GEMM would be
# (neither recovers precision lost at the store).
# Default is therefore "bf16" (byte-identical to pre-node-3.1 behavior).
# Set UNIFLASH_DS_DTYPE=fp8 to OPT IN to the smaller/potentially-faster but
# parity-gate-FAILING buffer -- e.g. for further research into a per-tile-
# scaled fp8 storage (a real fp8 training technique that could plausibly
# rescue this -- not implemented here, see Recommendation in the node 3.1
# report) or for a workload that provably tolerates a lossier dQ.
_DS_DTYPE_NAME = os.environ.get("UNIFLASH_DS_DTYPE", "bf16").lower()
if _DS_DTYPE_NAME == "fp8":
    _DS_DTYPE = torch.float8_e4m3fn
    _DS_ELEM_BYTES = 1
elif _DS_DTYPE_NAME == "bf16":
    _DS_DTYPE = torch.bfloat16
    _DS_ELEM_BYTES = 2
else:
    raise ValueError(
        f"UNIFLASH_DS_DTYPE={_DS_DTYPE_NAME!r} unrecognized; expected 'fp8' or 'bf16'"
    )


def _default_scale(head_dim: int) -> float:
    return 1.0 / (head_dim ** 0.5)


@triton.autotune(
    configs=[
        # BLOCK_N <= BLOCK_M so start_m*BLOCK_M is always a multiple of BLOCK_N
        # -> the unmasked/masked split point lands on a block boundary.
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
    ],
    key=["S", "HEAD_DIM"],
)
@triton.jit
def _fastflash_fwd(
    Q, K, V, O, L,
    qk_scale,                       # softmax_scale * log2e (folded for exp2)
    stride_z, stride_m, stride_d,
    S, HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_base = Q + off_hz * stride_z
    k_base = K + off_hz * stride_z
    v_base = V + off_hz * stride_z
    o_base = O + off_hz * stride_z

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_valid = offs_m < S

    q_ptrs = q_base + offs_m[:, None] * stride_m + offs_d[None, :] * stride_d
    q = tl.load(q_ptrs, mask=m_valid[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # -- split the key range into an unmasked BULK and a masked BOUNDARY --------
    if IS_CAUSAL:
        # bulk: key blocks fully below the diagonal need no mask at all.
        # split aligned to BLOCK_N (BLOCK_N<=BLOCK_M guarantees exact alignment).
        n_bulk = start_m * BLOCK_M            # exclusive; multiple of BLOCK_N
        n_end = tl.minimum((start_m + 1) * BLOCK_M, S)
    else:
        n_bulk = (S // BLOCK_N) * BLOCK_N     # all full blocks are unmasked
        n_end = S

    # ---- unmasked bulk: plain loads, no tl.where in the hot loop -------------
    for start_n in range(0, n_bulk, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kt_ptrs = k_base + offs_d[:, None] * stride_d + offs_n[None, :] * stride_m
        kt = tl.load(kt_ptrs)
        qk = tl.dot(q, kt, out_dtype=tl.float32) * qk_scale
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v_ptrs = v_base + offs_n[:, None] * stride_m + offs_d[None, :] * stride_d
        v = tl.load(v_ptrs)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        m_i = m_new

    # ---- masked boundary: diagonal (causal) and/or ragged tail (bounds) ------
    for start_n in range(n_bulk, n_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < S
        kt_ptrs = k_base + offs_d[:, None] * stride_d + offs_n[None, :] * stride_m
        kt = tl.load(kt_ptrs, mask=n_valid[None, :], other=0.0)
        qk = tl.dot(q, kt, out_dtype=tl.float32) * qk_scale
        mask = n_valid[None, :]
        if IS_CAUSAL:
            mask = mask & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v_ptrs = v_base + offs_n[:, None] * stride_m + offs_d[None, :] * stride_d
        v = tl.load(v_ptrs, mask=n_valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        m_i = m_safe

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    o_ptrs = o_base + offs_m[:, None] * stride_m + offs_d[None, :] * stride_d
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=m_valid[:, None])

    if RETURN_LSE:
        # base-2 logsumexp: M_i = m_i + log2(l_i); P_ij later = exp2(qk_ij - M_i)
        m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        lse = m_i_safe + tl.log2(l_safe)
        l_ptrs = L + off_hz * S + offs_m
        tl.store(l_ptrs, lse, mask=m_valid)


def fastflash_attn(q, k, v, causal: bool, return_lse: bool = False, scale=None):
    """Unified exact bf16 flash-attention forward. [B,H,S,D] bf16 -> [B,H,S,D] bf16.

    ``scale`` overrides the default ``1/sqrt(D)`` softmax scale (matching
    ``F.scaled_dot_product_attention``). When ``return_lse`` is True, also
    returns the per-row base-2 logsumexp ``M_i = m_i + log2(l_i)`` as an fp32
    tensor of shape [B,H,S] (used by the backward pass to recompute softmax
    probabilities via ``exp2(qk - M)``).
    """
    B, H, S, D = q.shape
    assert D <= 128 and (D & (D - 1)) == 0, "HEAD_DIM must be a power of two <= 128"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    q, k, v = (t.contiguous() for t in (q, k, v))
    qf = q.reshape(B * H, S, D)
    kf = k.reshape(B * H, S, D)
    vf = v.reshape(B * H, S, D)

    o = torch.empty_like(qf)
    if return_lse:
        lse = torch.empty(B * H, S, device=q.device, dtype=torch.float32)
    else:
        lse = torch.empty(1, device=q.device, dtype=torch.float32)  # dummy ptr
    sm = _default_scale(D) if scale is None else float(scale)
    qk_scale = sm * LOG2E
    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B * H)
    _fastflash_fwd[grid](
        qf, kf, vf, o, lse,
        qk_scale,
        qf.stride(0), qf.stride(1), qf.stride(2),
        S, HEAD_DIM=D,
        IS_CAUSAL=causal,
        RETURN_LSE=return_lse,
    )
    o = o.reshape(B, H, S, D)
    if return_lse:
        return o, lse.reshape(B, H, S)
    return o


# ======================= BACKWARD PASS ===================================== #

@triton.jit
def _uni_bwd_preprocess(
    O, DO, Delta,
    stride_z, stride_m, stride_d,
    S, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_valid = offs_m < S
    base = off_hz * stride_z + offs_m[:, None] * stride_m + offs_d[None, :] * stride_d
    o = tl.load(O + base, mask=m_valid[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + base, mask=m_valid[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * S + offs_m, delta, mask=m_valid)


def _bwd_preprocess(o, do, BLOCK_M: int = 128):
    """delta[i] = sum_d o[i,d]*do[i,d].  o,do: [B*H,S,D] bf16 -> [B*H,S] fp32."""
    BH, S, D = o.shape
    delta = torch.empty(BH, S, device=o.device, dtype=torch.float32)
    grid = (triton.cdiv(S, BLOCK_M), BH)
    _uni_bwd_preprocess[grid](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2),
        S, HEAD_DIM=D, BLOCK_M=BLOCK_M,
    )
    return delta


# Backward autotune space. dK/dV grids over N (BLOCK_N tiles) and loops M;
# dQ grids over M (BLOCK_M tiles) and loops N. Both hold fp32 D-wide
# accumulators, so the D=128 configs need enough warps to avoid register
# spilling — the single fixed 64x64/4-warp config was the pre-tuning bottleneck.
_BWD_CONFIGS = [
    # --- short-N / low-occupancy fillers: small & asymmetric tiles give more
    #     CTAs (grid ~ S/BLOCK * BH) to fill the SMs when S is small and BH=16,
    #     and low warp counts cut per-CTA launch/scheduling overhead. The bwd
    #     kernels mask the causal diagonal, so BLOCK_N need not be <= BLOCK_M.
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    # --- original space (long-N winners kept intact) -----------------------
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_BWD_CONFIGS, key=["S", "HEAD_DIM", "IS_CAUSAL"])
@triton.jit
def _uni_bwd_dkdv(
    Q, K, V, DO, DK, DV, L, Delta, DS,
    qk_scale, softmax_scale,
    stride_z, stride_m, stride_d,
    stride_ds_z, stride_ds_m, stride_ds_n,
    S, HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    STORE_DS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    zoff = off_hz * stride_z
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    n_valid = offs_n < S

    kv_ptrs = zoff + offs_n[:, None] * stride_m + offs_d[None, :] * stride_d
    k = tl.load(K + kv_ptrs, mask=n_valid[:, None], other=0.0)   # [BLOCK_N, D]
    v = tl.load(V + kv_ptrs, mask=n_valid[:, None], other=0.0)   # [BLOCK_N, D]
    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)

    # causal: only query blocks with some i >= min key in this block contribute.
    if IS_CAUSAL:
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    for start_m in range(lo, S, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_valid = offs_m < S
        q_ptrs = zoff + offs_m[:, None] * stride_m + offs_d[None, :] * stride_d
        q = tl.load(Q + q_ptrs, mask=m_valid[:, None], other=0.0)    # [BLOCK_M, D]
        do = tl.load(DO + q_ptrs, mask=m_valid[:, None], other=0.0)  # [BLOCK_M, D]
        lse = tl.load(L + off_hz * S + offs_m, mask=m_valid, other=0.0)      # [BLOCK_M]
        delta = tl.load(Delta + off_hz * S + offs_m, mask=m_valid, other=0.0)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale  # [BLOCK_M, BLOCK_N]
        p = tl.exp2(qk - lse[:, None])
        mask = n_valid[None, :] & m_valid[:, None]
        if IS_CAUSAL:
            mask = mask & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(mask, p, 0.0)                                    # [BLOCK_M, BLOCK_N]

        dv += tl.dot(tl.trans(p).to(do.dtype), do, out_dtype=tl.float32)  # [BLOCK_N, D]
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)                # [BLOCK_M, BLOCK_N]
        ds = p * (dp - delta[:, None])                                    # [BLOCK_M, BLOCK_N]
        if STORE_DS:
            # publish the UNSCALED ds tile (exactly as used for dk) so the dq
            # kernel can read it instead of recomputing qk/p/dp. Each (m,n) is
            # written by exactly this one n-block -> no atomics, deterministic.
            # Quantized to DS's own storage dtype (fp8 e4m3 by default, see
            # _DS_DTYPE) via DS.dtype.element_ty -- same generic-cast idiom
            # already used below for DK/DV's output dtype. dK/dV (below) are
            # computed from the full-precision in-register `ds`, NOT this
            # quantized copy, so they are unaffected by the DS buffer's dtype.
            ds_ptrs = (off_hz * stride_ds_z
                       + offs_m[:, None] * stride_ds_m
                       + offs_n[None, :] * stride_ds_n)
            tl.store(DS + ds_ptrs, ds.to(DS.dtype.element_ty),
                     mask=m_valid[:, None] & n_valid[None, :])
        dk += tl.dot(tl.trans(ds).to(q.dtype), q, out_dtype=tl.float32)   # [BLOCK_N, D]

    dk = dk * softmax_scale
    out_ptrs = zoff + offs_n[:, None] * stride_m + offs_d[None, :] * stride_d
    tl.store(DK + out_ptrs, dk.to(DK.dtype.element_ty), mask=n_valid[:, None])
    tl.store(DV + out_ptrs, dv.to(DV.dtype.element_ty), mask=n_valid[:, None])


def _bwd_dkdv(q, k, v, do, lse, delta, causal, scale=None, ds=None, dk=None, dv=None):
    """dK, dV for the two-kernel split. All [B*H,S,D] bf16 / [B*H,S] fp32.

    If ``ds`` (a preallocated [B*H,S,S] buffer -- fp8 e4m3 by default, see
    ``_DS_DTYPE``; any dtype the kernel can ``.to()``-cast into works) is
    given, dkdv ALSO stores each unscaled ds tile into it (STORE_DS path) so a
    serially-following dQ can read ds instead of recomputing qk/p/dp (7->5
    matmul cut). The unused pointer on the dead branch is bound to ``q``
    (valid, never dereferenced).

    ``dk``/``dv`` may be preallocated output views (e.g. a head-group slice of a
    full [BH,S,D] result buffer). They must share q's [.,S,D] layout; the kernel
    uses q's strides for all these tensors, so a dim-0 slice (identical strides,
    offset base) is written correctly. Omitted -> allocated here (empty_like).
    """
    BH, S, D = q.shape
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    softmax_scale = _default_scale(D) if scale is None else float(scale)
    qk_scale = softmax_scale * LOG2E
    store_ds = ds is not None
    ds_arg = ds if store_ds else q
    sdz, sdm, sdn = (ds.stride(0), ds.stride(1), ds.stride(2)) if store_ds else (0, 0, 0)
    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_N"]), BH)
    _uni_bwd_dkdv[grid](
        q, k, v, do, dk, dv, lse, delta, ds_arg,
        qk_scale, softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        sdz, sdm, sdn,
        S, HEAD_DIM=D, IS_CAUSAL=causal, STORE_DS=store_ds,
    )
    return dk, dv


@triton.autotune(configs=_BWD_CONFIGS, key=["S", "HEAD_DIM", "IS_CAUSAL"])
@triton.jit
def _uni_bwd_dq(
    Q, K, V, DO, DQ, L, Delta, O, DS,
    qk_scale, softmax_scale,
    stride_z, stride_m, stride_d,
    stride_ds_z, stride_ds_m, stride_ds_n,
    S, HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WRITE_DELTA: tl.constexpr,
    READ_DS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    zoff = off_hz * stride_z
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_valid = offs_m < S

    q_ptrs = zoff + offs_m[:, None] * stride_m + offs_d[None, :] * stride_d
    if not READ_DS:
        # recompute path: needs q, do, lse (+ delta) to rebuild ds per n-tile.
        q = tl.load(Q + q_ptrs, mask=m_valid[:, None], other=0.0)     # [BLOCK_M, D]
        do = tl.load(DO + q_ptrs, mask=m_valid[:, None], other=0.0)   # [BLOCK_M, D]
        lse = tl.load(L + off_hz * S + offs_m, mask=m_valid, other=0.0)
        # dQ grids over M with no row overlap, so it is the natural place to
        # produce delta[m]=sum_d(o*do): fold the standalone preprocess kernel in
        # here and publish delta for the (serial) dK/dV kernel. Zero redundancy.
        if WRITE_DELTA:
            o_ = tl.load(O + q_ptrs, mask=m_valid[:, None], other=0.0)
            delta = tl.sum(o_.to(tl.float32) * do.to(tl.float32), axis=1)
            tl.store(Delta + off_hz * S + offs_m, delta, mask=m_valid)
        else:
            delta = tl.load(Delta + off_hz * S + offs_m, mask=m_valid, other=0.0)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, S)
    else:
        hi = S

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < S
        kv_ptrs = zoff + offs_n[:, None] * stride_m + offs_d[None, :] * stride_d
        k = tl.load(K + kv_ptrs, mask=n_valid[:, None], other=0.0)   # [BLOCK_N, D]

        if READ_DS:
            # dS-storage path: read the ds tile written by dkdv; 1 matmul only.
            # causal/bounds mask zeros any (m,n) with m<n (dkdv left it unwritten
            # or wrote 0) so garbage from the empty buffer never enters dq.
            ds_ptrs = (off_hz * stride_ds_z
                       + offs_m[:, None] * stride_ds_m
                       + offs_n[None, :] * stride_ds_n)
            mask = n_valid[None, :] & m_valid[:, None]
            if IS_CAUSAL:
                mask = mask & (offs_m[:, None] >= offs_n[None, :])
            ds = tl.load(DS + ds_ptrs, mask=mask, other=0.0)  # [BLOCK_M, BLOCK_N], DS.dtype (fp8 e4m3 by default)
            ds = ds.to(k.dtype)  # upcast fp8 -> bf16 before the matmul; no native
                                  # fp8xbf16 tl.dot path assumed on this stack, and
                                  # the quantization already happened at the STORE.
            dq += tl.dot(ds, k, out_dtype=tl.float32)                # [BLOCK_M, D]
        else:
            v = tl.load(V + kv_ptrs, mask=n_valid[:, None], other=0.0)   # [BLOCK_N, D]
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale  # [BLOCK_M, BLOCK_N]
            p = tl.exp2(qk - lse[:, None])
            mask = n_valid[None, :] & m_valid[:, None]
            if IS_CAUSAL:
                mask = mask & (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)

            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)           # [BLOCK_M, BLOCK_N]
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k, out_dtype=tl.float32)        # [BLOCK_M, D]

    dq = dq * softmax_scale
    tl.store(DQ + q_ptrs, dq.to(DQ.dtype.element_ty), mask=m_valid[:, None])


def _bwd_dq(q, k, v, do, lse, delta, causal, o=None, scale=None, ds=None, dq=None):
    """dQ for the two-kernel split. All [B*H,S,D] bf16 / [B*H,S] fp32.

    If ``o`` is given, dQ also computes delta[m]=sum_d(o*do) and writes it into
    ``delta`` (which must be a preallocated [B*H,S] fp32 buffer), folding the
    standalone preprocess launch away. The unused pointer on the dead branch is
    bound to ``q`` (valid, never dereferenced).

    If ``ds`` (a [B*H,S,S] buffer -- fp8 e4m3 by default -- written by dkdv's
    STORE_DS) is given, dQ READS ds (upcasting it to k's dtype right after the
    load) instead of recomputing qk/p/dp: dq = ds @ k (1 matmul, not 3). This
    is mutually exclusive with ``o`` (READ_DS needs no delta).

    ``dq`` may be a preallocated output view (e.g. a head-group slice of a full
    [BH,S,D] result buffer); it must share q's [.,S,D] layout. Omitted ->
    allocated here (empty_like).
    """
    BH, S, D = q.shape
    if dq is None:
        dq = torch.empty_like(q)
    softmax_scale = _default_scale(D) if scale is None else float(scale)
    qk_scale = softmax_scale * LOG2E
    write_delta = o is not None
    o_arg = o if o is not None else q
    read_ds = ds is not None
    ds_arg = ds if read_ds else q
    sdz, sdm, sdn = (ds.stride(0), ds.stride(1), ds.stride(2)) if read_ds else (0, 0, 0)
    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), BH)
    _uni_bwd_dq[grid](
        q, k, v, do, dq, lse, delta, o_arg, ds_arg,
        qk_scale, softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        sdz, sdm, sdn,
        S, HEAD_DIM=D, IS_CAUSAL=causal, WRITE_DELTA=write_delta, READ_DS=read_ds,
    )
    return dq


# Fused-kernel-only autotune space: _BWD_CONFIGS plus a few LARGER-BLOCK_N
# configs. Rationale specific to the atomic dQ path: each query row m
# receives one atomic_add per kv-block whose range overlaps it (causally,
# ceil((m+1)/BLOCK_N) of them) -- so, unlike the non-atomic dkdv/dq kernels
# (which only care about occupancy/reuse), a LARGER BLOCK_N here directly
# reduces the *number of atomic read-modify-writes contending on the same
# dQ row*, not just data reuse. Kept as a SEPARATE list (not appended to
# _BWD_CONFIGS) so this is zero-risk to the default path's autotune cache/
# behavior. Measured impact: see FastFlashAttnFusedAtomic notes.
_FUSED_BWD_CONFIGS = _BWD_CONFIGS + [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, num_stages=2),
]


# ================= FUSED dK/dV+dQ BACKWARD (opt-in, non-default) =========== #
# Explores whether the dS-storage insight (compute ds ONCE per (kv,q) tile,
# reuse it instead of a 7-matmul full recompute) can be pushed one step
# further: fuse _uni_bwd_dkdv and _uni_bwd_dq into ONE kernel launch so ds
# never leaves the chip (no HBM DS[G,S,S] buffer at any S -- register/shared
# memory only). dK/dV stay on their existing disjoint, deterministic,
# no-atomics accumulation (grid over kv-blocks, unchanged). The one thing that
# genuinely needs a cross-program reduction is dQ: many different kv-block
# programs (grid cells) all contribute to the same q-rows, so dQ is written
# via tl.atomic_add into a global fp32 accumulator -- this is the ONE piece
# that trades away bit-exact determinism (dQ only; dK/dV are unaffected).
#
# History: an earlier ("naive") atomic single-pass dQ attempt in this
# codebase (predates dS-storage) was reverted for being slower than the
# deterministic path (0.48-0.64x of FA2-default), blamed on global-atomic-add
# throttling for O(S^2/BLOCK_N) dQ traffic on sm_120. That attempt recomputed
# qk/p/dp from scratch for dQ (a separate simple kernel), i.e. it did NOT share
# ds with dK/dV -- structurally 7 matmuls' worth of work, not 5. This kernel
# is different: ds is computed exactly once per (kv,q) tile and used for BOTH
# dK/dV and the dQ atomic contribution in the SAME breath, matching the 5-
# matmul budget FA2's own (also-atomic) backward uses. See
# FastFlashAttnFusedAtomic / fastflash_attn_train_fused_atomic for the opt-in
# entry points (env var UNIFLASH_BWD_FUSED_ATOMIC=1 reroutes the STANDARD
# fastflash_attn_train/fast_attention entry points here too).
@triton.autotune(configs=_FUSED_BWD_CONFIGS, key=["S", "HEAD_DIM", "IS_CAUSAL"], reset_to_zero=["DQ"])
@triton.jit
def _uni_bwd_fused_atomic(
    Q, K, V, DO, DK, DV, DQ, L, Delta,
    qk_scale, softmax_scale,
    stride_z, stride_m, stride_d,
    S, HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Fused dK/dV + dQ: ds is computed on-chip and immediately consumed by
    both outputs -- it is never materialized to an HBM DS buffer.

    Grid over kv-blocks (as ``_uni_bwd_dkdv``), looping q-blocks inside. Per
    (kv,q) tile: dK/dV accumulate exactly as ``_uni_bwd_dkdv`` (disjoint,
    deterministic, unchanged); additionally ``(ds @ k) * softmax_scale`` is
    atomic_add-ed into the global fp32 ``DQ`` accumulator. Scaling PER TILE
    (rather than once at the end, as the non-fused ``_uni_bwd_dq`` does) is
    required here because no single program ever sees the full n-range for a
    given q-row -- but it is equivalent, since atomic_add is a running sum and
    scalar multiplication distributes over addition.

    ``DQ`` MUST already be a zeroed fp32 tensor sharing Q's [.,S,D] strides
    (atomic_add on bf16 is unsupported/lossy on this hw+Triton stack, hence
    the fp32 workspace -- see ``_bwd_fused_atomic``). Causal/bounds masking on
    the atomic_add's OWN address (``m_valid``) is required even though ``ds``
    is already algebraically zero on fully-masked rows, because ``offs_m``
    can run past ``S`` for the ragged tail and must not address out of bounds.
    """
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    zoff = off_hz * stride_z
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    n_valid = offs_n < S

    kv_ptrs = zoff + offs_n[:, None] * stride_m + offs_d[None, :] * stride_d
    k = tl.load(K + kv_ptrs, mask=n_valid[:, None], other=0.0)   # [BLOCK_N, D]
    v = tl.load(V + kv_ptrs, mask=n_valid[:, None], other=0.0)   # [BLOCK_N, D]
    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)

    # causal: only query blocks with some i >= min key in this block contribute.
    if IS_CAUSAL:
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    for start_m in range(lo, S, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_valid = offs_m < S
        q_ptrs = zoff + offs_m[:, None] * stride_m + offs_d[None, :] * stride_d
        q = tl.load(Q + q_ptrs, mask=m_valid[:, None], other=0.0)    # [BLOCK_M, D]
        do = tl.load(DO + q_ptrs, mask=m_valid[:, None], other=0.0)  # [BLOCK_M, D]
        lse = tl.load(L + off_hz * S + offs_m, mask=m_valid, other=0.0)      # [BLOCK_M]
        delta = tl.load(Delta + off_hz * S + offs_m, mask=m_valid, other=0.0)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale  # [BLOCK_M, BLOCK_N]
        p = tl.exp2(qk - lse[:, None])
        mask = n_valid[None, :] & m_valid[:, None]
        if IS_CAUSAL:
            mask = mask & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(mask, p, 0.0)                                    # [BLOCK_M, BLOCK_N]

        dv += tl.dot(tl.trans(p).to(do.dtype), do, out_dtype=tl.float32)  # [BLOCK_N, D]
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)                # [BLOCK_M, BLOCK_N]
        ds = p * (dp - delta[:, None])                                    # [BLOCK_M, BLOCK_N]
        dk += tl.dot(tl.trans(ds).to(q.dtype), q, out_dtype=tl.float32)   # [BLOCK_N, D]

        # dQ: reuse the SAME on-chip ds tile (no recompute, no HBM round-trip).
        # Cross-program reduction -> atomic_add -> dQ is non-deterministic.
        dq_tile = tl.dot(ds.to(k.dtype), k, out_dtype=tl.float32) * softmax_scale  # [BLOCK_M, D]
        tl.atomic_add(DQ + q_ptrs, dq_tile, mask=m_valid[:, None])

    dk = dk * softmax_scale
    out_ptrs = zoff + offs_n[:, None] * stride_m + offs_d[None, :] * stride_d
    tl.store(DK + out_ptrs, dk.to(DK.dtype.element_ty), mask=n_valid[:, None])
    tl.store(DV + out_ptrs, dv.to(DV.dtype.element_ty), mask=n_valid[:, None])


def _bwd_fused_atomic(q, k, v, do, lse, delta, causal, scale=None, dk=None, dv=None):
    """dQ/dK/dV via the fused single-kernel path. All [B*H,S,D] bf16 / [B*H,S] fp32.

    Returns ``(dq_fp32, dk, dv)`` -- ``dq`` is left in its fp32 accumulator
    dtype (the caller casts to bf16); dk/dv are already in their native dtype.

    ``dq``'s fp32 workspace is allocated HERE via ``torch.zeros`` (not
    ``empty``) on EVERY call: ``@triton.autotune``'s ``reset_to_zero`` only
    re-zeros the accumulator during ITS OWN tuning trials (so different
    candidate configs don't compound atomic-adds into each other during the
    search) -- it does NOT re-zero on ordinary cache-hit calls once a shape
    has already been tuned (verified against triton 3.7.1's Autotoner.run:
    the reset pre_hook only fires inside the `key not in self.cache` branch).
    So correctness of every call, not just the first, depends on the Python
    wrapper zeroing this buffer itself.

    ``dk``/``dv`` may be preallocated output views (same contract as
    ``_bwd_dkdv``): they must share q's [.,S,D] layout.
    """
    BH, S, D = q.shape
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    dq_acc = torch.zeros(BH, S, D, device=q.device, dtype=torch.float32)
    softmax_scale = _default_scale(D) if scale is None else float(scale)
    qk_scale = softmax_scale * LOG2E
    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_N"]), BH)
    _uni_bwd_fused_atomic[grid](
        q, k, v, do, dk, dv, dq_acc, lse, delta,
        qk_scale, softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        S, HEAD_DIM=D, IS_CAUSAL=causal,
    )
    return dq_acc, dk, dv


_SIDE_STREAM = {}  # device index -> cached side CUDA stream for bwd overlap


def _side_stream(device):
    s = _SIDE_STREAM.get(device.index)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _SIDE_STREAM[device.index] = s
    return s


class FastFlashAttn(torch.autograd.Function):
    """Differentiable fastflash: exact bf16 forward + two-kernel-split backward."""

    @staticmethod
    def forward(ctx, q, k, v, causal, scale=None):
        o, lse = fastflash_attn(q, k, v, causal, return_lse=True, scale=scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.scale = scale
        return o

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        causal = ctx.causal
        scale = ctx.scale
        B, H, S, D = q.shape
        do = do.contiguous()
        qf = q.reshape(B * H, S, D)
        kf = k.reshape(B * H, S, D)
        vf = v.reshape(B * H, S, D)
        of = o.reshape(B * H, S, D)
        dof = do.reshape(B * H, S, D)
        lf = lse.reshape(B * H, S)

        # Opt-in escape hatch (OFF by default): reroute the STANDARD entry
        # point through the fused-atomic path (see FastFlashAttnFusedAtomic)
        # instead of any of the deterministic strategies below. This is a
        # pure ADDITION ahead of the existing branching -- when the env var
        # is unset (the default), every line below is reached exactly as
        # before, unmodified. Trades away dQ's bit-exact determinism (dK/dV
        # are unaffected) for an O(BH*S*D) dQ workspace instead of the
        # dS-storage path's O(S^2)-bounded-but-nonzero DS buffer.
        if os.environ.get("UNIFLASH_BWD_FUSED_ATOMIC", "0") == "1":
            delta = _bwd_preprocess(of, dof)
            dq_acc, dk, dv = _bwd_fused_atomic(qf, kf, vf, dof, lf, delta, causal, scale=scale)
            return (dq_acc.to(qf.dtype).reshape(B, H, S, D),
                    dk.reshape(B, H, S, D),
                    dv.reshape(B, H, S, D),
                    None, None)

        # Two size-gated deterministic strategies (disjoint writes, no atomics):
        #   * LONG N (S*D >= 262144): precompute delta once, then OVERLAP dK/dV
        #     and dQ on two streams. Overlap packs the large kernels onto the SMs
        #     (+3-8%); a standalone preprocess is used because the two kernels run
        #     concurrently and neither can safely publish delta for the other.
        #   * SHORT N (< 262144): launch/overhead-bound (profiled: ~68% of wall
        #     is CPU launch + autograd engine). The two kernels run serially, so
        #     fold the preprocess INTO dQ: dQ grids over M with no row overlap, so
        #     it computes delta[m]=sum(o*do) once (zero redundancy) and publishes
        #     it for the serially-following dK/dV. 3 launches -> 2.
        #   * dS-STORAGE (S*D >= _DS_MIN_SD): the 7->5 matmul cut, which SUPERSEDES
        #     the long-N dual-stream above where it applies (measured faster for all
        #     S*D>=262144 on sm_120). Precompute delta (standalone; dkdv needs it to
        #     form ds), then dkdv writes dk, dv AND stores each unscaled ds tile to a
        #     bf16 buffer; dq then READS ds and does dq = ds@k (1 matmul instead of
        #     3). dq is SERIAL after dkdv (real data dependency on DS) -> no dual-
        #     stream. Each ds tile written by exactly one n-block: disjoint, no
        #     atomics, deterministic.
        #     HEAD-GROUP TILING: heads (the BH leading dim) are fully independent,
        #     so process BH in GROUPS of G heads reusing ONE bounded [G,S,S]
        #     buffer. This bounds the O(S^2) buffer (was the old O(BH*S^2),
        #     which forced large-BH to the slower fallback) so the dS win
        #     applies for ANY batch size. A dim-0 slice of qf/kf/... and of the
        #     full dq/dk/dv result buffers keeps strides identical (offset base
        #     only), so per-head math is BYTE-IDENTICAL to the ungrouped path
        #     regardless of G.
        #     SHAPE-AWARE BUDGET: G = max(ds_budget_bytes // (S*S*_DS_ELEM_BYTES),
        #     D-aware floor), where ds_budget_bytes = min(_DS_BUF_BYTES,
        #     _DS_BUDGET_FRACTION * baseline_bytes) ties the cap to the
        #     byte-size of one already-resident [B,H,S,D] tensor instead of a
        #     flat constant (see _DS_BUDGET_FRACTION above) -- this is what
        #     flattens the mid-S training-memory hump without touching the
        #     large-S regime -- and the floor (see _DS_MIN_GROUP_DXG above)
        #     protects small-D shapes the byte budget alone would starve.
        #     _DS_ELEM_BYTES (see above) is 1 for the default fp8 DS buffer
        #     (was always 2, bf16, before node 3.1) -- fp8 halves the bytes
        #     per element, doubling the G the SAME byte budget buys.
        baseline_bytes = B * H * S * D * 2   # one resident bf16 [B,H,S,D] tensor (Q/K/V/O
                                              # stay bf16 regardless of the DS buffer's dtype)
        ds_budget_bytes = min(_DS_BUF_BYTES, int(_DS_BUDGET_FRACTION * baseline_bytes))
        use_ds = (S * D >= _DS_MIN_SD) and (S * S * _DS_ELEM_BYTES <= ds_budget_bytes) \
            and (S * S * _DS_ELEM_BYTES <= _DS_MAX_BYTES)
        if use_ds:
            BH = B * H
            delta = _bwd_preprocess(of, dof)           # [BH,S] fp32, all heads once
            dq = torch.empty_like(qf)
            dk = torch.empty_like(kf)
            dv = torch.empty_like(vf)
            # G = heads per group from the shape-aware byte budget, RAISED to
            # the D-aware minimum (see _DS_MIN_GROUP_DXG above -- protects
            # small-D speed the byte budget alone would starve), and clamped
            # so the dS pointer offset (max ~ G*S*S elements) stays within
            # int32 -- Triton indexes in int32 by default, and G*S*S can
            # exceed 2^31 if a user raises the budget past ~4GiB (else:
            # illegal memory access). This clamp is in ELEMENTS, not bytes, so
            # it is unaffected by _DS_ELEM_BYTES.
            computed_G = ds_budget_bytes // (S * S * _DS_ELEM_BYTES)
            # the floor is capped by the OUTER _DS_BUF_BYTES ceiling (not the
            # K-scaled ds_budget_bytes) so it can override the budget for
            # small D without ever exceeding the real hard memory ceiling.
            g_floor = min(max(1, _DS_MIN_GROUP_DXG // D),
                          _DS_BUF_BYTES // (S * S * _DS_ELEM_BYTES))
            G = max(1, min(BH, max(computed_G, g_floor), (2 ** 31 - 1) // (S * S)))
            DS = torch.empty(G, S, S, device=q.device, dtype=_DS_DTYPE)
            for g0 in range(0, BH, G):
                g1 = min(g0 + G, BH)
                dsv = DS[: g1 - g0]                     # last partial group: view
                _bwd_dkdv(qf[g0:g1], kf[g0:g1], vf[g0:g1], dof[g0:g1],
                          lf[g0:g1], delta[g0:g1], causal, scale=scale, ds=dsv,
                          dk=dk[g0:g1], dv=dv[g0:g1])   # stores ds tiles
                _bwd_dq(qf[g0:g1], kf[g0:g1], vf[g0:g1], dof[g0:g1],
                        lf[g0:g1], delta[g0:g1], causal, scale=scale, ds=dsv,
                        dq=dq[g0:g1])                   # reads ds; dq = ds@k
            del DS
        elif S * D >= 262144:
            delta = _bwd_preprocess(of, dof)
            cur = torch.cuda.current_stream()
            side = _side_stream(q.device)
            side.wait_stream(cur)             # delta must be visible to dq
            with torch.cuda.stream(side):
                dq = _bwd_dq(qf, kf, vf, dof, lf, delta, causal, scale=scale)
            dk, dv = _bwd_dkdv(qf, kf, vf, dof, lf, delta, causal, scale=scale)  # overlaps
            cur.wait_stream(side)             # dq done before we return
            dq.record_stream(cur)             # dq allocated on side, freed on cur
        else:
            delta = torch.empty(B * H, S, device=q.device, dtype=torch.float32)
            dq = _bwd_dq(qf, kf, vf, dof, lf, delta, causal, o=of, scale=scale)  # writes delta
            dk, dv = _bwd_dkdv(qf, kf, vf, dof, lf, delta, causal, scale=scale)  # reads delta
        return (dq.reshape(B, H, S, D),
                dk.reshape(B, H, S, D),
                dv.reshape(B, H, S, D),
                None, None)


def fastflash_attn_train(q, k, v, causal: bool, scale=None):
    """Differentiable entry point. [B,H,S,D] bf16 -> [B,H,S,D] bf16, grads on q,k,v."""
    return FastFlashAttn.apply(q, k, v, causal, scale)


class FastFlashAttnFusedAtomic(torch.autograd.Function):
    """Opt-in, NON-default variant: fuses dK/dV and dQ into ONE kernel launch.

    Mechanism: ``_uni_bwd_fused_atomic`` grids over kv-blocks (as
    ``_uni_bwd_dkdv``) and, for each (kv,q) tile, computes ``ds`` ON-CHIP ONCE
    and uses it for BOTH the existing disjoint dK/dV accumulation (unchanged,
    still deterministic) AND an atomic_add of ``ds @ k`` into a global dQ
    accumulator -- so the dS-storage optimization's "1 matmul instead of 3 for
    dQ" insight is kept, but ``ds`` never round-trips through HBM (no
    O(S^2)/O(G*S^2) DS buffer at any S; peak dQ/dK/dV memory is O(BH*S*D),
    the same order FA2's own workspace uses).

    Trade-off: dQ is accumulated via cross-program ``atomic_add``, so it is
    NOT bit-exact deterministic run-to-run (dK/dV remain deterministic --
    their accumulation is still fully disjoint, unchanged). This mirrors
    FA2's OWN default (non-deterministic) backward structure (5 matmuls,
    atomic dQ), which is why memory here should land near FA2-default's, not
    the dS-storage path's O(S^2)-bounded-but-nonzero buffer.

    NOT used by ``fastflash_attn_train``/``fast_attention`` unless
    ``UNIFLASH_BWD_FUSED_ATOMIC=1`` is set. Call
    ``fastflash_attn_train_fused_atomic`` directly to opt in without the env
    var.
    """

    @staticmethod
    def forward(ctx, q, k, v, causal, scale=None):
        o, lse = fastflash_attn(q, k, v, causal, return_lse=True, scale=scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.scale = scale
        return o

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        causal = ctx.causal
        scale = ctx.scale
        B, H, S, D = q.shape
        do = do.contiguous()
        qf = q.reshape(B * H, S, D)
        kf = k.reshape(B * H, S, D)
        vf = v.reshape(B * H, S, D)
        of = o.reshape(B * H, S, D)
        dof = do.reshape(B * H, S, D)
        lf = lse.reshape(B * H, S)
        delta = _bwd_preprocess(of, dof)          # [BH,S] fp32, all heads once
        dq_acc, dk, dv = _bwd_fused_atomic(qf, kf, vf, dof, lf, delta, causal, scale=scale)
        dq = dq_acc.to(qf.dtype)
        return (dq.reshape(B, H, S, D),
                dk.reshape(B, H, S, D),
                dv.reshape(B, H, S, D),
                None, None)


def fastflash_attn_train_fused_atomic(q, k, v, causal: bool, scale=None):
    """Opt-in entry point for the fused dK/dV+dQ single-kernel backward.

    Same forward as ``fastflash_attn_train`` (exact bf16). The backward fuses
    dK/dV and dQ into one kernel launch so the ``ds`` tile never touches HBM:
    dK/dV stay deterministic, dQ is atomic-accumulated (NOT bit-exact
    reproducible run-to-run). See ``FastFlashAttnFusedAtomic``. Not wired into
    ``fast_attention``/``is_eligible``; call this directly, or set
    ``UNIFLASH_BWD_FUSED_ATOMIC=1`` to reroute the standard
    ``fastflash_attn_train``/``fast_attention`` entry points here instead.
    """
    return FastFlashAttnFusedAtomic.apply(q, k, v, causal, scale)
