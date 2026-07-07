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
_DS_BUF_BYTES = int(os.environ.get("UNIFLASH_DS_BUF_BYTES", str(2 * 1024 ** 3)))


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
            ds_ptrs = (off_hz * stride_ds_z
                       + offs_m[:, None] * stride_ds_m
                       + offs_n[None, :] * stride_ds_n)
            tl.store(DS + ds_ptrs, ds.to(tl.bfloat16),
                     mask=m_valid[:, None] & n_valid[None, :])
        dk += tl.dot(tl.trans(ds).to(q.dtype), q, out_dtype=tl.float32)   # [BLOCK_N, D]

    dk = dk * softmax_scale
    out_ptrs = zoff + offs_n[:, None] * stride_m + offs_d[None, :] * stride_d
    tl.store(DK + out_ptrs, dk.to(DK.dtype.element_ty), mask=n_valid[:, None])
    tl.store(DV + out_ptrs, dv.to(DV.dtype.element_ty), mask=n_valid[:, None])


def _bwd_dkdv(q, k, v, do, lse, delta, causal, scale=None, ds=None, dk=None, dv=None):
    """dK, dV for the two-kernel split. All [B*H,S,D] bf16 / [B*H,S] fp32.

    If ``ds`` (a preallocated [B*H,S,S] bf16 buffer) is given, dkdv ALSO stores
    each unscaled ds tile into it (STORE_DS path) so a serially-following dQ can
    read ds instead of recomputing qk/p/dp (7->5 matmul cut). The unused pointer
    on the dead branch is bound to ``q`` (valid, never dereferenced).

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
            ds = tl.load(DS + ds_ptrs, mask=mask, other=0.0)         # [BLOCK_M, BLOCK_N] bf16
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

    If ``ds`` (a [B*H,S,S] bf16 buffer written by dkdv's STORE_DS) is given, dQ
    READS ds instead of recomputing qk/p/dp: dq = ds @ k (1 matmul, not 3). This
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
        #     so process BH in GROUPS of G = _DS_BUF_BYTES // (S*S*2) heads reusing
        #     ONE bounded [G,S,S] buffer. This bounds the O(S^2) buffer (was the old
        #     O(BH*S^2), which forced large-BH to the slower fallback) so the dS win
        #     applies for ANY batch size. A dim-0 slice of qf/kf/... and of the full
        #     dq/dk/dv result buffers keeps strides identical (offset base only), so
        #     per-head math is BYTE-IDENTICAL to the ungrouped path regardless of G.
        use_ds = (S * D >= _DS_MIN_SD) and (S * S * 2 <= _DS_BUF_BYTES) \
            and (S * S * 2 <= _DS_MAX_BYTES)
        if use_ds:
            BH = B * H
            delta = _bwd_preprocess(of, dof)           # [BH,S] fp32, all heads once
            dq = torch.empty_like(qf)
            dk = torch.empty_like(kf)
            dv = torch.empty_like(vf)
            # G = heads per group from the byte budget, but also clamped so the
            # dS pointer offset (max ~ G*S*S elements) stays within int32 —
            # Triton indexes in int32 by default, and G*S*S can exceed 2^31 if a
            # user raises _DS_BUF_BYTES past ~4GiB (else: illegal memory access).
            G = max(1, min(BH, _DS_BUF_BYTES // (S * S * 2), (2 ** 31 - 1) // (S * S)))
            DS = torch.empty(G, S, S, device=q.device, dtype=torch.bfloat16)
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
