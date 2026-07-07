"""FastFlashAttention vs FlashAttention-2 benchmark (forward / backward / full-step).

Times three regimes against both FA2 baselines (default = non-deterministic
backward; deterministic=True) across a sequence-length sweep, using CUDA-event
timing (median +/- IQR, warmup excluded) from bench.common. Prints one table per
(regime, D, causal) block and appends every row to results/benchmark.jsonl.

    python -m bench.benchmark            # full grid (paper numbers)
    python -m bench.benchmark --quick    # small grid smoke test

Ratios are FA2/FastFlashAttention wall time: >1 means FastFlashAttention is faster.
"""
from __future__ import annotations

import argparse

import torch

from bench.common import (
    RESULTS_DIR, attn_flops_exact, cuda_time, env_header, log_result,
    matmul_roofline_tflops,
)
from fastflash_attention import fast_attention

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    HAS_FLASH = True
except Exception as _e:  # pragma: no cover
    HAS_FLASH = False
    _FLASH_ERR = _e

B, H = 4, 16
FULL_N = [512, 1024, 2048, 4096, 8192, 16384]
QUICK_N = [512, 2048]


def _flash2(q, k, v, causal, deterministic=False):
    """FA2 in [B,H,S,D]->[B,H,S,D] framing (transpose to seq-major and back)."""
    q2, k2, v2 = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    out = _flash_attn_func(q2, k2, v2, causal=causal, deterministic=deterministic)
    return out.transpose(1, 2)


def _make(B, H, S, D, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    def mk():
        return torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, generator=g)
    return mk(), mk(), mk(), mk()  # q, k, v, do


def _fwd_thunk(fn, q, k, v):
    return lambda: fn(q, k, v)


def _bwd_thunk(fn, q, k, v, do):
    """Prebuild the graph once; time only autograd.grad (retain_graph to repeat)."""
    qg, kg, vg = (t.clone().requires_grad_(True) for t in (q, k, v))
    out = fn(qg, kg, vg)
    return lambda: torch.autograd.grad(out, (qg, kg, vg), do, retain_graph=True)


def _step_thunk(fn, q, k, v, do):
    """Fresh forward + backward each iteration (end-to-end training step)."""
    qg, kg, vg = (t.clone().requires_grad_(True) for t in (q, k, v))
    def thunk():
        out = fn(qg, kg, vg)
        torch.autograd.grad(out, (qg, kg, vg), do)
    return thunk


def _tf(flops, ms):
    return flops / (ms / 1e3) / 1e12


def _pct(tf, roof):
    return round(100.0 * tf / roof, 1) if roof else float("nan")


def run(quick=False):
    assert torch.cuda.is_available(), "CUDA required"
    assert HAS_FLASH, f"flash_attn required for benchmarks: {_FLASH_ERR}"
    env = env_header()
    roof = matmul_roofline_tflops(torch.bfloat16)
    print("ENV:", {k: env[k] for k in ("torch", "cuda", "gpu", "cc")},
          f"| bf16 matmul roofline = {roof:.1f} TF/s\n")
    Ns = QUICK_N if quick else FULL_N
    Ds = [128] if quick else [64, 128]
    causals = [True] if quick else [True, False]
    outpath = RESULTS_DIR / "benchmark.jsonl"

    for causal in causals:
        for D in Ds:
            hdr = (f"# regime=fwd/bwd/step  causal={causal}  D={D}  B={B} H={H}\n"
                   f"{'N':>6} | {'ff_fwd':>8} {'fa2_fwd':>8} {'r_fwd':>6} "
                   f"| {'ff_bwd':>8} {'fa2_bwd':>8} {'r_def':>6} {'fa2d_bwd':>8} {'r_det':>6} "
                   f"| {'ff_step':>8} {'fa2_step':>8} {'rs_def':>6} {'fa2d_step':>9} {'rs_det':>6}")
            print(hdr)
            print("-" * len(hdr.splitlines()[-1]))
            for S in Ns:
                q, k, v, do = _make(B, H, S, D)
                f_fwd = attn_flops_exact(B, H, S, D, causal, fwd_only=True)
                f_bwd = attn_flops_exact(B, H, S, D, causal, fwd_only=False) - f_fwd

                def ff(a, b, c): return fast_attention(a, b, c, is_causal=causal)
                def fa2(a, b, c): return _flash2(a, b, c, causal, False)
                def fa2d(a, b, c): return _flash2(a, b, c, causal, True)

                with torch.no_grad():
                    t_ff_f = cuda_time(_fwd_thunk(ff, q, k, v))
                    t_fa2_f = cuda_time(_fwd_thunk(fa2, q, k, v))
                t_ff_b = cuda_time(_bwd_thunk(ff, q, k, v, do))
                t_fa2_b = cuda_time(_bwd_thunk(fa2, q, k, v, do))
                t_fa2d_b = cuda_time(_bwd_thunk(fa2d, q, k, v, do))
                t_ff_s = cuda_time(_step_thunk(ff, q, k, v, do))
                t_fa2_s = cuda_time(_step_thunk(fa2, q, k, v, do))
                t_fa2d_s = cuda_time(_step_thunk(fa2d, q, k, v, do))

                row = {
                    "config": {"B": B, "H": H, "S": S, "D": D, "causal": causal},
                    "roofline_tflops": round(roof, 1),
                    "fwd": {
                        "ff_ms": round(t_ff_f.median_ms, 4), "fa2_ms": round(t_fa2_f.median_ms, 4),
                        "ratio": round(t_fa2_f.median_ms / t_ff_f.median_ms, 3),
                        "ff_tflops": round(_tf(f_fwd, t_ff_f.median_ms), 1),
                        "ff_pct_roofline": _pct(_tf(f_fwd, t_ff_f.median_ms), roof),
                    },
                    "bwd": {
                        "ff_ms": round(t_ff_b.median_ms, 4),
                        "fa2_ms": round(t_fa2_b.median_ms, 4),
                        "fa2det_ms": round(t_fa2d_b.median_ms, 4),
                        "ratio_def": round(t_fa2_b.median_ms / t_ff_b.median_ms, 3),
                        "ratio_det": round(t_fa2d_b.median_ms / t_ff_b.median_ms, 3),
                        "ff_tflops": round(_tf(f_bwd, t_ff_b.median_ms), 1),
                        "ff_pct_roofline": _pct(_tf(f_bwd, t_ff_b.median_ms), roof),
                    },
                    "step": {
                        "ff_ms": round(t_ff_s.median_ms, 4),
                        "fa2_ms": round(t_fa2_s.median_ms, 4),
                        "fa2det_ms": round(t_fa2d_s.median_ms, 4),
                        "ratio_def": round(t_fa2_s.median_ms / t_ff_s.median_ms, 3),
                        "ratio_det": round(t_fa2d_s.median_ms / t_ff_s.median_ms, 3),
                    },
                }
                log_result(outpath, {"env": env, **row})
                fw, bw, sw = row["fwd"], row["bwd"], row["step"]
                print(f"{S:>6} | {fw['ff_ms']:>8} {fw['fa2_ms']:>8} {fw['ratio']:>6} "
                      f"| {bw['ff_ms']:>8} {bw['fa2_ms']:>8} {bw['ratio_def']:>6} "
                      f"{bw['fa2det_ms']:>8} {bw['ratio_det']:>6} "
                      f"| {sw['ff_ms']:>8} {sw['fa2_ms']:>8} {sw['ratio_def']:>6} "
                      f"{sw['fa2det_ms']:>9} {sw['ratio_det']:>6}")
                torch.cuda.empty_cache()
            print()


def main():
    ap = argparse.ArgumentParser(description="FastFlashAttention vs FA2 benchmark")
    ap.add_argument("--quick", action="store_true", help="small grid smoke test")
    run(**vars(ap.parse_args()))


if __name__ == "__main__":
    main()
