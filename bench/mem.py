"""Peak-memory benchmark: FastFlashAttention vs FlashAttention-2 at the same N.

Measures peak *allocated* VRAM for two regimes — forward-only (inference) and a
full training step (fwd+bwd). Each (N, regime) point is measured in a FRESH
subprocess: peak-memory stats are contaminated across shapes/autotune passes in a
long-lived process, so isolation is the only way to get stable, reproducible
numbers. Warmup inside each worker absorbs the Triton autotune pass. Config
matches the speed benchmark's headline: B=4, H=16, causal, D=128.

    python -m bench.mem            # full N sweep (spawns workers)
    python -m bench.mem --quick

Peak is total allocated (inputs + op footprint); inputs are identical for both
backends, so the delta is the kernel's own footprint. FA2 measures identically
whether fed native seq-major or transposed from [B,H,S,D], so the forward gap is
intrinsic, not a layout artifact.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import torch

from bench.common import RESULTS_DIR, env_header, log_result

B, H, D = 4, 16, 128
FULL_N = [512, 1024, 2048, 4096, 8192, 16384]
QUICK_N = [512, 2048]


# --------------------------------------------------------------------------- #
# Worker: measures ONE (N, regime) point in a clean process, prints JSON.
# --------------------------------------------------------------------------- #
def _worker(N, regime):
    from fastflash_attention import fast_attention
    from flash_attn import flash_attn_func

    def mk(shape, seed=0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=g)

    def flash2(q, k, v):
        q2, k2, v2 = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        return flash_attn_func(q2, k2, v2, causal=True).transpose(1, 2)

    def peak(step, warmup=6):
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e6

    q, k, v = mk((B, H, N, D)), mk((B, H, N, D)), mk((B, H, N, D))
    ours_fn = lambda a, b, c: fast_attention(a, b, c, is_causal=True)

    if regime == "fwd":
        def mk_fwd(fn):
            def step():
                with torch.no_grad():
                    fn(q, k, v)
            return step
        ours = peak(mk_fwd(ours_fn))
        fa2 = peak(mk_fwd(flash2))
    else:
        do = mk((B, H, N, D), seed=1)
        def mk_train(fn):
            def step():
                qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
                out = fn(qg, kg, vg)
                torch.autograd.grad(out, (qg, kg, vg), do)
            return step
        ours = peak(mk_train(ours_fn))
        fa2 = peak(mk_train(flash2))

    print(json.dumps({"N": N, "regime": regime,
                      "ours_mb": round(ours, 1), "fa2_mb": round(fa2, 1)}))


# --------------------------------------------------------------------------- #
# Orchestrator: spawns one worker per (N, regime), collects, tabulates.
# --------------------------------------------------------------------------- #
def run(quick=False):
    assert torch.cuda.is_available(), "CUDA required"
    env = env_header()
    print("ENV:", {k: env[k] for k in ("torch", "cuda", "gpu", "cc")},
          f"| B={B} H={H} D={D} causal=True  (each point in a fresh process)\n")
    Ns = QUICK_N if quick else FULL_N
    outpath = RESULTS_DIR / "memory.jsonl"

    def measure(N, regime):
        cp = subprocess.run([sys.executable, "-m", "bench.mem", "--worker",
                             str(N), "--regime", regime],
                            capture_output=True, text=True)
        line = [l for l in cp.stdout.splitlines() if l.strip().startswith("{")]
        if not line:
            raise RuntimeError(f"worker N={N} {regime} failed:\n{cp.stdout}\n{cp.stderr}")
        return json.loads(line[-1])

    hdr = (f"{'N':>6} | {'fwd_ours':>9} {'fwd_fa2':>9} {'fwd_save':>9} "
           f"| {'trn_ours':>9} {'trn_fa2':>9} {'trn_save':>9}")
    print(hdr)
    print("-" * len(hdr))
    for N in Ns:
        f = measure(N, "fwd")
        t = measure(N, "train")
        fsave = round(100 * (1 - f["ours_mb"] / f["fa2_mb"]), 1)
        tsave = round(100 * (1 - t["ours_mb"] / t["fa2_mb"]), 1)
        log_result(outpath, {"env": env,
                             "config": {"B": B, "H": H, "S": N, "D": D, "causal": True},
                             "fwd": {**{k: f[k] for k in ("ours_mb", "fa2_mb")}, "pct_save": fsave},
                             "train": {**{k: t[k] for k in ("ours_mb", "fa2_mb")}, "pct_save": tsave}})
        print(f"{N:>6} | {f['ours_mb']:>9} {f['fa2_mb']:>9} {fsave:>8}% "
              f"| {t['ours_mb']:>9} {t['fa2_mb']:>9} {tsave:>8}%")


def main():
    ap = argparse.ArgumentParser(description="FastFlashAttention vs FA2 peak memory")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--worker", type=int, help="(internal) measure one N")
    ap.add_argument("--regime", choices=("fwd", "train"))
    a = ap.parse_args()
    if a.worker is not None:
        _worker(a.worker, a.regime)
    else:
        run(quick=a.quick)


if __name__ == "__main__":
    main()
