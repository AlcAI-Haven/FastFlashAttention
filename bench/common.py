"""Shared measurement utilities for the attention benchmark harness.

Everything here follows the pytorch-trial timing rules:
  - CUDA-event timing only (never wall-clock around async GPU work)
  - >=10 warmup iters excluded
  - report distributions (median + IQR), not single points
  - every result carries an environment header
"""
from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Environment header
# --------------------------------------------------------------------------- #
def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"


def env_header() -> dict:
    """Compact environment fingerprint. Every result row carries this."""
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "cc": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "vram_gb": round(props.total_memory / 1e9, 1),
        "python": platform.python_version(),
        "os": platform.system(),
        "git": git_sha(),
    }


# --------------------------------------------------------------------------- #
# Timing (CUDA events)
# --------------------------------------------------------------------------- #
@dataclass
class TimeStats:
    median_ms: float
    iqr_ms: float
    min_ms: float
    n: int

    def as_dict(self) -> dict:
        return {
            "median_ms": round(self.median_ms, 4),
            "iqr_ms": round(self.iqr_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "n": self.n,
        }


def cuda_time(fn: Callable[[], object], warmup: int = 15, iters: int = 30) -> TimeStats:
    """Median +/- IQR latency of ``fn`` in ms, measured with CUDA events.

    Warmup absorbs cuDNN autotune, lazy init, allocator growth. We interleave
    nothing here (single fn), but keep runs back-to-back on an idle GPU.
    """
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    samples.sort()
    med = statistics.median(samples)
    q1 = statistics.median(samples[: len(samples) // 2])
    q3 = statistics.median(samples[(len(samples) + 1) // 2 :])
    return TimeStats(median_ms=med, iqr_ms=q3 - q1, min_ms=samples[0], n=iters)


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
def peak_mem_mb(fn: Callable[[], object]) -> float:
    """Peak *allocated* memory (MB) during a single call of ``fn``."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


# --------------------------------------------------------------------------- #
# FLOP accounting (analytic)
# --------------------------------------------------------------------------- #
def attn_flops_exact(B: int, H: int, S: int, D: int, causal: bool, fwd_only: bool = True) -> float:
    """FLOPs for *exact* softmax attention.

    Two matmuls dominate: QK^T and (softmax)V, each 2*B*H*S*S*D.
    Softmax elementwise cost is negligible and omitted (standard convention,
    matches FlashAttention paper counting).
    Backward is ~2.5x forward; we expose fwd_only for the sweep.
    """
    fwd = 4.0 * B * H * S * S * D
    if causal:
        fwd *= 0.5  # ~half the scores are masked
    return fwd if fwd_only else fwd * 3.5


def linear_attn_flops(B: int, H: int, S: int, D: int) -> float:
    """FLOPs for chunk/recurrent linear attention: O(N * D^2), not O(N^2 * D).

    Dominant terms per step: state update k^T v  (2*D*D) and query read q@state
    (2*D*D). Reported separately so cross-algorithm TFLOP/s is not conflated
    with exact attention (different total work).
    """
    return 2.0 * (2.0 * B * H * S * D * D)


# --------------------------------------------------------------------------- #
# Empirical roofline
# --------------------------------------------------------------------------- #
def matmul_roofline_tflops(dtype: torch.dtype, sizes=(4096, 8192, 12288), iters: int = 30) -> float:
    """Best achieved TFLOP/s of a large square matmul in ``dtype`` on this GPU.

    We use *achieved* matmul throughput as the roofline denominator rather than
    a spec-sheet number: it is what the same silicon actually delivers on a
    dense tensor-core op, so attention's %-of-roofline is an honest efficiency.
    We take the best over several sizes so the ceiling isn't undersold by a
    single shape that happens to be sub-optimal for the GEMM scheduler.
    """
    best = 0.0
    for n in sizes:
        try:
            a = torch.randn(n, n, device="cuda", dtype=dtype)
            b = torch.randn(n, n, device="cuda", dtype=dtype)
            st = cuda_time(lambda: a @ b, warmup=10, iters=iters)
            best = max(best, (2.0 * n ** 3) / (st.median_ms / 1e3) / 1e12)
            del a, b
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return best


# --------------------------------------------------------------------------- #
# Reference attention (fp32 ground truth) + parity metric
# --------------------------------------------------------------------------- #
def attn_reference(q, k, v):
    """Exact causal attention in fp32 -- the ground truth for parity tests."""
    qf, kf, vf = (t.float() for t in (q, k, v))
    scale = 1.0 / (qf.shape[-1] ** 0.5)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    B, H, S, _ = q.shape
    causal = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), 1)
    scores = scores.masked_fill(causal, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, vf)


def rel_l2(x, ref):
    return (torch.norm(x - ref) / torch.norm(ref)).item()


# --------------------------------------------------------------------------- #
# Results logging (append-only JSONL)
# --------------------------------------------------------------------------- #
def log_result(path: Path, row: dict) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(), **row}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def dtype_of(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]
