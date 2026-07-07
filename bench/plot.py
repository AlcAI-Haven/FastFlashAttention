"""Render the FastFlashAttention-vs-FA2 speedup figure from benchmark.jsonl.

Produces light- and dark-mode PNGs (assets/speedup_light.png, _dark.png) used in
the README. Three panels — forward / backward / full training step — each showing
speedup (FA2 / FastFlashAttention wall time; >1 = FastFlashAttention faster) across
sequence length, for the tuned headline config (causal, D=128). A dashed line at
1.0 marks parity with FlashAttention-2.

    python -m bench.plot        # after: python -m bench.benchmark

Colors are the dataviz reference categorical pair (blue = deterministic baseline,
orange = default baseline), validated CVD-safe on both surfaces.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
JSONL = ROOT / "results" / "benchmark.jsonl"
ASSETS = ROOT / "assets"

# dataviz reference palette (validated: light/dark surfaces, CVD ΔE ~97)
THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e",
              "grid": "#e3e2de", "blue": "#2a78d6", "orange": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "muted": "#c3c2b7",
             "grid": "#33332f", "blue": "#3987e5", "orange": "#d95926"},
}


def _load(causal=True, D=128):
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")]
    rows = [r for r in rows if r["config"]["causal"] == causal and r["config"]["D"] == D]
    rows.sort(key=lambda r: r["config"]["S"])
    return rows


def _fmt_n(n):
    return f"{n // 1024}k" if n >= 1024 else str(n)


def _panel(ax, t, xs, series, xlabels, c):
    """series: list of (values, color, label). Draws lines + endpoint value labels."""
    ax.axhline(1.0, ls=(0, (4, 3)), lw=1.2, color=c["muted"], zorder=1)
    for ys, color, _label in series:
        ax.plot(xs, ys, "-o", color=color, lw=2.0, ms=6.5, mfc=color,
                mec=c["surface"], mew=1.0, zorder=3)
        ax.annotate(f"{ys[-1]:.2f}×", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(6, 0), va="center",
                    ha="left", fontsize=8.5, color=c["muted"])
    ax.set_title(t, fontsize=12, color=c["ink"], pad=8, fontweight="medium")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8.5, color=c["muted"])
    ax.tick_params(axis="y", labelsize=8.5, colors=c["muted"], length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=c["grid"], lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.margins(x=0.16)
    ax.set_facecolor(c["surface"])


def make_figure(mode):
    c = THEME[mode]
    rows = _load(True, 128)
    xs = [r["config"]["S"] for r in rows]
    xlabels = [_fmt_n(n) for n in xs]

    fwd = [r["fwd"]["ratio"] for r in rows]
    bwd_det = [r["bwd"]["ratio_det"] for r in rows]
    bwd_def = [r["bwd"]["ratio_def"] for r in rows]
    stp_det = [r["step"]["ratio_det"] for r in rows]
    stp_def = [r["step"]["ratio_def"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), sharey=True)
    fig.patch.set_facecolor(c["surface"])
    fig.subplots_adjust(top=0.74, bottom=0.15, left=0.075, right=0.965, wspace=0.12)

    _panel(axes[0], "Forward", xs, [(fwd, c["orange"], "vs FA2")], xlabels, c)
    _panel(axes[1], "Backward", xs,
           [(bwd_det, c["blue"], "det"), (bwd_def, c["orange"], "def")], xlabels, c)
    _panel(axes[2], "Full training step", xs,
           [(stp_det, c["blue"], "det"), (stp_def, c["orange"], "def")], xlabels, c)

    lo = min(min(fwd), min(bwd_det), min(bwd_def), min(stp_det), min(stp_def), 1.0)
    hi = max(max(fwd), max(bwd_det), max(bwd_def), max(stp_det), max(stp_def))
    axes[0].set_ylim(lo * 0.9, hi * 1.10)
    axes[0].set_ylabel("speedup vs FlashAttention-2  (×)", fontsize=9.5, color=c["muted"])

    # figure-level legend: color = which FA2 baseline
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=c["blue"], lw=2.4, marker="o", ms=6,
               mec=c["surface"], label="vs FA2  (deterministic backward)"),
        Line2D([0], [0], color=c["orange"], lw=2.4, marker="o", ms=6,
               mec=c["surface"], label="vs FA2  (default / non-deterministic)"),
        Line2D([0], [0], color=c["muted"], lw=1.2, ls=(0, (4, 3)),
               label="parity (1.0×)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.01), labelcolor=c["muted"])
    fig.text(0.5, 0.93, "FastFlashAttention speedup over FlashAttention-2  ·  RTX 5090 (sm_120)  ·  causal, D=128, bf16",
             ha="center", fontsize=12.5, color=c["ink"])
    fig.text(0.5, 0.855, "higher is better — above the dashed parity line, FastFlashAttention is faster",
             ha="center", fontsize=9, color=c["muted"])

    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / f"speedup_{mode}.png"
    fig.savefig(out, dpi=200, facecolor=c["surface"])
    plt.close(fig)
    print("wrote", out)


def main():
    for mode in ("light", "dark"):
        make_figure(mode)


if __name__ == "__main__":
    main()
