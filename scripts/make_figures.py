"""Regenerate the two data-derived manuscript figures from real experiment
results (fig_performance_comparison.pdf, fig_ablation_study_plot.pdf) --
the earlier versions were built from the placeholder numbers this revision
replaces. Uses the validated colorblind-safe categorical palette (fixed hue
order, not auto-cycled) from the dataviz skill's reference palette.

Usage: python -m scripts.make_figures --out_dir ../paper
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette, fixed order (dataviz skill reference palette,
# light-mode slots 1-6): blue, orange, aqua, yellow, magenta, green.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

METHODS = ["FT", "EWC", "LwF", "DER", "LUD", "EAKD-CFND"]
DATASETS = ["PHEME-Event", "FNN-Poli-Time", "FNN-Gossip-Time"]

# real numbers, from code/runs/core_pheme.json / core_poli.json / core_gossip.json
PERF = {
    "PHEME-Event": {
        "FT": (16.1, 1.2, -85.3, 2.0), "EWC": (16.5, 0.5, -86.3, 1.6),
        "LwF": (16.5, 0.3, -87.1, 1.0), "DER": (84.8, 1.5, -3.2, 2.1),
        "LUD": (16.7, 0.3, -87.1, 0.9), "EAKD-CFND": (16.8, 0.4, -87.2, 1.1),
    },
    "FNN-Poli-Time": {
        "FT": (27.4, 2.8, -80.5, 3.4), "EWC": (28.6, 3.4, -82.1, 2.7),
        "LwF": (31.2, 1.0, -81.4, 2.4), "DER": (30.4, 0.9, -80.1, 4.0),
        "LUD": (31.2, 0.5, -82.2, 2.3), "EAKD-CFND": (29.8, 2.8, -81.6, 2.0),
    },
    "FNN-Gossip-Time": {
        "FT": (29.0, 0.2, -84.6, 2.0), "EWC": (28.9, 0.3, -84.8, 1.6),
        "LwF": (28.9, 0.5, -84.3, 1.5), "DER": (29.4, 0.3, -85.4, 0.5),
        "LUD": (29.0, 0.3, -84.8, 1.5), "EAKD-CFND": (29.1, 0.4, -84.6, 1.7),
    },
}

# real numbers, from code/runs/ablation.json (FNN-Poli-Time)
ABLATION_ORDER = [
    "EAKD-CFND\n(Full)", "Standard KD\n($\\lambda=1$)", "Random\n$\\alpha(x)$",
    "Loss-based\n$\\alpha$", "w/o External\nVerification", "Fine-tuning\n(No KD)",
]
ABLATION = {
    "EAKD-CFND\n(Full)": (29.8, 2.8, -81.6, 2.0),
    "Standard KD\n($\\lambda=1$)": (31.2, 1.0, -81.4, 2.4),
    "Random\n$\\alpha(x)$": (29.0, 2.5, -81.1, 1.2),
    "Loss-based\n$\\alpha$": (28.8, 2.3, -80.8, 2.5),
    "w/o External\nVerification": (29.5, 2.6, -81.6, 2.0),
    "Fine-tuning\n(No KD)": (27.4, 2.8, -80.5, 3.4),
}


def style_axis(ax, ylabel):
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def make_performance_figure(out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    n_methods = len(METHODS)
    n_datasets = len(DATASETS)
    width = 0.8 / n_methods
    x = range(n_datasets)

    for m_idx, method in enumerate(METHODS):
        accs = [PERF[d][method][0] for d in DATASETS]
        acc_errs = [PERF[d][method][1] for d in DATASETS]
        bwts = [PERF[d][method][2] for d in DATASETS]
        bwt_errs = [PERF[d][method][3] for d in DATASETS]
        offsets = [xi + (m_idx - n_methods / 2 + 0.5) * width for xi in x]
        axes[0].bar(offsets, accs, width=width * 0.9, yerr=acc_errs, capsize=2,
                    color=PALETTE[m_idx], label=method, zorder=3,
                    error_kw={"ecolor": MUTED, "elinewidth": 0.8})
        axes[1].bar(offsets, bwts, width=width * 0.9, yerr=bwt_errs, capsize=2,
                    color=PALETTE[m_idx], label=method, zorder=3,
                    error_kw={"ecolor": MUTED, "elinewidth": 0.8})

    for ax, ylabel, title in ((axes[0], "Avg. Accuracy (%)", "Average Accuracy"),
                               (axes[1], "BWT (%)", "Backward Transfer")):
        style_axis(ax, ylabel)
        ax.set_xticks(list(x))
        ax.set_xticklabels(DATASETS, fontsize=9, color=INK)
        ax.set_title(title, fontsize=11, color=INK)
        ax.axhline(0, color=GRID, linewidth=0.8, zorder=1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Continual Learning Performance Comparison (real rerun, 5 seeds)",
                 fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def make_ablation_figure(out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    accs = [ABLATION[c][0] for c in ABLATION_ORDER]
    acc_errs = [ABLATION[c][1] for c in ABLATION_ORDER]
    bwts = [ABLATION[c][2] for c in ABLATION_ORDER]
    bwt_errs = [ABLATION[c][3] for c in ABLATION_ORDER]
    x = range(len(ABLATION_ORDER))
    colors = [PALETTE[0] if c.startswith("EAKD-CFND") else PALETTE[i % len(PALETTE)]
              for i, c in enumerate(ABLATION_ORDER)]

    axes[0].bar(x, accs, yerr=acc_errs, capsize=3, color=colors, zorder=3,
                error_kw={"ecolor": MUTED, "elinewidth": 0.8})
    axes[1].bar(x, bwts, yerr=bwt_errs, capsize=3, color=colors, zorder=3,
                error_kw={"ecolor": MUTED, "elinewidth": 0.8})

    for ax, ylabel, title in ((axes[0], "Avg. Accuracy (%)", "Average Accuracy"),
                               (axes[1], "BWT (%)", "Backward Transfer")):
        style_axis(ax, ylabel)
        ax.set_xticks(list(x))
        ax.set_xticklabels(ABLATION_ORDER, fontsize=7.5, color=INK)
        ax.set_title(title, fontsize=11, color=INK)
        ax.axhline(0, color=GRID, linewidth=0.8, zorder=1)

    fig.suptitle("Ablation Study on FNN-Poli-Time (real rerun, 5 seeds)",
                 fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="../paper")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    make_performance_figure(out_dir / "fig_performance_comparison.pdf")
    make_ablation_figure(out_dir / "fig_ablation_study_plot.pdf")


if __name__ == "__main__":
    main()
