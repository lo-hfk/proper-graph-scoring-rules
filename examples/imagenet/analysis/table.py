"""
Produce the aggregate heatmap table of mean scores across all models.

Reads data/results/<model>.csv, computes per-model mean scores,
and saves figures/table.pdf.

Usage
-----
    python analysis/table.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Display names for columns
COL_NAMES = {
    "top1_err": "Top-1 err",
    "top5_err": "Top-5 err",
    "dist_at_1": "Dist@1",
    "dist_at_5": "Dist@5",
    "log_loss": "Log-loss",
    "brier": "Brier",
    "hxe": "HXE",
    "resistance": "Resistance",
}

# Display names for models (row labels)
MODEL_NAMES = {
    "convnext_tiny": "ConvNeXt-T",
    "swin_tiny": "Swin-T",
    "vit_small_patch16_224": "ViT-S/16",
    "resnet50_timm": "ResNet-50",
    "efficientnet_b2": "EffNet-B2",
    "efficientnet_b0": "EffNet-B0",
    "mobilenet_v3_large": "MobileNet-L",
    "resnet18": "ResNet-18",
    "shufflenet_v2_x1_0": "ShuffleNet",
    "mobilenet_v3_small": "MobileNet-S",
}


def load_means():
    rows = {}
    for csv in sorted(RESULTS_DIR.glob("*.csv")):
        name = csv.stem
        df = pd.read_csv(csv)
        # exract pd.Series of means over all 50000 validation images (index = score):
        rows[name] = df.mean()
    return pd.DataFrame(rows).T


def plot_heatmap(df):
    # normalise each column: 0 = best (min), 1 = worst (max)
    norm = (df - df.min()) / (df.max() - df.min())

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "heatmap", ["#ffffff", "#1a7a3e"], N=256
    )

    n_rows, n_cols = df.shape
    fig, ax = plt.subplots(figsize=(12, 5))

    for i in range(n_rows):
        for j in range(n_cols):
            val  = df.iloc[i, j]
            nval = norm.iloc[i, j]
            color = cmap(nval)
            rect = plt.Rectangle((j, n_rows - 1 - i), 1, 1,
                                  facecolor=color,
                                  edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
            brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            tc = "black" if brightness > 0.5 else "white"
            ax.text(j + 0.5, n_rows - 1 - i + 0.5,
                    f"{val:.3f}", ha="center", va="center",
                    fontsize=12, color=tc)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([COL_NAMES.get(c, c) for c in df.columns],
                       rotation=30, ha="right", fontsize=13)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([MODEL_NAMES.get(r, r) for r in df.index[::-1]],
                       fontsize=13)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    df = load_means()
    # sort models by resistance score (best first)
    df = df.sort_values("resistance")
    print(df.round(4).to_string())

    fig = plot_heatmap(df)
    out = FIGURES_DIR / "table.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=600)
    fig.savefig(FIGURES_DIR / "table.png", bbox_inches="tight", dpi=600)
    print(f"\nSaved to {out}")
