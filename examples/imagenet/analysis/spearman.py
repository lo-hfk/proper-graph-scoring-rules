"""
Per-image Spearman correlation heatmap between metrics.

Computed on misclassified images only (top-1 wrong), averaged over all models.
Reads data/results/<model>.csv and saves:
  - figures/spearman_avg.pdf : average correlation heatmap
  - figures/spearman_per_model.pdf : per-model grid (2 rows x 5 cols)

Usage
-----
    python analysis/spearman.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["dist_at_1", "dist_at_5", "log_loss", "brier", "hxe", "resistance"]
LABELS  = ["Dist@1", "Dist@5", "Log-loss", "Brier", "HXE", "Resistance"]

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


def spearman_matrix(df):
    mask = df["top1_err"] == 1
    sub  = df.loc[mask, METRICS].values
    k    = len(METRICS)
    mat  = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            rho, _ = spearmanr(sub[:, i], sub[:, j])
            mat[i, j] = mat[j, i] = rho
    return mat


def draw_heatmap(ax, mat, title,
                 show_xlabels=False, show_ylabels=False,
                 fontsize=8):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "corr", ["#ffffff", "#2166ac"], N=256
    )
    k = len(LABELS)
    for i in range(k):
        for j in range(k):
            val   = mat[i, j]
            color = cmap(np.clip(val, 0, 1))
            rect  = plt.Rectangle((j, k - 1 - i), 1, 1,
                                   facecolor=color,
                                   edgecolor="black", linewidth=0.3)
            ax.add_patch(rect)
            brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            tc = "black" if brightness > 0.5 else "white"
            ax.text(j + 0.5, k - 1 - i + 0.5, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=fontsize - 1, color=tc)

    ax.set_xlim(0, k)
    ax.set_ylim(0, k)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_xlabels:
        ax.set_xticks(np.arange(k) + 0.5)
        ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=fontsize)
    else:
        ax.set_xticks([])

    if show_ylabels:
        ax.set_yticks(np.arange(k) + 0.5)
        ax.set_yticklabels(LABELS[::-1], fontsize=fontsize)
    else:
        ax.set_yticks([])

    ax.set_title(title, fontsize=fontsize + 1, pad=4)


if __name__ == "__main__":
    csvs = sorted(RESULTS_DIR.glob("*.csv"))
    all_mats  = []
    model_mats = {}

    for csv in csvs:
        df  = pd.read_csv(csv)
        mat = spearman_matrix(df)
        all_mats.append(mat)
        model_mats[csv.stem] = mat
        print(f"  {csv.stem}: {(df['top1_err']==1).sum()} misclassified")

    avg_mat = np.mean(all_mats, axis=0)

    # --- Figure 1: average heatmap ---
    fig1, ax1 = plt.subplots(figsize=(5.5, 4.8))
    draw_heatmap(ax1, avg_mat, title="", show_xlabels=True, show_ylabels=True, fontsize=12)
    plt.tight_layout()
    fig1.savefig(FIGURES_DIR / "spearman_avg.pdf", bbox_inches="tight", dpi=600)
    fig1.savefig(FIGURES_DIR / "spearman_avg.png", bbox_inches="tight", dpi=600)
    print(f"\nSaved spearman_avg")

    print("\nAverage Spearman correlation (misclassified images):")
    print(pd.DataFrame(avg_mat, index=LABELS, columns=LABELS).round(3).to_string())

    # --- Figure 2: per-model grid (2 rows x 5 cols) ---
    models = list(model_mats.keys())  # 10 models, sorted alphabetically
    n_rows, n_cols = 2, 5
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.8 * n_rows))

    for idx, name in enumerate(models):
        row = idx // n_cols
        col = idx  % n_cols
        ax  = axes[row, col]
        draw_heatmap(
            ax,
            model_mats[name],
            title=MODEL_NAMES.get(name, name),
            show_xlabels=(row == n_rows - 1),  # only bottom row
            show_ylabels=(col == 0),  # only leftmost column
            fontsize=13,
        )

    plt.tight_layout()
    fig2.savefig(FIGURES_DIR / "spearman_per_model.pdf", bbox_inches="tight", dpi=600)
    fig2.savefig(FIGURES_DIR / "spearman_per_model.png", bbox_inches="tight", dpi=600)
    print("Saved spearman_per_model")
