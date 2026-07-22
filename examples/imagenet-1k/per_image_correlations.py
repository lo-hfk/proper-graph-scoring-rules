"""
Per-image Spearman correlation between metrics, computed on misclassified
images only (top-1 wrong), separately for each model.

Metrics compared (all continuous):
  dist@1, dist@5, log-loss, HXE (alpha=0.5), Brier, Resistance

Produces:
  1. Average correlation heatmap across all models
  2. Per-model heatmap grid (all models in one figure)
  3. CSV of per-model correlation matrices

Run from the imagenet-1k directory:
    python per_image_correlations.py --graph tree
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("./wordnet_graph")
OUT_DIR  = Path("./model_validation_outputs")
GRAPH_FILES = {
    "tree":  ("tree.gpickle",  "tree_leaf_synsets.json"),
    "ctree": ("ctree.gpickle", "ctree_leaf_synsets.json"),
}

sys.path.insert(0, str(Path(__file__).parent))
from propnetscore.batched_scorer import GraphScorer
import evaluate as ev


# ---------------------------------------------------------------------------
# Per-image metric computation
# ---------------------------------------------------------------------------

def per_image_metrics(probs, true_labels, dist_matrix,
                       scorer, leaf_full_idx, hxe_paths, alpha=0.5):
    """
    Returns dict of (n_examples,) arrays, one per metric.
    All metrics are continuous and lower = better.
    """
    n, N = probs.shape
    top5 = np.argsort(-probs, axis=1)[:, :5]
    top1 = top5[:, 0]

    # dist@1: LCA-height distance between top-1 and true (0 when correct)
    d1 = dist_matrix[true_labels, top1].astype(float)

    # dist@5: mean LCA-height distance over top-5
    d5 = np.mean(dist_matrix[true_labels[:, None], top5], axis=1).astype(float)

    # log-loss (flat)
    p_true = probs[np.arange(n), true_labels]
    logloss = -np.log(np.clip(p_true, 1e-12, 1.0))

    # Brier (flat)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), true_labels] = 1.0
    brier = np.sum((probs - onehot) ** 2, axis=1)

    # HXE
    hxe = ev.hxe_score(probs, true_labels, hxe_paths)

    # Resistance
    n_full = scorer.size
    p_full = np.zeros((n, n_full), dtype=np.float32)
    p_full[:, leaf_full_idx] = probs.astype(np.float32)
    true_full = np.array([leaf_full_idx[i] for i in true_labels], dtype=np.int64)
    resist = scorer.resistance_score(p_full, true_full).astype(float)

    return {
        "dist@1":      d1,
        "dist@5":      d5,
        "Log-loss":    logloss,
        "HXE (α=0.5)": hxe,
        "Brier":       brier,
        "Resistance":  resist,
    }


def spearman_matrix(metric_dict, mask):
    """Compute Spearman correlation matrix over masked examples."""
    keys = list(metric_dict.keys())
    k = len(keys)
    mat = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                mat[i, j] = 1.0
            elif j > i:
                a = metric_dict[keys[i]][mask]
                b = metric_dict[keys[j]][mask]
                rho, _ = spearmanr(a, b)
                mat[i, j] = mat[j, i] = rho
    return mat, keys


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_corr_heatmap(ax, mat, labels, title, vmin=0.0, vmax=1.0,
                      fontsize=8.5, show_values=True):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "corr", ["#ffffff", "#2166ac"], N=256
    )
    n = len(labels)
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            norm = (val - vmin) / (vmax - vmin)
            color = cmap(np.clip(norm, 0, 1))
            rect = plt.Rectangle((j, n - 1 - i), 1, 1,
                                  facecolor=color,
                                  edgecolor="black", linewidth=0.4)
            ax.add_patch(rect)
            if show_values:
                brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
                tc = "black" if brightness > 0.5 else "white"
                ax.text(j + 0.5, n - 1 - i + 0.5,
                        f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=fontsize - 1, color=tc)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=fontsize)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels[::-1], fontsize=fontsize)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=fontsize + 1, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="tree", choices=["tree", "ctree"])
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    graph_variant = args.graph
    alpha = args.alpha

    # Load graph
    graph_file, synsets_file = GRAPH_FILES[graph_variant]
    import json, networkx as nx
    with open(DATA_DIR / graph_file, "rb") as f:
        G = pickle.load(f)
    with open(DATA_DIR / synsets_file) as f:
        leaf_info = json.load(f)
    leaf_names = leaf_info["class_idx_to_synset_name"]

    print("Building scorer...")
    ev.OUT_DIR  = OUT_DIR
    ev.DATA_DIR = DATA_DIR
    scorer, _, leaf_full_idx = ev.build_full_graph_scorer(G, leaf_names, graph_variant)

    print("Loading distance matrix...")
    dist_matrix = ev.precompute_leaf_distance_matrix(G, leaf_names, graph_variant)

    print(f"Building HXE cache (alpha={alpha})...")
    hxe_paths = ev.build_hxe_cache(G, leaf_names, alpha=alpha,
                                    graph_variant=graph_variant)

    models = ev.discover_available_models()
    print(f"Found {len(models)} models: {models}")

    all_mats = []
    model_mats = {}
    metric_keys = None

    for model_name in models:
        print(f"  Computing per-image metrics: {model_name}")
        probs, true_labels = ev.load_model_outputs(model_name)

        metrics = per_image_metrics(
            probs, true_labels, dist_matrix,
            scorer, leaf_full_idx, hxe_paths, alpha=alpha
        )
        if metric_keys is None:
            metric_keys = list(metrics.keys())

        # Misclassified mask: top-1 wrong
        top1 = probs.argmax(axis=1)
        mask = (top1 != true_labels)
        n_wrong = mask.sum()
        print(f"    Misclassified: {n_wrong}/{len(true_labels)} "
              f"({100*n_wrong/len(true_labels):.1f}%)")

        mat, keys = spearman_matrix(metrics, mask)
        all_mats.append(mat)
        model_mats[model_name] = mat

    # Average correlation matrix
    avg_mat = np.mean(all_mats, axis=0)

    # -----------------------------------------------------------------------
    # Figure 1: Average heatmap
    # -----------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(5.5, 4.8))
    draw_corr_heatmap(ax1, avg_mat, metric_keys,
                      "Average Spearman correlation\n(misclassified images)",
                      vmin=0.0, vmax=1.0, fontsize=9)
    plt.tight_layout()
    out1 = OUT_DIR / f"spearman_avg_{graph_variant}.pdf"
    fig1.savefig(out1, bbox_inches="tight", dpi=600)
    fig1.savefig(str(out1).replace(".pdf", ".png"), bbox_inches="tight", dpi=600)
    print(f"Saved: {out1}")

    # -----------------------------------------------------------------------
    # Figure 2: Per-model grid
    # -----------------------------------------------------------------------
    n_models = len(models)
    n_cols_grid = 4
    n_rows_grid = math.ceil(n_models / n_cols_grid)
    fig2, axes = plt.subplots(n_rows_grid, n_cols_grid,
                               figsize=(4.5 * n_cols_grid, 4.2 * n_rows_grid))
    axes = axes.flatten()

    # Use same vmin/vmax as average for comparability
    vmin_all = min(m[np.triu_indices(len(metric_keys), k=1)].min()
                   for m in all_mats)
    vmin_all = max(0.0, vmin_all - 0.05)

    for idx, model_name in enumerate(models):
        draw_corr_heatmap(axes[idx], model_mats[model_name], metric_keys,
                          model_name, vmin=vmin_all, vmax=1.0,
                          fontsize=7.5)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    fig2.suptitle(
        f"Per-model Spearman correlations on misclassified images [{graph_variant}]",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    out2 = OUT_DIR / f"spearman_per_model_{graph_variant}.pdf"
    fig2.savefig(out2, bbox_inches="tight", dpi=600)
    fig2.savefig(str(out2).replace(".pdf", ".png"), bbox_inches="tight", dpi=600)
    print(f"Saved: {out2}")

    # -----------------------------------------------------------------------
    # Print average matrix
    # -----------------------------------------------------------------------
    import pandas as pd
    df = pd.DataFrame(avg_mat, index=metric_keys, columns=metric_keys)
    print("\nAverage Spearman correlation (misclassified images):")
    print(df.round(3).to_string())

    # Save CSV
    df.to_csv(OUT_DIR / f"spearman_avg_{graph_variant}.csv")
    print(f"\nSaved CSV: {OUT_DIR}/spearman_avg_{graph_variant}.csv")


if __name__ == "__main__":
    main()