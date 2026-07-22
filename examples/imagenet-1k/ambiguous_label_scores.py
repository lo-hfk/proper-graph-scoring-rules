"""
For a single image with an ambiguous true label, compute per-example scores
under both the standard (point-mass) ground truth and a user-specified
soft true distribution, then display a comparison table.

Usage:
    python ambiguous_label_scores.py \
        --image-idx 44101 \
        --model-a vit_small_patch16_224 \
        --model-b convnext_tiny \
        --true-labels "vizsla.n.01:0.5,moped.n.01:0.5" \
        --graph tree

The --true-labels argument is a comma-separated list of synset:weight pairs.
Weights are normalised to sum to 1.
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path("./wordnet_graph")
OUT_DIR  = Path("./model_validation_outputs")
sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev

COLOR_A = "\033[94m"   # blue
COLOR_B = "\033[91m"   # red
RESET   = "\033[0m"
BOLD    = "\033[1m"


def score_under_distribution(probs, true_dist, dist_matrix,
                              scorer, leaf_full_idx, hxe_paths):
    """
    Compute expected scores under a soft true distribution.

    true_dist: list of (leaf_idx, weight) pairs, weights sum to 1.

    Returns dict of metric_name -> expected score.
    """
    scores = {}

    # Top-1 error: 1 - sum_i q_i * 1(argmax p == i)
    top1 = int(np.argmax(probs))
    scores["Top-1 error"] = 1.0 - sum(w for (i, w) in true_dist if i == top1)

    # Top-5 error: 1 - sum_i q_i * 1(i in top5)
    top5 = set(np.argsort(-probs)[:5])
    scores["Top-5 error"] = 1.0 - sum(w for (i, w) in true_dist if i in top5)

    # distance@1: E_{i~q}[height(LCA(top1, i))]
    scores["distance@1"] = sum(w * dist_matrix[i, top1]
                                for (i, w) in true_dist)

    # distance@5: E_{i~q}[ mean_{j in top5} height(LCA(j, i)) ]
    top5_list = list(np.argsort(-probs)[:5])
    scores["distance@5"] = sum(
        w * np.mean([dist_matrix[i, j] for j in top5_list])
        for (i, w) in true_dist
    )

    # Log-loss: E_{i~q}[-log p_i]
    scores["Log-loss"] = sum(
        w * (-math.log(max(probs[i], 1e-12)))
        for (i, w) in true_dist
    )

    # HXE: E_{i~q}[HXE(p, i)]
    n_true = len(true_dist)
    probs_tiled = np.tile(probs[np.newaxis, :], (n_true, 1))
    hxe_vals = ev.hxe_score(
        probs_tiled,
        np.array([i for (i, _) in true_dist]),
        hxe_paths
    )
    scores["HXE (α=0.5)"] = sum(w * hxe_vals[k]
                                  for k, (_, w) in enumerate(true_dist))

    # Brier: E_{i~q}[||p - e_i||^2]
    scores["Brier score"] = sum(
        w * float(np.sum((probs - np.eye(len(probs))[i]) ** 2))
        for (i, w) in true_dist
    )

    # Resistance: E_{i~q}[S_res(p, i)]
    n_full = scorer.size
    p_full = np.zeros((1, n_full), dtype=np.float32)
    p_full[0, leaf_full_idx] = probs.astype(np.float32)
    resist_vals = scorer.resistance_score(
        np.tile(p_full, (len(true_dist), 1)),
        np.array([leaf_full_idx[i] for (i, _) in true_dist], dtype=np.int64)
    )
    scores["Resistance score"] = sum(w * float(resist_vals[k])
                                      for k, (_, w) in enumerate(true_dist))

    return scores


def print_comparison_table(scores_point_a, scores_point_b,
                            scores_soft_a,  scores_soft_b,
                            model_a_name, model_b_name,
                            true_label_name, soft_label_str):
    metric_order = ["Top-1 error", "Top-5 error",
                    "distance@1", "distance@5",
                    "Log-loss", "HXE (α=0.5)",
                    "Brier score", "Resistance score"]

    sep = "-" * 80
    print()
    print(sep)
    print(f"  Scores for image — standard ground truth: {true_label_name}")
    print(f"  Soft ground truth: {soft_label_str}")
    print(sep)
    print(f"  {'Metric':<22}  {'':>4}  "
          f"{'Point: ' + model_a_name:>18}  {'Point: ' + model_b_name:>18}  "
          f"{'Soft: ' + model_a_name:>18}  {'Soft: ' + model_b_name:>18}")
    print(sep)

    for m in metric_order:
        pa = scores_point_a.get(m, float("nan"))
        pb = scores_point_b.get(m, float("nan"))
        sa = scores_soft_a.get(m,  float("nan"))
        sb = scores_soft_b.get(m,  float("nan"))

        def fmt(v, better):
            s = f"{v:8.4f}"
            return f"{BOLD}{s}{RESET}" if better else s

        point_a_better = pa < pb
        point_b_better = pb < pa
        soft_a_better  = sa < sb
        soft_b_better  = sb < sa

        print(f"  {m:<22}  "
              f"  {fmt(pa, point_a_better):>18}  {fmt(pb, point_b_better):>18}"
              f"  {fmt(sa, soft_a_better):>18}  {fmt(sb, soft_b_better):>18}")
    print(sep)
    print()


def draw_table_png(scores_point_a, scores_point_b,
                   scores_soft_a,  scores_soft_b,
                   model_a_name, model_b_name,
                   true_label_name, soft_label_str,
                   out_path):
    import matplotlib.pyplot as plt

    metric_order = ["Top-1 error", "Top-5 error",
                    "distance@1", "distance@5",
                    "Log-loss", "HXE (\u03b1=0.5)",
                    "Brier score", "Resistance score"]

    color_a = "#1565C0"
    color_b = "#BF360C"
    n = len(metric_order)
    row_h = 0.42
    col_w = [2.0, 1.5, 1.5, 1.5, 1.5]
    pad = 0.15
    fig_w = sum(col_w) + 2 * pad
    # rows: title + group header + model header + n data rows + bottom padding
    fig_h = row_h * (n + 3) + 0.35

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    x0 = pad
    cx = [x0 + col_w[0] / 2]
    acc = x0 + col_w[0]
    for w in col_w[1:]:
        cx.append(acc + w / 2)
        acc += w

    def row_y(i):
        """y-centre for display row i (0=top)."""
        return fig_h - row_h * (i + 0.5)

    # Row 0: title
    ax.text(fig_w / 2, row_y(0),
            f"Point ground truth: {true_label_name}     "
            f"Soft ground truth: {soft_label_str}",
            ha="center", va="center", fontsize=7.5, color="#444444",
            style="italic")

    # Row 1: group labels
    mid_point = (cx[1] + cx[2]) / 2
    mid_soft  = (cx[3] + cx[4]) / 2
    ax.text(mid_point, row_y(1), "Point ground truth",
            ha="center", va="center", fontsize=8, color="#555555",
            fontweight="bold")
    ax.text(mid_soft,  row_y(1), "Soft ground truth",
            ha="center", va="center", fontsize=8, color="#555555",
            fontweight="bold")

    # Horizontal rule between group labels and model names
    rule1_y = row_y(1) - row_h * 0.55
    ax.axhline(rule1_y, color="#aaaaaa", lw=0.6)

    # Row 2: model names
    for j, (name, col) in enumerate([(model_a_name, color_a),
                                       (model_b_name, color_b),
                                       (model_a_name, color_a),
                                       (model_b_name, color_b)]):
        ax.text(cx[j + 1], row_y(2), name,
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color=col)

    # Horizontal rule below model names (above data)
    rule2_y = row_y(2) - row_h * 0.55
    ax.axhline(rule2_y, color="black", lw=0.8)

    # Bottom rule — sits below the last data row
    last_row_y = row_y(n + 2)
    bottom_y = last_row_y - row_h * 0.5
    ax.axhline(bottom_y, color="black", lw=0.8)

    # Data rows
    all_scores = [scores_point_a, scores_point_b,
                  scores_soft_a,  scores_soft_b]
    for i, m in enumerate(metric_order):
        y = row_y(i + 3)
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle(
                (0, y - row_h / 2), fig_w, row_h,
                facecolor="#f4f4f4", edgecolor="none", zorder=0))

        ax.text(x0 + 0.1, y, m, ha="left", va="center",
                fontsize=8, color="#222222")

        vals = [s.get(m, float("nan")) for s in all_scores]
        better = [vals[0] < vals[1], vals[1] < vals[0],
                  vals[2] < vals[3], vals[3] < vals[2]]
        colors = [color_a, color_b, color_a, color_b]
        for j in range(4):
            # 1 decimal for Resistance, 4 for everything else
            fmt = f"{vals[j]:.1f}" if m == "Resistance score" else f"{vals[j]:.4f}"
            ax.text(cx[j + 1], y, fmt,
                    ha="center", va="center", fontsize=8,
                    color=colors[j],
                    fontweight="bold" if better[j] else "normal")

    # Vertical rules — from below model names to bottom rule
    vx1 = x0 + col_w[0]
    vx2 = x0 + col_w[0] + col_w[1] + col_w[2]
    vx3 = vx2 + col_w[3]
    for x, lw, col in [(vx1, 0.5, "#cccccc"),
                        (vx2, 0.8, "#888888"),
                        (vx3, 0.5, "#cccccc")]:
        ax.plot([x, x], [bottom_y, rule2_y], color=col, lw=lw)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-idx",   type=int, required=True)
    parser.add_argument("--model-a",     default="vit_small_patch16_224")
    parser.add_argument("--model-b",     default="convnext_tiny")
    parser.add_argument("--graph",       default="tree",
                        choices=["tree", "ctree", "dag"])
    parser.add_argument("--true-labels",
                        help="synset_name:weight,... e.g. 'vizsla.n.01:0.5,moped.n.01:0.5'")
    parser.add_argument("--alpha",       type=float, default=0.5)
    args = parser.parse_args()

    import json, networkx as nx
    graph_files = {
        "tree":  ("tree.gpickle",  "tree_leaf_synsets.json"),
        "ctree": ("ctree.gpickle", "ctree_leaf_synsets.json"),
        "dag":   ("graph.gpickle", "leaf_synsets.json"),
    }
    gf, sf = graph_files[args.graph]
    with open(DATA_DIR / gf, "rb") as f:
        G = pickle.load(f)
    with open(DATA_DIR / sf) as f:
        leaf_info = json.load(f)
    leaf_names = leaf_info["class_idx_to_synset_name"]

    ev.OUT_DIR  = OUT_DIR
    ev.DATA_DIR = DATA_DIR
    scorer, _, leaf_full_idx = ev.build_full_graph_scorer(G, leaf_names, args.graph)
    dist_matrix = ev.precompute_leaf_distance_matrix(G, leaf_names, args.graph)
    hxe_paths   = ev.build_hxe_cache(G, leaf_names, alpha=args.alpha,
                                       graph_variant=args.graph)

    probs_a, labels = ev.load_model_outputs(args.model_a)
    probs_b, _      = ev.load_model_outputs(args.model_b)

    img  = args.image_idx
    pa   = probs_a[img]
    pb   = probs_b[img]
    true_idx = int(labels[img])
    true_name = leaf_names[true_idx]
    print(f"Image {img}: true label = {true_name} (idx {true_idx})")

    # Point ground truth
    point_dist = [(true_idx, 1.0)]

    # Soft ground truth
    if args.true_labels:
        raw = args.true_labels.split(",")
        soft_pairs = []
        for item in raw:
            synset, weight = item.rsplit(":", 1)
            synset = synset.strip()
            if synset not in leaf_names:
                raise ValueError(f"Synset '{synset}' not found in leaf_names. "
                                  f"Check spelling.")
            soft_pairs.append((leaf_names.index(synset), float(weight)))
        total = sum(w for _, w in soft_pairs)
        soft_dist = [(i, w / total) for (i, w) in soft_pairs]
        soft_label_str = args.true_labels
    else:
        soft_dist = point_dist
        soft_label_str = true_name

    print("Computing scores...")
    sp_a = score_under_distribution(pa, point_dist, dist_matrix,
                                     scorer, leaf_full_idx, hxe_paths)
    sp_b = score_under_distribution(pb, point_dist, dist_matrix,
                                     scorer, leaf_full_idx, hxe_paths)
    ss_a = score_under_distribution(pa, soft_dist,  dist_matrix,
                                     scorer, leaf_full_idx, hxe_paths)
    ss_b = score_under_distribution(pb, soft_dist,  dist_matrix,
                                     scorer, leaf_full_idx, hxe_paths)

    from qualitative_examples_dendrogram import model_display_name
    ma_name = model_display_name(args.model_a)
    mb_name = model_display_name(args.model_b)

    print_comparison_table(sp_a, sp_b, ss_a, ss_b,
                            ma_name, mb_name,
                            true_name, soft_label_str)

    out_path = OUT_DIR / "qualitative_examples" / f"ambiguous_scores_{img}_{args.model_a}_vs_{args.model_b}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    draw_table_png(sp_a, sp_b, ss_a, ss_b,
                   ma_name, mb_name,
                   true_name, soft_label_str,
                   out_path)


if __name__ == "__main__":
    main()