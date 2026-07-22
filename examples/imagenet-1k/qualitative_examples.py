"""
Step 8: Qualitative example visualisation.

For each selected validation image, produces two output files:
  qualitative_examples/image_<idx>.png
  qualitative_examples/subtree_<idx>_<modelA>_vs_<modelB>.png

The subtree figure shows:
  - The local WordNet subtree connecting all leaves in the 90%-mass coverage
    of either model, the true label, and the shared wrong top-1.
  - Probability bars on each leaf node, on a SHARED ABSOLUTE SCALE across
    both models -- bar height = actual predicted probability.
  - A metrics panel below the tree showing top-1, top-5, Brier, log-loss,
    mistake severity, and resistance for both models on this image.
  - Internal nodes drawn as circles with NO labels (they clutter the figure).

Usage:
    python qualitative_examples.py
    python qualitative_examples.py --n-examples 8 --cumulative-threshold 0.85
    python qualitative_examples.py --direction b_better
    python qualitative_examples.py --model-a efficientnet_b2 --model-b vit_small_patch16_224
"""

import argparse
import io
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

GRAPH_DIR   = Path("./wordnet_graph")
DATA_DIR    = Path("C:/Users/ma1201204/Documents/Data/imagenet-1k")
OUT_DIR     = Path("./model_validation_outputs")
EXAMPLE_DIR = OUT_DIR / "qualitative_examples"

MAX_SUBTREE_LEAVES = 20  # trim if the union grows too large


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tree():
    with open(GRAPH_DIR / "tree.gpickle", "rb") as f:
        tree = pickle.load(f)
    with open(GRAPH_DIR / "tree_leaf_synsets.json") as f:
        info = json.load(f)
    return tree, info["class_idx_to_synset_name"], info["class_idx_to_label"]


def load_probs(model_name):
    probs  = np.load(OUT_DIR / f"probs_{model_name}.npy")
    labels = np.load(OUT_DIR / f"true_labels_{model_name}.npy")
    return probs, labels


def load_image_from_parquet(image_idx):
    import pyarrow.parquet as pq
    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR}")
    shard_sizes = [pq.read_metadata(f).num_rows for f in files]
    cumulative  = [0]
    for s in shard_sizes:
        cumulative.append(cumulative[-1] + s)
    if image_idx >= cumulative[-1]:
        raise IndexError(f"image_idx {image_idx} out of range "
                         f"(dataset has {cumulative[-1]} rows)")
    shard  = next(i for i in range(len(files))
                  if cumulative[i] <= image_idx < cumulative[i + 1])
    within = image_idx - cumulative[shard]
    row = pq.read_table(files[shard]).slice(within, 1).to_pydict()
    img_field = row["image"][0]
    img_bytes = img_field["bytes"] if isinstance(img_field, dict) else img_field
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# ---------------------------------------------------------------------------
# Per-example metrics (flat ones can be computed from saved probs+labels)
# ---------------------------------------------------------------------------
def per_example_metrics(probs_row, true_label_idx, model_name,
                         leaf_names, dist_matrix, scorer, leaf_full_idx):
    """Compute all metrics for a single image."""
    n_leaves = len(leaf_names)
    top_indices = np.argsort(-probs_row)

    top1_acc = float(top_indices[0] == true_label_idx)
    top5_acc = float(true_label_idx in top_indices[:5])

    # Brier
    onehot = np.zeros(n_leaves); onehot[true_label_idx] = 1.0
    brier = float(np.sum((probs_row - onehot) ** 2))

    # Log-loss
    log_loss = float(-np.log(max(probs_row[true_label_idx], 1e-12)))

    # Mistake severity (only defined when wrong)
    pred_idx = top_indices[0]
    if pred_idx != true_label_idx:
        severity = float(dist_matrix[true_label_idx, pred_idx])
    else:
        severity = 0.0

    # Resistance score
    n_full = scorer.size
    p_full = np.zeros((1, n_full))
    p_full[0, leaf_full_idx] = probs_row
    true_full = np.array([leaf_full_idx[true_label_idx]])
    resistance = float(scorer.resistance_score(p_full, true_full)[0])

    return {
        "top1_acc":        top1_acc,
        "top5_acc":        top5_acc,
        "brier":           brier,
        "log_loss":        log_loss,
        "mistake_severity": severity,
        "resistance":      resistance,
    }


# ---------------------------------------------------------------------------
# Subtree construction
# ---------------------------------------------------------------------------
def leaves_for_cumulative_threshold(probs_row, leaf_names, threshold):
    order = np.argsort(-probs_row)
    cumsum, selected = 0.0, []
    for idx in order:
        selected.append(leaf_names[idx])
        cumsum += probs_row[idx]
        if cumsum >= threshold:
            break
    return selected


def minimal_subtree(tree, nodes_to_span):
    tree_u = tree.to_undirected()
    steiner = set(nodes_to_span)
    nodes_list = list(nodes_to_span)
    for i in range(len(nodes_list)):
        for j in range(i + 1, len(nodes_list)):
            try:
                steiner.update(nx.shortest_path(tree_u, nodes_list[i], nodes_list[j]))
            except nx.NetworkXNoPath:
                pass
    return tree.subgraph(steiner).copy()


def subtree_layout(subtree, root):
    tree_u  = subtree.to_undirected()
    depths  = nx.single_source_shortest_path_length(tree_u, root)
    max_dep = max(depths.values()) if depths else 1
    children = {n: list(subtree.predecessors(n)) for n in subtree.nodes()}

    x_pos, counter = {}, [0]

    def assign_x(node):
        kids = children[node]
        if not kids:
            x_pos[node] = counter[0]; counter[0] += 1
        else:
            for k in sorted(kids): assign_x(k)
            x_pos[node] = np.mean([x_pos[k] for k in kids])

    assign_x(root)
    return {n: (x_pos[n], max_dep - depths[n]) for n in subtree.nodes()}


# ---------------------------------------------------------------------------
# Figure: subtree + metrics panel
# ---------------------------------------------------------------------------
def draw_subtree_figure(subtree, pos,
                         true_label, shared_top1,
                         probs_a, probs_b,
                         leaf_names, leaf_labels,
                         model_a_name, model_b_name,
                         metrics_a, metrics_b,
                         image_idx, out_path):
    """
    Subtree figure with shared-scale probability bars and metrics panel.

    Bar heights use a SINGLE SHARED ABSOLUTE SCALE: the tallest bar across
    all leaves and both models fills the available slot height.
    This makes bars directly visually comparable -- a taller bar means
    higher predicted probability, regardless of which model or leaf.
    Absolute probabilities are printed above each bar.

    Internal nodes are drawn as plain circles with NO text labels.
    """
    leaf_set = set(leaf_names)
    subtree_leaves   = sorted(
        [n for n in subtree.nodes()
         if n in leaf_set and subtree.in_degree(n) == 0],
        key=lambda n: pos[n][0]
    )
    subtree_internal = [n for n in subtree.nodes() if n not in leaf_set]

    n_leaves = len(subtree_leaves)
    fig_w    = max(13, n_leaves * 1.7)

    # Layout: tree panel (top ~72%), metrics panel (bottom ~22%)
    fig = plt.figure(figsize=(fig_w, 10))
    ax_tree    = fig.add_axes([0.01, 0.28, 0.98, 0.68])
    ax_metrics = fig.add_axes([0.01, 0.02, 0.98, 0.22])
    ax_tree.set_axis_off()
    ax_metrics.set_axis_off()

    # Edges
    for u, v in subtree.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ax_tree.plot([x0, x1], [y0, y1], color="#cccccc", lw=1.2, zorder=1)

    # Internal nodes: dot only, NO label
    for node in subtree_internal:
        x, y = pos[node]
        ax_tree.plot(x, y, "o", color="#e0e0e0", markersize=9,
                     markeredgecolor="#aaaaaa", lw=0.8, zorder=2)

    # SHARED ABSOLUTE SCALE for bars
    all_p = []
    for node in subtree_leaves:
        idx = leaf_names.index(node)
        all_p.extend([probs_a[idx], probs_b[idx]])
    global_max = max(all_p) if all_p else 1.0
    bar_scale  = 0.82  # height in y-units when p == global_max

    x_vals = sorted(set(pos[n][0] for n in subtree_leaves))
    bar_w  = min(0.40, 0.75 * (x_vals[1] - x_vals[0])) if len(x_vals) > 1 else 0.38

    color_a, color_b = "#1565C0", "#E65100"  # deep blue / deep orange

    for node in subtree_leaves:
        x, y   = pos[node]
        idx    = leaf_names.index(node)
        p_a, p_b = probs_a[idx], probs_b[idx]
        h_a    = (p_a / global_max) * bar_scale
        h_b    = (p_b / global_max) * bar_scale

        ax_tree.bar(x - bar_w / 2, h_a, width=bar_w * 0.46,
                    bottom=y, color=color_a, alpha=0.82, zorder=3)
        ax_tree.bar(x + bar_w / 2, h_b, width=bar_w * 0.46,
                    bottom=y, color=color_b, alpha=0.82, zorder=3)

        fs = 6.5
        if p_a >= 0.004:
            ax_tree.text(x - bar_w / 2, y + h_a + 0.02, f"{p_a:.3f}",
                         ha="center", va="bottom", fontsize=fs,
                         color=color_a, fontweight="bold", zorder=5)
        if p_b >= 0.004:
            ax_tree.text(x + bar_w / 2, y + h_b + 0.02, f"{p_b:.3f}",
                         ha="center", va="bottom", fontsize=fs,
                         color=color_b, fontweight="bold", zorder=5)

        is_true = (node == true_label)
        is_top1 = (node == shared_top1)
        fill = "#FFF176" if is_true else ("#FFCDD2" if is_top1 else "#f5f5f5")
        edge = "#F9A825" if is_true else ("#c62828" if is_top1 else "#aaaaaa")
        ax_tree.plot(x, y, "s", color=fill, markersize=13,
                     markeredgecolor=edge, markeredgewidth=1.5, zorder=2)

        label  = leaf_labels[idx].replace("_", " ")
        suffix = " ★" if is_true else (" ✗" if is_top1 else "")
        ax_tree.text(x, y - 0.14, label + suffix,
                     ha="center", va="top", fontsize=6.5, rotation=30,
                     fontweight="bold" if (is_true or is_top1) else "normal",
                     color="#111111", zorder=5)

    # Legend + scale note
    winner = model_a_name if metrics_a["resistance"] < metrics_b["resistance"] else model_b_name
    legend_elements = [
        mpatches.Patch(color=color_a, label=model_a_name),
        mpatches.Patch(color=color_b, label=model_b_name),
        mpatches.Patch(color="#FFF176", label="★ true label"),
        mpatches.Patch(color="#FFCDD2", label="✗ shared wrong top-1"),
    ]
    ax_tree.legend(handles=legend_elements, loc="upper right",
                   fontsize=8, framealpha=0.95)
    ax_tree.text(0.01, 0.01,
                 f"Bars: shared absolute scale  (max p = {global_max:.3f})",
                 transform=ax_tree.transAxes, fontsize=6.5,
                 color="#777777", va="bottom")

    true_name = true_label.split(".")[0].replace("_", " ")
    top1_name = shared_top1.split(".")[0].replace("_", " ")
    ax_tree.set_title(
        f"Image {image_idx}  —  Local WordNet subtree\n"
        f"Both models predict: '{top1_name}' (WRONG)    "
        f"True label: '{true_name}'\n"
        f"Better resistance score: {winner}",
        fontsize=9, pad=8
    )

    # Divider
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.265, 0.265],
                               transform=fig.transFigure,
                               color="#cccccc", lw=0.8))

    # Metrics panel
    metric_rows = [
        ("Top-1 accuracy",    "top1_acc",         True),
        ("Top-5 accuracy",    "top5_acc",         True),
        ("Brier score",       "brier",            False),
        ("Log-loss",          "log_loss",         False),
        ("Mistake severity",  "mistake_severity", False),
        ("Resistance score",  "resistance",       False),
    ]
    col_x  = [0.05, 0.40, 0.70]   # label | model_a | model_b
    row_ys = np.linspace(0.88, 0.10, len(metric_rows))

    ax_metrics.text(col_x[0], 0.96, "Per-example metric",
                    ha="left", va="top", fontsize=8,
                    fontweight="bold", color="#333333",
                    transform=ax_metrics.transAxes)
    ax_metrics.text(col_x[1], 0.96, model_a_name,
                    ha="center", va="top", fontsize=8,
                    fontweight="bold", color=color_a,
                    transform=ax_metrics.transAxes)
    ax_metrics.text(col_x[2], 0.96, model_b_name,
                    ha="center", va="top", fontsize=8,
                    fontweight="bold", color=color_b,
                    transform=ax_metrics.transAxes)

    for row_y, (label, key, higher_better) in zip(row_ys, metric_rows):
        val_a = metrics_a.get(key, float("nan"))
        val_b = metrics_b.get(key, float("nan"))
        a_wins = (higher_better and val_a > val_b) or (not higher_better and val_a < val_b)
        b_wins = (higher_better and val_b > val_a) or (not higher_better and val_b < val_a)

        ax_metrics.text(col_x[0], row_y, label,
                        ha="left", va="center", fontsize=7.5, color="#333333",
                        transform=ax_metrics.transAxes)
        ax_metrics.text(col_x[1], row_y, f"{val_a:.4f}",
                        ha="center", va="center", fontsize=7.5,
                        color=color_a, fontweight="bold" if a_wins else "normal",
                        transform=ax_metrics.transAxes)
        ax_metrics.text(col_x[2], row_y, f"{val_b:.4f}",
                        ha="center", va="center", fontsize=7.5,
                        color=color_b, fontweight="bold" if b_wins else "normal",
                        transform=ax_metrics.transAxes)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default="efficientnet_b2")
    parser.add_argument("--model-b", default="vit_small_patch16_224")
    parser.add_argument("--graph",   default="tree",
                        choices=["dag", "tree", "ctree"])
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--cumulative-threshold", type=float, default=0.90)
    parser.add_argument("--sort-by", default="percentile",
                        choices=["raw", "zscore", "percentile"])
    parser.add_argument("--min-resistance-gap-pct", type=float, default=3.0)
    parser.add_argument("--cases-csv", default=None,
                        help="Path to a CSV of pre-filtered cases to visualise. "
                             "If provided, the --sort-by, --min-resistance-gap-pct, "
                             "and --direction filters are still applied but the "
                             "input comes from this file instead of the default "
                             "same_top1_cases_<graph>.csv. Use this to pass in "
                             "the output of find_classically_equivalent_cases.py "
                             "or find_entropy_controlled_cases.py.")
    parser.add_argument("--direction", default="any",
                        choices=["any", "a_better", "b_better"],
                        help="any: all cases. a_better/b_better: restrict to "
                             "cases where that model has the lower resistance score.")
    args = parser.parse_args()

    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tree, leaf info, graph scorer...")
    tree, leaf_names, leaf_labels = load_tree()
    leaf_set = set(leaf_names)

    # Load the graph scorer and distance matrix (needed for per-example metrics)
    import evaluate as ev
    ev.OUT_DIR  = OUT_DIR
    ev.DATA_DIR = GRAPH_DIR
    G, leaf_info = ev.load_graph_and_leaf_info(args.graph)
    scorer, _, leaf_full_idx = ev.build_full_graph_scorer(G, leaf_names, args.graph)
    dist_matrix = ev.precompute_leaf_distance_matrix(G, leaf_names, args.graph)

    print(f"Loading model outputs...")
    probs_a, labels_a = load_probs(args.model_a)
    probs_b, labels_b = load_probs(args.model_b)
    assert np.array_equal(labels_a, labels_b), "Label mismatch!"

    # Load cases -- custom pre-filtered file or default same-top1 CSV
    if args.cases_csv:
        cases_csv = Path(args.cases_csv)
        if not cases_csv.exists():
            raise FileNotFoundError(f"--cases-csv not found: {cases_csv}")
        print(f"Using custom cases file: {cases_csv}")
    else:
        cases_csv = OUT_DIR / f"same_top1_cases_{args.graph}.csv"
        if not cases_csv.exists():
            raise FileNotFoundError(
                f"{cases_csv} not found. Run:\n"
                f"  python same_top1_analysis.py "
                f"--models {args.model_a} {args.model_b} --graph {args.graph}"
            )
    df = pd.read_csv(cases_csv)

    mask = (
        ((df["model_A"] == args.model_a) & (df["model_B"] == args.model_b)) |
        ((df["model_A"] == args.model_b) & (df["model_B"] == args.model_a))
    )
    df = df[mask].copy().reset_index(drop=True)
    print(f"Found {len(df)} same-top1 cases for this pair")

    # Normalise so model_A = args.model_a throughout
    swap = df["model_A"] == args.model_b
    for pair in [("model_A","model_B"), ("resistance_A","resistance_B"),
                 ("resistance_A_zscore","resistance_B_zscore"),
                 ("resistance_A_pct","resistance_B_pct")]:
        if pair[0] in df.columns:
            df.loc[swap, list(pair)] = df.loc[swap, list(pair)].values[:, ::-1]

    # Minimum gap filter
    df["res_gap_pct"] = (
        (df["resistance_B"] - df["resistance_A"]).abs()
        / df[["resistance_A","resistance_B"]].min(axis=1) * 100
    )
    df = df[df["res_gap_pct"] >= args.min_resistance_gap_pct].copy()

    # Direction filter
    if args.direction == "b_better":
        df = df[df["resistance_B"] < df["resistance_A"]].copy()
        print(f"Cases where {args.model_b} scores better: {len(df)}")
    elif args.direction == "a_better":
        df = df[df["resistance_A"] < df["resistance_B"]].copy()
        print(f"Cases where {args.model_a} scores better: {len(df)}")

    # Sort
    sort_col = {
        "raw":        "resistance_abs_diff",
        "zscore":     "resistance_zscore_abs_diff",
        "percentile": "resistance_pct_abs_diff",
    }.get(args.sort_by, "resistance_pct_abs_diff")
    if sort_col not in df.columns:
        sort_col = "resistance_abs_diff"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    selected = df.head(args.n_examples)
    print(f"\nVisualising {len(selected)} examples...\n")

    for rank, (_, row) in enumerate(selected.iterrows(), 1):
        img_idx         = int(row["image_idx"])
        true_idx        = int(row["true_label"])
        top1_idx        = int(row["shared_top1_pred"])
        true_name       = leaf_names[true_idx]
        top1_name       = leaf_names[top1_idx]

        print(f"[{rank}/{len(selected)}] image {img_idx}")
        print(f"  True:   {leaf_labels[true_idx]}")
        print(f"  Top-1:  {leaf_labels[top1_idx]}")

        p_a = probs_a[img_idx]
        p_b = probs_b[img_idx]

        # Per-example metrics
        ma = per_example_metrics(p_a, true_idx, args.model_a,
                                  leaf_names, dist_matrix, scorer, leaf_full_idx)
        mb = per_example_metrics(p_b, true_idx, args.model_b,
                                  leaf_names, dist_matrix, scorer, leaf_full_idx)
        print(f"  {args.model_a}: res={ma['resistance']:.1f}  brier={ma['brier']:.4f}")
        print(f"  {args.model_b}: res={mb['resistance']:.1f}  brier={mb['brier']:.4f}")

        # Build subtree
        leaves_a = leaves_for_cumulative_threshold(p_a, leaf_names, args.cumulative_threshold)
        leaves_b = leaves_for_cumulative_threshold(p_b, leaf_names, args.cumulative_threshold)
        span = set(leaves_a) | set(leaves_b) | {true_name, top1_name}
        print(f"  Leaf union: {len(span)} nodes")

        subtree = minimal_subtree(tree, span)
        sub_root = [n for n in subtree.nodes() if subtree.out_degree(n) == 0]
        if not sub_root:
            print("  ERROR: no root found, skipping"); continue
        pos = subtree_layout(subtree, sub_root[0])
        print(f"  Subtree: {subtree.number_of_nodes()} nodes")

        # Image
        try:
            img = load_image_from_parquet(img_idx)
            img_path = EXAMPLE_DIR / f"image_{img_idx:05d}.png"
            img.save(img_path)
            print(f"  Image: {img_path.name}")
        except Exception as e:
            print(f"  WARNING: image load failed: {e}")

        # Subtree figure
        fig_path = EXAMPLE_DIR / (
            f"subtree_{img_idx:05d}_{args.model_a}_vs_{args.model_b}.png"
        )
        draw_subtree_figure(
            subtree=subtree, pos=pos,
            true_label=true_name, shared_top1=top1_name,
            probs_a=p_a, probs_b=p_b,
            leaf_names=leaf_names, leaf_labels=leaf_labels,
            model_a_name=args.model_a, model_b_name=args.model_b,
            metrics_a=ma, metrics_b=mb,
            image_idx=img_idx, out_path=fig_path,
        )
        print()

    print(f"Done. Output in: {EXAMPLE_DIR}/")


if __name__ == "__main__":
    main()