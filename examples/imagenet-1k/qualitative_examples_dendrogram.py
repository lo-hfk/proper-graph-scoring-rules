"""
Dendrogram-style qualitative example visualisation.

Like qualitative_examples.py but using a horizontal dendrogram layout
that scales to large leaf sets (100+ leaves) without becoming unreadable.

Layout (left to right):
  [dendrogram tree]  |  [leaf labels]  |  [model A bars]  |  [model B bars]
                                                                ^ metrics panel below

Usage:
    python qualitative_examples_dendrogram.py
    python qualitative_examples_dendrogram.py --direction b_better --n-examples 24
    python qualitative_examples_dendrogram.py \\
        --cases-csv model_validation_outputs/classically_equivalent_cases_tree.csv \\
        --direction b_better --n-examples 24
    python qualitative_examples_dendrogram.py --cumulative-threshold 0.85
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


# ---------------------------------------------------------------------------
# Shared helpers (identical to qualitative_examples.py)
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
    """
    Load a single image by global index from HF parquet shards.

    Handles unequal shard sizes correctly by reading each shard's
    actual row count from its metadata (fast -- no data is loaded).
    This is important because HF parquet splits are NOT guaranteed to
    be equal-sized: the final shard is typically smaller than the rest.
    Assuming equal shard sizes (as a naive implementation would) gives
    the correct shard/offset for early shards but silently returns the
    wrong image for any index in the later shards.
    """
    import pyarrow.parquet as pq
    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR}")

    # Build cumulative row counts from metadata only (no data loaded)
    shard_sizes = [pq.read_metadata(f).num_rows for f in files]
    cumulative  = [0]
    for s in shard_sizes:
        cumulative.append(cumulative[-1] + s)

    total_rows = cumulative[-1]
    if image_idx >= total_rows:
        raise IndexError(
            f"image_idx {image_idx} out of range "
            f"(dataset has {total_rows} rows across {len(files)} shards)"
        )

    # Find which shard contains this global index
    shard = next(i for i in range(len(files))
                 if cumulative[i] <= image_idx < cumulative[i + 1])
    within = image_idx - cumulative[shard]

    row = pq.read_table(files[shard]).slice(within, 1).to_pydict()
    img_field = row["image"][0]
    img_bytes = img_field["bytes"] if isinstance(img_field, dict) else img_field
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def per_example_metrics(probs_row, true_label_idx, leaf_names,
                         dist_matrix, scorer, leaf_full_idx,
                         hxe_paths=None):
    import math as _math
    n = len(leaf_names)
    top_idx   = np.argsort(-probs_row)
    top1_acc  = float(top_idx[0] == true_label_idx)
    top5_acc  = float(true_label_idx in top_idx[:5])
    onehot    = np.zeros(n); onehot[true_label_idx] = 1.0
    brier     = float(np.sum((probs_row - onehot) ** 2))
    log_loss  = float(-np.log(max(probs_row[true_label_idx], 1e-12)))
    # distance@k: Karthik convention, all examples, 0 when correct
    dist1 = float(dist_matrix[true_label_idx, top_idx[0]])
    dist5 = float(np.mean([dist_matrix[true_label_idx, top_idx[j]]
                            for j in range(5)]))
    n_full    = scorer.size
    p_full    = np.zeros((1, n_full))
    p_full[0, leaf_full_idx] = probs_row
    resistance = float(scorer.resistance_score(
        p_full, np.array([leaf_full_idx[true_label_idx]]))[0])

    result = dict(top1_acc=top1_acc, top5_acc=top5_acc, brier=brier,
                  log_loss=log_loss, distance_at_1=dist1, distance_at_5=dist5,
                  resistance=resistance)

    if hxe_paths is not None:
        path_info = hxe_paths[true_label_idx]
        s = 0.0
        for sub_idx, w in path_info:
            P_j = probs_row[sub_idx].sum()
            s -= w * _math.log(max(P_j, 1e-12))
        result["hxe"] = s

    return result


MODEL_DISPLAY_NAMES = {
    "convnext_tiny":          "ConvNeXt-Tiny",
    "swin_tiny":              "Swin-Tiny",
    "resnet50_timm":          "ResNet-50",
    "vit_small_patch16_224":  "ViT-S/16",
    "efficientnet_b2":        "EfficientNet-B2",
    "efficientnet_b0":        "EfficientNet-B0",
    "mobilenet_v3_large":     "MobileNetV3-L",
    "resnet18":               "ResNet-18",
    "shufflenet_v2_x1_0":     "ShuffleNet-V2",
    "mobilenet_v3_small":     "MobileNetV3-S",
}


def model_display_name(name):
    return MODEL_DISPLAY_NAMES.get(name, name)


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
    tree_u    = tree.to_undirected()
    steiner   = set(nodes_to_span)
    nodes_lst = list(nodes_to_span)
    for i in range(len(nodes_lst)):
        for j in range(i + 1, len(nodes_lst)):
            try:
                steiner.update(nx.shortest_path(tree_u, nodes_lst[i], nodes_lst[j]))
            except nx.NetworkXNoPath:
                pass
    return tree.subgraph(steiner).copy()


# ---------------------------------------------------------------------------
# Dendrogram layout
# ---------------------------------------------------------------------------
def dendrogram_layout(subtree, root):
    """
    Horizontal dendrogram layout.
    x = depth from root  (root leftmost, leaves rightmost)
    y = DFS order of leaves, equally spaced; internal nodes at mean-y of children.

    Returns:
      pos : dict  node -> (x, y)   -- for drawing edges and nodes
      leaf_order : list of leaf node names top-to-bottom (increasing y)
    """
    tree_u = subtree.to_undirected()
    depths = nx.single_source_shortest_path_length(tree_u, root)

    # children of each node (predecessors in child->parent DiGraph)
    children = {n: list(subtree.predecessors(n)) for n in subtree.nodes()}
    leaf_set = set(n for n in subtree.nodes() if not children[n])

    # Assign y by DFS leaf order
    y_pos  = {}
    counter = [0]

    def assign_y(node):
        kids = sorted(children[node])   # sort for reproducible left-right order
        if not kids:                    # leaf
            y_pos[node] = counter[0]
            counter[0] += 1
        else:
            for k in kids:
                assign_y(k)
            y_pos[node] = np.mean([y_pos[k] for k in kids])

    assign_y(root)

    pos = {n: (depths[n], y_pos[n]) for n in subtree.nodes()}
    leaf_order = sorted(leaf_set, key=lambda n: y_pos[n])
    return pos, leaf_order


def draw_dendrogram_edges(ax, subtree, pos, children):
    """
    Draw L-shaped dendrogram connections:
      vertical segment at parent's x connecting all children's y positions,
      horizontal segment from each child's y to the child's x position.
    """
    for node in subtree.nodes():
        kids = children[node]
        if not kids:
            continue
        px, py = pos[node]
        ys = [pos[k][1] for k in kids]
        # Vertical bar at parent x
        ax.plot([px, px], [min(ys), max(ys)],
                color="#bbbbbb", lw=0.8, zorder=1)
        # Horizontal arm to each child
        for k in kids:
            cx, cy = pos[k]
            ax.plot([px, cx], [cy, cy],
                    color="#bbbbbb", lw=0.8, zorder=1)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def draw_dendrogram_figure(subtree, pos, leaf_order,
                            true_label, top1_a, top1_b,
                            probs_a, probs_b,
                            leaf_names, leaf_labels,
                            model_a_name, model_b_name,
                            metrics_a, metrics_b,
                            image_idx, out_path,
                            full_tree_root_depth=0,
                            full_tree_max_depth=18):
    """
    Horizontal dendrogram with probability bars.

    full_tree_root_depth : depth of subtree root in the full tree
                           (so x=0 always corresponds to the imaginary full root)
    full_tree_max_depth  : max depth of any leaf in the full tree
                           (leaves always drawn at the rightmost x position)
    """
    children  = {n: list(subtree.predecessors(n)) for n in subtree.nodes()}
    n_leaves  = len(leaf_order)
    font_size    = 8.5
    bar_label_fs = 7.5

    color_a   = "#1565C0"
    color_b   = "#BF360C"
    bar_alpha = 0.80

    # Fixed x layout — constant across all images
    x_scale     = 1.0
    tree_width  = full_tree_max_depth * x_scale
    label_start = tree_width + 0.4
    label_width = 3.5
    bar_gap     = 0.6
    bar_A_start = label_start + label_width + bar_gap
    bar_width   = 4.0
    bar_B_start = bar_A_start + bar_width + bar_gap
    total_x     = bar_B_start + bar_width + 0.8

    # Fixed row spacing — constant across all images
    # Leaves are at integer y positions 0..n_leaves-1 in data coords.
    # We fix inches_per_leaf so that one data unit always maps to the same height.
    inches_per_leaf = 0.22        # fixed; figure grows with n_leaves
    bar_h_data      = 0.50        # bar height in data units (fraction of row spacing)

    # (other) row: fixed gap below y=0
    other_gap = 1.5               # in data units = ~1.5 * inches_per_leaf gap
    y_other   = -other_gap
    y_sep     = -other_gap / 2

    def tx(node):
        return (pos[node][0] + full_tree_root_depth) * x_scale

    def ty(node):
        return pos[node][1]

    # Figure height: n_leaves rows + gap + (other) row + margins
    # ylim will be [y_other - 0.8, n_leaves - 0.5 + 0.8]
    y_span       = (n_leaves - 0.5 + 0.8) - (y_other - 0.8)
    fig_h_inches = y_span * inches_per_leaf
    fig_w_inches = 14

    fig = plt.figure(figsize=(fig_w_inches, fig_h_inches))
    ax  = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_axis_off()

    # Dendrogram edges
    for node in subtree.nodes():
        kids = children[node]
        if not kids: continue
        px, ys = tx(node), [ty(k) for k in kids]
        ax.plot([px, px], [min(ys), max(ys)], color="#bbbbbb", lw=0.7, zorder=1)
        for k in kids:
            ax.plot([px, tx(k)], [ty(k), ty(k)], color="#bbbbbb", lw=0.7, zorder=1)

    # Internal nodes
    for node in subtree.nodes():
        if children[node]:
            ax.plot(tx(node), ty(node), "o", color="#e0e0e0",
                    markersize=4, markeredgecolor="#aaaaaa", lw=0.5, zorder=2)

    # Bar scaling
    shown_idx = [leaf_names.index(n) for n in leaf_order]
    pa_shown  = sum(probs_a[i] for i in shown_idx)
    pb_shown  = sum(probs_b[i] for i in shown_idx)
    pa_other  = max(0.0, 1.0 - pa_shown)
    pb_other  = max(0.0, 1.0 - pb_shown)
    all_p     = ([probs_a[i] for i in shown_idx] +
                 [probs_b[i] for i in shown_idx] +
                 [pa_other, pb_other])
    global_max = max(all_p) if all_p else 1.0

    # Reserve space for the label "0.000" to the right of every bar.
    # We do this by normalising so the widest bar only occupies (bar_width - label_room).
    label_room = 0.55   # data units reserved for the label to the right
    usable_bar = bar_width - label_room

    shown_idx = [leaf_names.index(n) for n in leaf_order]
    pa_shown  = sum(probs_a[i] for i in shown_idx)
    pb_shown  = sum(probs_b[i] for i in shown_idx)
    pa_other  = max(0.0, 1.0 - pa_shown)
    pb_other  = max(0.0, 1.0 - pb_shown)
    all_p     = ([probs_a[i] for i in shown_idx] +
                 [probs_b[i] for i in shown_idx] +
                 [pa_other, pb_other])
    raw_max   = max(all_p) if all_p else 1.0
    # Scale so that raw_max maps to usable_bar
    global_max = raw_max / usable_bar   # divide by this to get bar width in data units

    ms = 6

    for leaf in leaf_order:
        y    = ty(leaf)
        lx   = tx(leaf)
        lidx = leaf_names.index(leaf)
        p_a  = probs_a[lidx]
        p_b  = probs_b[lidx]
        is_true = (leaf == true_label)

        ax.plot(lx, y, "s", color="#f0f0f0", markersize=ms,
                markeredgecolor="#aaaaaa", markeredgewidth=0.5, zorder=3)

        label = leaf_labels[lidx].replace("_", " ")
        fw    = "bold" if is_true else "normal"
        ax.text(label_start, y, label, ha="left", va="center",
                fontsize=font_size, fontweight=fw, color="#111111",
                zorder=4, clip_on=True)

        bw_a = p_a / global_max
        ax.barh(y, bw_a, height=bar_h_data, left=bar_A_start,
                color=color_a, alpha=bar_alpha, zorder=3)
        if p_a >= 0.005:
            ax.text(bar_A_start + bw_a + 0.07, y, f"{p_a:.3f}",
                    ha="left", va="center", fontsize=bar_label_fs,
                    color=color_a, fontweight="bold", zorder=5)

        bw_b = p_b / global_max
        ax.barh(y, bw_b, height=bar_h_data, left=bar_B_start,
                color=color_b, alpha=bar_alpha, zorder=3)
        if p_b >= 0.005:
            ax.text(bar_B_start + bw_b + 0.07, y, f"{p_b:.3f}",
                    ha="left", va="center", fontsize=bar_label_fs,
                    color=color_b, fontweight="bold", zorder=5)

    # Dashed separator midway between bottom leaf and (other)
    ax.plot([label_start, total_x], [y_sep, y_sep],
            color="#cccccc", lw=0.6, linestyle="--", zorder=0)

    # (other) row
    ax.text(label_start, y_other, "(other)", ha="left", va="center",
            fontsize=font_size, color="#888888", style="italic", zorder=4)

    bw_ao = pa_other / global_max
    ax.barh(y_other, bw_ao, height=bar_h_data, left=bar_A_start,
            color=color_a, alpha=0.35, zorder=3)
    if pa_other >= 0.005:
        ax.text(bar_A_start + bw_ao + 0.07, y_other, f"{pa_other:.3f}",
                ha="left", va="center", fontsize=bar_label_fs,
                color=color_a, zorder=5)

    bw_bo = pb_other / global_max
    ax.barh(y_other, bw_bo, height=bar_h_data, left=bar_B_start,
            color=color_b, alpha=0.35, zorder=3)
    if pb_other >= 0.005:
        ax.text(bar_B_start + bw_bo + 0.07, y_other, f"{pb_other:.3f}",
                ha="left", va="center", fontsize=bar_label_fs,
                color=color_b, zorder=5)

    # Model name headers
    header_y = n_leaves - 0.5 + 0.7
    ax.text(bar_A_start + bar_width / 2, header_y, model_a_name,
            ha="center", va="bottom", fontsize=font_size,
            color=color_a, fontweight="bold")
    ax.text(bar_B_start + bar_width / 2, header_y, model_b_name,
            ha="center", va="bottom", fontsize=font_size,
            color=color_b, fontweight="bold")

    ax.axvline(bar_A_start - bar_gap / 2, color="#dddddd", lw=0.5, zorder=0)
    ax.set_xlim(-0.3, total_x)
    ax.set_ylim(y_other - 0.8, n_leaves - 0.5 + 0.8)

    true_name_short = true_label.split(".")[0].replace("_", " ")
    ax.set_title(f"Image {image_idx}  \u2014  {true_name_short}",
                 fontsize=12, pad=6, loc="left", fontweight="bold")

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")
def draw_metrics_table(metrics_a, metrics_b, model_a_name, model_b_name,
                        out_path):
    """Standalone metrics comparison table as a PNG."""
    metric_rows = [
        ("Top-1 error",      "top1_acc"),
        ("Top-5 error",      "top5_acc"),
        ("distance@1",       "distance_at_1"),
        ("distance@5",       "distance_at_5"),
        ("Log-loss",         "log_loss"),
        ("HXE (\u03b1=0.5)", "hxe"),
        ("Brier score",      "brier"),
        ("Resistance score", "resistance"),
    ]

    def to_error(d):
        out = dict(d)
        for k in ["top1_acc", "top5_acc"]:
            if k in out:
                out[k] = 1.0 - out[k]
        return out

    ma = to_error(metrics_a)
    mb = to_error(metrics_b)

    color_a = "#1565C0"
    color_b = "#BF360C"

    n_rows = len(metric_rows)
    row_h  = 0.45
    col_w  = [2.2, 1.8, 1.8]   # name, model_a, model_b
    pad    = 0.15
    fig_w  = sum(col_w) + 2 * pad
    fig_h  = row_h * (n_rows + 1) + 0.15

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    x0 = pad
    cx = [x0 + col_w[0] / 2,
          x0 + col_w[0] + col_w[1] / 2,
          x0 + col_w[0] + col_w[1] + col_w[2] / 2]

    header_y  = fig_h - row_h / 2
    header_bl = fig_h - row_h        # y of header bottom rule
    table_bl  = 0.08                 # y of bottom rule

    def row_y(i):
        return fig_h - row_h * (i + 1.5)

    # Header
    ax.text(cx[1], header_y, model_a_name, ha="center", va="center",
            fontsize=8, fontweight="bold", color=color_a)
    ax.text(cx[2], header_y, model_b_name, ha="center", va="center",
            fontsize=8, fontweight="bold", color=color_b)

    # Horizontal rules
    ax.axhline(header_bl, color="black", lw=0.8)
    ax.axhline(table_bl,  color="black", lw=0.8)

    # Data rows
    for i, (label, key) in enumerate(metric_rows):
        y = row_y(i)
        va_val = ma.get(key, float("nan"))
        vb_val = mb.get(key, float("nan"))
        a_better = va_val < vb_val
        b_better = vb_val < va_val

        if i % 2 == 0:
            ax.add_patch(plt.Rectangle(
                (0, y - row_h / 2), fig_w, row_h,
                facecolor="#f4f4f4", edgecolor="none", zorder=0))

        ax.text(x0 + 0.1, y, label, ha="left", va="center",
                fontsize=8, color="#222222")
        ax.text(cx[1], y, f"{va_val:.4f}", ha="center", va="center",
                fontsize=8, color=color_a,
                fontweight="bold" if a_better else "normal")
        ax.text(cx[2], y, f"{vb_val:.4f}", ha="center", va="center",
                fontsize=8, color=color_b,
                fontweight="bold" if b_better else "normal")

    # Vertical rules — only between header_bl and table_bl
    for x in [x0 + col_w[0], x0 + col_w[0] + col_w[1]]:
        ax.plot([x, x], [table_bl, header_bl], color="#cccccc", lw=0.5)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default="efficientnet_b2")
    parser.add_argument("--model-b", default="vit_small_patch16_224")
    parser.add_argument("--graph",   default="tree",
                        choices=["dag", "tree", "ctree"])
    parser.add_argument("--n-examples",           type=int,   default=5)
    parser.add_argument("--cumulative-threshold", type=float, default=0.90,
                        help="Include leaves until cumulative prob >= this (default 0.90).")
    parser.add_argument("--top-k-classes", type=int, default=None,
                        help="Include exactly the top-K classes by max(p_a, p_b). "
                             "Overrides --cumulative-threshold.")
    parser.add_argument("--prob-threshold", type=float, default=None,
                        help="Include classes where max(p_a, p_b) >= threshold. "
                             "Overrides --cumulative-threshold.")
    parser.add_argument("--sort-by", default="percentile",
                        choices=["raw", "zscore", "percentile"])
    parser.add_argument("--min-resistance-gap-pct", type=float, default=3.0)
    parser.add_argument("--direction", default="any",
                        choices=["any", "a_better", "b_better"])
    parser.add_argument("--cases-csv", default=None,
                        help="Pre-filtered cases CSV (output of "
                             "find_classically_equivalent_cases.py or "
                             "find_entropy_controlled_cases.py). "
                             "If omitted, uses same_top1_cases_<graph>.csv.")
    parser.add_argument("--image-indices", default=None,
                        help="Comma-separated image indices to visualise directly, "
                             "bypassing any CSV. E.g. --image-indices 12345,23456")
    args = parser.parse_args()

    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tree and leaf info...")
    tree, leaf_names, leaf_labels = load_tree()

    print("Loading graph scorer and distance matrix...")
    import evaluate as ev
    ev.OUT_DIR  = OUT_DIR
    ev.DATA_DIR = GRAPH_DIR
    G, leaf_info = ev.load_graph_and_leaf_info(args.graph)
    scorer, _, leaf_full_idx = ev.build_full_graph_scorer(G, leaf_names, args.graph)
    dist_matrix = ev.precompute_leaf_distance_matrix(G, leaf_names, args.graph)

    hxe_alpha = 0.5
    print(f"Loading HXE cache (alpha={hxe_alpha})...")
    hxe_paths = ev.build_hxe_cache(G, leaf_names, alpha=hxe_alpha, graph_variant=args.graph)

    print(f"Loading model outputs: {args.model_a}, {args.model_b}...")
    probs_a, labels_a = load_probs(args.model_a)
    probs_b, labels_b = load_probs(args.model_b)
    assert np.array_equal(labels_a, labels_b), "Label mismatch!"

    # Build cases dataframe
    if args.image_indices:
        # Direct image index mode — bypass CSV entirely
        indices = [int(x.strip()) for x in args.image_indices.split(",")]
        df = pd.DataFrame({
            "image_idx":  indices,
            "true_label": [int(labels_a[i]) for i in indices],
        })
        is_same_top1_schema = False
        selected = df
        print(f"Direct mode: visualising {len(selected)} specified images")
    else:
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

        # Detect which CSV schema we have:
        is_same_top1_schema = "model_A" in df.columns

        if is_same_top1_schema:
            mask = (
                ((df["model_A"] == args.model_a) & (df["model_B"] == args.model_b)) |
                ((df["model_A"] == args.model_b) & (df["model_B"] == args.model_a))
            )
            df = df[mask].copy().reset_index(drop=True)
            print(f"Cases found for this pair: {len(df)}")

            swap = df["model_A"] == args.model_b
            for pair in [("model_A", "model_B"), ("resistance_A", "resistance_B"),
                         ("resistance_A_zscore", "resistance_B_zscore"),
                         ("resistance_A_pct", "resistance_B_pct")]:
                if pair[0] in df.columns:
                    df.loc[swap, list(pair)] = df.loc[swap, list(pair)].values[:, ::-1]

            df["res_gap_pct"] = (
                (df["resistance_B"] - df["resistance_A"]).abs()
                / df[["resistance_A", "resistance_B"]].min(axis=1) * 100
            )
            df = df[df["res_gap_pct"] >= args.min_resistance_gap_pct].copy()

            if args.direction == "b_better":
                df = df[df["resistance_B"] < df["resistance_A"]].copy()
            elif args.direction == "a_better":
                df = df[df["resistance_A"] < df["resistance_B"]].copy()

            sort_col = {"raw": "resistance_abs_diff", "zscore": "resistance_zscore_abs_diff",
                        "percentile": "resistance_pct_abs_diff"}.get(args.sort_by, "resistance_pct_abs_diff")
            if sort_col not in df.columns:
                sort_col = "resistance_abs_diff"
            if sort_col in df.columns:
                df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        else:
            print(f"Pre-filtered CSV: {len(df)} cases (no model_A/B filtering applied)")
            if "ge_improvement_gap" in df.columns:
                df = df.sort_values("ge_improvement_gap", ascending=False).reset_index(drop=True)
            elif "pct_ge_gap" in df.columns:
                df = df.sort_values("pct_ge_gap", ascending=False).reset_index(drop=True)

        selected = df.head(args.n_examples)
    print(f"\nVisualising {len(selected)} examples...\n")

    for rank, (_, row) in enumerate(selected.iterrows(), 1):
        img_idx  = int(row["image_idx"])
        true_idx = int(row["true_label"])
        true_name = leaf_names[true_idx]

        # Determine the "shared top-1" label to mark on the figure.
        # In same_top1 schema it's the shared wrong prediction;
        # in ge_contrast schema we use top1_A (model A's prediction)
        # as the primary marker, and also mark top1_B if different.
        if is_same_top1_schema and "shared_top1_pred" in row.index:
            top1_idx  = int(row["shared_top1_pred"])
            top1_name = leaf_names[top1_idx]
        elif "top1_A" in row.index:
            top1_idx  = int(row["top1_A"])
            top1_name = leaf_names[top1_idx]
        else:
            top1_idx  = int(probs_a[img_idx].argmax())
            top1_name = leaf_names[top1_idx]

        print(f"[{rank}/{len(selected)}] image {img_idx}")
        print(f"  True: {leaf_labels[true_idx]}")
        print(f"  Top-1 marker: {leaf_labels[top1_idx]}"
              + (" (shared wrong top-1)" if is_same_top1_schema else
                 f" ({args.model_a} prediction)"))

        p_a = probs_a[img_idx]
        p_b = probs_b[img_idx]

        # Compute both models' top-1 predictions independently
        top1_idx_a = int(p_a.argmax())
        top1_idx_b = int(p_b.argmax())
        top1_name_a = leaf_names[top1_idx_a]
        top1_name_b = leaf_names[top1_idx_b]

        print(f"  {args.model_a} top-1: {leaf_labels[top1_idx_a]}"
              + (" (correct)" if top1_idx_a == true_idx else " (wrong)"))
        print(f"  {args.model_b} top-1: {leaf_labels[top1_idx_b]}"
              + (" (correct)" if top1_idx_b == true_idx else " (wrong)"))

        ma = per_example_metrics(p_a, true_idx, leaf_names,
                                  dist_matrix, scorer, leaf_full_idx,
                                  hxe_paths=hxe_paths)
        mb = per_example_metrics(p_b, true_idx, leaf_names,
                                  dist_matrix, scorer, leaf_full_idx,
                                  hxe_paths=hxe_paths)
        print(f"  {args.model_a}: res={ma['resistance']:.1f}  brier={ma['brier']:.4f}")
        print(f"  {args.model_b}: res={mb['resistance']:.1f}  brier={mb['brier']:.4f}")

        # Build subtree: include both models' top-1 nodes
        # Select leaves to show
        if args.top_k_classes is not None:
            # Top-K by max(p_a, p_b)
            max_p = {n: max(p_a[leaf_names.index(n)], p_b[leaf_names.index(n)])
                     for n in leaf_names}
            top_k = sorted(max_p, key=max_p.get, reverse=True)[:args.top_k_classes]
            leaves_a = leaves_b = top_k
        elif args.prob_threshold is not None:
            # All classes where max(p_a, p_b) >= threshold
            leaves_a = [n for n in leaf_names
                        if max(p_a[leaf_names.index(n)],
                               p_b[leaf_names.index(n)]) >= args.prob_threshold]
            leaves_b = leaves_a
        else:
            leaves_a = leaves_for_cumulative_threshold(p_a, leaf_names, args.cumulative_threshold)
            leaves_b = leaves_for_cumulative_threshold(p_b, leaf_names, args.cumulative_threshold)
        span = (set(leaves_a) | set(leaves_b)
                | {true_name, top1_name_a, top1_name_b})
        print(f"  Leaf union: {len(span)}")

        subtree = minimal_subtree(tree, span)
        sub_root = [n for n in subtree.nodes() if subtree.out_degree(n) == 0]
        if not sub_root:
            print("  ERROR: no root, skipping"); continue
        pos, leaf_order = dendrogram_layout(subtree, sub_root[0])
        print(f"  Subtree: {subtree.number_of_nodes()} nodes, "
              f"{len(leaf_order)} leaves")

        # Save image
        try:
            img = load_image_from_parquet(img_idx)
            img_path = EXAMPLE_DIR / f"image_{img_idx:05d}.png"
            img.save(img_path)
            print(f"  Image: {img_path.name}")
        except Exception as e:
            print(f"  WARNING: image load failed: {e}")

        # Save dendrogram figure
        stub     = f"{args.model_a}_vs_{args.model_b}"
        fig_path = EXAMPLE_DIR / f"dendro_{img_idx:05d}_{stub}.png"

        # Compute absolute depth of subtree root and max depth of full tree
        full_tree_u   = tree.to_undirected()
        full_root     = [n for n in tree.nodes() if tree.out_degree(n) == 0][0]
        abs_depths    = nx.single_source_shortest_path_length(full_tree_u, full_root)
        sub_root_depth  = abs_depths.get(sub_root[0], 0)
        full_tree_max_d = max(abs_depths.values())

        draw_dendrogram_figure(
            subtree=subtree, pos=pos, leaf_order=leaf_order,
            true_label=true_name,
            top1_a=top1_name_a,
            top1_b=top1_name_b,
            probs_a=p_a, probs_b=p_b,
            leaf_names=leaf_names, leaf_labels=leaf_labels,
            model_a_name=model_display_name(args.model_a),
            model_b_name=model_display_name(args.model_b),
            metrics_a=ma, metrics_b=mb,
            image_idx=img_idx, out_path=fig_path,
            full_tree_root_depth=sub_root_depth,
            full_tree_max_depth=full_tree_max_d,
        )
        # Separate metrics table PNG
        table_path = EXAMPLE_DIR / fig_path.name.replace(".png", "_scores.png")
        draw_metrics_table(
            metrics_a=ma, metrics_b=mb,
            model_a_name=model_display_name(args.model_a),
            model_b_name=model_display_name(args.model_b),
            out_path=table_path,
        )
        print()

    print(f"Done. Output in: {EXAMPLE_DIR}/")


if __name__ == "__main__":
    main()