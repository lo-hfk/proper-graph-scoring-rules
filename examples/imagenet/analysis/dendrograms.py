"""
Dendrogram-style qualitative example visualisation.

For a given pair of models and one or more image indices, produces:
  - figures/dendro_<idx>_<model_a>_vs_<model_b>.png           : dendrogram (publication)
  - figures/dendro_<idx>_<model_a>_vs_<model_b>_annotated.png : with internal node labels
  - figures/dendro_<idx>_<model_a>_vs_<model_b>_scores.png    : scores comparison table
  - figures/image_<idx>.png                                    : extracted image

Leaf selection (choose one; default: cumulative threshold 0.85):
  --prob-threshold T      show all classes assigned probability >= T by either model
  --top-k-classes K       show K classes with highest max(prob_model_a, prob_model_b)
  --cumulative-threshold T  show classes covering fraction T of probability mass

Soft ground truth (optional):
  --soft-gt "vizsla:0.5,moped:0.5"   produce an additional scores table under a
                                      mixed ground truth distribution

Usage
-----
    python analysis/dendrograms.py --model-a resnet50_timm --model-b swin_tiny \
        --image-indices 16763
    python analysis/dendrograms.py --model-a vit_small_patch16_224 --model-b convnext_tiny \
        --image-indices 44101 --prob-threshold 0.01 \
        --soft-gt "vizsla:0.5,moped:0.5"
"""

import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))  # so config.py can be found
from config import IMAGENET_VAL_DIR

DATA_DIR    = Path(__file__).parent.parent / "data"
TREE_DIR    = DATA_DIR / "tree"
PROBS_DIR   = DATA_DIR / "probs"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DISPLAY_NAMES = {
    "convnext_tiny":         "ConvNeXt-Tiny",
    "swin_tiny":             "Swin-Tiny",
    "resnet50_timm":         "ResNet-50",
    "vit_small_patch16_224": "ViT-S/16",
    "efficientnet_b2":       "EfficientNet-B2",
    "efficientnet_b0":       "EfficientNet-B0",
    "mobilenet_v3_large":    "MobileNetV3-L",
    "resnet18":              "ResNet-18",
    "shufflenet_v2_x1_0":   "ShuffleNet-V2",
    "mobilenet_v3_small":   "MobileNetV3-S",
}

HXE_ALPHA = 0.5

# -----------------------------------------------------------------
# Layout constants (easy to adjust)
# -----------------------------------------------------------------
TREE_FRAC  = 0.40   # fraction of total figure width for the dendrogram
LABEL_FRAC = 0.60   # remaining fraction for labels + probability bars
EDGE_COLOR = "#222222"   # near-black dendrogram edges
NODE_COLOR = "#555555"   # internal node markers


# -----------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------
def load_tree():
    with open(TREE_DIR / "tree.gpickle", "rb") as f:
        tree = pickle.load(f)
    with open(TREE_DIR / "leaf_synsets.json") as f:
        info = json.load(f)
    return tree, info["synset_name"], info["label"]


def load_probs(model_name):
    return np.load(PROBS_DIR / f"{model_name}.npy")


def load_image(image_idx):
    """Load a single image by index from HF parquet shards."""
    val_dir = Path(IMAGENET_VAL_DIR)
    files   = sorted(val_dir.glob("validation-*.parquet"))
    # read shard sizes from metadata (no data loaded)
    sizes = [pq.read_metadata(f).num_rows for f in files]
    cumulative = [0]
    for s in sizes:
        cumulative.append(cumulative[-1] + s)
    shard  = next(i for i in range(len(files))
                  if cumulative[i] <= image_idx < cumulative[i + 1])
    within = image_idx - cumulative[shard]
    row    = pq.read_table(files[shard]).slice(within, 1).to_pydict()
    img_field = row["image"][0]
    img_bytes = img_field["bytes"]
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# -----------------------------------------------------------------
# Leaf selection
# -----------------------------------------------------------------
def leaves_for_threshold(probs_row, leaf_names, threshold):
    order  = np.argsort(-probs_row)
    cumsum, selected = 0.0, []
    for idx in order:
        selected.append(leaf_names[idx])
        cumsum += probs_row[idx]
        if cumsum >= threshold:
            break
    return selected


def minimal_subtree(tree, nodes_to_span):
    tree_u  = tree.to_undirected()
    steiner = set(nodes_to_span)
    nodes_lst = list(nodes_to_span)
    for i in range(len(nodes_lst)):
        for j in range(i + 1, len(nodes_lst)):
            steiner.update(nx.shortest_path(tree_u, nodes_lst[i], nodes_lst[j]))
    return tree.subgraph(steiner).copy()


# -----------------------------------------------------------------
# Per-example metrics using HierarchicalScorer
# -----------------------------------------------------------------
def per_example_metrics(probs_row, true_idx, scorer):
    P = probs_row[np.newaxis]
    i = np.array([true_idx])
    return {
        "top1_acc":    float(1 - scorer.top_k_error(P, i, k=1)[0]),
        "top5_acc":    float(1 - scorer.top_k_error(P, i, k=5)[0]),
        "distance_at_1": float(scorer.distance_at_k(P, i, k=1)[0]),
        "distance_at_5": float(scorer.distance_at_k(P, i, k=5)[0]),
        "log_loss":    float(scorer.logarithmic_score(P, i)[0]),
        "brier":       float(scorer.brier_score(P, i)[0]),
        "hxe":         float(scorer.hxe_score(P, i, alpha=HXE_ALPHA)[0]),
        "resistance":  float(scorer.resistance_score(P, i)[0]),
    }


# -----------------------------------------------------------------
# Dendrogram layout (parent->child convention)
# -----------------------------------------------------------------
def dendrogram_layout(subtree, root):
    """
    Horizontal dendrogram layout.
    x = depth from root (root leftmost, leaves rightmost)
    y = DFS leaf order (equally spaced); internal nodes at mean-y of children.
    Returns pos dict and leaf_order list (top to bottom).
    """
    tree_u = subtree.to_undirected()
    depths = nx.single_source_shortest_path_length(tree_u, root)

    # children: successors in parent->child DiGraph
    children = {n: list(subtree.successors(n)) for n in subtree.nodes()}
    leaf_set  = {n for n in subtree.nodes() if not children[n]}

    y_pos   = {}
    counter = [0]

    def assign_y(node):
        kids = sorted(children[node])
        if not kids:
            y_pos[node] = counter[0]
            counter[0] += 1
        else:
            for k in kids:
                assign_y(k)
            y_pos[node] = np.mean([y_pos[k] for k in kids])

    assign_y(root)

    pos        = {n: (depths[n], y_pos[n]) for n in subtree.nodes()}
    leaf_order = sorted(leaf_set, key=lambda n: y_pos[n])
    return pos, leaf_order


# -----------------------------------------------------------------
# Main dendrogram figure
# -----------------------------------------------------------------
def draw_dendrogram_figure(subtree, pos, leaf_order,
                            true_label, top1_a, top1_b,
                            probs_a, probs_b,
                            leaf_names, leaf_labels,
                            model_a_name, model_b_name,
                            metrics_a, metrics_b,
                            image_idx, out_path,
                            full_tree_root_depth=0,
                            full_tree_max_depth=18,
                            annotate_nodes=False):

    children  = {n: list(subtree.successors(n)) for n in subtree.nodes()}
    n_leaves  = len(leaf_order)
    font_size    = 8.5
    bar_label_fs = 7.5
    color_a   = "#1565C0"
    color_b   = "#BF360C"
    bar_alpha = 0.80

    # Layout: total_x is the full figure width in data units.
    # TREE_FRAC of it goes to the dendrogram, LABEL_FRAC to labels + bars.
    fig_w_inches = 14
    total_x      = fig_w_inches   # 1 data unit = 1 inch
    tree_x_end   = total_x * TREE_FRAC
    x_scale      = tree_x_end / (full_tree_max_depth or 1)

    label_start  = tree_x_end + 0.4
    label_width  = 3.5
    bar_gap      = 0.6
    bar_A_start  = label_start + label_width + bar_gap
    bar_width    = (total_x * LABEL_FRAC - label_width - 3 * bar_gap) / 2
    bar_B_start  = bar_A_start + bar_width + bar_gap

    inches_per_leaf = 0.22
    bar_h_data      = 0.50
    other_gap = 1.5
    y_other   = -other_gap
    y_sep     = -other_gap / 2

    def tx(node): return (pos[node][0] + full_tree_root_depth) * x_scale
    def ty(node): return pos[node][1]

    y_span       = (n_leaves - 0.5 + 0.8) - (y_other - 0.8)
    fig = plt.figure(figsize=(fig_w_inches, y_span * inches_per_leaf))
    ax  = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_axis_off()

    # Dendrogram edges
    for node in subtree.nodes():
        kids = children[node]
        if not kids: continue
        px, ys = tx(node), [ty(k) for k in kids]
        ax.plot([px, px], [min(ys), max(ys)], color=EDGE_COLOR, lw=0.8, zorder=1)
        for k in kids:
            ax.plot([px, tx(k)], [ty(k), ty(k)], color=EDGE_COLOR, lw=0.8, zorder=1)

    # Internal nodes (and optional annotations)
    for node in subtree.nodes():
        if children[node]:
            ax.plot(tx(node), ty(node), "o", color=NODE_COLOR,
                    markersize=3, markeredgecolor=NODE_COLOR, lw=0.5, zorder=2)
            if annotate_nodes:
                label = node.split(".")[0].replace("_", " ")
                ax.text(tx(node) + 0.05, ty(node), label,
                        ha="left", va="center", fontsize=4.5,
                        color="#444444", rotation=15, zorder=3)

    # Bar scaling
    shown_idx = [leaf_names.index(n) for n in leaf_order]
    pa_other  = max(0.0, 1.0 - sum(probs_a[i] for i in shown_idx))
    pb_other  = max(0.0, 1.0 - sum(probs_b[i] for i in shown_idx))
    all_p     = ([probs_a[i] for i in shown_idx] + [probs_b[i] for i in shown_idx]
                 + [pa_other, pb_other])
    label_room = 0.55
    raw_max    = max(all_p) if all_p else 1.0
    global_max = raw_max / (bar_width - label_room)

    for leaf in leaf_order:
        y    = ty(leaf)
        lx   = tx(leaf)
        lidx = leaf_names.index(leaf)
        p_a  = probs_a[lidx]
        p_b  = probs_b[lidx]
        is_true = (leaf == true_label)

        ax.plot(lx, y, "s", color="#f0f0f0", markersize=6,
                markeredgecolor="#aaaaaa", markeredgewidth=0.5, zorder=3)

        label = leaf_labels[lidx].replace("_", " ")
        ax.text(label_start, y, label, ha="left", va="center",
                fontsize=font_size, fontweight="bold" if is_true else "normal",
                color="#111111", zorder=4, clip_on=True)

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

    # Separator and (other) row
    ax.plot([label_start, total_x], [y_sep, y_sep],
            color="#cccccc", lw=0.6, linestyle="--", zorder=0)
    ax.text(label_start, y_other, "(other)", ha="left", va="center",
            fontsize=font_size, color="#888888", style="italic", zorder=4)

    bw_ao = pa_other / global_max
    ax.barh(y_other, bw_ao, height=bar_h_data, left=bar_A_start,
            color=color_a, alpha=0.35, zorder=3)
    if pa_other >= 0.005:
        ax.text(bar_A_start + bw_ao + 0.07, y_other, f"{pa_other:.3f}",
                ha="left", va="center", fontsize=bar_label_fs, color=color_a, zorder=5)

    bw_bo = pb_other / global_max
    ax.barh(y_other, bw_bo, height=bar_h_data, left=bar_B_start,
            color=color_b, alpha=0.35, zorder=3)
    if pb_other >= 0.005:
        ax.text(bar_B_start + bw_bo + 0.07, y_other, f"{pb_other:.3f}",
                ha="left", va="center", fontsize=bar_label_fs, color=color_b, zorder=5)

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

    plt.savefig(out_path, dpi=1200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# -----------------------------------------------------------------
# Scores comparison table
# -----------------------------------------------------------------
def draw_scores_table(metrics_a, metrics_b, model_a_name, model_b_name, out_path):
    metric_rows = [
        ("Top-1 error",       "top1_acc"),
        ("Top-5 error",       "top5_acc"),
        ("Distance@1",        "distance_at_1"),
        ("Distance@5",        "distance_at_5"),
        ("Log-loss",          "log_loss"),
        (f"HXE (\u03b1={HXE_ALPHA})", "hxe"),
        ("Brier score",       "brier"),
        ("Resistance score",  "resistance"),
    ]

    def to_error(d):
        out = dict(d)
        for k in ["top1_acc", "top5_acc"]:
            if k in out:
                out[k] = 1.0 - out[k]
        return out

    ma = to_error(metrics_a)
    mb = to_error(metrics_b)
    color_a, color_b = "#1565C0", "#BF360C"

    n_rows = len(metric_rows)
    row_h  = 0.45
    col_w  = [2.2, 1.8, 1.8]
    pad    = 0.15
    fig_w  = sum(col_w) + 2 * pad
    fig_h  = row_h * (n_rows + 1) + 0.15

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis("off")

    x0 = pad
    cx = [x0 + col_w[0] / 2,
          x0 + col_w[0] + col_w[1] / 2,
          x0 + col_w[0] + col_w[1] + col_w[2] / 2]

    header_y  = fig_h - row_h / 2
    header_bl = fig_h - row_h
    table_bl  = 0.08

    def row_y(i): return fig_h - row_h * (i + 1.5)

    ax.text(cx[1], header_y, model_a_name, ha="center", va="center",
            fontsize=8, fontweight="bold", color=color_a)
    ax.text(cx[2], header_y, model_b_name, ha="center", va="center",
            fontsize=8, fontweight="bold", color=color_b)
    ax.axhline(header_bl, color="black", lw=0.8)
    ax.axhline(table_bl,  color="black", lw=0.8)

    for i, (label, key) in enumerate(metric_rows):
        y      = row_y(i)
        va_val = ma.get(key, float("nan"))
        vb_val = mb.get(key, float("nan"))

        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y - row_h / 2), fig_w, row_h,
                                        facecolor="#f7f7f7", zorder=0))

        ax.text(x0 + 0.05, y, label, ha="left", va="center", fontsize=7.5)

        for col, val, c, is_better in [
            (1, va_val, color_a, va_val <= vb_val),
            (2, vb_val, color_b, vb_val <= va_val),
        ]:
            fw = "bold" if (is_better and va_val != vb_val) else "normal"
            ax.text(cx[col], y, f"{val:.3f}", ha="center", va="center",
                    fontsize=7.5, color=c, fontweight=fw)

    plt.savefig(out_path, dpi=1200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--image-indices", required=True,
                        help="Comma-separated image indices, e.g. 16763,44101")
    parser.add_argument("--prob-threshold", type=float, default=None,
                        help="Show all classes where max(p_a,p_b) >= threshold")
    parser.add_argument("--top-k-classes", type=int, default=None,
                        help="Show top-k classes by max(p_a,p_b)")
    parser.add_argument("--cumulative-threshold", type=float, default=0.85,
                        help="Show classes covering this fraction of probability mass")
    parser.add_argument("--soft-gt", type=str, default=None,
                        help='Soft ground truth, e.g. "vizsla:0.5,moped:0.5"')
    args = parser.parse_args()

    from propnetscore import HierarchicalScorer
    print("Loading tree and scorer...")
    tree, leaf_names, leaf_labels = load_tree()
    scorer = HierarchicalScorer(tree, leaf_names)

    true_labels = np.load(PROBS_DIR / "true_labels.npy")
    probs_a = load_probs(args.model_a)
    probs_b = load_probs(args.model_b)

    model_a_name = MODEL_DISPLAY_NAMES.get(args.model_a, args.model_a)
    model_b_name = MODEL_DISPLAY_NAMES.get(args.model_b, args.model_b)

    # full tree depth info for consistent x-axis
    full_tree_u   = tree.to_undirected()
    full_root     = next(n for n in tree.nodes() if tree.in_degree(n) == 0)
    abs_depths    = nx.single_source_shortest_path_length(full_tree_u, full_root)
    full_tree_max = max(abs_depths.values())

    indices = [int(x.strip()) for x in args.image_indices.split(",")]

    for img_idx in indices:
        true_idx  = int(true_labels[img_idx])
        true_name = leaf_names[true_idx]
        p_a = probs_a[img_idx]
        p_b = probs_b[img_idx]

        print(f"\nImage {img_idx}: true={leaf_labels[true_idx]}")
        print(f"  {args.model_a} top-1: {leaf_labels[p_a.argmax()]}")
        print(f"  {args.model_b} top-1: {leaf_labels[p_b.argmax()]}")

        ma = per_example_metrics(p_a, true_idx, scorer)
        mb = per_example_metrics(p_b, true_idx, scorer)
        print(f"  resistance: {args.model_a}={ma['resistance']:.4f}, "
              f"{args.model_b}={mb['resistance']:.4f}")

        # select leaves to show
        if args.top_k_classes is not None:
            max_p = np.maximum(p_a, p_b)
            span_leaves = [leaf_names[i] for i in np.argsort(-max_p)[:args.top_k_classes]]
        elif args.prob_threshold is not None:
            max_p = np.maximum(p_a, p_b)
            span_leaves = [leaf_names[i] for i in range(len(leaf_names))
                           if max_p[i] >= args.prob_threshold]
        else:
            span_leaves = list(set(
                leaves_for_threshold(p_a, leaf_names, args.cumulative_threshold) +
                leaves_for_threshold(p_b, leaf_names, args.cumulative_threshold)
            ))

        span = set(span_leaves) | {true_name,
                                    leaf_names[p_a.argmax()],
                                    leaf_names[p_b.argmax()]}
        subtree  = minimal_subtree(tree, span)
        sub_root = next(n for n in subtree.nodes() if subtree.in_degree(n) == 0)
        pos, leaf_order = dendrogram_layout(subtree, sub_root)
        print(f"  Subtree: {subtree.number_of_nodes()} nodes, {len(leaf_order)} leaves")

        stub     = f"{args.model_a}_vs_{args.model_b}"
        fig_path = FIGURES_DIR / f"dendro_{img_idx:05d}_{stub}.png"
        tbl_path = FIGURES_DIR / f"dendro_{img_idx:05d}_{stub}_scores.png"

        kw = dict(
            subtree=subtree, pos=pos, leaf_order=leaf_order,
            true_label=true_name,
            top1_a=leaf_names[p_a.argmax()],
            top1_b=leaf_names[p_b.argmax()],
            probs_a=p_a, probs_b=p_b,
            leaf_names=leaf_names, leaf_labels=leaf_labels,
            model_a_name=model_a_name, model_b_name=model_b_name,
            metrics_a=ma, metrics_b=mb,
            image_idx=img_idx,
            full_tree_root_depth=abs_depths.get(sub_root, 0),
            full_tree_max_depth=full_tree_max,
        )
        draw_dendrogram_figure(**kw, out_path=fig_path, annotate_nodes=False)
        # annotated version for internal use only
        ann_path = FIGURES_DIR / f"dendro_{img_idx:05d}_{stub}_annotated.png"
        draw_dendrogram_figure(**kw, out_path=ann_path, annotate_nodes=True)
        draw_scores_table(ma, mb, model_a_name, model_b_name, tbl_path)

        # soft ground truth scores table
        if args.soft_gt:
            # parse "vizsla:0.5,moped:0.5" -> {leaf_idx: weight}
            label_to_idx = {lbl: k for k, lbl in enumerate(leaf_labels)}
            q = np.zeros(len(leaf_names))
            for part in args.soft_gt.split(","):
                lbl, wt = part.strip().split(":")
                lbl = lbl.strip()
                if lbl not in label_to_idx:
                    raise ValueError(f"Label '{lbl}' not found in leaf_labels")
                q[label_to_idx[lbl]] = float(wt)
            q /= q.sum()  # normalise
            # expected scores under soft GT
            def expected_metrics(probs_row):
                result = {}
                for true_idx_s, wt in enumerate(q):
                    if wt == 0: continue
                    m = per_example_metrics(probs_row, true_idx_s, scorer)
                    for k, v in m.items():
                        result[k] = result.get(k, 0.0) + wt * v
                return result
            ma_soft = expected_metrics(p_a)
            mb_soft = expected_metrics(p_b)
            soft_path = FIGURES_DIR / f"dendro_{img_idx:05d}_{stub}_scores_soft.png"
            draw_scores_table(ma_soft, mb_soft, model_a_name, model_b_name, soft_path)

        # save image
        try:
            img = load_image(img_idx)
            img.save(FIGURES_DIR / f"image_{img_idx:05d}.png")
        except Exception as e:
            print(f"  WARNING: image load failed: {e}")


if __name__ == "__main__":
    main()
