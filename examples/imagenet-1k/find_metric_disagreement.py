"""
Find images where two metrics disagree on which of two models is better.

Example: HXE favours model A strongly, but Resistance favours model B strongly.

Usage:
    python find_metric_disagreement.py \
        --model-a vit_small_patch16_224 \
        --model-b resnet50_timm \
        --metric-1 hxe \
        --metric-2 resistance \
        --graph tree \
        --top-n 30

Output: images where metric-1 favours model-a but metric-2 favours model-b
        (or vice versa), ranked by total disagreement magnitude.
"""
import argparse, pickle, sys, json, math
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("./wordnet_graph")
OUT_DIR  = Path("./model_validation_outputs")
sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev


def compute_metrics(probs, labels, scorer, leaf_full_idx,
                    dist_matrix, hxe_paths, n_full):
    N = len(labels)
    top1 = probs.argmax(axis=1)
    top5 = np.argsort(-probs, axis=1)[:, :5]

    pf = np.zeros((N, n_full), dtype=np.float32)
    pf[:, leaf_full_idx] = probs.astype(np.float32)
    true_full = np.array([leaf_full_idx[int(l)] for l in labels], dtype=np.int64)

    resist  = scorer.resistance_score(pf, true_full).astype(float)
    hxe     = ev.hxe_score(probs, labels.astype(np.int64), hxe_paths)
    p_true  = probs[np.arange(N), labels]
    logloss = -np.log(np.clip(p_true, 1e-12, 1.0))
    onehot  = np.zeros_like(probs)
    onehot[np.arange(N), labels] = 1.0
    brier   = np.sum((probs - onehot)**2, axis=1)
    dist1   = dist_matrix[labels, top1].astype(float)
    dist5   = np.mean(dist_matrix[labels[:, None], top5], axis=1).astype(float)
    top1_correct = (top1 == labels).astype(float)

    return {
        "hxe":        hxe,
        "resistance": resist,
        "dist1":      dist1,
        "dist5":      dist5,
        "brier":      brier,
        "logloss":    logloss,
        "top1_correct": top1_correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default="vit_small_patch16_224")
    parser.add_argument("--model-b", default="resnet50_timm")
    parser.add_argument("--metric-1", default="hxe",
                        choices=["hxe","resistance","dist1","dist5","brier","logloss"])
    parser.add_argument("--metric-2", default="resistance",
                        choices=["hxe","resistance","dist1","dist5","brier","logloss"])
    parser.add_argument("--graph", default="tree",
                        choices=["tree","ctree","dag"])
    parser.add_argument("--same-top1", action="store_true",
                        help="Only include images where both models have same top-1")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    gf = {"tree":  ("tree.gpickle",  "tree_leaf_synsets.json"),
          "ctree": ("ctree.gpickle", "ctree_leaf_synsets.json"),
          "dag":   ("graph.gpickle", "leaf_synsets.json")}[args.graph]

    with open(DATA_DIR / gf[0], "rb") as f: G = pickle.load(f)
    with open(DATA_DIR / gf[1]) as f: leaf_info = json.load(f)
    leaf_names  = leaf_info["class_idx_to_synset_name"]
    leaf_labels = leaf_info.get("class_idx_to_label",
                                [n.split(".")[0] for n in leaf_names])

    ev.OUT_DIR = OUT_DIR; ev.DATA_DIR = DATA_DIR
    scorer, _, leaf_full_idx = ev.build_full_graph_scorer(G, leaf_names, args.graph)
    dist_matrix = ev.precompute_leaf_distance_matrix(G, leaf_names, args.graph)
    hxe_paths   = ev.build_hxe_cache(G, leaf_names, alpha=args.alpha,
                                      graph_variant=args.graph)

    probs_a, labels = ev.load_model_outputs(args.model_a)
    probs_b, _      = ev.load_model_outputs(args.model_b)
    N = len(labels)
    n_full = scorer.size

    print(f"Computing metrics for {args.model_a}...")
    ma = compute_metrics(probs_a, labels, scorer, leaf_full_idx,
                         dist_matrix, hxe_paths, n_full)
    print(f"Computing metrics for {args.model_b}...")
    mb = compute_metrics(probs_b, labels, scorer, leaf_full_idx,
                         dist_matrix, hxe_paths, n_full)

    # Per-image signed gaps: positive = model A better (lower score)
    # gap_m1[i] > 0 means metric-1 favours A on image i
    # gap_m2[i] > 0 means metric-2 favours A on image i
    # Disagreement: gap_m1 and gap_m2 have opposite signs
    gap_m1 = mb[args.metric_1] - ma[args.metric_1]   # pos = A better on metric1
    gap_m2 = mb[args.metric_2] - ma[args.metric_2]   # pos = A better on metric2

    # Normalise gaps to make them comparable: divide by the mean absolute value
    gap_m1_norm = gap_m1 / (np.abs(gap_m1).mean() + 1e-10)
    gap_m2_norm = gap_m2 / (np.abs(gap_m2).mean() + 1e-10)

    # Disagreement score: large when m1 strongly favours A but m2 strongly favours B
    # = gap_m1_norm * (-gap_m2_norm) when they disagree
    # Capture both directions: use -gap_m1_norm * gap_m2_norm (large when opposite sign)
    disagree = -gap_m1_norm * gap_m2_norm

    top1_a = probs_a.argmax(axis=1)
    top1_b = probs_b.argmax(axis=1)

    rows = []
    for i in range(N):
        if args.same_top1 and top1_a[i] != top1_b[i]:
            continue
        rows.append({
            "image_idx":      i,
            "true_label":     int(labels[i]),
            "true_name":      leaf_labels[int(labels[i])],
            "top1_a_correct": bool(top1_a[i] == labels[i]),
            "top1_b_correct": bool(top1_b[i] == labels[i]),
            f"{args.metric_1}_a": float(ma[args.metric_1][i]),
            f"{args.metric_1}_b": float(mb[args.metric_1][i]),
            f"{args.metric_2}_a": float(ma[args.metric_2][i]),
            f"{args.metric_2}_b": float(mb[args.metric_2][i]),
            "gap_m1_norm":    float(gap_m1_norm[i]),
            "gap_m2_norm":    float(gap_m2_norm[i]),
            "disagree":       float(disagree[i]),
        })

    df = pd.DataFrame(rows)

    # Two interesting directions:
    # (1) metric-1 favours A, metric-2 favours B: gap_m1>0, gap_m2<0 -> disagree>0
    # (2) metric-1 favours B, metric-2 favours A: gap_m1<0, gap_m2>0 -> disagree>0
    # Sort by disagree score (most disagreement first), then split by which direction

    df_dir1 = df[(df["gap_m1_norm"] > 0) & (df["gap_m2_norm"] < 0)].copy()
    df_dir1 = df_dir1.sort_values("disagree", ascending=False)

    df_dir2 = df[(df["gap_m1_norm"] < 0) & (df["gap_m2_norm"] > 0)].copy()
    df_dir2 = df_dir2.sort_values("disagree", ascending=False)

    ma_name = args.model_a.replace("_timm","")
    mb_name = args.model_b.replace("_timm","")

    def print_table(dff, title):
        print(f"\n{title}")
        print(f"{'idx':>7}  {'true class':<28}  {'A✓':>3} {'B✓':>3}  "
              f"{args.metric_1+'_A':>10}  {args.metric_1+'_B':>10}  "
              f"{args.metric_2+'_A':>12}  {args.metric_2+'_B':>12}")
        print("-" * 105)
        m1a = f"{args.metric_1}_a"; m1b = f"{args.metric_1}_b"
        m2a = f"{args.metric_2}_a"; m2b = f"{args.metric_2}_b"
        for _, r in dff.head(args.top_n).iterrows():
            print(f"{int(r.image_idx):>7}  {r.true_name:<28}  "
                  f"{'✓' if r.top1_a_correct else '✗':>3} "
                  f"{'✓' if r.top1_b_correct else '✗':>3}  "
                  f"{r[m1a]:>10.4f}  {r[m1b]:>10.4f}  "
                  f"{r[m2a]:>12.2f}  {r[m2b]:>12.2f}")

    print_table(df_dir1,
        f"Direction 1: {args.metric_1} favours {ma_name}, "
        f"{args.metric_2} favours {mb_name}")
    print_table(df_dir2,
        f"Direction 2: {args.metric_1} favours {mb_name}, "
        f"{args.metric_2} favours {ma_name}")

    # Save both
    out1 = OUT_DIR / (f"disagree_{args.metric_1}_vs_{args.metric_2}"
                      f"_{args.model_a}_vs_{args.model_b}_dir1.csv")
    out2 = str(out1).replace("dir1","dir2")
    df_dir1.to_csv(out1, index=False)
    df_dir2.to_csv(out2, index=False)
    print(f"\nSaved:\n  {out1}\n  {out2}")


if __name__ == "__main__":
    main()