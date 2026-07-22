"""
Find validation images where two models have similar flat quadratic
self-spread (0.5 * p'p) but very different per-example GE improvement.

This isolates cases where the GRAPH STRUCTURE of residual uncertainty
differs between models, independently of how peaked or spread-out each
model's distribution is overall on that image.

The intended use is to illustrate the score decomposition finding:
that some architectures direct their residual uncertainty toward
graph-near classes (high GE improvement) while others do not (low GE
improvement), even when the total amount of residual uncertainty is the
same.

Unlike find_classically_equivalent_cases.py, this script does NOT
require the same wrong top-1 prediction. Both correct and incorrect
predictions are considered, because the GE improvement is meaningful
regardless of whether the top-1 prediction is right.

Usage:
    python find_ge_contrast_cases.py
    python find_ge_contrast_cases.py \\
        --model-a vit_small_patch16_224 --model-b convnext_tiny \\
        --max-flat-quad-diff 0.02 --min-ge-improvement-gap 0.3 \\
        --top-n 20

Requires:
    model_validation_outputs/decomposition_<model>_<graph>.npz
        (from score_decomposition.py --recompute)
    model_validation_outputs/probs_<model>.npy
    model_validation_outputs/true_labels_<model>.npy
    wordnet_graph/<graph>_leaf_synsets.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR   = Path("./model_validation_outputs")
GRAPH_DIR = Path("./wordnet_graph")


def load_decomposition(model_name, graph_variant):
    path = OUT_DIR / f"decomposition_{model_name}_{graph_variant}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Run: python score_decomposition.py "
            f"--models {model_name} --graph {graph_variant} --recompute"
        )
    data = np.load(path)
    if "flat_quad" not in data:
        raise ValueError(
            f"Cache {path} uses old format (flat_entropy instead of flat_quad). "
            f"Re-run score_decomposition.py with --recompute."
        )
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default="vit_small_patch16_224",
                        help="Model with expected HIGH GE improvement (graph-coherent)")
    parser.add_argument("--model-b", default="convnext_tiny",
                        help="Model with expected LOW GE improvement (graph-diffuse)")
    parser.add_argument("--graph", default="tree",
                        choices=["dag", "tree", "ctree"])
    parser.add_argument("--max-flat-quad-diff", type=float, default=0.02,
                        help="Max allowed |flat_quad_A - flat_quad_B| on the same "
                             "image. Controls for differing overall confidence levels. "
                             "Default 0.02.")
    parser.add_argument("--min-ge-improvement-gap", type=float, default=0.20,
                        help="Min required ge_improvement_A - ge_improvement_B "
                             "(positive = model A has more graph-directed uncertainty). "
                             "Default 0.20.")
    parser.add_argument("--min-flat-quad", type=float, default=0.05,
                        help="Min flat_quad for both models: ensures there is "
                             "meaningful non-top-1 mass to compare. Default 0.05.")
    parser.add_argument("--require-same-top1", action="store_true", default=False,
                        help="If set, only include images where both models make "
                             "the same top-1 prediction.")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--sort-by", default="ge_improvement_gap",
                        choices=["ge_improvement_gap", "pct_ge_gap", "flat_quad_diff"])
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    print(f"Loading decomposition for {args.model_a}...")
    data_a = load_decomposition(args.model_a, args.graph)
    ge_a   = data_a["ge_improvement"]      # per-example GE improvement
    fq_a   = data_a["flat_quad"]           # per-example 0.5*p'p
    gn_a   = data_a["ge_null"]             # per-example GE null baseline

    print(f"Loading decomposition for {args.model_b}...")
    data_b = load_decomposition(args.model_b, args.graph)
    ge_b   = data_b["ge_improvement"]
    fq_b   = data_b["flat_quad"]
    gn_b   = data_b["ge_null"]

    # Load probs for top-1 and true labels
    probs_a     = np.load(OUT_DIR / f"probs_{args.model_a}.npy")
    probs_b     = np.load(OUT_DIR / f"probs_{args.model_b}.npy")
    true_labels = np.load(OUT_DIR / f"true_labels_{args.model_a}.npy")

    top1_a = probs_a.argmax(axis=1)
    top1_b = probs_b.argmax(axis=1)

    n = len(true_labels)
    assert len(ge_a) == n == len(ge_b), "Array length mismatch"

    # Per-example percentage GE improvement
    pct_ge_a = np.where(gn_a > 1e-8, ge_a / gn_a * 100, 0.0)
    pct_ge_b = np.where(gn_b > 1e-8, ge_b / gn_b * 100, 0.0)

    # Derived quantities
    fq_diff          = np.abs(fq_a - fq_b)
    ge_imp_gap       = ge_a - ge_b          # positive = model A more graph-directed
    pct_ge_gap       = pct_ge_a - pct_ge_b  # positive = model A more graph-directed (%)
    both_fq_ok       = (fq_a >= args.min_flat_quad) & (fq_b >= args.min_flat_quad)
    fq_similar       = fq_diff <= args.max_flat_quad_diff
    ge_gap_large     = ge_imp_gap >= args.min_ge_improvement_gap

    mask = both_fq_ok & fq_similar & ge_gap_large
    if args.require_same_top1:
        mask &= (top1_a == top1_b)

    idx = np.where(mask)[0]
    print(f"\nCases passing all filters: {len(idx)} "
          f"(out of {n} images)")
    print(f"  |flat_quad_A - flat_quad_B| <= {args.max_flat_quad_diff}")
    print(f"  both flat_quad >= {args.min_flat_quad}")
    print(f"  ge_improvement_A - ge_improvement_B >= {args.min_ge_improvement_gap}")
    if args.require_same_top1:
        print(f"  same top-1 prediction")

    if len(idx) == 0:
        print("\nNo cases found. Try relaxing filters:")
        print("  --max-flat-quad-diff 0.05  --min-ge-improvement-gap 0.1")
        return

    # Sort
    if args.sort_by == "pct_ge_gap":
        order = np.argsort(-pct_ge_gap[idx])
    elif args.sort_by == "flat_quad_diff":
        order = np.argsort(fq_diff[idx])   # ascending: tightest match first
    else:
        order = np.argsort(-ge_imp_gap[idx])
    idx = idx[order]

    # Load leaf labels for display
    with open(GRAPH_DIR / f"{args.graph}_leaf_synsets.json") as f:
        leaf_info = json.load(f)
    leaf_labels = leaf_info["class_idx_to_label"]

    # Build output dataframe
    rows = []
    for i in idx:
        rows.append({
            "image_idx":       int(i),
            "true_label":      int(true_labels[i]),
            "true_label_name": leaf_labels[int(true_labels[i])].replace("_"," "),
            "top1_A":          int(top1_a[i]),
            "top1_A_name":     leaf_labels[int(top1_a[i])].replace("_"," "),
            "top1_B":          int(top1_b[i]),
            "top1_B_name":     leaf_labels[int(top1_b[i])].replace("_"," "),
            "top1_correct_A":  int(top1_a[i] == true_labels[i]),
            "top1_correct_B":  int(top1_b[i] == true_labels[i]),
            "flat_quad_A":     float(fq_a[i]),
            "flat_quad_B":     float(fq_b[i]),
            "flat_quad_diff":  float(fq_diff[i]),
            "ge_improvement_A":   float(ge_a[i]),
            "ge_improvement_B":   float(ge_b[i]),
            "ge_improvement_gap": float(ge_imp_gap[i]),
            "pct_ge_A":        float(pct_ge_a[i]),
            "pct_ge_B":        float(pct_ge_b[i]),
            "pct_ge_gap":      float(pct_ge_gap[i]),
            "ge_null_A":       float(gn_a[i]),
            "ge_null_B":       float(gn_b[i]),
        })
    df = pd.DataFrame(rows)

    # Print top-n
    print(f"\n{'='*80}")
    print(f"TOP {min(args.top_n, len(df))} CASES  [sorted by {args.sort_by}]")
    print(f"Model A (high GE): {args.model_a}")
    print(f"Model B (low GE):  {args.model_b}")
    print(f"{'='*80}")
    display_cols = [
        "image_idx", "true_label_name",
        "top1_A_name", "top1_correct_A",
        "top1_B_name", "top1_correct_B",
        "flat_quad_A", "flat_quad_B",
        "pct_ge_A", "pct_ge_B", "pct_ge_gap",
    ]
    print(df.head(args.top_n)[display_cols].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n--- Summary over all {len(df)} cases ---")
    print(f"ge_improvement_gap:  "
          f"mean={df['ge_improvement_gap'].mean():.3f}  "
          f"median={df['ge_improvement_gap'].median():.3f}")
    print(f"pct_ge_gap:          "
          f"mean={df['pct_ge_gap'].mean():.1f}%  "
          f"median={df['pct_ge_gap'].median():.1f}%")
    print(f"flat_quad_diff:      "
          f"mean={df['flat_quad_diff'].mean():.4f}  "
          f"max={df['flat_quad_diff'].max():.4f}")
    same_top1 = (df['top1_A_name'] == df['top1_B_name']).mean()
    print(f"Fraction same top-1: {same_top1:.3f}")
    both_correct = ((df['top1_correct_A']==1) & (df['top1_correct_B']==1)).mean()
    print(f"Fraction both correct: {both_correct:.3f}")

    out_csv = args.out_csv or (
        OUT_DIR / f"ge_contrast_cases_{args.model_a}_vs_{args.model_b}_{args.graph}.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} cases to {out_csv}")
    print(f"\nTo visualise:")
    print(f"  python qualitative_examples_dendrogram.py \\")
    print(f"    --model-a {args.model_a} --model-b {args.model_b} \\")
    print(f"    --graph {args.graph} \\")
    print(f"    --cases-csv {out_csv} \\")
    print(f"    --n-examples 5")


if __name__ == "__main__":
    main()