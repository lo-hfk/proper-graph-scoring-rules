"""
Step 9: Find same-top1 cases where resistance scores differ substantially
but flat entropy is similar between the two models.

This isolates cases where the GRAPH-STRUCTURAL placement of probability
mass differs between models, independently of how much total spread each
model has -- i.e. cases where the two models spread their uncertainty by
roughly the same amount (similar flat entropy) but in very different
directions in the WordNet tree (different resistance score).

These are the cleanest examples for the paper: they show the resistance
score detecting something that neither top-1 accuracy NOR Brier score
(which is closely related to flat entropy) can distinguish.

Requires:
  - model_validation_outputs/same_top1_cases_<graph>.csv
  - model_validation_outputs/decomposition_<model>_<graph>.npz
    (from score_decomposition.py, provides per-example flat_entropy)

Usage:
    python find_entropy_controlled_cases.py
    python find_entropy_controlled_cases.py --model-a efficientnet_b2 --model-b vit_small_patch16_224
    python find_entropy_controlled_cases.py --max-entropy-diff 0.1 --min-resistance-gap-pct 10 --top-n 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR  = Path("./model_validation_outputs")
GRAPH_DIR = Path("./wordnet_graph")


def load_decomposition(model_name, graph_variant):
    """Load per-example flat_entropy from score_decomposition output."""
    path = OUT_DIR / f"decomposition_{model_name}_{graph_variant}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Run first: python score_decomposition.py "
            f"--models {model_name} --graph {graph_variant}"
        )
    data = np.load(path)
    return data["flat_entropy"]   # shape (50000,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default="efficientnet_b2")
    parser.add_argument("--model-b", default="vit_small_patch16_224")
    parser.add_argument("--graph", default="tree",
                        choices=["dag", "tree", "ctree"])
    parser.add_argument("--max-entropy-diff", type=float, default=0.15,
                        help="Maximum allowed |flat_entropy_A - flat_entropy_B| "
                             "for a case to be included. Smaller = stricter "
                             "entropy matching. Default 0.15.")
    parser.add_argument("--min-resistance-gap-pct", type=float, default=5.0,
                        help="Minimum resistance score gap as %% of the smaller "
                             "value. Default 5%%.")
    parser.add_argument("--direction", default="any",
                        choices=["any", "a_better", "b_better"])
    parser.add_argument("--top-n", type=int, default=20,
                        help="Print this many top cases.")
    parser.add_argument("--sort-by", default="resistance_gap_pct",
                        choices=["resistance_gap_pct", "resistance_abs_diff",
                                 "entropy_diff"],
                        help="Sort output by resistance gap (default) or "
                             "by entropy difference (smallest first).")
    parser.add_argument("--out-csv", default=None,
                        help="Save filtered cases to this CSV. Default: "
                             "model_validation_outputs/entropy_controlled_cases_<graph>.csv")
    args = parser.parse_args()

    # Load same-top1 cases
    cases_csv = OUT_DIR / f"same_top1_cases_{args.graph}.csv"
    if not cases_csv.exists():
        raise FileNotFoundError(
            f"{cases_csv} not found.\n"
            f"Run first: python same_top1_analysis.py "
            f"--models {args.model_a} {args.model_b} --graph {args.graph}"
        )
    df = pd.read_csv(cases_csv)

    # Filter to our two models
    mask = (
        ((df["model_A"] == args.model_a) & (df["model_B"] == args.model_b)) |
        ((df["model_A"] == args.model_b) & (df["model_B"] == args.model_a))
    )
    df = df[mask].copy().reset_index(drop=True)
    print(f"Same-top1 cases for {args.model_a} vs {args.model_b}: {len(df)}")

    if len(df) == 0:
        print("No cases found. Run same_top1_analysis.py for this pair.")
        return

    # Normalise: model_A = args.model_a throughout
    swap = df["model_A"] == args.model_b
    for pair in [("model_A", "model_B"),
                 ("resistance_A", "resistance_B"),
                 ("resistance_A_zscore", "resistance_B_zscore"),
                 ("resistance_A_pct", "resistance_B_pct")]:
        if pair[0] in df.columns:
            df.loc[swap, list(pair)] = df.loc[swap, list(pair)].values[:, ::-1]

    # Load per-example flat entropy for both models
    print("Loading per-example flat entropy from decomposition files...")
    fe_a = load_decomposition(args.model_a, args.graph)
    fe_b = load_decomposition(args.model_b, args.graph)

    # Add per-example flat entropy to the dataframe
    img_indices = df["image_idx"].values.astype(int)
    df["flat_entropy_A"] = fe_a[img_indices]
    df["flat_entropy_B"] = fe_b[img_indices]
    df["flat_entropy_diff"] = (df["flat_entropy_A"] - df["flat_entropy_B"]).abs()
    df["flat_entropy_mean"] = (df["flat_entropy_A"] + df["flat_entropy_B"]) / 2

    # Compute resistance gap
    df["resistance_gap_pct"] = (
        (df["resistance_B"] - df["resistance_A"]).abs()
        / df[["resistance_A", "resistance_B"]].min(axis=1) * 100
    )

    # Filter: entropy similar, resistance gap large
    n_before = len(df)
    df_filtered = df[
        (df["flat_entropy_diff"] <= args.max_entropy_diff) &
        (df["resistance_gap_pct"] >= args.min_resistance_gap_pct)
    ].copy()
    print(f"After filtering (|Δentropy| ≤ {args.max_entropy_diff}, "
          f"resistance gap ≥ {args.min_resistance_gap_pct}%): "
          f"{len(df_filtered)} cases  (from {n_before})")

    # Direction filter
    if args.direction == "b_better":
        df_filtered = df_filtered[df_filtered["resistance_B"] < df_filtered["resistance_A"]]
        print(f"  → {len(df_filtered)} where {args.model_b} scores better")
    elif args.direction == "a_better":
        df_filtered = df_filtered[df_filtered["resistance_A"] < df_filtered["resistance_B"]]
        print(f"  → {len(df_filtered)} where {args.model_a} scores better")

    if len(df_filtered) == 0:
        print("\nNo cases passed all filters.")
        print("Try relaxing --max-entropy-diff or --min-resistance-gap-pct.")
        return

    # Sort
    if args.sort_by == "entropy_diff":
        df_filtered = df_filtered.sort_values("flat_entropy_diff", ascending=True)
    elif args.sort_by == "resistance_abs_diff":
        df_filtered = df_filtered.sort_values("resistance_abs_diff", ascending=False)
    else:
        df_filtered = df_filtered.sort_values("resistance_gap_pct", ascending=False)
    df_filtered = df_filtered.reset_index(drop=True)

    # Load leaf labels for display
    with open(GRAPH_DIR / f"{args.graph}_leaf_synsets.json") as f:
        leaf_info = json.load(f)
    leaf_labels = leaf_info["class_idx_to_label"]

    # Print summary
    print(f"\n{'='*80}")
    print(f"TOP {min(args.top_n, len(df_filtered))} CASES  "
          f"[sorted by {args.sort_by}]")
    print(f"Entropy controlled: |ΔH| ≤ {args.max_entropy_diff},  "
          f"resistance gap ≥ {args.min_resistance_gap_pct}%")
    print(f"{'='*80}")

    cols_display = [
        "image_idx", "true_label_name", "shared_pred_name",
        "flat_entropy_A", "flat_entropy_B", "flat_entropy_diff",
        "resistance_A", "resistance_B", "resistance_gap_pct"
    ]
    top = df_filtered.head(args.top_n).copy()
    top["true_label_name"]  = top["true_label"].apply(
        lambda i: leaf_labels[int(i)].replace("_"," "))
    top["shared_pred_name"] = top["shared_top1_pred"].apply(
        lambda i: leaf_labels[int(i)].replace("_"," "))

    print(top[cols_display].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n--- Distribution of resistance_gap_pct (all {len(df_filtered)} cases) ---")
    for pct in [25, 50, 75, 90, 95]:
        print(f"  {pct}th percentile: {np.percentile(df_filtered['resistance_gap_pct'], pct):.2f}%")

    print(f"\n--- Distribution of flat_entropy_diff (all {len(df_filtered)} cases) ---")
    print(f"  mean={df_filtered['flat_entropy_diff'].mean():.4f}, "
          f"median={df_filtered['flat_entropy_diff'].median():.4f}, "
          f"max={df_filtered['flat_entropy_diff'].max():.4f}")

    # Save
    out_csv = args.out_csv or (
        OUT_DIR / f"entropy_controlled_cases_{args.graph}.csv"
    )
    df_filtered.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df_filtered)} filtered cases to {out_csv}")
    print(f"\nTo visualise the top examples, run:")
    print(f"  python qualitative_examples.py "
          f"--model-a {args.model_a} --model-b {args.model_b} "
          f"--graph {args.graph} --direction {args.direction} --n-examples 5")
    print(f"  (the qualitative script uses same_top1_cases_{args.graph}.csv;")
    print(f"   the entropy-controlled list here is a subset you can inspect "
          f"to pick the most interesting image indices manually)")


if __name__ == "__main__":
    main()