"""
Step 10: Find validation examples where two models agree on all classical metrics
but differ substantially on resistance score.

Specifically, filters same-top1 cases to those where ADDITIONALLY:
  - Both models have top-5 accuracy = 1 (true label in both top-5 sets)
  - Brier scores are similar (|brier_A - brier_B| <= --max-brier-diff)
  - Resistance scores differ substantially (gap >= --min-resistance-gap-pct)

Mistake severity is automatically identical whenever top-1 is the same
(since severity = dist(true_label, shared_top1_pred), which is a fixed
graph property independent of the model's probability values).  It is
computed and shown in the output as a sanity check.

All these constraints together mean: the two models are indistinguishable
by top-1, top-5, Brier, and mistake severity, yet your resistance score
separates them.  These are the cleanest possible illustrations that
resistance score detects something real that no existing metric can see.

Usage:
    python find_classically_equivalent_cases.py
    python find_classically_equivalent_cases.py \\
        --model-a efficientnet_b2 --model-b vit_small_patch16_224 \\
        --max-brier-diff 0.05 --min-resistance-gap-pct 10 --top-n 20
    python find_classically_equivalent_cases.py --direction b_better

Requires:
    model_validation_outputs/same_top1_cases_<graph>.csv
    model_validation_outputs/probs_<model>.npy
    model_validation_outputs/true_labels_<model>.npy
    model_validation_outputs/graph_scorer_cache_<graph>.pkl
    wordnet_graph/<graph>_leaf_synsets.json
    wordnet_graph/leaf_distance_matrix_<graph>.npy
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR   = Path("./model_validation_outputs")
GRAPH_DIR = Path("./wordnet_graph")


def load_per_example_metrics(model_name, probs, true_labels,
                              dist_matrix, scorer, leaf_full_idx):
    """
    Compute per-example top-5 accuracy and Brier score for all 50k examples.
    Returns two arrays of shape (n_examples,).
    """
    n = len(true_labels)
    n_leaves = probs.shape[1]

    # Top-5 accuracy: 1 if true label is in the top-5 predicted classes
    top5_indices = np.argpartition(-probs, kth=4, axis=1)[:, :5]
    top5_acc = (top5_indices == true_labels[:, None]).any(axis=1).astype(float)

    # Brier score: sum of squared differences from one-hot
    # = sum(p^2) - 2*p[true] + 1
    brier = (np.sum(probs ** 2, axis=1)
             - 2 * probs[np.arange(n), true_labels]
             + 1.0)

    # Mistake severity: dist(true_label, top1_pred), 0 if correct
    top1 = probs.argmax(axis=1)
    wrong = top1 != true_labels
    severity = np.zeros(n)
    severity[wrong] = dist_matrix[true_labels[wrong], top1[wrong]]

    return top5_acc, brier, severity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default="efficientnet_b2")
    parser.add_argument("--model-b", default="vit_small_patch16_224")
    parser.add_argument("--graph", default="tree",
                        choices=["dag", "tree", "ctree"])
    parser.add_argument("--require-top5", action="store_true", default=True,
                        help="Require both models to have top-5 accuracy = 1 "
                             "(true label in both top-5 sets). Default: True.")
    parser.add_argument("--no-require-top5", dest="require_top5",
                        action="store_false",
                        help="Drop the top-5 requirement.")
    parser.add_argument("--max-brier-diff", type=float, default=0.05,
                        help="Maximum allowed |brier_A - brier_B|. "
                             "Smaller = stricter Brier matching. Default 0.05.")
    parser.add_argument("--min-resistance-gap-pct", type=float, default=5.0,
                        help="Minimum resistance gap as %% of the smaller value. "
                             "Default 5%%.")
    parser.add_argument("--direction", default="any",
                        choices=["any", "a_better", "b_better"],
                        help="any: both directions. "
                             "b_better: only cases where model-b has lower "
                             "(better) resistance. a_better: vice versa.")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top cases to print. Default 20.")
    parser.add_argument("--sort-by", default="resistance_gap_pct",
                        choices=["resistance_gap_pct", "resistance_abs_diff",
                                 "brier_diff"],
                        help="Sort output by resistance gap %% (default), "
                             "absolute resistance gap, or Brier difference.")
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    # --- Load same-top1 base cases ---
    cases_csv = OUT_DIR / f"same_top1_cases_{args.graph}.csv"
    if not cases_csv.exists():
        raise FileNotFoundError(
            f"{cases_csv} not found.\n"
            f"Run first:\n"
            f"  python same_top1_analysis.py "
            f"--models {args.model_a} {args.model_b} --graph {args.graph}"
        )
    df = pd.read_csv(cases_csv)

    mask = (
        ((df["model_A"] == args.model_a) & (df["model_B"] == args.model_b)) |
        ((df["model_A"] == args.model_b) & (df["model_B"] == args.model_a))
    )
    df = df[mask].copy().reset_index(drop=True)
    print(f"Same-top1 cases for {args.model_a} vs {args.model_b}: {len(df)}")

    # Normalise so model_A = args.model_a
    swap = df["model_A"] == args.model_b
    for pair in [("model_A", "model_B"),
                 ("resistance_A", "resistance_B"),
                 ("resistance_A_zscore", "resistance_B_zscore"),
                 ("resistance_A_pct", "resistance_B_pct")]:
        if pair[0] in df.columns:
            df.loc[swap, list(pair)] = df.loc[swap, list(pair)].values[:, ::-1]

    # --- Load probs and compute per-example metrics ---
    print("Loading model outputs and computing per-example metrics...")
    probs_a  = np.load(OUT_DIR / f"probs_{args.model_a}.npy")
    probs_b  = np.load(OUT_DIR / f"probs_{args.model_b}.npy")
    true_labels = np.load(OUT_DIR / f"true_labels_{args.model_a}.npy")
    assert np.array_equal(true_labels,
                          np.load(OUT_DIR / f"true_labels_{args.model_b}.npy")), \
        "Label mismatch between models!"

    # Load scorer and distance matrix via evaluate module
    import evaluate as ev
    ev.OUT_DIR  = OUT_DIR
    ev.DATA_DIR = GRAPH_DIR
    G, leaf_info = ev.load_graph_and_leaf_info(args.graph)
    leaf_names = leaf_info["class_idx_to_synset_name"]
    scorer, _, leaf_full_idx = ev.build_full_graph_scorer(G, leaf_names, args.graph)
    dist_matrix = ev.precompute_leaf_distance_matrix(G, leaf_names, args.graph)

    top5_a, brier_a, sev_a = load_per_example_metrics(
        args.model_a, probs_a, true_labels, dist_matrix, scorer, leaf_full_idx)
    top5_b, brier_b, sev_b = load_per_example_metrics(
        args.model_b, probs_b, true_labels, dist_matrix, scorer, leaf_full_idx)

    # --- Add per-example metrics to dataframe ---
    idx = df["image_idx"].values.astype(int)
    df["top5_A"]    = top5_a[idx]
    df["top5_B"]    = top5_b[idx]
    df["brier_A"]   = brier_a[idx]
    df["brier_B"]   = brier_b[idx]
    df["severity_A"] = sev_a[idx]   # should equal severity_B since top1 is shared
    df["severity_B"] = sev_b[idx]
    df["brier_diff"] = (df["brier_A"] - df["brier_B"]).abs()
    df["resistance_gap_pct"] = (
        (df["resistance_B"] - df["resistance_A"]).abs()
        / df[["resistance_A", "resistance_B"]].min(axis=1) * 100
    )

    # Sanity check: severity should be identical when top-1 is the same
    max_sev_diff = (df["severity_A"] - df["severity_B"]).abs().max()
    if max_sev_diff > 1e-6:
        print(f"WARNING: max |severity_A - severity_B| = {max_sev_diff:.6f} "
              f"(should be 0 when top-1 is identical; check your distance matrix)")
    else:
        print(f"Sanity check: mistake severity identical for all same-top1 pairs ✓")

    # --- Apply filters ---
    n_start = len(df)

    if args.require_top5:
        df = df[(df["top5_A"] == 1) & (df["top5_B"] == 1)].copy()
        print(f"After requiring top-5 = 1 for both models: {len(df)} cases "
              f"(from {n_start})")

    df = df[df["brier_diff"] <= args.max_brier_diff].copy()
    print(f"After |ΔBrier| ≤ {args.max_brier_diff}: {len(df)} cases")

    df = df[df["resistance_gap_pct"] >= args.min_resistance_gap_pct].copy()
    print(f"After resistance gap ≥ {args.min_resistance_gap_pct}%: {len(df)} cases")

    if args.direction == "b_better":
        df = df[df["resistance_B"] < df["resistance_A"]].copy()
        print(f"After direction filter ({args.model_b} better): {len(df)} cases")
    elif args.direction == "a_better":
        df = df[df["resistance_A"] < df["resistance_B"]].copy()
        print(f"After direction filter ({args.model_a} better): {len(df)} cases")

    if len(df) == 0:
        print("\nNo cases passed all filters.")
        print("Try relaxing --max-brier-diff or --min-resistance-gap-pct,")
        print("or use --no-require-top5.")
        return

    # --- Sort ---
    df = df.sort_values(args.sort_by, ascending=(args.sort_by == "brier_diff"))
    df = df.reset_index(drop=True)

    # --- Display ---
    with open(GRAPH_DIR / f"{args.graph}_leaf_synsets.json") as f:
        leaf_labels = json.load(f)["class_idx_to_label"]

    top = df.head(args.top_n).copy()
    top["true_label_name"]   = top["true_label"].apply(
        lambda i: leaf_labels[int(i)].replace("_", " "))
    top["shared_pred_name"]  = top["shared_top1_pred"].apply(
        lambda i: leaf_labels[int(i)].replace("_", " "))

    display_cols = [
        "image_idx", "true_label_name", "shared_pred_name",
        "brier_A", "brier_B", "brier_diff",
        "severity_A",
        "resistance_A", "resistance_B", "resistance_gap_pct",
    ]

    print(f"\n{'='*90}")
    print(f"TOP {min(args.top_n, len(df))} CASES  [sorted by {args.sort_by}]")
    print(f"Filters: same top-1 (wrong) | "
          f"{'top-5 = 1 for both | ' if args.require_top5 else ''}"
          f"|ΔBrier| ≤ {args.max_brier_diff} | "
          f"resistance gap ≥ {args.min_resistance_gap_pct}%")
    print(f"Models: {args.model_a} (A) vs {args.model_b} (B)")
    print(f"{'='*90}")
    print(top[display_cols].to_string(index=False,
          float_format=lambda x: f"{x:.4f}"))

    print(f"\n--- Summary over all {len(df)} filtered cases ---")
    print(f"resistance_gap_pct:  "
          f"mean={df['resistance_gap_pct'].mean():.1f}%  "
          f"median={df['resistance_gap_pct'].median():.1f}%  "
          f"max={df['resistance_gap_pct'].max():.1f}%")
    print(f"brier_diff:          "
          f"mean={df['brier_diff'].mean():.4f}  "
          f"median={df['brier_diff'].median():.4f}  "
          f"max={df['brier_diff'].max():.4f}")
    print(f"Direction breakdown: "
          f"{(df['resistance_B'] < df['resistance_A']).sum()} cases where "
          f"{args.model_b} better, "
          f"{(df['resistance_A'] < df['resistance_B']).sum()} cases where "
          f"{args.model_a} better")

    out_csv = args.out_csv or (
        OUT_DIR / f"classically_equivalent_cases_{args.graph}.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} cases to {out_csv}")
    print(f"\nTo visualise top examples:")
    print(f"  python qualitative_examples.py "
          f"--model-a {args.model_a} --model-b {args.model_b} "
          f"--graph {args.graph} --direction {args.direction} --n-examples 5")


if __name__ == "__main__":
    main()