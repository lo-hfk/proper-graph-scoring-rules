"""
Find images where:
  - Both models predict the CORRECT top-1 class (top-1 acc = 1 for both)
  - Model A (e.g. resnet50) has higher Brier score than model B
    (i.e. model A is overconfident/sharp)
  - Model B wins on Resistance score by a large margin
  - Optionally: model A and B have similar log-loss (so it's not just a quality gap)

Usage:
    python find_hedging_cases.py \
        --model-a resnet50_timm \
        --model-b convnext_tiny \
        --graph tree \
        --min-resistance-gap-pct 10 \
        --top-n 20
"""
import argparse, pickle, sys, json, math
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("./wordnet_graph")
OUT_DIR  = Path("./model_validation_outputs")
sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default="resnet50_timm",
                        help="The 'sharp' model (typically ResNet-50)")
    parser.add_argument("--model-b", default="convnext_tiny",
                        help="The 'hierarchically better' model")
    parser.add_argument("--graph", default="tree",
                        choices=["tree", "ctree", "dag"])
    parser.add_argument("--min-resistance-gap-pct", type=float, default=10.0,
                        help="Min %% gap in resistance score (B better than A)")
    parser.add_argument("--min-excess-gap", type=float, default=5.0,
                        help="Min excess of resistance gap over Brier gap (both in %%). "
                             "Filters out cases where resistance gap is just due to "
                             "overall confidence difference.")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Show top N cases")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    gf = {"tree": ("tree.gpickle", "tree_leaf_synsets.json"),
          "ctree": ("ctree.gpickle", "ctree_leaf_synsets.json"),
          "dag":   ("graph.gpickle", "leaf_synsets.json")}[args.graph]

    with open(DATA_DIR / gf[0], "rb") as f: G = pickle.load(f)
    with open(DATA_DIR / gf[1]) as f: leaf_info = json.load(f)
    leaf_names = leaf_info["class_idx_to_synset_name"]
    leaf_labels = leaf_info.get("class_idx_to_label",
                                [n.split(".")[0] for n in leaf_names])

    ev.OUT_DIR = OUT_DIR; ev.DATA_DIR = DATA_DIR
    scorer, _, leaf_full_idx = ev.build_full_graph_scorer(G, leaf_names, args.graph)
    hxe_paths = ev.build_hxe_cache(G, leaf_names, alpha=args.alpha,
                                    graph_variant=args.graph)

    probs_a, labels = ev.load_model_outputs(args.model_a)
    probs_b, _      = ev.load_model_outputs(args.model_b)
    N = len(labels)

    print(f"Computing per-image metrics for {args.model_a} vs {args.model_b}...")

    top1_a = probs_a.argmax(axis=1)
    top1_b = probs_b.argmax(axis=1)

    # Filter: both correct
    both_correct = (top1_a == labels) & (top1_b == labels)
    idxs = np.where(both_correct)[0]
    print(f"Both correct: {len(idxs)} / {N}")

    # Compute resistance for filtered set
    n_full = scorer.size
    rows = []
    for img in idxs:
        true_idx = int(labels[img])
        pa = probs_a[img]; pb = probs_b[img]

        # Resistance
        pf_a = np.zeros(n_full, dtype=np.float32)
        pf_a[leaf_full_idx] = pa
        pf_b = np.zeros(n_full, dtype=np.float32)
        pf_b[leaf_full_idx] = pb
        true_full = leaf_full_idx[true_idx]

        res_a = float(scorer.resistance_score(
            pf_a[np.newaxis], np.array([true_full]))[0])
        res_b = float(scorer.resistance_score(
            pf_b[np.newaxis], np.array([true_full]))[0])

        # Brier
        onehot = np.zeros(len(leaf_names)); onehot[true_idx] = 1.0
        brier_a = float(np.sum((pa - onehot)**2))
        brier_b = float(np.sum((pb - onehot)**2))

        # Log-loss
        logloss_a = float(-math.log(max(pa[true_idx], 1e-12)))
        logloss_b = float(-math.log(max(pb[true_idx], 1e-12)))

        # Resistance gap: B better than A means res_b < res_a
        if res_a <= 0: continue
        gap_pct = (res_a - res_b) / res_a * 100  # positive = B better

        rows.append({
            "image_idx": int(img),
            "true_label": int(true_idx),
            "true_name": leaf_labels[true_idx],
            "resistance_a": res_a,
            "resistance_b": res_b,
            "resistance_gap_pct": gap_pct,
            "brier_a": brier_a,
            "brier_b": brier_b,
            "brier_gap": brier_a - brier_b,   # positive = A worse (sharper but wrong)
            "logloss_a": logloss_a,
            "logloss_b": logloss_b,
        })

    df = pd.DataFrame(rows)
    print(f"Cases computed: {len(df)}")

    # Filter: B wins on resistance by at least min_gap%
    df = df[df["resistance_gap_pct"] >= args.min_resistance_gap_pct].copy()
    print(f"After resistance gap filter (>={args.min_resistance_gap_pct}%): {len(df)}")

    # Key filter: resistance gap much larger than Brier gap
    # Normalise both gaps to make them comparable:
    # resistance_gap_pct already in %; compute Brier gap as % of model A's Brier
    df["brier_gap_pct"] = (df["brier_a"] - df["brier_b"]) / df["brier_a"].clip(lower=1e-6) * 100
    # We want: resistance_gap_pct >> brier_gap_pct
    # i.e. resistance explains something beyond just overall confidence
    df["excess_resistance_gap"] = df["resistance_gap_pct"] - df["brier_gap_pct"]
    df = df[df["excess_resistance_gap"] >= args.min_excess_gap].copy()
    print(f"After excess resistance gap filter (>={args.min_excess_gap}%): {len(df)}")

    # Sort by excess gap descending
    df = df.sort_values("excess_resistance_gap", ascending=False)

    out_csv = OUT_DIR / f"hedging_cases_{args.model_a}_vs_{args.model_b}_{args.graph}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(f"\nTop {args.top_n} cases (sorted by excess resistance gap):")
    print(f"{'idx':>7}  {'true class':<25}  {'res_A':>8}  {'res_B':>8}  "
          f"{'res_gap%':>8}  {'brier_gap%':>10}  {'excess%':>8}  "
          f"{'brier_A':>7}  {'brier_B':>7}")
    for _, r in df.head(args.top_n).iterrows():
        print(f"{int(r.image_idx):>7}  {r.true_name:<25}  "
              f"{r.resistance_a:>8.1f}  {r.resistance_b:>8.1f}  "
              f"{r.resistance_gap_pct:>8.1f}%  "
              f"{r.brier_gap_pct:>10.1f}%  "
              f"{r.excess_resistance_gap:>8.1f}%  "
              f"{r.brier_a:>7.4f}  {r.brier_b:>7.4f}")


if __name__ == "__main__":
    main()