"""
Compute all evaluation metrics for each model and save per-model CSVs.

Loads data/tree/tree.gpickle and data/probs/<model>.npy and computes:
  - top-1 error, top-5 error
  - distance@1, distance@5  (LCA-height, improper)
  - log-loss, Brier         (flat proper scores)
  - HXE (alpha=0.5)         (hierarchical proper, log-score family)
  - Resistance score        (hierarchical proper, Brier family)

Saves one CSV per model to data/results/<model>.csv.

Usage
-----
    python data_prep/05_evaluate.py
"""

import json
import pickle
from pathlib import Path

import numpy as np

from propnetscore import GraphScorer, HierarchicalScorer

DATA_DIR    = Path(__file__).parent.parent / "data"
TREE_DIR    = DATA_DIR / "tree"
PROBS_DIR   = DATA_DIR / "probs"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HXE_ALPHA = 0.5


def load_tree_and_scorer():
    with open(TREE_DIR / "tree.gpickle", "rb") as f:
        tree = pickle.load(f)
    with open(TREE_DIR / "leaf_synsets.json") as f:
        leaf_info = json.load(f)
    leaf_names = leaf_info["synset_name"]   # list of 1000 synset names in class-index order
    scorer = HierarchicalScorer(tree, leaf_names)
    return scorer, leaf_names


def evaluate_model(model_name, scorer, true_labels):
    probs_path = PROBS_DIR / f"{model_name}.npy"
    if not probs_path.exists():
        print(f"  Skipping {model_name}: no probs file found.")
        return None

    out_path = RESULTS_DIR / f"{model_name}.csv"
    if out_path.exists():
        print(f"  {model_name}: already evaluated, skipping.")
        return

    print(f"  Evaluating {model_name}...")
    P = np.load(probs_path)
    i = true_labels

    # compute per-image scores (used both for saving and for printing means)
    per_image = {
        "top1_err":   scorer.top_k_error_rate(P, i, k=1),
        "top5_err":   scorer.top_k_error_rate(P, i, k=5),
        "dist_at_1":  scorer.distance_at_k(P, i, k=1),
        "dist_at_5":  scorer.distance_at_k(P, i, k=5),
        "log_loss":   scorer.logarithmic_score(P, i),
        "brier":      scorer.brier_score(P, i),
        "hxe":        scorer.hxe_score(P, i, alpha=HXE_ALPHA),
        "resistance": scorer.resistance_score(P, i),
    }

    import pandas as pd
    pd.DataFrame(per_image).to_csv(out_path, index=False)

    for k, v in per_image.items():
        print(f"    {k:15s} {v.mean():.4f}")
    print(f"    Saved to {out_path}")


if __name__ == "__main__":
    print("Loading tree and building scorer (one-time cost)...")
    scorer, leaf_names = load_tree_and_scorer()
    print(f"  Tree: {scorer.tree.number_of_nodes()} nodes, {scorer.size} leaves")

    true_labels = np.load(PROBS_DIR / "true_labels.npy")

    model_names = [p.stem for p in sorted(PROBS_DIR.glob("*.npy"))
                   if p.stem != "true_labels"]
    print(f"  Found {len(model_names)} models: {model_names}")

    for name in model_names:
        evaluate_model(name, scorer, true_labels)

    print("\nDone.")