"""
Step 6 (obsolete): Graph-structure sensitivity analysis.

For each model, for each validation example, computes:

  improvement_from_structure(p, v) = E_pi[S(p_pi, v)] - S(p, v)

where:
  - S(p, v) is the real resistance_score of the model's actual prediction
  - p_pi is a "permuted" version of p: the top-1 leaf's probability is held
    fixed, but the remaining (999) probability values are randomly reassigned
    to different leaf positions -- i.e. the same MULTISET of non-top-1
    probabilities, placed at uniformly random leaves
  - E_pi[...] averages over n_trials such permutations

Positive value = model's actual hedging is BETTER than random graph placement
  of the same probability mass (the model's non-top-1 uncertainty is
  directed toward graph-near alternatives)
Negative value = model's hedging is WORSE than random (anti-correlated with
  the graph -- its uncertainty is specifically directed toward graph-far classes)
Zero = indistinguishable from random graph placement (non-top-1 mass carries
  no graph-structural information beyond what you'd expect by chance)

NOTE on what's being controlled for: the permutation preserves the EXACT
multiset of non-top-1 probability values (their magnitudes), only randomizing
which LEAF POSITIONS they land on. This means:
  - Flat Brier score: INVARIANT under this permutation (depends only on
    the probability assigned to the true leaf, and on the sum of squares of
    all probabilities -- the latter is permutation-invariant)
  - Log score: INVARIANT (depends only on p(true_leaf), which we don't touch)
  - Top-k accuracy: INVARIANT (top-1 is held fixed; top-k for k>1 changes
    but the permutation is of a different quantity than these metrics use)
  - Resistance score (p' Omega p TERM): changes, because Omega encodes
    graph distances -- the same mass at different graph positions gives
    different p'Omega p values
  - Resistance score ((p'Omega)_v TERM): changes, because graph distance
    from permuted mass positions to the true node v changes

So "improvement_from_structure" specifically measures what resistance_score
captures that all classical metrics structurally CANNOT: whether the model's
non-top-1 probability mass is placed at graph-near vs graph-far leaves.

Terminology note: what we are permuting is the FLAT distribution of
probability masses. In the score's own terms, the "self-spread" term
p'Omega p is graph-aware (not flat entropy) -- so this test controls for
flat spread and isolates graph-directed placement, but does NOT decompose
the score into its own two components (p'Omega p vs (p'Omega)_v).

Usage:
    python3 graph_structure_sensitivity.py --all
    python3 graph_structure_sensitivity.py --model resnet50 --n-trials 20

Output:
    outputs/structure_sensitivity_<model>.npz  -- per-example arrays
    outputs/structure_sensitivity_summary.csv  -- per-model aggregate table
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import evaluate

DATA_DIR = Path("./wordnet_graph")
OUT_DIR = Path("./model_validation_outputs")


def compute_permuted_control_chunk(probs_leaf_chunk, true_labels_leaf_chunk,
                                    n_trials, leaf_full_idx, n_full,
                                    scorer, rng):
    """
    Compute E_pi[resistance_score(p_pi, v)] for a chunk of examples.

    Processes chunk_size * n_trials rows in a single batched resistance_score
    call for efficiency (avoids per-example overhead).

    Returns: np.ndarray of shape (chunk_size,) -- mean permuted score per example.
    """
    chunk_size = len(probs_leaf_chunk)
    total_rows = chunk_size * n_trials

    P_all = np.zeros((total_rows, n_full), dtype=np.float64)
    true_all = np.zeros(total_rows, dtype=int)

    for ex in range(chunk_size):
        p_leaf = probs_leaf_chunk[ex]
        top1 = np.argmax(p_leaf)

        non_top1_mask = np.ones(len(p_leaf), dtype=bool)
        non_top1_mask[top1] = False
        non_top1_vals = p_leaf[non_top1_mask]            # (n_leaves-1,)
        non_top1_full_pos = leaf_full_idx[non_top1_mask] # (n_leaves-1,)

        # generate all n_trials permutations at once
        noise = rng.rand(n_trials, len(p_leaf) - 1)
        perm_indices = np.argsort(noise, axis=1)          # (n_trials, n_leaves-1)

        row_start = ex * n_trials
        row_end = row_start + n_trials

        # place top-1 probability at its fixed full-space position
        P_all[row_start:row_end, leaf_full_idx[top1]] = p_leaf[top1]
        # scatter permuted non-top-1 values
        # non_top1_vals[perm_indices] has shape (n_trials, n_leaves-1)
        P_all[row_start:row_end][:, non_top1_full_pos] = non_top1_vals[perm_indices]

        true_all[row_start:row_end] = leaf_full_idx[true_labels_leaf_chunk[ex]]

    scores_all = scorer.resistance_score(P_all, true_all)
    return scores_all.reshape(chunk_size, n_trials).mean(axis=1)


def compute_real_scores_chunk(probs_leaf_chunk, true_labels_leaf_chunk,
                               leaf_full_idx, n_full, scorer):
    """Compute real resistance_score for a chunk, for the 'improvement' denominator."""
    chunk_size = len(probs_leaf_chunk)
    P_full = np.zeros((chunk_size, n_full), dtype=np.float64)
    P_full[:, leaf_full_idx] = probs_leaf_chunk
    true_full = leaf_full_idx[true_labels_leaf_chunk]
    return scorer.resistance_score(P_full, true_full)


def run_one_model(model_name, scorer, leaf_full_idx, n_trials, chunk_size, rng_seed):
    """
    Compute per-example real scores, permuted-control means, and their
    difference (improvement_from_structure) for one model's saved predictions.
    """
    probs, true_labels = evaluate.load_model_outputs(model_name)
    n_examples = len(true_labels)
    n_full = scorer.size

    real_scores = np.zeros(n_examples)
    control_means = np.zeros(n_examples)

    rng = np.random.RandomState(rng_seed)

    n_chunks = (n_examples + chunk_size - 1) // chunk_size
    for chunk_idx in tqdm(range(n_chunks), desc=f"{model_name}", unit="chunk"):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_examples)
        chunk_probs = probs[start:end]
        chunk_labels = true_labels[start:end]

        real_scores[start:end] = compute_real_scores_chunk(
            chunk_probs, chunk_labels, leaf_full_idx, n_full, scorer
        )
        control_means[start:end] = compute_permuted_control_chunk(
            chunk_probs, chunk_labels, n_trials, leaf_full_idx, n_full, scorer, rng
        )

    improvement = control_means - real_scores

    cache_path = OUT_DIR / f"structure_sensitivity_{model_name}_{graph_variant}.npz"
    np.savez(cache_path,
             real_scores=real_scores,
             control_means=control_means,
             improvement=improvement,
             true_labels=true_labels)

    return real_scores, control_means, improvement, true_labels


def summarize(model_name, real_scores, control_means, improvement, true_labels,
              probs=None):
    """
    Compute per-model summary statistics for the improvement_from_structure.
    """
    n = len(improvement)
    summary = {
        "model": model_name,
        "n_examples": n,
        # aggregate improvement (positive = model's hedging is on average
        # graph-near relative to random; negative = graph-far)
        "mean_improvement": improvement.mean(),
        "median_improvement": np.median(improvement),
        # fraction where the model's actual hedging beats random placement
        "frac_better_than_random": (improvement > 0).mean(),
        # real score and control baseline (for context / sanity check)
        "mean_real_score": real_scores.mean(),
        "mean_control_score": control_means.mean(),
        # how much does the control baseline exceed the real score, in % terms?
        "pct_improvement_vs_control": (
            (control_means.mean() - real_scores.mean()) / control_means.mean() * 100
        ),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--graph", choices=["dag", "tree", "ctree"], default="dag",
                        help="Which graph to use (dag/tree/ctree). Output files are suffixed accordingly.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    graph_variant = args.graph

    available = evaluate.discover_available_models()
    if not available:
        raise RuntimeError("No model outputs found in outputs/. Run run_inference.py first.")

    model_names = available if args.all else (args.models or [available[0]])
    for m in model_names:
        assert m in available, f"No outputs for '{m}'. Available: {available}"

    print(f"Loading graph and building GraphScorer [{graph_variant}]...")
    G, leaf_info = evaluate.load_graph_and_leaf_info(graph_variant)
    leaf_names = leaf_info["class_idx_to_synset_name"]
    scorer, _, leaf_full_idx = evaluate.build_full_graph_scorer(G, leaf_names, graph_variant)

    all_summaries = []
    for model_name in model_names:
        cache = OUT_DIR / f"structure_sensitivity_{model_name}_{graph_variant}.npz"
        if cache.exists() and not args.recompute:
            print(f"\nLoading cached results for {model_name} [{graph_variant}] from {cache}")
            data = np.load(cache)
            real_scores = data["real_scores"]
            control_means = data["control_means"]
            improvement = data["improvement"]
            true_labels = data["true_labels"]
        else:
            print(f"\nComputing for {model_name} [{graph_variant}] "
                  f"({args.n_trials} trials, chunk_size={args.chunk_size})...")
            real_scores, control_means, improvement, true_labels = run_one_model(
                model_name, scorer, leaf_full_idx,
                n_trials=args.n_trials,
                chunk_size=args.chunk_size,
                rng_seed=args.seed,
            )

        summary = summarize(model_name, real_scores, control_means, improvement, true_labels)
        all_summaries.append(summary)

        print(f"\n  mean_improvement (control - real): {summary['mean_improvement']:.2f}")
        print(f"  frac better than random:           {summary['frac_better_than_random']:.3f}")
        print(f"  pct improvement vs control:        {summary['pct_improvement_vs_control']:.2f}%")
        print(f"  mean real score:                   {summary['mean_real_score']:.2f}")
        print(f"  mean control score:                {summary['mean_control_score']:.2f}")

    summary_df = pd.DataFrame(all_summaries).set_index("model")
    out_csv = OUT_DIR / f"structure_sensitivity_summary_{graph_variant}.csv"
    summary_df.to_csv(out_csv)
    print(f"\n{'='*70}\nSUMMARY TABLE  [{graph_variant}]\n{'='*70}")
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()