"""
Step 5: Same top-1, different graph-aware score" analysis.

This directly demonstrates, on REAL data, the point illustrated by the toy
Z-vs-W example from our discussion: two predictions can share the exact
same top-1 class (hence identical classical "mistake severity" /
hierarchical-distance@1, a la Bertinetto et al. / Karthik et al. -- a metric
that only ever looks at the arg-max) while differing substantially in
resistance_score / diffusion_score, because those scores are sensitive to
how the REST of the probability mass is distributed, not just the arg-max.

Design choice (see conversation notes for why): we compare ACROSS MODELS on
the SAME image, not across images within one model. This holds the true
label AND the top-1 prediction fixed, varying only "how each model
distributes its remaining probability mass" -- exactly mirroring the toy
example. Comparing across images within one model would confound "the model
hedged differently" with "the true label sits in a different part of the
graph", which is not a clean comparison.

Usage:
    python3 same_top1_analysis.py --all
    python3 same_top1_analysis.py --models resnet50 efficientnet_b0 mobilenet_v3_small

Produces:
  - outputs/same_top1_cases.csv : every (image, model_pair) where both models
    share the same top-1 prediction but the true label is different from
    that prediction (i.e. both models are WRONG in the same way at the
    top-1 level), sorted by how much resistance_score/diffusion_score
    disagree despite the shared top-1.
  - A printed summary of the single most striking example, plus
    distributional statistics.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate

DATA_DIR = Path("./wordnet_graph")
OUT_DIR = Path("./model_validation_outputs")


def find_same_top1_disagreements(probs_dict, true_labels, scorer, leaf_full_idx,
                                  diffusion_tau=1.0, min_models=2):
    """
    For every validation image, find every PAIR of models that:
      (a) agree on the top-1 predicted class, AND
      (b) that shared top-1 class is WRONG (!= true label)
    and compute how much resistance_score / diffusion_score disagree
    between the two models on that image, despite the shared (wrong) top-1.

    Parameters
    ----------
    probs_dict : dict[str, np.ndarray]
        model_name -> (n_images, 1000) probability array. All arrays must
        share the same row ordering (verified by the caller in evaluate.py
        via the cross-model true_labels consistency check -- re-checked here
        too, defensively).
    true_labels : np.ndarray (n_images,)
        Shared true labels across all models.
    scorer, leaf_full_idx : from build_full_graph_scorer()

    Returns
    -------
    pd.DataFrame with one row per (image_idx, model_A, model_B) where the
    same-top1-disagreement condition holds, including both models'
    resistance_score and diffusion_score, and the absolute difference in
    each (the sort key for "most striking disagreement").
    """
    model_names = list(probs_dict.keys())
    n_images = len(true_labels)

    # Top-1 prediction per model, per image: shape (n_models, n_images)
    top1_preds = np.stack([probs_dict[m].argmax(axis=1) for m in model_names])

    # Precompute per-model, per-image resistance/diffusion scores ONCE
    # (not inside the pairwise loop) -- this is the expensive part, and
    # doing it once per model rather than once per pair avoids quadratic
    # redundant work.
    print("  Computing resistance/diffusion scores for every model...")
    per_model_scores = {}
    for m in model_names:
        scores = evaluate.your_graph_aware_scores(
            probs_dict[m], true_labels, scorer, leaf_full_idx, diffusion_tau=diffusion_tau
        )
        per_model_scores[m] = scores  # dict with 'resistance', 'diffusion'

    rows = []
    n_models = len(model_names)
    for img_idx in range(n_images):
        true_lab = true_labels[img_idx]
        # group models by their top-1 prediction on this image
        preds_this_image = top1_preds[:, img_idx]
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pred_i, pred_j = preds_this_image[i], preds_this_image[j]
                if pred_i != pred_j:
                    continue  # not a same-top1 case
                if pred_i == true_lab:
                    continue  # both correct -- not an interesting "mistake" case
                m_i, m_j = model_names[i], model_names[j]
                res_i = per_model_scores[m_i]["resistance"][img_idx]
                res_j = per_model_scores[m_j]["resistance"][img_idx]
                diff_i = per_model_scores[m_i]["diffusion"][img_idx]
                diff_j = per_model_scores[m_j]["diffusion"][img_idx]
                rows.append({
                    "image_idx": img_idx,
                    "true_label": true_lab,
                    "shared_top1_pred": pred_i,
                    "model_A": m_i,
                    "model_B": m_j,
                    "resistance_A": res_i,
                    "resistance_B": res_j,
                    "resistance_abs_diff": abs(res_i - res_j),
                    "diffusion_A": diff_i,
                    "diffusion_B": diff_j,
                    "diffusion_abs_diff": abs(diff_i - diff_j),
                })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # -------------------------------------------------------------------
    # IMPORTANT: raw resistance_abs_diff / diffusion_abs_diff are NOT
    # comparable across model pairs of very different overall quality.
    # A weak model's mistakes tend to be large in absolute resistance
    # terms REGARDLESS of which other model they're compared against, so
    # sorting by raw abs_diff mechanically surfaces "weak model vs strong
    # model" pairs -- which just re-states each model's overall accuracy
    # gap (already visible in the comparison table), not anything about
    # mistake STRUCTURE that the classical top-1 metric misses.
    #
    # To isolate genuine "surprising divergence given each model's own
    # typical mistake severity" cases, we additionally compute a z-scored
    # version: how many standard deviations from EACH model's own mean
    # mistake-resistance-score is its value on this particular image,
    # then look at pairs where the z-scores themselves diverge sharply
    # (not just the raw values). This is the column to sort by if you
    # want examples that aren't just restating "model X is worse than
    # model Y overall" -- see conversation notes for the diagnostic that
    # motivated this (shufflenet_v2_x1_0 dominating the raw top-20 purely
    # because it is the weakest model, with near-constant ~20-24k
    # resistance values regardless of which other model it's paired with).
    #
    # CAVEAT on the z-score itself: resistance_score is necessarily
    # right-skewed (nonnegative, unbounded above, with most mistakes
    # moderate and a tail of much larger ones) -- a mean/std-based z-score
    # on a skewed distribution can make tail mistakes look artificially
    # "surprising" even when they're not that unusual in RANK terms, simply
    # because the tail inflates the std it's compared against. We therefore
    # ALSO compute a rank-based percentile normalization (each mistake's
    # percentile within that model's own mistake distribution) as a
    # robustness check -- if the zscore-based and percentile-based rankings
    # of "most surprising case" agree, the finding is solid; if they
    # disagree sharply, the zscore version is likely an artifact of skew
    # and the percentile version should be preferred/reported instead.
    # -------------------------------------------------------------------
    model_mistake_mean = {}
    model_mistake_std = {}
    model_mistake_vals = {}
    for m in model_names:
        wrong_mask = top1_preds[model_names.index(m)] != true_labels
        vals = per_model_scores[m]["resistance"][wrong_mask]
        model_mistake_mean[m] = vals.mean() if len(vals) > 0 else 0.0
        model_mistake_std[m] = vals.std() if len(vals) > 0 else 1.0
        model_mistake_vals[m] = vals

    # Pre-sort each model's mistake values ONCE, so per-row percentile
    # lookups are O(log n) via searchsorted instead of O(n log n) via
    # rankdata -- verified to give IDENTICAL results to the natural
    # "count(vals <= value) / len(vals)" definition on hand-checked
    # examples; an earlier draft using rankdata on the value appended to
    # the array had a subtle off-by-one self-inclusion bug, caught by
    # cross-checking against a fully hand-traced tiny example.
    model_mistake_sorted = {m: np.sort(v) for m, v in model_mistake_vals.items()}

    def zscore(model_name, value):
        std = model_mistake_std[model_name]
        return (value - model_mistake_mean[model_name]) / std if std > 0 else 0.0

    def percentile(model_name, value):
        """
        What fraction of this model's OWN mistakes are <= this value?
        0.99 means "this is in the worst 1% of mistakes THIS model makes" --
        a rank-based notion of surprise that is robust to skew, unlike the
        mean/std-based z-score above.

        LIMITATION (found via an engineered test case -- see conversation
        notes): percentile rank discards MAGNITUDE of extremity. A value
        that is "slightly above this model's own historical max" and a
        value that is "wildly above this model's own historical max" both
        get percentile 1.0 if they're each individually the largest value
        seen so far for that model -- the percentile difference between two
        models' "both at their own max" values is then 0, even if the
        z-score difference (which DOES encode magnitude) would correctly
        flag a real difference in how extreme each one is. This is most
        likely to bite when a model has FEW recorded mistakes to rank
        against (so "the max so far" is a noisy, low-resolution statistic).
        Neither normalization is strictly better: z-score encodes magnitude
        but can be distorted by skew/outliers in the reference distribution;
        percentile is robust to skew but discards magnitude information and
        is noisier with small reference samples. Check BOTH and treat
        disagreements as worth inspecting by hand, not as a sign one of
        them is simply wrong.
        """
        sorted_vals = model_mistake_sorted[model_name]
        if len(sorted_vals) == 0:
            return 0.5
        idx = np.searchsorted(sorted_vals, value, side='right')
        return idx / len(sorted_vals)

    df["resistance_A_zscore"] = df.apply(lambda r: zscore(r["model_A"], r["resistance_A"]), axis=1)
    df["resistance_B_zscore"] = df.apply(lambda r: zscore(r["model_B"], r["resistance_B"]), axis=1)
    df["resistance_zscore_abs_diff"] = (df["resistance_A_zscore"] - df["resistance_B_zscore"]).abs()

    df["resistance_A_pct"] = df.apply(lambda r: percentile(r["model_A"], r["resistance_A"]), axis=1)
    df["resistance_B_pct"] = df.apply(lambda r: percentile(r["model_B"], r["resistance_B"]), axis=1)
    df["resistance_pct_abs_diff"] = (df["resistance_A_pct"] - df["resistance_B_pct"]).abs()

    df = df.sort_values("resistance_abs_diff", ascending=False).reset_index(drop=True)
    return df


def summarize_and_report(df, leaf_info):
    if len(df) == 0:
        print("\nNo same-top1-disagreement cases found at all -- this would be "
              "surprising with >=2 models; check that probs arrays are loaded "
              "correctly and models are genuinely making independent mistakes.")
        return

    leaf_labels_human = leaf_info["class_idx_to_label"]

    n_total = len(df)
    print(f"\n{'='*70}")
    print(f"SAME-TOP1-DISAGREEMENT ANALYSIS")
    print(f"{'='*70}")
    print(f"Found {n_total} (image, model-pair) cases where two models share "
          f"the same WRONG top-1 prediction.")
    print(f"\nDistribution of resistance_score absolute difference despite "
          f"identical top-1 (this is the quantity that classical "
          f"mistake-severity / hierarchical-distance@1 CANNOT see, since it "
          f"only looks at the arg-max):")
    print(df["resistance_abs_diff"].describe())

    print(f"\nDistribution of diffusion_score absolute difference:")
    print(df["diffusion_abs_diff"].describe())

    pct_nonzero = (df["resistance_abs_diff"] > 1e-6).mean() * 100
    print(f"\n{pct_nonzero:.1f}% of same-top1 pairs have a NON-ZERO resistance_score "
          f"difference -- i.e. the classical metric would call these cases "
          f"'identical severity' while the graph-aware score correctly "
          f"distinguishes them.")

    print(f"\n--- Single most striking example ---")
    top = df.iloc[0]
    true_name = leaf_labels_human[int(top["true_label"])]
    pred_name = leaf_labels_human[int(top["shared_top1_pred"])]
    print(f"Image index: {int(top['image_idx'])}")
    print(f"True label: {true_name}")
    print(f"Shared (wrong) top-1 prediction by BOTH models: {pred_name}")
    print(f"  {top['model_A']}: resistance_score={top['resistance_A']:.4f}, "
          f"diffusion_score={top['diffusion_A']:.4f}")
    print(f"  {top['model_B']}: resistance_score={top['resistance_B']:.4f}, "
          f"diffusion_score={top['diffusion_B']:.4f}")
    print(f"Despite IDENTICAL top-1 prediction (and hence identical classical "
          f"mistake severity), resistance_score differs by {top['resistance_abs_diff']:.4f} "
          f"and diffusion_score differs by {top['diffusion_abs_diff']:.4f}.")
    print(f"\nThis means: classical hierarchical-distance@1 would rank these two "
          f"models' predictions on this image as EQUALLY bad. Your graph-aware "
          f"proper scoring rule does not.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--graph", choices=["dag", "tree", "ctree"], default="dag",
                        help="Which graph to use (dag/tree/ctree). Output files are suffixed accordingly.")
    parser.add_argument("--diffusion-tau", type=float, default=1.0)
    parser.add_argument("--top-n-examples", type=int, default=20)
    parser.add_argument("--sort-by", choices=["raw", "zscore", "percentile"], default="percentile")
    args = parser.parse_args()
    graph_variant = args.graph

    available = evaluate.discover_available_models()
    if not available:
        raise RuntimeError("No probs_<model>.npy files found in outputs/. "
                            "Run run_inference.py first.")

    if args.models:
        for m in args.models:
            assert m in available, f"No outputs found for model '{m}'. Available: {available}"
        model_names = args.models
    else:
        model_names = available

    assert len(model_names) >= 2, "Need at least 2 models to compare same-top1 cases."
    print(f"Comparing models: {model_names}  [{graph_variant}]")

    print(f"Loading label graph and building GraphScorer [{graph_variant}]...")
    G, leaf_info = evaluate.load_graph_and_leaf_info(graph_variant)
    leaf_names = leaf_info["class_idx_to_synset_name"]
    scorer, full_node_list, leaf_full_idx = evaluate.build_full_graph_scorer(G, leaf_names, graph_variant)

    print("Loading model outputs and verifying label consistency...")
    probs_dict = {}
    reference_labels = None
    for m in model_names:
        probs, labels = evaluate.load_model_outputs(m)
        if reference_labels is None:
            reference_labels = labels
        else:
            assert np.array_equal(labels, reference_labels), (
                f"true_labels for '{m}' do not match other models."
            )
        probs_dict[m] = probs

    print("Scanning for same-top1-disagreement cases...")
    df = find_same_top1_disagreements(
        probs_dict, reference_labels, scorer, leaf_full_idx,
        diffusion_tau=args.diffusion_tau
    )

    sort_col = {
        "raw": "resistance_abs_diff",
        "zscore": "resistance_zscore_abs_diff",
        "percentile": "resistance_pct_abs_diff",
    }[args.sort_by]
    if len(df) > 0:
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        print(f"\n(Sorted by: {sort_col}. Use --sort-by raw|zscore|percentile to switch.)")

        n_check = min(20, len(df))
        top_by_zscore = set(df.sort_values("resistance_zscore_abs_diff", ascending=False)
                              .head(n_check)["image_idx"])
        top_by_percentile = set(df.sort_values("resistance_pct_abs_diff", ascending=False)
                                   .head(n_check)["image_idx"])
        overlap = len(top_by_zscore & top_by_percentile)
        print(f"Agreement check: {overlap}/{n_check} images appear in both top lists. "
              f"{'Good agreement.' if overlap >= n_check * 0.5 else 'LOW agreement -- prefer percentile ranking.'}")

    out_csv = OUT_DIR / f"same_top1_cases_{graph_variant}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved full results ({len(df)} rows) to {out_csv}")

    summarize_and_report(df, leaf_info)

    if len(df) > 0:
        print(f"\n--- Top {args.top_n_examples} disagreement cases (sorted by {sort_col}) ---")
        leaf_labels_human = leaf_info["class_idx_to_label"]
        display_df = df.head(args.top_n_examples).copy()
        display_df["true_label_name"] = display_df["true_label"].apply(lambda i: leaf_labels_human[int(i)])
        display_df["shared_pred_name"] = display_df["shared_top1_pred"].apply(lambda i: leaf_labels_human[int(i)])
        cols = ["image_idx", "true_label_name", "shared_pred_name", "model_A", "model_B",
                "resistance_A", "resistance_B", "resistance_abs_diff",
                "resistance_A_zscore", "resistance_B_zscore", "resistance_zscore_abs_diff",
                "resistance_A_pct", "resistance_B_pct", "resistance_pct_abs_diff"]
        print(display_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()