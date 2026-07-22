"""
Step 7: Score decomposition analysis.

Decomposes the resistance score S(p, v) into its three constituent terms for
every validation example, for every model:

  S(p, v) = calibration(p, v) - graph_entropy(p) + const

where:
  calibration(p, v) = (p' Omega)_v   = sum_i Omega_{v,i} p_i
    -- resistance-distance-weighted sum of probabilities as seen from the
       true node v. LOW = model's mass is graph-near to the truth.
       GRAPH-AWARE, DEPENDS ON v.

  graph_entropy(p)  = 0.5 * p' Omega p
    -- how spread the distribution is relative to the graph metric.
       LOW = mass concentrated on graph-near leaves (even if flat entropy
       is high). HIGH = mass spread over graph-far leaves.
       GRAPH-AWARE, DOES NOT DEPEND ON v.

  flat_quad(p)      = 0.5 * p'p  (Brier self-term; quadratic, graph-unaware)
    -- how spread the distribution is, ignoring graph structure entirely.
       This is the natural flat analogue of graph_entropy: both are quadratic
       in p, both measure spread, and they coincide (up to a constant) on K_N.
       GRAPH-UNAWARE, DOES NOT DEPEND ON v.

These three quantities cleanly separate what's going on:
  - calibration tracks how well the model directs its uncertainty toward
    the truth specifically (the "direction" component)
  - graph_entropy tracks how diffuse the model's uncertainty is in graph
    terms (the "magnitude in graph metric" component, independent of v)
  - flat_quad tracks how diffuse in flat quadratic terms (graph-unaware baseline)

The key diagnostic is the ratio graph_entropy / flat_quad:
  > 1: model's mass is spread over graph-DISTANT classes (relative to flat spread)
  < 1: model's mass is concentrated on graph-NEAR classes (relative to flat spread)

The permutation test in 06_graph_structure_sensitivity.py is essentially a
Monte Carlo estimate of the expected graph_entropy under random leaf placement
of the same flat probability values -- the direct computation here is exact
and much cheaper.

Usage:
    python3 score_decomposition.py --all
    python3 score_decomposition.py --model efficientnet_b0

Output:
    outputs/decomposition_<model>.npz   -- per-example arrays
    outputs/decomposition_summary.csv   -- per-model summary table
    outputs/decomposition_histograms.png -- combined figure
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import evaluate

DATA_DIR = Path("./wordnet_graph")
OUT_DIR = Path("./model_validation_outputs")


# ---------------------------------------------------------------------------
# Precompute graph-only sums needed for the analytic null baseline.
# These depend only on Omega, not on any data -- compute once, use for
# all examples and all models.
# ---------------------------------------------------------------------------
def precompute_graph_sums(Omega, leaf_full_idx):
    """
    Precompute the four graph-only quantities needed for the analytic
    expected graph entropy under random permutation of non-top-1 mass.

    IMPORTANT: sums are over LEAF positions only (the n_leaves positions
    that can receive non-zero probability), NOT over all n_full positions.
    Internal graph nodes always have p=0 and must be excluded -- including
    them would give wrong results because the permutation only operates over
    leaf positions, not over internal nodes.

    All returned arrays have shape (n_leaves,), indexed by the top-1 LEAF
    index (0..n_leaves-1), not the full-graph index.

    Derivation:
      E_pi[0.5 p_pi' Omega p_pi] where pi permutes the n_leaves-1 non-top-1
      LEAF probability values across the n_leaves-1 non-top-1 LEAF positions,
      with all internal-node positions always fixed at zero:

        0.5 * [ p0^2 * Omega[k0_full, k0_full]
              + 2*p0*(S1/n_other)*sum_{j_leaf != k0} Omega[k0_full, j_full]
              + (S2/n_other)*sum_{i_leaf != k0} Omega[i_full, i_full]
              + ((S1^2-S2)/(n_other*(n_other-1)))*
                sum_{i_leaf != k0, j_leaf != k0, i!=j} Omega[i_full, j_full] ]

    where S1 = sum of non-top-1 leaf probs, S2 = sum of their squares,
    n_other = n_leaves - 1 = number of non-top-1 LEAF positions.
    """
    n_leaves = len(leaf_full_idx)

    # Submatrix of Omega restricted to leaf positions only: (n_leaves, n_leaves)
    Omega_leaf = Omega[np.ix_(leaf_full_idx, leaf_full_idx)]

    omega_self       = np.diag(Omega_leaf)                        # (n_leaves,)
    omega_cross      = Omega_leaf.sum(axis=1) - omega_self        # (n_leaves,) sum_{j_leaf != k} Omega_leaf[k,j]
    omega_diag_rest  = Omega_leaf.trace() - omega_self            # (n_leaves,) sum_{i_leaf != k} Omega_leaf[i,i]

    # sum_{i_leaf != k, j_leaf != k, i!=j} Omega_leaf[i,j]
    # = total_leaf_sum - 2*(row sum of k) + Omega_leaf[k,k] - omega_diag_rest[k]
    total_leaf = Omega_leaf.sum()
    row_sums = Omega_leaf.sum(axis=1)
    omega_offdiag_rest = total_leaf - 2*row_sums + omega_self - omega_diag_rest

    return omega_self, omega_cross, omega_diag_rest, omega_offdiag_rest


def expected_ge_null_batch(probs_leaf, leaf_full_idx,
                            omega_self, omega_cross, omega_diag_rest, omega_offdiag_rest):
    """
    Analytic expected graph entropy E_pi[0.5 p_pi' Omega p_pi] for a batch
    of examples, computed exactly (no Monte Carlo needed).

    probs_leaf: (batch, n_leaves)  -- leaf-only probability vectors
    All omega_* arrays are indexed by LEAF index (0..n_leaves-1), as returned
    by precompute_graph_sums.

    Returns: (batch,) array of expected graph entropies under random permutation
    of the non-top-1 leaf probability values.
    """
    batch_size, n_leaves = probs_leaf.shape
    n_other = n_leaves - 1

    top1_leaf = probs_leaf.argmax(axis=1)              # (batch,) -- leaf index
    p0 = probs_leaf[np.arange(batch_size), top1_leaf]  # (batch,)

    S1 = 1.0 - p0                                      # sum of non-top-1 leaf probs
    S2 = np.sum(probs_leaf**2, axis=1) - p0**2         # sum of squares of non-top-1 leaf probs

    # look up precomputed leaf-indexed sums at the top-1 LEAF position
    t1 = p0**2 * omega_self[top1_leaf]
    t2 = 2 * p0 * (S1 / n_other) * omega_cross[top1_leaf]
    t3_diag = (S2 / n_other) * omega_diag_rest[top1_leaf]
    e_pi_pj = (S1**2 - S2) / (n_other * (n_other - 1))
    t3_offdiag = e_pi_pj * omega_offdiag_rest[top1_leaf]

    return 0.5 * (t1 + t2 + t3_diag + t3_offdiag)


# ---------------------------------------------------------------------------
# Core decomposition -- all three terms in one pass
# ---------------------------------------------------------------------------
def compute_decomposition_chunk(probs_leaf_chunk, true_labels_leaf_chunk,
                                 leaf_full_idx, n_full, scorer,
                                 omega_self, omega_cross, omega_diag_rest, omega_offdiag_rest):
    """
    For a chunk of examples, compute all three terms of the resistance score
    decomposition PLUS the analytic null baseline for graph entropy,
    all in a single batched pass.

    Returns five arrays of shape (chunk_size,):
      flat_entropy, graph_entropy, calibration,
      ge_null (expected graph entropy under random permutation),
      ge_improvement (ge_null - graph_entropy; positive = better than random)
    """
    chunk_size = len(probs_leaf_chunk)

    P_full = np.zeros((chunk_size, n_full), dtype=np.float64)
    P_full[:, leaf_full_idx] = probs_leaf_chunk
    true_full = leaf_full_idx[true_labels_leaf_chunk]

    # flat quadratic spread: 0.5 * p'p  (the Brier score's self-term)
    # This is the natural flat analogue of the graph entropy 0.5 * p'Omega p:
    # both are quadratic in p, both measure spread, and they coincide (up to
    # a constant) on K_N where Omega = I - (1/N)11'.
    # Concretely, 0.5*p'p = 0.5 * sum_j p_j^2, which equals (1 - Brier)/2
    # when the true label has probability p_true.
    # We use LEAF probabilities only (no internal nodes receive mass).
    flat_quad = 0.5 * np.sum(probs_leaf_chunk ** 2, axis=1)

    # graph entropy and calibration (both graph-aware)
    P_Omega = P_full @ scorer.resistance          # (chunk_size, n_full)
    graph_entropy = 0.5 * np.sum(P_Omega * P_full, axis=1)
    rows = np.arange(chunk_size)
    calibration = P_Omega[rows, true_full]

    # analytic null: expected graph entropy if non-top-1 mass were placed randomly
    ge_null = expected_ge_null_batch(
        probs_leaf_chunk, leaf_full_idx,
        omega_self, omega_cross, omega_diag_rest, omega_offdiag_rest
    )

    # improvement: how much BETTER is the model's actual placement than random?
    # positive = model's hedging is more graph-clustered than chance
    ge_improvement = ge_null - graph_entropy

    return flat_quad, graph_entropy, calibration, ge_null, ge_improvement


def run_one_model(model_name, scorer, leaf_full_idx, graph_variant="dag",
                   chunk_size=500, omega_self=None, omega_cross=None,
                   omega_diag_rest=None, omega_offdiag_rest=None):
    probs, true_labels = evaluate.load_model_outputs(model_name)
    n_examples = len(true_labels)
    n_full = scorer.size

    flat_quad      = np.zeros(n_examples)
    graph_entropy  = np.zeros(n_examples)
    calibration    = np.zeros(n_examples)
    ge_null        = np.zeros(n_examples)
    ge_improvement = np.zeros(n_examples)

    n_chunks = (n_examples + chunk_size - 1) // chunk_size
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_examples)
        fq, ge, cal, gen, gei = compute_decomposition_chunk(
            probs[start:end], true_labels[start:end],
            leaf_full_idx, n_full, scorer,
            omega_self, omega_cross, omega_diag_rest, omega_offdiag_rest
        )
        flat_quad[start:end]      = fq
        graph_entropy[start:end]  = ge
        calibration[start:end]    = cal
        ge_null[start:end]        = gen
        ge_improvement[start:end] = gei

        if chunk_idx % 20 == 0:
            print(f"  {model_name}: chunk {chunk_idx+1}/{n_chunks}", end="\r")
    print()

    resistance_reconstructed = calibration - graph_entropy

    cache_path = OUT_DIR / f"decomposition_{model_name}_{graph_variant}.npz"
    np.savez(cache_path,
             flat_quad=flat_quad,
             graph_entropy=graph_entropy,
             calibration=calibration,
             ge_null=ge_null,
             ge_improvement=ge_improvement,
             resistance_reconstructed=resistance_reconstructed,
             true_labels=true_labels)

    return flat_quad, graph_entropy, calibration, ge_null, ge_improvement


def summarize(model_name, flat_quad, graph_entropy, calibration, ge_null, ge_improvement):
    return {
        "model": model_name,
        # flat quadratic spread: 0.5*p'p (Brier self-term, graph-unaware baseline)
        "flat_quad_mean":             flat_quad.mean(),
        "flat_quad_median":           np.median(flat_quad),
        # graph entropy: 0.5*p'Omega p (graph-aware self-spread)
        "graph_entropy_mean":         graph_entropy.mean(),
        "graph_entropy_median":       np.median(graph_entropy),
        # ratio: how much larger is graph spread than flat quadratic spread?
        # > 1 means mass is spread over graph-far classes
        # < 1 means mass is concentrated on graph-near classes
        "graph_over_flat_mean":       graph_entropy.mean() / flat_quad.mean(),
        "graph_over_flat_median":     np.median(graph_entropy) / np.median(flat_quad),
        "ge_null_mean":               ge_null.mean(),
        "ge_null_median":             np.median(ge_null),
        "ge_improvement_mean":        ge_improvement.mean(),
        "ge_improvement_median":      np.median(ge_improvement),
        "frac_ge_better_than_null":   (ge_improvement > 0).mean(),
        "pct_ge_improvement":         ge_improvement.mean() / ge_null.mean() * 100,
        "calibration_mean":           calibration.mean(),
        "calibration_median":         np.median(calibration),
        "calibration_over_graph_entropy": calibration.mean() / graph_entropy.mean(),
    }


def make_histograms(all_models_data, summary_df, graph_variant="dag"):
    """
    Three rows: flat_entropy, graph_entropy (with null overlay), ge_improvement.
    One column per model, ordered by top-1 accuracy descending (best left).
    Uses whatever models are in all_models_data -- no hardcoded model list.
    """
    # Order by top-1 accuracy if available in summary_df, otherwise alphabetical.
    # summary_df may not have top1_acc since it's a decomposition summary;
    # fall back to sorting by mean real resistance score (lower = better) or
    # just use the order they were computed in.
    if "top1_acc" in summary_df.columns:
        model_order = summary_df.sort_values("top1_acc", ascending=False).index.tolist()
    else:
        model_order = sorted(all_models_data.keys())
    # Keep only models we actually have data for
    model_order = [m for m in model_order if m in all_models_data]
    n_models = len(model_order)

    terms = ["flat_quad", "graph_entropy_with_null", "ge_improvement", "graph_over_flat"]
    term_labels = {
        "flat_quad":               "Flat quadratic spread $\\frac{1}{2}\\mathbf{p}^\\top\\mathbf{p}$\n(Brier self-term; graph-unaware)",
        "graph_entropy_with_null": "Graph entropy $\\frac{1}{2}\\mathbf{p}^\\top\\Omega\\mathbf{p}$\nvs analytic null (grey)",
        "ge_improvement":          "GE improvement over null\n$\\mathrm{GE}_{\\mathrm{null}} - \\frac{1}{2}\\mathbf{p}^\\top\\Omega\\mathbf{p}$\n(positive = better than random)",
        "graph_over_flat":         "Ratio $\\frac{1}{2}\\mathbf{p}^\\top\\Omega\\mathbf{p}\\;/\\;\\frac{1}{2}\\mathbf{p}^\\top\\mathbf{p}$\n(>1: mass on graph-far classes;\n<1: mass on graph-near classes)",
    }

    fig, axes = plt.subplots(4, n_models,
                              figsize=(max(3.2 * n_models, 20), 12),
                              sharex="row")
    fig.suptitle(f"Score decomposition across models [{graph_variant}]",
                 fontsize=13, y=1.01)

    colors = {"flat_quad": "#4C72B0", "graph_entropy_with_null": "#DD8452",
              "ge_improvement": "#55A868", "graph_over_flat": "#9370DB"}

    for row, term in enumerate(terms):
        if term == "graph_entropy_with_null":
            data_key = "graph_entropy"
        elif term == "graph_over_flat":
            data_key = "graph_over_flat"  # computed on the fly below
        else:
            data_key = term

        # For ratio, compute on the fly from stored arrays
        def get_vals(model_name, data_key):
            if data_key == "graph_over_flat":
                ge = all_models_data[model_name]["graph_entropy"]
                fq = all_models_data[model_name]["flat_quad"]
                return ge / np.maximum(fq, 1e-12)
            return all_models_data[model_name][data_key]

        all_real = np.concatenate([get_vals(m, data_key) for m in model_order])
        xmax = np.percentile(all_real, 99)
        xmin = np.percentile(all_real, 1)

        for col, model_name in enumerate(model_order):
            ax = axes[row, col]
            vals = get_vals(model_name, data_key)
            vals_clipped = np.clip(vals, xmin, xmax)
            color = colors[term]

            ax.hist(vals_clipped, bins=60, color=color, alpha=0.75,
                    density=True, edgecolor="none",
                    label=f"real (med={np.median(vals):.2f})")

            if term == "graph_entropy_with_null":
                null_vals = all_models_data[model_name]["ge_null"]
                null_clipped = np.clip(null_vals, xmin, xmax)
                ax.hist(null_clipped, bins=60, color="grey", alpha=0.4,
                        density=True, edgecolor="none",
                        label=f"null (med={np.median(null_vals):.2f})")

            ax.axvline(vals.mean(), color="black", lw=1.5, ls="-",
                       label=f"mean={vals.mean():.2f}")
            ax.axvline(np.median(vals), color="black", lw=1.5, ls="--")

            if term == "ge_improvement":
                ax.axvline(0, color="red", lw=1.0, ls=":", alpha=0.7,
                           label="random=0")
            if term == "graph_over_flat":
                ax.axvline(1.0, color="red", lw=1.0, ls=":", alpha=0.7,
                           label="flat=1")

            ax.legend(fontsize=6.5, loc="upper right")
            if row == 0:
                short = (model_name
                         .replace("mobilenet_v3_", "mbv3_")
                         .replace("shufflenet_v2_x1_0", "shufflenet")
                         .replace("efficientnet_b0", "effnet_b0")
                         .replace("efficientnet_b2", "effnet_b2")
                         .replace("vit_small_patch16_224", "vit_s16")
                         .replace("swin_tiny", "swin_t")
                         .replace("convnext_tiny", "convnext_t")
                         .replace("resnet50_timm", "resnet50")
                         .replace("resnet18", "resnet18"))
                ax.set_title(short, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(term_labels[term], fontsize=8)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    out_path = OUT_DIR / f"decomposition_histograms_{graph_variant}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved histogram figure to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--graph", choices=["dag", "tree", "ctree"], default="dag",
                        help="Which graph to use (dag/tree/ctree). Output files are suffixed accordingly.")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    graph_variant = args.graph

    available = evaluate.discover_available_models()
    if not available:
        raise RuntimeError("No model outputs found. Run run_inference.py first.")
    model_names = available if args.all else (args.models or [available[0]])
    for m in model_names:
        assert m in available, f"No outputs for '{m}'. Available: {available}"

    print(f"Loading graph and building GraphScorer [{graph_variant}]...")
    G, leaf_info = evaluate.load_graph_and_leaf_info(graph_variant)
    leaf_names = leaf_info["class_idx_to_synset_name"]
    scorer, _, leaf_full_idx = evaluate.build_full_graph_scorer(G, leaf_names, graph_variant)

    print("Precomputing graph-only sums for analytic null baseline...")
    omega_self, omega_cross, omega_diag_rest, omega_offdiag_rest = \
        precompute_graph_sums(scorer.resistance, leaf_full_idx)

    all_summaries = []
    all_models_data = {}

    for model_name in model_names:
        cache = OUT_DIR / f"decomposition_{model_name}_{graph_variant}.npz"
        if cache.exists() and not args.recompute:
            print(f"Loading cached decomposition for {model_name} [{graph_variant}]...")
            data = np.load(cache)
            # Support both old (flat_entropy) and new (flat_quad) cache format
            if "flat_quad" in data:
                flat_quad = data["flat_quad"]
            else:
                print(f"  WARNING: cache uses old flat_entropy format; "
                      f"use --recompute to regenerate with flat_quad.")
                flat_quad = data["flat_entropy"]
            graph_entropy  = data["graph_entropy"]
            calibration    = data["calibration"]
            ge_null        = data["ge_null"]
            ge_improvement = data["ge_improvement"]
        else:
            print(f"\nComputing decomposition for {model_name} [{graph_variant}]...")
            flat_quad, graph_entropy, calibration, ge_null, ge_improvement = \
                run_one_model(model_name, scorer, leaf_full_idx,
                              graph_variant=graph_variant,
                              chunk_size=args.chunk_size,
                              omega_self=omega_self, omega_cross=omega_cross,
                              omega_diag_rest=omega_diag_rest,
                              omega_offdiag_rest=omega_offdiag_rest)

        all_models_data[model_name] = {
            "flat_quad":     flat_quad,
            "graph_entropy": graph_entropy,
            "calibration":   calibration,
            "ge_null":       ge_null,
            "ge_improvement":ge_improvement,
        }

        summary = summarize(model_name, flat_quad, graph_entropy, calibration,
                             ge_null, ge_improvement)
        all_summaries.append(summary)

        ratio_mean   = graph_entropy.mean() / max(flat_quad.mean(), 1e-12)
        ratio_median = np.median(graph_entropy) / max(np.median(flat_quad), 1e-12)
        print(f"\n  {model_name} [{graph_variant}]:")
        print(f"    flat_quad      mean={flat_quad.mean():.4f}  median={np.median(flat_quad):.4f}")
        print(f"    graph_entropy  mean={graph_entropy.mean():.4f}  median={np.median(graph_entropy):.4f}")
        print(f"    graph/flat ratio  mean={ratio_mean:.3f}  median={ratio_median:.3f}")
        print(f"    ge_null        mean={ge_null.mean():.2f}  median={np.median(ge_null):.2f}")
        print(f"    ge_improvement mean={ge_improvement.mean():.2f}  median={np.median(ge_improvement):.2f}")
        print(f"    frac better than null: {(ge_improvement>0).mean():.3f}")
        print(f"    pct GE improvement: {ge_improvement.mean()/ge_null.mean()*100:.2f}%")
        print(f"    calibration    mean={calibration.mean():.2f}  median={np.median(calibration):.2f}")

    summary_df = pd.DataFrame(all_summaries).set_index("model")
    out_csv = OUT_DIR / f"decomposition_summary_{graph_variant}.csv"
    summary_df.to_csv(out_csv)
    print(f"\n{'='*70}\nDECOMPOSITION SUMMARY TABLE  [{graph_variant}]\n{'='*70}")
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))

    if len(model_names) >= 2:
        make_histograms(all_models_data, summary_df, graph_variant)
    else:
        print("(Need >= 2 models to generate comparison histograms.)")


if __name__ == "__main__":
    main()