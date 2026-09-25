"""
Compute expected scores under the true distribution and produce the heatmap table.

For each model, computes E_{i~q}[S(p_model, i)] = sum_i q_i * S(p_model, i)
where q = true_prob from the snapshot CSV.

Scores used, as in the paper:
  - error_distance, top_1_err, top_3_err, top_5_err, rank
  - brier, resistance
  - diffusion at tau = 0.1, 0.3, 1, 3, 10, 30

Produces figures/score_table_<snapshot>.pdf

Usage
-----
    python analysis/score_table.py --snapshot source=0_run=46
    python analysis/score_table.py --snapshot source=0_run=46 \\
        --models static_gnn_n=10000 piecewise_uniform soft_margin_a=0.1_n=1000 monte_carlo_n=1000
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from propnetscore import GraphScorer

DATA_DIR = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Scores to compute
DIFFUSION_TAUS = [0.1, 0.3, 1, 3, 10, 30]

# Display names for rows (scores) and columns (models)
SCORE_DISPLAY = {
    "error_distance": "Error dist.",
    "top_1_err": "Top-1 err",
    "top_3_err": "Top-3 err",
    "top_5_err": "Top-5 err",
    "rank": "Rank",
    "brier": "Brier",
    "resistance": "Resistance",
    **{f"diff_{str(t).replace('.', '')}": f"Diff. (τ={t})"
       for t in DIFFUSION_TAUS},
}

MODEL_DISPLAY = {
    "true_prob": "True",
    "static_gnn_n=10000": "GNN",
    "piecewise_uniform": "Piecewise uniform",
    "soft_margin_a=0.1_n=1000": "SME",
    "monte_carlo_n=1000": "MCMF",
}


def load_data(snapshot_name):
    with open(DATA_DIR / "network.gpickle", "rb") as f:
        G = pickle.load(f)
    csv_path = DATA_DIR / "interesting_snapshots" / f"{snapshot_name}.csv"
    df = pd.read_csv(csv_path).set_index("node")
    return G, df


def build_piecewise_uniform(df):
    """Construct the piecewise-uniform distribution (snapshot-specific)."""
    pu = df["true_prob"].copy()
    mass_4_5_6_16 = df.loc[[4, 5, 6, 16], "true_prob"].sum()
    mass_1_17_21 = df.loc[[1, 17, 21], "true_prob"].sum()
    pu.loc[[4, 5, 6, 16]] = mass_4_5_6_16 / 4
    pu.loc[[1, 17, 21]] = mass_1_17_21 / 3
    pu.loc[11] = df.loc[11, "true_prob"]
    pu.loc[0] = df.loc[0, "true_prob"]
    return pu


def compute_expected_scores(G, df, model_columns):
    """Compute expected scores E_{i~q}[S(p, i)] for each model.

    Returns a DataFrame with models as columns and scores as rows.
    """
    A = nx.to_numpy_array(G, weight=None)
    scorer = GraphScorer(A)
    # regularisation parameter
    mu = 0.8  # pragmatic choice: value that gives values of similar order of magnitude for large and small tau
    node_order = list(G.nodes())
    N = len(node_order)

    df_ord = df.reindex(node_order).fillna(0.0)
    true_prob = df_ord["true_prob"].values  # q vector
    true_nodes = np.arange(N) # score against every node

    results = {}
    for model in model_columns:
        p = df_ord[model].values  # (N,) probability vector
        P = np.tile(p, (N, 1))   # (N, N): same p for every true node

        scores = {
            "error_distance": scorer.error_distance(P, true_nodes),
            "top_1_err": scorer.top_k_error(P, true_nodes, k=1),
            "top_3_err": scorer.top_k_error(P, true_nodes, k=3),
            "top_5_err": scorer.top_k_error(P, true_nodes, k=5),
            "rank": scorer.rank_score(P, true_nodes),
            "brier": scorer.brier_score(P, true_nodes),
            "resistance": scorer.resistance_score(P, true_nodes),
        }
        for tau in DIFFUSION_TAUS:
            key = f"diff_{str(tau).replace('.', '')}"
            scores[key] = scorer.diffusion_score(P, true_nodes, tau=tau, mu=mu)

        # expected score under true distribution q
        results[model] = {k: float(np.dot(true_prob, v)) for k, v in scores.items()}

    return pd.DataFrame(results).T  # rows = models, cols = scores


def plot_heatmap(df, snapshot_name):
    """Heatmap with models as columns, scores as rows. Same style as ImageNet table."""
    norm = (df - df.min()) / (df.max() - df.min())

    cmap = plt.cm.Greens

    n_rows, n_cols = df.shape
    col_labels = [SCORE_DISPLAY.get(c, c) for c in df.columns]
    row_labels = [MODEL_DISPLAY.get(r, r) for r in df.index]

    # fixed total width like ImageNet table, height scales with number of rows
    fig, ax = plt.subplots(figsize=(10, n_rows * 0.9))

    for i in range(n_rows):
        for j in range(n_cols):
            val = df.iloc[i, j]
            nval = norm.iloc[i, j]
            color = cmap(nval)
            rect = plt.Rectangle((j, n_rows - 1 - i), 1, 1,
                                  facecolor=color, edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            tc = "black" if brightness > 0.5 else "white"
            ax.text(j + 0.5, n_rows - 1 - i + 0.5, f"{val:.3f}",
                    ha="center", va="center", fontsize=11, color=tc)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=13)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(row_labels[::-1], fontsize=13)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / f"score_table_{snapshot_name}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=600)
    fig.savefig(FIGURES_DIR / f"score_table_{snapshot_name}.png",
                bbox_inches="tight", dpi=600)
    plt.close()
    print(f"Saved: {out.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="source=0_run=46")
    parser.add_argument("--models", nargs="+",
                        default=["true_prob", "static_gnn_n=10000", "piecewise_uniform",
                                 "soft_margin_a=0.1_n=1000", "monte_carlo_n=1000"])
    args = parser.parse_args()

    G, df = load_data(args.snapshot)

    if "piecewise_uniform" in args.models and "piecewise_uniform" not in df.columns:
        df["piecewise_uniform"] = build_piecewise_uniform(df)

    print(f"Computing expected scores for {args.snapshot}...")
    scores_df = compute_expected_scores(G, df, args.models)
    print(scores_df.round(4).to_string())

    plot_heatmap(scores_df, args.snapshot)
