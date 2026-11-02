"""
Produce appendix figures for every snapshot in data/interesting_snapshots/:
  figures/appendix/network_<snapshot>.pdf    — infection status + true distribution
  figures/appendix/score_table_<snapshot>.pdf — full score heatmap (all models)

Usage
-----
    python analysis/figures_appendix.py                          # all snapshots
    python analysis/figures_appendix.py --snapshots source=0_run=46 source=1_run=12
"""

import argparse
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from propnetscore import GraphScorer

DATA_DIR    = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures" / "appendix"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Visual parameters (matching visualise.py)
TRANSFORMATION_A  = 0.5
CMAP              = plt.cm.Grays
NODE_SIZE         = 400
LINEWIDTHS        = 1.5
EDGE_COLOR        = "gray"
PROB_LABEL_FS     = 8
PROB_LABEL_OFFSET = 0.005
PROB_DECIMALS     = 2
NETWORK_SEED      = 42

DIFFUSION_TAUS = [0.1, 0.3, 1, 3, 10, 30]

SCORE_DISPLAY = {
    "error_distance": "Error dist.",
    "top_1_err":      "Top-1 err",
    "top_3_err":      "Top-3 err",
    "top_5_err":      "Top-5 err",
    "rank":           "Rank",
    "brier":          "Brier",
    "resistance":     "Resistance",
    **{f"diff_{str(t).replace('.', '')}": f"Diff. (τ={t})" for t in DIFFUSION_TAUS},
}

# Default model order: outer n, inner GNN/MCMF/SME, innermost a ascending
_NS     = [10, 100, 1000, 10000]
_ALPHAS = [0.1, 0.15, 0.2, 0.3, 0.4]
DEFAULT_MODEL_ORDER = ["true_prob"] + [
    col
    for n in _NS
    for col in (
        [f"static_gnn_n={n}", f"monte_carlo_n={n}"]
        + [f"soft_margin_a={a}_n={n}" for a in _ALPHAS]
    )
]

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

def _model_label(col):
    if col == "true_prob":
        return "True"
    m = re.fullmatch(r"static_gnn_n=(\d+)", col)
    if m:
        return f"GNN{m.group(1).translate(_SUB)}"
    m = re.fullmatch(r"monte_carlo_n=(\d+)", col)
    if m:
        return f"MCMF{m.group(1).translate(_SUB)}"
    m = re.fullmatch(r"soft_margin_a=([\d.]+)_n=(\d+)", col)
    if m:
        a = m.group(1).rstrip("0").rstrip(".")
        return f"SME{m.group(2).translate(_SUB)}({a})"
    return col


def _draw_panel(ax, G, pos, node_order, probs_dict, title, infection_colors=None):
    if infection_colors is not None:
        nx.draw(G, pos=pos, ax=ax, node_color=infection_colors, with_labels=False,
                node_size=NODE_SIZE, edgecolors="black", linewidths=LINEWIDTHS,
                edge_color=EDGE_COLOR)
    else:
        probs = np.array([probs_dict.get(n, 0.0) for n in node_order])
        nx.draw_networkx_nodes(G, pos=pos, ax=ax, nodelist=node_order,
                               node_color=np.power(probs, TRANSFORMATION_A),
                               cmap=CMAP, node_size=NODE_SIZE,
                               vmin=0, vmax=1.0,
                               edgecolors="black", linewidths=LINEWIDTHS)
        nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color=EDGE_COLOR)
        for n in node_order:
            p = probs_dict.get(n, 0.0)
            if p > 0:
                x, y = pos[n]
                ax.text(x, y - PROB_LABEL_OFFSET, f"{p:.{PROB_DECIMALS}f}",
                        ha="center", va="center", fontsize=PROB_LABEL_FS, color="black")
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_frame_on(False)


def plot_network(G, df, snapshot_name, pos):
    node_order = list(G.nodes())
    df_full = df.reindex(node_order).fillna(0.0)
    infected_colors = ["red" if df_full.loc[n, "infected"] == 1 else "blue" for n in node_order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    _draw_panel(axes[0], G, pos, node_order, {}, "(a) Infection status",
                infection_colors=infected_colors)
    _draw_panel(axes[1], G, pos, node_order, df_full["true_prob"].to_dict(),
                "(b) True distribution")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"network_{snapshot_name}.pdf", bbox_inches="tight", dpi=1200)
    plt.close()
    print(f"  Saved network_{snapshot_name}.pdf")


def compute_expected_scores(df, model_columns, scorer, node_order):
    N = len(node_order)
    df_ord     = df.loc[node_order]
    true_prob  = df_ord["true_prob"].values
    true_nodes = np.arange(N)
    results = {}
    for model in model_columns:
        P = np.tile(df_ord[model].values, (N, 1))
        scores = {
            "error_distance": scorer.error_distance(P, true_nodes),
            "top_1_err":      scorer.top_k_error(P, true_nodes, k=1),
            "top_3_err":      scorer.top_k_error(P, true_nodes, k=3),
            "top_5_err":      scorer.top_k_error(P, true_nodes, k=5),
            "rank":           scorer.rank_score(P, true_nodes),
            "brier":          scorer.brier_score(P, true_nodes),
            "resistance":     scorer.resistance_score(P, true_nodes),
            **{f"diff_{str(tau).replace('.', '')}": scorer.diffusion_score(P, true_nodes, tau=tau, mu=0.8)
               for tau in DIFFUSION_TAUS},
        }
        results[model] = {k: float(np.dot(true_prob, v)) for k, v in scores.items()}
    return pd.DataFrame(results).T


def plot_heatmap(scores_df, snapshot_name):
    norm = (scores_df - scores_df.min()) / (scores_df.max() - scores_df.min())
    cmap = plt.cm.Greens
    n_rows, n_cols = scores_df.shape

    fig, ax = plt.subplots(figsize=(10, n_rows * 0.5))
    for i in range(n_rows):
        for j in range(n_cols):
            color = cmap(norm.iloc[i, j])
            ax.add_patch(plt.Rectangle((j, n_rows - 1 - i), 1, 1,
                                       facecolor=color, edgecolor="white", linewidth=0.5))
            brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            ax.text(j + 0.5, n_rows - 1 - i + 0.5, f"{scores_df.iloc[i, j]:.3f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if brightness > 0.5 else "white")

    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([SCORE_DISPLAY.get(c, c) for c in scores_df.columns],
                       rotation=30, ha="right", fontsize=11)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([_model_label(r) for r in scores_df.index][::-1], fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"score_table_{snapshot_name}.pdf", bbox_inches="tight", dpi=1200)
    plt.close()
    print(f"  Saved score_table_{snapshot_name}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", nargs="+", default=["all"])
    args = parser.parse_args()

    if args.snapshots == ["all"]:
        snapshots = sorted(p.stem for p in (DATA_DIR / "interesting_snapshots").glob("*.csv"))
    else:
        snapshots = args.snapshots

    G          = pickle.load(open(DATA_DIR / "network.gpickle", "rb"))
    node_order = sorted(G.nodes())
    scorer     = GraphScorer(nx.to_numpy_array(G, nodelist=node_order, weight=None))
    pos        = nx.spring_layout(G, seed=NETWORK_SEED)

    for snap in snapshots:
        print(f"\n{snap}")
        df   = pd.read_csv(DATA_DIR / "interesting_snapshots" / f"{snap}.csv").set_index("node")
        cols = [c for c in DEFAULT_MODEL_ORDER if c in df.columns]
        plot_network(G, df, snap, pos)
        plot_heatmap(compute_expected_scores(df, cols, scorer, node_order), snap)
