"""
Reproduce the toy graph comparison figures from the paper.

Produces:
  - figures/toy_graphs.pdf : 2x3 grid of network plots with the 6 distributions
  - figures/toy_table.pdf : 2x3 grid of score heatmaps
  - figures/toy_table_extended.pdf : 2x4 grid including log-loss and error distance

The graph has 10 nodes. The true node is always 0. Six distributions
each place equal mass on 4 nodes, forming a 2x3 array:
  top row (a,b,c): true node IN support
  bottom row (d,e,f): true node NOT in support
Within each row, mass moves progressively further from the true node.

Usage
-----
    python toy_example.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from propnetscore import GraphScorer

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Diffusion regularisation parameter ((MU_INDEX+1)-smallest Laplacian eigenvalue)
# chosen so diffusion scores are on similar scale across all tau values
MU_INDEX = 2

# Score heatmap colormap
CMAP = "Greens"

# -----------------------------------------------------------------
# Graph definition
# -----------------------------------------------------------------
def build_graph():
    G = nx.Graph()
    G.add_nodes_from(range(10))
    G.add_edges_from([
        (0,1),(0,2),(0,4),(1,4),(2,4),(3,4),(4,5),
        (5,6),(5,9),(6,7),(6,9),(6,8),(8,9)
    ])
    G.remove_node(9)
    G.add_edge(7, 5)
    G.add_node(9)
    G.add_edges_from([(1,9),(4,9)])
    return G


# -----------------------------------------------------------------
# Distributions (true node = 0 throughout)
# -----------------------------------------------------------------
TRUE_NODE = 0

DISTRIBUTIONS = {
    "a": {0: .25, 1: .25, 2: .25, 4: .25},  # true node in support, mass nearby
    "b": {0: .25, 2: .25, 3: .25, 4: .25},  # true node in support, mass medium
    "c": {0: .25, 3: .25, 4: .25, 5: .25},  # true node in support, mass far
    "d": {1: .25, 2: .25, 3: .25, 4: .25},  # true node NOT in support, mass nearby
    "e": {1: .25, 2: .25, 4: .25, 5: .25},  # true node NOT in support, mass medium
    "f": {2: .25, 4: .25, 5: .25, 9: .25},  # true node NOT in support, mass far
}


def make_prob_vector(G, dist):
    p = np.zeros(len(G))
    for node, prob in dist.items():
        p[node] = prob
    return p


# -----------------------------------------------------------------
# Network figure
# -----------------------------------------------------------------
def plot_graphs(G, distributions, pos):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5), dpi=200)
    for ax, (label, dist) in zip(axs.flatten(), distributions.items()):
        p = make_prob_vector(G, dist)
        node_colors  = ["gray" if p[n] > 0 else "white" for n in G.nodes()]
        edge_colors  = ["red" if n == TRUE_NODE else "black" for n in G.nodes()]
        linewidths   = [4.0 if n == TRUE_NODE else 1.5 for n in G.nodes()]
        nx.draw(G, pos=pos, ax=ax,
                node_color=node_colors,
                edgecolors=edge_colors,
                linewidths=linewidths,
                edge_color="gray",
                node_size=400,
                with_labels=False)
        ax.set_axis_off()
    plt.tight_layout()
    out = FIGURES_DIR / "toy_graphs.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=600)
    fig.savefig(FIGURES_DIR / "toy_graphs.png", bbox_inches="tight", dpi=600)
    plt.close()
    print(f"Saved: {out.name}")


# -----------------------------------------------------------------
# Score table figure
# -----------------------------------------------------------------
def compute_scores(scorer, P, true_nodes, mu):
    scores = {}
    scores["Top-4 error rate"] = scorer.top_k_error(P, true_nodes, k=4)
    scores["Brier"] = scorer.brier_score(P, true_nodes)
    scores["Resistance"] = scorer.resistance_score(P, true_nodes)
    scores[r"Diffusion($\tau$=0.1)"] = scorer.diffusion_score(P, true_nodes, tau=0.1, mu=mu)
    scores[r"Diffusion($\tau$=1)"] = scorer.diffusion_score(P, true_nodes, tau=1, mu=mu)
    scores[r"Diffusion($\tau$=10)"] = scorer.diffusion_score(P, true_nodes, tau=10, mu=mu)
    return scores


def _draw_score_panel(ax, values, title):
    """Draw one score heatmap panel, handling inf values gracefully."""
    data = values.reshape(2, 3)
    has_inf = not np.all(np.isfinite(data))
    data_finite = np.where(np.isfinite(data), data, np.nan)
    finite_max = float(np.nanmax(data_finite)) if np.any(np.isfinite(data)) else 1.0
    if has_inf:
        vmax = finite_max * 100 if finite_max > 0 else 1.0
    else:
        vmax = finite_max
    data_plot = np.where(np.isfinite(data), data, vmax)
    ax.imshow(data_plot, vmin=0, vmax=vmax, cmap=CMAP)
    for i in range(2):
        for j in range(3):
            val = data[i, j]
            nval = data_plot[i, j]
            if not np.isfinite(val):
                tc = "white"
            else:
                tc = "white" if vmax > 0 and nval / vmax > 0.9 else "black"
            if not np.isfinite(val):
                formatted = "∞"
            elif abs(val) < 0.0005:
                formatted = "0"
            else:
                formatted = f"{val:.3g}"
                if "e" not in formatted:
                    formatted = f"{float(formatted):.3f}".rstrip("0").rstrip(".")
            ax.text(j, i, formatted, ha="center", va="center",
                    color=tc, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def plot_score_table(scores):
    fig, axs = plt.subplots(2, 3, figsize=(6, 3.5), dpi=400)
    for ax, (title, values) in zip(axs.flatten(), scores.items()):
        _draw_score_panel(ax, values, title)
    plt.tight_layout()
    out = FIGURES_DIR / "toy_table.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=600)
    fig.savefig(FIGURES_DIR / "toy_table.png", bbox_inches="tight", dpi=600)
    plt.close()
    print(f"Saved: {out.name}")


def plot_score_table_extended(scores):
    """2x4 score table: row 1 = traditional scores, row 2 = graph-aware proper scores."""
    fig, axs = plt.subplots(2, 4, figsize=(8, 3.5), dpi=400)
    for ax, (title, values) in zip(axs.flatten(), scores.items()):
        _draw_score_panel(ax, values, title)
    plt.tight_layout()
    out = FIGURES_DIR / "toy_table_extended.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=600)
    fig.savefig(FIGURES_DIR / "toy_table_extended.png", bbox_inches="tight", dpi=600)
    plt.close()
    print(f"Saved: {out.name}")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
if __name__ == "__main__":
    G = build_graph()
    pos = nx.kamada_kawai_layout(G)

    A = nx.to_numpy_array(G)
    scorer = GraphScorer(A)

    eigenvalues = np.sort(np.linalg.eigvalsh(scorer.laplacian))
    mu = float(eigenvalues[MU_INDEX])  # NB: eigenvalues[2] = 1.0 exactly for this graph
    print(f"Graph: {len(G)} nodes, {G.number_of_edges()} edges")
    print(f"Regularisation mu = eigenvalues[{MU_INDEX}] = {mu:.4f}")

    P = np.array([make_prob_vector(G, d) for d in DISTRIBUTIONS.values()])
    true_nodes = np.full(6, TRUE_NODE, dtype=int)

    plot_graphs(G, DISTRIBUTIONS, pos)

    scores = compute_scores(scorer, P, true_nodes, mu)
    plot_score_table(scores)

    # extended 2x4 table (row 1: flat scores, row 2: graph-aware proper scores)
    scores_ext = {
        "Log-loss": scorer.logarithmic_score(P, true_nodes),
        "Top-4 error": scorer.top_k_error(P, true_nodes, k=4),
        "Brier": scorer.brier_score(P, true_nodes),
        "Error distance": scorer.error_distance(P, true_nodes),
        "Resistance": scorer.resistance_score(P, true_nodes),
        r"Diffusion($\tau$=0.1)": scorer.diffusion_score(P, true_nodes, tau=0.1, mu=mu),
        r"Diffusion($\tau$=1)": scorer.diffusion_score(P, true_nodes, tau=1, mu=mu),
        r"Diffusion($\tau$=10)": scorer.diffusion_score(P, true_nodes, tau=10, mu=mu),
    }
    plot_score_table_extended(scores_ext)
