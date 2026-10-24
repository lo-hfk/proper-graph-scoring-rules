"""
Visualise a source detection snapshot and model estimates on the Karate Club network.

Produces:
  - figures/network_<snapshot>.pdf  : 2x3 grid of network plots (infection status,
                                      true distribution, and 4 model estimates)
  - figures/calibration_<snapshot>.pdf : 2x2 calibration plots (predicted vs true prob)

The paper example uses:
    python analysis/visualise.py --snapshot source=0_run=46

Usage
-----
    python analysis/visualise.py --snapshot source=0_run=46
    python analysis/visualise.py --snapshot source=0_run=46 \\
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

DATA_DIR    = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Visual parameters
TRANSFORMATION_A = 0.5 # probability shading: p^a (a=1: linear, a<1: emphasises small values)
CMAP = plt.cm.Grays
NODE_SIZE = 400
LINEWIDTHS = 1.5
EDGE_COLOR = "gray"
PROB_LABEL_FS = 8     # font size for probability labels inside nodes
PROB_LABEL_OFFSET = 0.005  # vertical offset of prob labels below node center (data coords)
PROB_DECIMALS = 2 # decimal places for probability labels
NETWORK_SEED = 42 # spring_layout seed for reproducibility


def load_data(snapshot_name):
    with open(DATA_DIR / "network.gpickle", "rb") as f:
        G = pickle.load(f)
    csv_path = DATA_DIR / "interesting_snapshots" / f"{snapshot_name}.csv"
    df = pd.read_csv(csv_path).set_index("node")
    return G, df


def build_piecewise_uniform(df):
    """Construct the piecewise-uniform heuristic distribution. (This
    is specific to the source=0_run=46 snapshot used in the paper.)

    Places the correct total probability mass of each infected cluster
    uniformly within that cluster.
    """
    pu = df["true_prob"].copy()
    mass_4_5_6_16 = df.loc[[4,  5,  6, 16], "true_prob"].sum()
    mass_1_17_21 = df.loc[[1, 17, 21], "true_prob"].sum()
    prob_11 = df.loc[11, "true_prob"]
    prob_0 = df.loc[0,  "true_prob"]
    pu.loc[[4,  5,  6, 16]] = mass_4_5_6_16 / 4
    pu.loc[[1, 17, 21]] = mass_1_17_21  / 3
    pu.loc[11] = prob_11
    pu.loc[0] = prob_0
    return pu


def transform(p):
    return np.power(np.asarray(p, dtype=float), TRANSFORMATION_A)


def draw_network_ax(ax, G, pos, node_order, probs_dict, title,
                    infection_colors=None):
    """Draw one network panel.

    If infection_colors is given, use it for node colors (infection status panel).
    Otherwise use greyscale shading by transformed probability.
    Node labels show rounded probability for all nodes with prob > 0.
    """
    if infection_colors is not None:
        nx.draw(G, pos=pos, ax=ax,
                node_color=infection_colors,
                with_labels=False,
                node_size=NODE_SIZE,
                edgecolors="black",
                linewidths=LINEWIDTHS,
                edge_color=EDGE_COLOR)
    else:
        probs = np.array([probs_dict.get(n, 0.0) for n in node_order])
        nx.draw_networkx_nodes(G, pos=pos, ax=ax,
                               nodelist=node_order,
                               node_color=transform(probs),
                               cmap=CMAP,
                               node_size=NODE_SIZE,
                               vmin=0, vmax=transform([1])[0],
                               edgecolors="black",
                               linewidths=LINEWIDTHS)
        nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color=EDGE_COLOR)

        # probability labels inside nodes (all nodes with nonzero prob)
        for n in node_order:
            p = probs_dict.get(n, 0.0)
            if p > 0:
                x, y = pos[n]
                ax.text(x, y - PROB_LABEL_OFFSET,
                        f"{p:.{PROB_DECIMALS}f}",
                        ha="center", va="center",
                        fontsize=PROB_LABEL_FS, color="black")

    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_frame_on(False)


def plot_networks(G, df, model_columns, snapshot_name, pos):
    """Produce the 2x3 network figure."""
    node_order = list(G.nodes())
    df_full = df.reindex(node_order).fillna(0.0)

    infected_colors = ["red" if df_full.loc[n, "infected"] == 1
                       else "blue" for n in node_order]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # (a) infection status
    draw_network_ax(axes[0, 0], G, pos, node_order,
                    probs_dict={},
                    title="(a) Infection status",
                    infection_colors=infected_colors)

    # (b) true distribution
    true_probs = df_full["true_prob"].to_dict()
    draw_network_ax(axes[1, 0], G, pos, node_order,
                    probs_dict=true_probs,
                    title="(b) True distribution")

    # (c)-(f) model estimates
    panels = [(0, 1), (1, 1), (0, 2), (1, 2)]
    labels = ["(c)", "(d)", "(e)", "(f)"]
    for (row, col), lbl, model in zip(panels, labels, model_columns):
        probs = df_full[model].to_dict() if model in df_full.columns else {}
        draw_network_ax(axes[row, col], G, pos, node_order,
                        probs_dict=probs,
                        title=f"{lbl} {model}")

    plt.tight_layout()
    out = FIGURES_DIR / f"network_{snapshot_name}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=600)
    fig.savefig(FIGURES_DIR / f"network_{snapshot_name}.png",
                bbox_inches="tight", dpi=600)
    plt.close()
    print(f"Saved: {out.name}")


def plot_calibration(df, model_columns, snapshot_name):
    """Produce 2x2 calibration plots (predicted vs true probability)."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axs = plt.subplots(2, 2, figsize=(4, 4), dpi=600,
                             sharex=True, sharey=True)
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    for ax, model in zip(axs.flatten(), model_columns):
        y = df[model].values if model in df.columns else np.zeros(len(df))
        ax.scatter(df["true_prob"].values, y, s=20, alpha=1,
                   facecolors="none", edgecolors=color)
        ax.plot([0, 1], [0, 1], color="black", lw=1.0,
                linestyle="--", zorder=0)
        ax.set_title(model, fontsize=8)
        ax.set_xlim(1e-7, 1); ax.set_ylim(1e-7, 1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([1e-6, 1e-3, 1])
        ax.set_xticklabels([r"$10^{-6}$", r"$10^{-3}$", "1"])
        ax.set_yticks([1e-6, 1e-3, 1])
        ax.set_yticklabels([r"$10^{-6}$", r"$10^{-3}$", "1"])
        ax.tick_params(direction="out", length=3, width=0.8)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")

    plt.tight_layout()
    out = FIGURES_DIR / f"calibration_{snapshot_name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")
    plt.rcParams.update(plt.rcParamsDefault)


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="source=0_run=46",
                        help="Snapshot filename (without .csv)")
    parser.add_argument("--models", nargs=4,
                        default=["static_gnn_n=10000", "piecewise_uniform",
                                 "soft_margin_a=0.1_n=1000", "monte_carlo_n=1000"],
                        help="Exactly 4 model columns to plot (c, d, e, f).")
    args = parser.parse_args()

    G, df = load_data(args.snapshot)

    # build piecewise_uniform if requested and not in CSV
    if "piecewise_uniform" in args.models and "piecewise_uniform" not in df.columns:
        df["piecewise_uniform"] = build_piecewise_uniform(df)
        print("  Built piecewise_uniform distribution.")

    pos = nx.spring_layout(G, seed=NETWORK_SEED)

    print(f"Snapshot: {args.snapshot}")
    print(f"Models:   {args.models}")

    plot_networks(G, df, args.models, args.snapshot, pos)
    plot_calibration(df, args.models, args.snapshot)
