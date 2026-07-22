"""
Step 1b: Build WordNet trees from the full DAG for use with hierarchy-aware scoring rules.

Produces two variants, selectable via --graph:

  tree   (default) -- Bertinetto path-selection only, NO single-child collapse.
                      Preserves full chain lengths, so resistance distance
                      correctly penalises long relay paths.  This is the
                      recommended variant for distance-based scoring rules.

  ctree            -- Bertinetto path-selection PLUS single-child collapse,
                      exactly replicating what Bertinetto et al. (CVPR 2020)
                      used for their HXE / soft-label losses.  Collapse
                      shortens distances along relay chains (a known distortion
                      for distance-based metrics -- see paper discussion) but
                      is included for direct comparability with the literature.

Both variants are proper trees (each node has at most one parent), so the
key mathematical property holds for both: resistance distance == shortest-path
distance (verified explicitly).

Output files (in wordnet_graph/):
  tree variant:
    tree.gpickle, tree_leaf_synsets.json, tree_stats.json
  ctree variant:
    ctree.gpickle, ctree_leaf_synsets.json, ctree_stats.json

Usage:
    python build_tree_graph.py --graph tree     # recommended
    python build_tree_graph.py --graph ctree    # Bertinetto-exact replication

Reads:  wordnet_graph/graph.gpickle
        wordnet_graph/leaf_synsets.json
"""

import argparse
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np

GRAPH_DIR = Path("./wordnet_graph")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_dag():
    with open(GRAPH_DIR / "graph.gpickle", "rb") as f:
        G = pickle.load(f)
    with open(GRAPH_DIR / "leaf_synsets.json") as f:
        leaf_info = json.load(f)
    return G, leaf_info


# ---------------------------------------------------------------------------
# Bertinetto path selection (shared by both variants)
# ---------------------------------------------------------------------------
def all_root_paths(G, node):
    paths = []
    def dfs(current, path_so_far):
        parents = list(G.successors(current))
        if not parents:
            paths.append(list(path_so_far))
            return
        for parent in parents:
            path_so_far.append(parent)
            dfs(parent, path_so_far)
            path_so_far.pop()
    dfs(node, [node])
    return paths


def select_best_path(paths, nodes_already_in_tree):
    if not paths:
        return []
    max_len = max(len(p) for p in paths)
    longest = [p for p in paths if len(p) == max_len]
    if len(longest) == 1:
        return longest[0]
    def n_new(path):
        return sum(1 for n in path if n not in nodes_already_in_tree)
    return min(longest, key=n_new)


def build_bertinetto_tree(G, leaf_names):
    tree = nx.DiGraph()
    nodes_in_tree = set()
    for leaf in leaf_names:
        if leaf not in G:
            raise ValueError(f"Leaf {leaf} not found in the DAG")
        paths = all_root_paths(G, leaf)
        best_path = select_best_path(paths, nodes_in_tree)
        for node in best_path:
            if node not in tree:
                attrs = G.nodes[node] if node in G.nodes else {}
                tree.add_node(node, **attrs)
            nodes_in_tree.add(node)
        for i in range(len(best_path) - 1):
            child, parent = best_path[i], best_path[i + 1]
            if not tree.has_edge(child, parent):
                tree.add_edge(child, parent)
    return tree


# ---------------------------------------------------------------------------
# Single-child collapse (ctree only)
#
# NOTE: this shortens relay-chain distances, which is a distortion for
# distance-based scoring rules.  Included only for Bertinetto replication.
# See paper discussion section for full argument.
# ---------------------------------------------------------------------------
def collapse_single_child_nodes(tree, leaf_names):
    leaf_set = set(leaf_names)
    changed = True
    while changed:
        changed = False
        for node in list(tree.nodes()):
            if node in leaf_set:
                continue
            children = list(tree.predecessors(node))
            parents  = list(tree.successors(node))
            if len(children) == 1 and len(parents) == 1:
                tree.add_edge(children[0], parents[0])
                tree.remove_node(node)
                changed = True
                break
    return tree


# ---------------------------------------------------------------------------
# Verification (shared)
# ---------------------------------------------------------------------------
def verify_tree(tree, leaf_names):
    assert nx.is_directed_acyclic_graph(tree), "Not a DAG!"
    roots = [n for n in tree.nodes() if tree.out_degree(n) == 0]
    assert len(roots) == 1, f"Expected 1 root, found {len(roots)}: {roots}"
    bad = [n for n in tree.nodes()
           if tree.out_degree(n) not in (0, 1)]
    assert not bad, f"Nodes with unexpected out-degree: {bad[:5]}"
    missing = [l for l in leaf_names if l not in tree]
    assert not missing, f"Missing leaves: {missing[:5]}"
    n = tree.number_of_nodes()
    assert tree.number_of_edges() == n - 1, \
        f"Tree on {n} nodes must have {n-1} edges, got {tree.number_of_edges()}"
    return roots[0]


def verify_resistance_equals_shortest_path(tree, n_sample=30, seed=42):
    rng = np.random.RandomState(seed)
    nodes = list(tree.nodes())
    n = len(nodes)
    G_u = tree.to_undirected()

    A = nx.to_numpy_array(G_u, nodelist=nodes)
    L = np.diag(A.sum(axis=1)) - A
    L_pinv = np.linalg.pinv(L)
    diag = np.diag(L_pinv)
    resistance = diag[:, None] + diag[None, :] - 2 * L_pinv

    sp = dict(nx.all_pairs_shortest_path_length(G_u))
    idx_pairs = [(rng.randint(0, n), rng.randint(0, n)) for _ in range(n_sample)]

    max_diff = 0.0
    for i, j in idx_pairs:
        diff = abs(resistance[i, j] - sp[nodes[i]].get(nodes[j], np.inf))
        max_diff = max(max_diff, diff)

    assert max_diff < 1e-6, \
        f"Resistance != shortest-path on tree! Max diff: {max_diff:.2e}"
    return max_diff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph", choices=["tree", "ctree"], default="tree",
        help=(
            "tree  (default): Bertinetto path selection only, no collapse. "
            "Preserves relay-chain lengths -- recommended for distance-based "
            "scoring rules. "
            "ctree: path selection + single-child collapse, exactly replicating "
            "Bertinetto et al. (2020). Shortens relay chains -- a known "
            "distortion for distance-based metrics; included for literature "
            "comparability only."
        ),
    )
    args = parser.parse_args()
    variant = args.graph  # "tree" or "ctree"

    # Clear labels for output and terminal
    VARIANT_LABEL = {
        "tree":  "TREE (no collapse)",
        "ctree": "CTREE (Bertinetto-exact, with single-child collapse)",
    }[variant]

    print(f"{'='*70}")
    print(f"Building: {VARIANT_LABEL}")
    print(f"{'='*70}")

    print("\nLoading full DAG...")
    G_dag, leaf_info = load_dag()
    leaf_names  = leaf_info["class_idx_to_synset_name"]
    leaf_wnids  = leaf_info["class_idx_to_wnid"]
    leaf_labels = leaf_info["class_idx_to_label"]
    print(f"  DAG: {G_dag.number_of_nodes()} nodes, {G_dag.number_of_edges()} edges")
    print(f"  Leaves: {len(leaf_names)}")

    print("\nStep 1: selecting best root path per leaf (Bertinetto rule)...")
    result = build_bertinetto_tree(G_dag, leaf_names)
    print(f"  After path selection: {result.number_of_nodes()} nodes, "
          f"{result.number_of_edges()} edges")

    if variant == "ctree":
        print("\nStep 2: collapsing single-child internal nodes "
              "(ctree variant -- see docstring for distance-distortion caveat)...")
        n_before = result.number_of_nodes()
        result = collapse_single_child_nodes(result, leaf_names)
        n_collapsed = n_before - result.number_of_nodes()
        print(f"  After collapse: {result.number_of_nodes()} nodes, "
              f"{result.number_of_edges()} edges "
              f"({n_collapsed} nodes removed)")
    else:
        print(f"\nStep 2: skipping collapse (tree variant -- relay chains preserved).")
        print(f"  Single-child internal nodes: "
              f"{sum(1 for n in result.nodes() if result.in_degree(n)==1 and result.out_degree(n)==1 and n not in set(leaf_names))} "
              f"(retained intentionally)")

    print(f"\nStep 3: verifying tree properties ({variant})...")
    root = verify_tree(result, leaf_names)
    print(f"  Root: {root}")
    print(f"  Tree properties confirmed.")

    print(f"\nStep 4: verifying resistance == shortest-path distance...")
    max_diff = verify_resistance_equals_shortest_path(result, n_sample=30)
    print(f"  Max |resistance - shortest_path|: {max_diff:.2e}  -- confirmed.")

    # Diagnostics
    dag_multi_parent = [n for n in G_dag.nodes() if G_dag.out_degree(n) > 1]
    depths = nx.single_source_shortest_path_length(result.reverse(), root)
    max_depth = max(depths.values())

    stats = {
        "variant": variant,
        "variant_label": VARIANT_LABEL,
        "dag_n_nodes": G_dag.number_of_nodes(),
        "dag_n_edges": G_dag.number_of_edges(),
        "dag_n_multi_parent_nodes": len(dag_multi_parent),
        f"{variant}_n_nodes": result.number_of_nodes(),
        f"{variant}_n_edges": result.number_of_edges(),
        f"{variant}_n_nodes_vs_dag": G_dag.number_of_nodes() - result.number_of_nodes(),
        f"{variant}_root": root,
        f"{variant}_max_depth": max_depth,
        f"{variant}_n_leaves": len(leaf_names),
        "collapse_applied": variant == "ctree",
        "resistance_equals_shortest_path_confirmed": True,
        "resistance_equals_shortest_path_max_diff": float(max_diff),
        "note": (
            "tree: relay chains preserved -- recommended for resistance/diffusion scores. "
            "ctree: relay chains collapsed -- Bertinetto et al. exact replication, "
            "but shortens distances along chains (see paper discussion)."
        ),
    }

    print(f"\n{'='*70}")
    print(f"Summary ({VARIANT_LABEL}):")
    print(f"{'='*70}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save -- filenames clearly encode the variant
    graph_file    = GRAPH_DIR / f"{variant}.gpickle"
    synsets_file  = GRAPH_DIR / f"{variant}_leaf_synsets.json"
    stats_file    = GRAPH_DIR / f"{variant}_stats.json"

    with open(graph_file, "wb") as f:
        pickle.dump(result, f)
    with open(synsets_file, "w") as f:
        json.dump({
            "class_idx_to_wnid":        leaf_wnids,
            "class_idx_to_label":       leaf_labels,
            "class_idx_to_synset_name": leaf_names,
        }, f, indent=2)
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved {variant}.gpickle  ({result.number_of_nodes()} nodes)")
    print(f"Saved {variant}_leaf_synsets.json  (leaf ordering identical to DAG)")
    print(f"Saved {variant}_stats.json")