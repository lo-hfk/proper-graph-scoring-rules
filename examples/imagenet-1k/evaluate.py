"""
Step 4: Evaluation harness.

Loads the saved probs_<model>.npy / true_labels_<model>.npy from Step 2
(one or more models) and the graph from Step 1, computes:
  (a) standard baselines: top-1 acc, top-5 acc, log-loss / Brier (flat,
      graph-unaware proper scoring rule, as a reference point)
  (b) YOUR graph-aware proper scoring rules: resistance_score and
      diffusion_score (from batched_scorer.GraphScorer), evaluated over the
      FULL WordNet node set (leaves + internal ancestors), not a pruned
      leaf-only graph -- see the comment block above build_full_graph_scorer
      for why this matters.
  (c) a hierarchical "mistake severity" baseline a la Bertinetto et al., for
      direct comparison against the literature's existing attempt at this

Run for a single model:
    python3 04_evaluate.py --model resnet50

Run for ALL models you have inference outputs for, producing a single
comparison table across models (the main table you'll want for the paper):
    python3 04_evaluate.py --all

Produces a results table (per-model, or combined comparison) and diagnostic
plots.

Requires batched_scorer.py (GraphScorer) to be in the same directory.
"""

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from propnetscore.batched_scorer import GraphScorer

DATA_DIR = Path("./wordnet_graph")
OUT_DIR = Path("./model_validation_outputs")

# Mapping from --graph argument to the graph and leaf-synset filenames
GRAPH_FILES = {
    "dag":   ("graph.gpickle",  "leaf_synsets.json"),
    "tree":  ("tree.gpickle",   "tree_leaf_synsets.json"),
    "ctree": ("ctree.gpickle",  "ctree_leaf_synsets.json"),
}


def discover_available_models():
    """
    Scans outputs/ for probs_<model>.npy files and returns the list of
    model names found, by inspecting filenames rather than hardcoding,
    so this stays in sync with whatever you've actually run in Step 2.
    """
    found = []
    for f in sorted(OUT_DIR.glob("probs_*.npy")):
        name = f.stem.replace("probs_", "")
        if (OUT_DIR / f"true_labels_{name}.npy").exists():
            found.append(name)
    return found


def load_model_outputs(model_name):
    probs = np.load(OUT_DIR / f"probs_{model_name}.npy")
    true_labels = np.load(OUT_DIR / f"true_labels_{model_name}.npy")
    return probs, true_labels


def load_graph_and_leaf_info(graph_variant="dag"):
    graph_file, synsets_file = GRAPH_FILES[graph_variant]
    with open(DATA_DIR / graph_file, "rb") as f:
        G = pickle.load(f)
    with open(DATA_DIR / synsets_file) as f:
        leaf_info = json.load(f)
    return G, leaf_info


# ---------------------------------------------------------------------------
# (a) Standard baselines
# ---------------------------------------------------------------------------
def top_k_accuracy(probs, true_labels, k):
    topk = np.argsort(-probs, axis=1)[:, :k]
    hits = (topk == true_labels[:, None]).any(axis=1)
    return hits.mean()


def brier_score_flat(probs, true_labels):
    """
    Standard (graph-unaware) multiclass Brier score:
      S(p, v) = ||p - e_v||^2 = sum_j (p_j - 1[j=v])^2
    A proper scoring rule, but treats all non-true classes symmetrically --
    exactly the baseline your graph-aware rule should beat.
    """
    n, k = probs.shape
    onehot = np.zeros((n, k), dtype=np.float32)
    onehot[np.arange(n), true_labels] = 1.0
    return ((probs - onehot) ** 2).sum(axis=1)  # per-instance, then .mean() outside


def log_loss_flat(probs, true_labels, eps=1e-12):
    """Standard log score: S(p, v) = log p_v. Also graph-unaware."""
    p_true = probs[np.arange(len(true_labels)), true_labels]
    return -np.log(np.clip(p_true, eps, 1.0))


def build_hxe_cache(G, leaf_names, alpha=0.5, graph_variant="tree"):
    """
    Precompute the HXE score components for all leaf classes.

    HXE_alpha(f, y) = -sum_{j in a(y), j != root} lambda(j) * log(P_j / P_{p(j)})
    where lambda(j) = exp(-alpha * depth(j)), depth from root (root=0).

    In the L_T representation (Urbani et al.), the EFFECTIVE weights on
    log(P_j) are:
        w_j = lambda(j) - lambda(child_of_j_on_path)  for internal nodes
        w_j = lambda(j)                                for leaf nodes

    These weights are balanced: sum_{j in a(y)} w_j = exp(-alpha) for any leaf y.

    We precompute for each leaf y:
        - The list of ancestor nodes (excluding root) on the path from y to root
        - The effective weight w_j for each ancestor

    Then HXE(f, y) = -sum_{j in a(y)} w_j * log(P_j(f))
    where P_j(f) = sum_{k in subtree(j)} f_k.

    We store:
        - paths_weights: list of (ancestor_leaf_indices, weights) per leaf
          where ancestor_leaf_indices[j] = list of leaf indices in subtree(j)
    This lets us compute P_j(f) as f[ancestor_leaf_indices[j]].sum()
    """
    cache_path = OUT_DIR / f"hxe_cache_{graph_variant}_alpha{alpha:.2f}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    print(f"  Building HXE cache (alpha={alpha}, graph={graph_variant})...")
    # G is a DiGraph with child->parent edges
    # root = node with out_degree 0
    roots = [n for n in G.nodes() if G.out_degree(n) == 0]
    assert len(roots) == 1, f"Expected 1 root, got {len(roots)}"
    root = roots[0]

    # Build parent map and children map
    parent = {n: list(G.successors(n))[0] if G.out_degree(n) > 0 else None
              for n in G.nodes()}

    # Depth from root for each node
    G_undirected = G.to_undirected()
    depth = nx.single_source_shortest_path_length(G_undirected, root)

    # Subtree leaves for each node: walk descendants
    children = {n: list(G.predecessors(n)) for n in G.nodes()}

    def subtree_leaves(node):
        """Return all leaf synset names in subtree of node."""
        result = []
        stack = [node]
        while stack:
            cur = stack.pop()
            kids = children[cur]
            if not kids:  # leaf
                result.append(cur)
            else:
                stack.extend(kids)
        return result

    leaf_set = set(leaf_names)
    leaf_to_idx = {name: i for i, name in enumerate(leaf_names)}

    # For each leaf, compute path to root (excluding root) with effective weights
    # path = [leaf, parent_of_leaf, ..., child_of_root]
    # effective weight at node j:
    #   leaf: w = exp(-alpha * depth[j])
    #   internal: w = exp(-alpha * depth[j]) - exp(-alpha * depth[child_on_path])
    # Since child_on_path has depth[j]+1 (tree is depth-labelled from root):
    #   internal: w = exp(-alpha*d) - exp(-alpha*(d+1)) = exp(-alpha*d)*(1 - exp(-alpha))

    # For efficiency: precompute for each leaf the list of (subtree_leaf_indices, weight)
    hxe_paths = []  # one entry per leaf in leaf_names order
    for leaf_name in leaf_names:
        path_nodes = []
        cur = leaf_name
        while cur != root:
            path_nodes.append(cur)
            cur = parent[cur]
        # path_nodes = [leaf, ..., child_of_root], ordered leaf->root (excl root)

        path_info = []
        for k, node in enumerate(path_nodes):
            d = depth[node]
            if k == 0:  # the leaf itself
                w = math.exp(-alpha * d)
            else:
                # internal node: w = exp(-alpha*d) - exp(-alpha*(d+1))
                w = math.exp(-alpha * d) * (1.0 - math.exp(-alpha))
            # subtree leaf indices
            sub_leaves = subtree_leaves(node)
            sub_idx = np.array([leaf_to_idx[l] for l in sub_leaves
                                 if l in leaf_to_idx], dtype=np.int32)
            path_info.append((sub_idx, w))

        hxe_paths.append(path_info)

    with open(cache_path, "wb") as fh:
        pickle.dump(hxe_paths, fh)
    print(f"  HXE cache saved: {cache_path}")
    return hxe_paths


def hxe_score(probs, true_labels, hxe_paths, eps=1e-12):
    """
    Compute per-example HXE score.

    HXE(f, y) = -sum_{j in a(y)} w_j * log(P_j(f))
    where P_j(f) = f[subtree_leaves(j)].sum()

    probs: (n_examples, n_leaves)
    true_labels: (n_examples,)  -- indices into leaf_names
    hxe_paths: precomputed from build_hxe_cache

    Returns: (n_examples,) array of per-example HXE scores
    """
    n = len(true_labels)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y = true_labels[i]
        path_info = hxe_paths[y]
        s = 0.0
        for sub_idx, w in path_info:
            P_j = probs[i, sub_idx].sum()
            s -= w * math.log(max(P_j, eps))
        scores[i] = s
    return scores


# ---------------------------------------------------------------------------
# (b) YOUR graph-aware proper scoring rule, via GraphScorer (resistance_score
#     and diffusion_score), evaluated over the FULL WordNet node set --
#     leaves AND internal ancestor nodes -- not just the 1000 leaves.
#
# Why the full node set, not just the leaves:
#   The whole point of keeping the DAG un-pruned (rather than collapsing to
#   a tree a la Bertinetto et al.) is that internal nodes can serve as
#   alternate connecting routes between leaves -- e.g. two leaves sharing
#   TWO different immediate parents (a "diamond") are genuinely closer in
#   resistance/diffusion distance than two leaves connected only through a
#   single shared ancestor. Collapsing to a leaf-only graph would throw this
#   structure away. See 03_graph_utils.py Test 4 for the verified example.
#
# A model's predictions live only over the 1000 leaves (no classifier
# assigns probability to an internal WordNet concept like "mammal" as a
# literal output class), so we zero-pad each leaf-only probability vector
# into the full node space before scoring. This is a valid probability
# distribution over the full node set (it still sums to 1), and is exactly
# the correct representation of "this model never predicts internal nodes".
# ---------------------------------------------------------------------------
def build_full_graph_scorer(G, leaf_names, graph_variant="dag"):
    """
    Builds a GraphScorer over the FULL node set of the WordNet graph (leaves +
    internal ancestor nodes), converting the directed is-a graph into a
    simple undirected graph first.

    The cache filename is suffixed with the graph variant so DAG, tree, and
    ctree scorers are stored and loaded independently.
    """
    cache_path = OUT_DIR / f"graph_scorer_cache_{graph_variant}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        return cached["scorer"], cached["full_node_list"], cached["leaf_full_idx"]

    print("  Converting DiGraph -> simple undirected graph...")
    G_undirected = G.to_undirected()
    # to_undirected() on a DiGraph with parallel child->parent / parent->child
    # pairs (there are none here, edges only ever go child->parent) just
    # drops direction; no multi-edge collapsing surprises to worry about.
    assert isinstance(G_undirected, nx.Graph) and not G_undirected.is_directed()

    full_node_list = list(G_undirected.nodes())
    print(f"  Full node set size: {len(full_node_list)} "
          f"({len(leaf_names)} leaves + {len(full_node_list) - len(leaf_names)} internal nodes)")

    print("  Building dense adjacency matrix...")
    A_full = nx.to_numpy_array(G_undirected, nodelist=full_node_list)

    n_components = nx.number_connected_components(G_undirected)
    if n_components > 1:
        print(f"  WARNING: graph has {n_components} connected components -- "
              f"resistance distance between nodes in different components "
              f"is technically infinite; np.linalg.pinv handles this "
              f"gracefully (via the Moore-Penrose pseudoinverse) but double "
              f"check this is intended rather than a bug in graph construction.")

    name_to_full_idx = {name: i for i, name in enumerate(full_node_list)}
    leaf_full_idx = np.array([name_to_full_idx[name] for name in leaf_names])

    print("  Building GraphScorer (computing Laplacian pseudoinverse -- "
          "one-time cost, ~1-3s at this scale)...")
    scorer = GraphScorer(A_full)
    _ = scorer.resistance  # force eager computation now, with a clear log line,
                            # rather than a silent delay on first score() call

    with open(cache_path, "wb") as f:
        pickle.dump({
            "scorer": scorer,
            "full_node_list": full_node_list,
            "leaf_full_idx": leaf_full_idx,
        }, f)

    return scorer, full_node_list, leaf_full_idx


def embed_leaf_probs_into_full_graph(probs_leaf, leaf_full_idx, n_full_nodes):
    """
    Zero-pad a (batch, 1000) leaf-only probability array into a
    (batch, n_full_nodes) array, placing each leaf's probability at its
    corresponding index in the full node space and leaving all internal
    (non-leaf) node positions at exactly zero.

    Sum of each row is preserved exactly (still a valid distribution).
    """
    batch_size = probs_leaf.shape[0]
    probs_full = np.zeros((batch_size, n_full_nodes), dtype=np.float64)
    probs_full[:, leaf_full_idx] = probs_leaf
    return probs_full


def embed_leaf_labels_into_full_graph(true_labels_leaf, leaf_full_idx):
    """Map true leaf class indices (0..999) to their index in the full node space."""
    return leaf_full_idx[true_labels_leaf]


# ---------------------------------------------------------------------------
# (a-graph) Leaf-to-leaf shortest-path distance matrix, still used for the
# Bertinetto-style mistake_severity baseline below (that metric is defined
# via shortest-path graph distance, not resistance/diffusion -- kept as-is
# for direct literature comparability).
# ---------------------------------------------------------------------------
def precompute_leaf_distance_matrix(G, leaf_names, graph_variant="dag"):
    """
    Precompute the pairwise LCA-height mistake severity matrix between all
    1000 leaf classes, matching Bertinetto et al. (2020)'s definition:

        severity(u, v) = height(LCA(u, v))

    where height(node) is the length of the longest downward path from that
    node to any leaf in its subtree (leaves have height 0), and LCA(u,v) is
    the lowest common ancestor of u and v in the rooted taxonomy tree.

    NOTE: this is NOT the same as shortest-path distance d(u,v).  On the
    WordNet tree, the two metrics agree on the ranking of mistakes only when
    the tree is balanced; on the real unbalanced WordNet tree they can disagree
    substantially.  We use LCA-height for direct comparability with the
    literature (Bertinetto 2020, Karthik 2021, Russakovsky 2015).

    This function requires G to be a tree (not a DAG).  If called with
    graph_variant='dag', it falls back to shortest-path distance with a
    warning, since height/LCA are not well-defined on a DAG.
    """
    cache_path = OUT_DIR / f"leaf_distance_matrix_{graph_variant}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    if graph_variant == "dag":
        import warnings
        warnings.warn(
            "LCA-height mistake severity is not well-defined on a DAG "
            "(a node can have multiple LCAs). Falling back to shortest-path "
            "distance for graph_variant='dag'. Use --graph tree for "
            "Bertinetto-compatible mistake severity.",
            stacklevel=2
        )
        G_undirected = G.to_undirected()
        n = len(leaf_names)
        dist_matrix = np.zeros((n, n), dtype=np.float32)
        for i, name in enumerate(leaf_names):
            lengths = nx.shortest_path_length(G_undirected, source=name)
            for j, other_name in enumerate(leaf_names):
                dist_matrix[i, j] = lengths.get(other_name, np.inf)
        np.save(cache_path, dist_matrix)
        return dist_matrix

    # For tree/ctree: compute LCA-height correctly.
    # G has child->parent edges; root = the unique node with out_degree 0.
    roots = [n for n in G.nodes() if G.out_degree(n) == 0]
    assert len(roots) == 1, f"Expected 1 root, found {len(roots)}"
    root = roots[0]

    # Build children dict (parent -> list of children) for height computation
    children = {n: list(G.predecessors(n)) for n in G.nodes()}

    # Compute height of every node (memoised)
    _height_cache = {}
    def node_height(n):
        if n in _height_cache:
            return _height_cache[n]
        kids = children[n]
        h = 0 if not kids else 1 + max(node_height(k) for k in kids)
        _height_cache[n] = h
        return h

    # Precompute heights for all nodes
    for n in G.nodes():
        node_height(n)

    # Compute ancestors (inclusive) for each leaf -- stored as sets for O(1) lookup
    # We walk child->parent edges upward.
    def ancestors_inclusive(n):
        visited = set()
        queue = [n]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for par in G.successors(cur):  # child->parent direction
                queue.append(par)
        return visited

    # For LCA on a tree, a fast approach: precompute depth of each node,
    # then for each pair walk up from both nodes until they meet.
    # But for 1000x1000 pairs we need something efficient.
    # Since it's a tree, we can use the following:
    # ancestors(u) intersect ancestors(v) = all common ancestors;
    # LCA = the deepest one = max depth in tree (root has depth 0).
    G_undirected = G.to_undirected()
    depths = nx.single_source_shortest_path_length(G_undirected, root)

    # Precompute ancestor sets for all 1000 leaves
    print("  Precomputing ancestor sets for LCA computation...")
    leaf_ancestors = {}
    for name in leaf_names:
        leaf_ancestors[name] = ancestors_inclusive(name)

    n = len(leaf_names)
    dist_matrix = np.zeros((n, n), dtype=np.float32)

    print("  Computing pairwise LCA heights...")
    for i, u in enumerate(leaf_names):
        anc_u = leaf_ancestors[u]
        for j, v in enumerate(leaf_names):
            if i == j:
                dist_matrix[i, j] = 0.0
                continue
            if j < i:
                dist_matrix[i, j] = dist_matrix[j, i]
                continue
            # LCA = deepest common ancestor
            common = anc_u & leaf_ancestors[v]
            lca_node = max(common, key=lambda n: depths[n])
            dist_matrix[i, j] = float(_height_cache[lca_node])

    np.save(cache_path, dist_matrix)
    return dist_matrix


def your_graph_aware_scores(probs_leaf, true_labels_leaf, scorer, leaf_full_idx,
                             diffusion_tau=1.0):
    """
    Your actual graph-aware proper scoring rules: resistance_score and
    diffusion_score, evaluated over the full WordNet node set.

    Both follow the same "lower is better" convention as brier_score /
    log_loss_flat elsewhere in this file (verified explicitly against
    GraphScorer's behaviour: a prediction concentrated exactly on the true
    node always scores 0, the best possible value, for both).

    Parameters
    ----------
    probs_leaf : np.ndarray (m, 1000)
        Model's predicted probabilities over the 1000 leaf classes.
    true_labels_leaf : np.ndarray (m,)
        True leaf class indices, 0..999.
    scorer : GraphScorer
        Built once over the full node set via build_full_graph_scorer().
    leaf_full_idx : np.ndarray (1000,)
        Maps leaf class index -> index in the full node space.
    diffusion_tau : float
        Inverse length scale for the diffusion kernel. Larger tau = sharper
        discrimination at short range but less sensitivity at long range
        (see conversation notes / verification sweep over tau in
        scoring_refactor/ for the empirical behaviour across tau values --
        near-misses score better than far-misses across the whole range
        tested, 0.1 to 5.0; tau=1.0 is a reasonable default, not a value
        that needed to be tuned for correctness).

    Returns
    -------
    dict with keys 'resistance' and 'diffusion', each an (m,) array of
    per-instance scores.
    """
    n_full_nodes = scorer.size
    probs_full = embed_leaf_probs_into_full_graph(probs_leaf, leaf_full_idx, n_full_nodes)
    true_full = embed_leaf_labels_into_full_graph(true_labels_leaf, leaf_full_idx)

    resistance = scorer.resistance_score(probs_full, true_full)
    diffusion = scorer.diffusion_score(probs_full, true_full, tau=diffusion_tau, convention="positive")

    return {"resistance": resistance, "diffusion": diffusion}


# ---------------------------------------------------------------------------
# (c) Hierarchical mistake severity (Bertinetto et al. style) -- for
#     direct, literature-grounded comparison.
# ---------------------------------------------------------------------------
def distance_at_k(probs, true_labels, dist_matrix, k=1):
    """
    Average LCA-height distance between the top-K predictions and the true
    label, following Karthik et al. 2021 convention: computed over ALL
    examples (including correct ones, where distance=0 for k=1).

    For k=1: dist_matrix[true, top1_pred], which is 0 when correct.
    For k>1: mean of dist_matrix[true, top_j] over j=1..k.
    """
    n = len(true_labels)
    top_k_preds = np.argsort(-probs, axis=1)[:, :k]  # (n, k)
    total = np.zeros(n)
    for j in range(k):
        total += dist_matrix[true_labels, top_k_preds[:, j]]
    return (total / k).mean()


# ---------------------------------------------------------------------------
# Compute the full metric suite for ONE model's outputs.
# ---------------------------------------------------------------------------
def compute_all_metrics(model_name, probs, true_labels, dist_matrix,
                         scorer, leaf_full_idx, hxe_paths=None,
                         diffusion_tau=1.0):
    top1 = top_k_accuracy(probs, true_labels, 1)
    top5 = top_k_accuracy(probs, true_labels, 5)
    brier = brier_score_flat(probs, true_labels).mean()
    logloss = log_loss_flat(probs, true_labels).mean()
    dist1 = distance_at_k(probs, true_labels, dist_matrix, k=1)
    dist5 = distance_at_k(probs, true_labels, dist_matrix, k=5)

    graph_scores = your_graph_aware_scores(
        probs, true_labels, scorer, leaf_full_idx, diffusion_tau=diffusion_tau
    )

    result = {
        "model":            model_name,
        "top1_acc":         top1,
        "top5_acc":         top5,
        "brier_flat":       brier,
        "logloss_flat":     logloss,
        "distance_at_1":    dist1,
        "distance_at_5":    dist5,
        "resistance_score": graph_scores["resistance"].mean(),
    }

    if hxe_paths is not None:
        result["hxe_score"] = hxe_score(probs, true_labels, hxe_paths).mean()

    return result


def make_distance_at_1_plot(model_name, probs, true_labels, dist_matrix):
    preds = probs.argmax(axis=1)
    dists = dist_matrix[true_labels, preds]  # 0 for correct, >0 for wrong

    plt.figure(figsize=(7, 5))
    plt.hist(dists, bins=range(0, int(dists.max()) + 2), edgecolor="black")
    plt.xlabel("LCA-height distance between predicted and true class (all examples)")
    plt.ylabel("Count")
    plt.title(f"Hierarchical distance@1 distribution -- {model_name}")
    plt.tight_layout()
    out_path = OUT_DIR / f"distance_at_1_distribution_{model_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Main: compute everything for one or all models, print/save comparison table
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=None,
        help="Evaluate a single model.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate ALL models found in outputs/.",
    )
    parser.add_argument(
        "--graph", choices=["dag", "tree", "ctree"], default="dag",
        help=(
            "Which graph to use for the graph-aware scores and distance matrix. "
            "dag: full WordNet ancestor-closure DAG (default). "
            "tree: Bertinetto path-selection tree, no single-child collapse "
            "(recommended for distance-based scores -- preserves relay chains). "
            "ctree: Bertinetto tree WITH single-child collapse (exact replication "
            "of Bertinetto et al. 2020 -- relay chains shortened, use for "
            "literature comparability only). "
            "All output files are suffixed with the graph variant so results "
            "from dag/tree/ctree never overwrite each other."
        ),
    )
    args = parser.parse_args()
    graph_variant = args.graph

    available = discover_available_models()
    if not available:
        raise RuntimeError(
            "No probs_<model>.npy files found in outputs/. "
            "Run run_inference.py --model <name> (or --all) first."
        )
    print(f"Found inference outputs for: {available}")
    print(f"Graph variant: {graph_variant}")

    if args.all:
        models_to_run = available
    elif args.model is not None:
        assert args.model in available, (
            f"No outputs found for model '{args.model}'. Available: {available}"
        )
        models_to_run = [args.model]
    else:
        models_to_run = [available[0]]
        print(f"No --model specified, defaulting to '{models_to_run[0]}'")

    print(f"Loading label graph ({graph_variant})...")
    G, leaf_info = load_graph_and_leaf_info(graph_variant)
    leaf_names = leaf_info["class_idx_to_synset_name"]

    print("Computing/loading leaf-to-leaf shortest-path distance matrix...")
    dist_matrix = precompute_leaf_distance_matrix(G, leaf_names, graph_variant)

    print(f"Building GraphScorer ({graph_variant})...")
    scorer, full_node_list, leaf_full_idx = build_full_graph_scorer(G, leaf_names, graph_variant)

    hxe_alpha = 0.5
    print(f"Building HXE cache (alpha={hxe_alpha})...")
    hxe_paths = build_hxe_cache(G, leaf_names, alpha=hxe_alpha, graph_variant=graph_variant)

    all_results = []
    reference_labels = None
    for model_name in models_to_run:
        print(f"\n{'='*70}\nEvaluating: {model_name}  [{graph_variant}]\n{'='*70}")
        probs, true_labels = load_model_outputs(model_name)

        if reference_labels is None:
            reference_labels = true_labels
        else:
            assert np.array_equal(true_labels, reference_labels), (
                f"true_labels for '{model_name}' do not match the labels "
                f"from the first model evaluated."
            )

        metrics = compute_all_metrics(
            model_name, probs, true_labels, dist_matrix, scorer, leaf_full_idx,
            hxe_paths=hxe_paths
        )
        for k, v in metrics.items():
            if k != "model":
                print(f"  {k:30s} {v:.4f}")

        all_results.append(metrics)
        plot_path = make_distance_at_1_plot(model_name, probs, true_labels, dist_matrix)
        print(f"  Saved plot: {plot_path}")

    results_df = pd.DataFrame(all_results).set_index("model")
    out_csv = OUT_DIR / (
        f"results_table_all_models_{graph_variant}.csv" if len(models_to_run) > 1
        else f"results_table_{models_to_run[0]}_{graph_variant}.csv"
    )
    results_df.to_csv(out_csv)
    print(f"\n{'='*70}\nCOMPARISON TABLE  [{graph_variant}]\n{'='*70}")
    print(results_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved combined results table to {out_csv}")


if __name__ == "__main__":
    main()