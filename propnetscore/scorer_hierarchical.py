"""
Scoring rules for hierarchical classification (node selection on trees).

Also provides utility functions to construct a rooted tree as a
networkx DiGraph from common input formats.

Convention: edges in the tree are directed parent -> child.
"""

import math
import warnings

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path

from .scorer import GraphScorer


# -----------------------------------------------------------------
# Tree construction utilities
# -----------------------------------------------------------------

def _check_is_tree(G: nx.DiGraph):
    """Raise error if G is not a rooted tree (edges parent -> child)."""
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph is not a DAG (contains cycles).")
    in_degrees = dict(G.in_degree())
    roots = [n for n, d in in_degrees.items() if d == 0]
    if len(roots) != 1:
        raise ValueError(
            f"Graph has {len(roots)} roots (nodes with in-degree 0); "
            f"a rooted tree must have exactly one."
        )
    if not nx.is_weakly_connected(G):
        raise ValueError("Graph is not connected.")
    n = G.number_of_nodes()
    e = G.number_of_edges()
    if e != n - 1:
        raise ValueError(
            f"Graph has {n} nodes and {e} edges; a tree requires exactly "
            f"n-1={n-1} edges. (The graph may be a DAG but not a tree.)"
        )


def tree_from_edges(edges, check: bool = True) -> nx.DiGraph:
    """Build a rooted tree DiGraph from a list of (parent, child) pairs.

    Parameters
    ----------
    edges : iterable of (parent, child) pairs
        Node identifiers can be any hashable type.
    check : bool
        If True (default), verify that the result is a rooted tree.

    Returns
    -------
    nx.DiGraph with edges directed from parent to child.
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)
    if check:
        _check_is_tree(G)
    return G


def tree_from_parent_dict(parents: dict, check: bool = True) -> nx.DiGraph:
    """Build a rooted tree DiGraph from a {child: parent} dict.

    Parameters
    ----------
    parents : dict
        Maps each non-root node to its parent. The root is the unique
        node that does not appear as a key.
    check : bool
        If True (default), verify that the result is a rooted tree.

    Returns
    -------
    nx.DiGraph with edges directed from parent to child.
    """
    edges = [(parent, child) for child, parent in parents.items()]
    return tree_from_edges(edges, check=check)


# -----------------------------------------------------------------
# HierarchicalScorer
# -----------------------------------------------------------------

class HierarchicalScorer:
    """Scoring rules for hierarchical classification on a rooted tree.

    Precomputes tree-level quantities (LCA-height distance matrix, HXE
    path weights) on construction and caches them. All scoring methods
    share the same signature as the ones from GraphScorer:

    Parameters
    ----------
    P : np.ndarray (m, n_leaves)
        Batch of m probability distributions over the leaf nodes.
    true_nodes : np.ndarray (m,), dtype int
        True node index for each row of P. Values in [0, n_leaves).

    Returns
    -------
    np.ndarray (m,)
        One score per row. For all scores, lower values indicate better performance.

    Note: this class implements the mandatory leaf-node prediction (MLP)
    setting, i.e. P must be a distribution over leaf nodes.
    For the non-mandatory leaf-node prediction (NMLP) setting, use
    GraphScorer directly with a distribution over all tree nodes.
    """

    def __init__(self, tree: nx.DiGraph, leaves: list):
        """
        Parameters
        ----------
        tree : nx.DiGraph
            Rooted tree with edges directed from parent to child.
            Use tree_from_edges or tree_from_parent_dict to construct.
        leaves : list
            Ordered list of leaf node identifiers. The order defines
            the column ordering of P: P[:, k] are the probabilities
            assigned to the leaf leaves[k].
        """
        _check_is_tree(tree)
        tree_leaves = {n for n in tree.nodes() if tree.out_degree(n) == 0}
        if set(leaves) != tree_leaves:
            raise ValueError(
                f"'leaves' does not match the leaf nodes of the tree. "
                f"Expected {len(tree_leaves)} leaves, got {len(leaves)}. "
                f"Extra in leaves: {set(leaves) - tree_leaves}. "
                f"Missing from leaves: {tree_leaves - set(leaves)}."
            )
        self.tree = tree
        self.leaves = leaves
        self.size = len(leaves)  # careful: not the number of all nodes in the tree
        self._leaf_to_idx = {leaf: k for k, leaf in enumerate(leaves)}

        # Prepare an internal instance of GraphScorer for resistance score.
        nodes = list(tree.nodes())
        self._node_to_idx = {n: k for k, n in enumerate(nodes)}  # this defines the node indexing 0,...,n-1 in the full graph
        A = nx.to_numpy_array(tree.to_undirected(), nodelist=nodes)  # adjacency matrix of the undirected tree with the same indexing
        self._graph_scorer = GraphScorer(A)
        # On a tree the resistance distance equals the shortest-path length, which is more stable to compute:
        self._graph_scorer._resistance = shortest_path(A, method="D", directed=False, unweighted=True)
        self._leaf_to_graph_idx = np.array(
            [self._node_to_idx[l] for l in leaves], dtype=np.int64  # map leaves to their corresponding indices in the full graph
        )

        # Lazy quantities
        self._root = None
        self._depths = None  # dict: node -> depth from root
        self._heights = None  # dict: node -> height (longest path to leaf)
        self._subtree_leaves = None  # dict: node -> np.array of corresponding leaf indices
        self._lca_dist_matrix = None  # (n_leaves, n_leaves) float array of LCA height distances between leaves
        self._hxe_paths = None  # list of length n_leaves, each containing a list of tuples (node, sub_idx, depth); see hxe_paths property

    # -----------------------------------------------------------------
    # Tree utilities (internal)
    # -----------------------------------------------------------------

    def _get_root(self):
        """Return the unique root node (in-degree 0)."""
        if self._root is None:
            for node in self.tree.nodes():
                if self.tree.in_degree(node) == 0:
                    self._root = node
                    break
        return self._root

    def _get_depths(self):
        """Depth of each node from root (root = 0)."""
        if self._depths is None:
            root = self._get_root()
            self._depths = nx.single_source_shortest_path_length(
                self.tree.to_undirected(), root
            )
        return self._depths

    def _get_heights(self):
        """Height of each node = length of longest downward path to a leaf."""
        if self._heights is None:
            heights = {}
            def height(node):
                if node in heights:
                    return heights[node]
                children = list(self.tree.successors(node))
                h = 0 if not children else 1 + max(height(c) for c in children)
                heights[node] = h
                return h
            for node in self.tree.nodes():
                height(node)
            self._heights = heights
        return self._heights

    def _get_subtree_leaves(self):
        """For each node, sorted array of leaf indices in its subtree."""
        if self._subtree_leaves is None:
            result = {}
            # process in reverse topological order (leaves first)
            for node in reversed(list(nx.topological_sort(self.tree))):
                children = list(self.tree.successors(node))
                if children:  # node is internal
                    result[node] = np.concatenate(
                        [result[c] for c in children]
                    ).astype(np.int32)
                else:  # node is a leaf
                    result[node] = np.array(
                        [self._leaf_to_idx[node]]
                    ).astype(np.int32)
            self._subtree_leaves = result
        return self._subtree_leaves

    # -----------------------------------------------------------------
    # Precomputed quantities
    # -----------------------------------------------------------------

    @property
    def lca_dist_matrix(self):
        """LCA-height distance matrix between all pairs of leaves.

        dist_matrix[j, k] = height(LCA(j, k))

        where height(node) is the length of the longest downward path
        from that node to any leaf in its subtree (leaves have height 0)
        and LCA(j, k) denotes the lowest common ancestor of j and k.

        Shape: (n_leaves, n_leaves), dtype int32.
        """
        if self._lca_dist_matrix is None:
            depths = self._get_depths()
            heights = self._get_heights()
            subtree_leaves = self._get_subtree_leaves()
            D = np.zeros((self.size, self.size), dtype=np.int32)

            # LCA(u, v) = deepest node in the intersection of
            # ancestors(u) and ancestors(v). Can be found by walking up
            # from both nodes using ancestor sets.
            def ancestors_inclusive(node):
                """Generate complete ancestor set of node (including root and itself)."""
                anc = set()
                cur = node
                while cur is not None:
                    anc.add(cur)
                    pars = list(self.tree.predecessors(cur))
                    cur = pars[0] if pars else None
                return anc

            leaf_ancestors = {
                leaf: ancestors_inclusive(leaf) for leaf in self.leaves
            }

            for i, u in enumerate(self.leaves):
                for j, v in enumerate(self.leaves):
                    if i == j:
                        continue
                    if j < i:
                        D[i, j] = D[j, i]
                        continue
                    common = leaf_ancestors[u] & leaf_ancestors[v]
                    lca = max(common, key=lambda x: depths[x])
                    D[i, j] = int(heights[lca])
            self._lca_dist_matrix = D
        return self._lca_dist_matrix

    @property
    def hxe_paths(self):
        """Path information for HXE scoring, precomputed for each leaf.

        hxe_paths[k] is a list of D tuples (node, sub_idx, depth), one per
        ancestor of leaf k from depth 1 (just below root) to depth D
        (the leaf itself), where sub_idx is a numpy array of leaf-level
        indices of all leaves descending from that ancestor.

        Used by hxe_score to efficiently compute subtree probabilities
        P_j = p[sub_idx].sum() at each level of the hierarchy.
        """
        if self._hxe_paths is None:
            depths = self._get_depths()
            subtree_leaves = self._get_subtree_leaves()
            self._hxe_paths = []
            for leaf in self.leaves:
                # walk from leaf up to root, collect path
                path = []
                cur = leaf
                while cur is not None:
                    path.append(cur)
                    pars = list(self.tree.predecessors(cur))
                    cur = pars[0] if pars else None
                path.reverse()  # now ordered from root to leaf

                path_info = []
                for node in path[1:]:  # exclude root: P_root = 1 always, contributes 0
                    sub_idx = subtree_leaves[node]
                    d = depths[node]
                    path_info.append((node, sub_idx, d))
                self._hxe_paths.append(path_info)
        return self._hxe_paths

    # -----------------------------------------------------------------
    # Input checking
    # -----------------------------------------------------------------

    def _check_inputs(self, P: np.ndarray, true_nodes: np.ndarray):
        """Check that P and true_nodes are compatible with the tree."""
        if P.ndim != 2:
            raise ValueError(f"P must be 2D, got shape {P.shape}.")
        if P.shape[1] != self.size:
            raise ValueError(
                f"P has {P.shape[1]} columns but tree has {self.size} leaves."
            )
        if true_nodes.ndim != 1:
            raise ValueError(f"true_nodes must be 1D, got shape {true_nodes.shape}.")
        if P.shape[0] != true_nodes.shape[0]:
            raise ValueError(
                f"P has {P.shape[0]} rows but true_nodes has {true_nodes.shape[0]} elements."
            )
        if np.any(true_nodes < 0) or np.any(true_nodes >= self.size):
            raise ValueError(
                f"true_nodes contains indices outside [0, {self.size})."
            )

    # -----------------------------------------------------------------
    # Embedding in full tree and delegation to GraphScorer
    # -----------------------------------------------------------------

    def _embed(self, P: np.ndarray, true_nodes: np.ndarray):
        """Embed (P, true_nodes) from leaf-space into full tree for GraphScorer."""
        m = P.shape[0]
        n_full = self._graph_scorer.size
        P_full = np.zeros((m, n_full))
        P_full[:, self._leaf_to_graph_idx] = P
        true_nodes_full = self._leaf_to_graph_idx[true_nodes]
        return P_full, true_nodes_full

    def resistance_score(self, P: np.ndarray, true_nodes: np.ndarray
                         ) -> np.ndarray:
        """Resistance score on the tree (delegates to GraphScorer)."""
        self._check_inputs(P, true_nodes)
        P_full, true_full = self._embed(P, true_nodes)
        return self._graph_scorer.resistance_score(P_full, true_full)

    def brier_score(self, P: np.ndarray, true_nodes: np.ndarray):
        self._check_inputs(P, true_nodes)
        m = P.shape[0]
        onehot = np.zeros((m, self.size))
        onehot[np.arange(m), true_nodes] = 1.0
        residuals = onehot - P
        return np.einsum('mi,mi->m', residuals, residuals)

    def logarithmic_score(self, P: np.ndarray, true_nodes: np.ndarray):
        """Logarithmic score, also known as cross-entropy."""
        self._check_inputs(P, true_nodes)
        return -np.log(P[np.arange(P.shape[0]), true_nodes])

    def top_k_error(self, P: np.ndarray, true_nodes: np.ndarray, k: int = 1):
        """ Top-k miss indicator:
        Returns 1 if the true node is not in the top-k nodes according to P; 0 else.
        """
        self._check_inputs(P, true_nodes)
        top_k_idx = np.argpartition(-P, kth=k - 1, axis=1)[:, :k]  # argpartition is more efficient than a full argsort
        return (~(top_k_idx == true_nodes[:, None]).any(axis=1)).astype(float)

    # -----------------------------------------------------------------
    # Scoring methods
    # -----------------------------------------------------------------

    def distance_at_k(self, P: np.ndarray, true_nodes: np.ndarray,
                      k: int = 1) -> np.ndarray:
        """Hierarchical distance@k.

        Average LCA-height distance between the true node and each of
        the top-k predicted nodes, following [Bertinetto et al. 2020]
        and [Karthik et al. 2021].

        Warning: not a proper scoring rule! Kept for comparison purposes.
        """
        self._check_inputs(P, true_nodes)
        top_k_idx = np.argpartition(-P, kth=k - 1, axis=1)[:, :k]
        m = P.shape[0]
        out = np.empty(m)
        D = self.lca_dist_matrix
        for row in range(m):
            out[row] = D[true_nodes[row], top_k_idx[row]].mean()
        return out

    def urbani_score(self, P: np.ndarray, true_nodes: np.ndarray,
                     weights: dict) -> np.ndarray:
        """Urbani score with arbitrary node weights.

        Strictly proper log-score-type scoring rule for hierarchical
        classification [Urbani et al. 2024].

            S_Urb(p, i) = -sum_{j on path root->i, j != root} w_j * log(P_j)

        where P_j = p[subtree_leaf_indices(j)].sum() and w_j = weights[j].

        Strict propriety holds when the weights are balanced, i.e. the
        total weight on every root-to-leaf path is the same.

        Parameters
        ----------
        weights : dict {node: float}
            Weight for each non-root node in the tree. Must contain
            an entry for every node.
        """
        self._check_inputs(P, true_nodes)
        paths = self.hxe_paths

        # check balancedness
        path_totals = [
            sum(weights[node] for node, _, _ in leaf_paths)
            for leaf_paths in paths
        ]
        if not np.allclose(path_totals, path_totals[0]):
            warnings.warn(
                "Urbani score: weights are not balanced (total weight differs "
                "across root-to-leaf paths). The score may not be strictly proper.",
                UserWarning
            )

        m = P.shape[0]
        out = np.zeros(m)

        for row in range(m):
            p = P[row]
            path_info = paths[true_nodes[row]]
            score = 0.0
            for node, sub_idx, depth in path_info:
                P_j = p[sub_idx].sum()
                score -= weights[node] * math.log(max(P_j, 1e-300))
            out[row] = score

        return out

    def hxe_score(self, P: np.ndarray, true_nodes: np.ndarray,
                  alpha: float) -> np.ndarray:
        """Hierarchical cross-entropy score (HXE).

        Special case of urbani_score with exponential weights:
            internal node j: w_j = exp(-alpha * depth(j)) * (1 - exp(-alpha))
            leaf node j:     w_j = exp(-alpha * depth(j))

        Parameters
        ----------
        alpha : float
            Decay parameter.
        """
        self._check_inputs(P, true_nodes)
        paths = self.hxe_paths
        decay = math.exp(-alpha)
        m = P.shape[0]
        out = np.zeros(m)

        for row in range(m):
            p = P[row]
            path_info = paths[true_nodes[row]]
            depth_true = len(path_info)  # depth of the true node
            score = 0.0
            for _, sub_idx, depth in path_info:
                P_j = p[sub_idx].sum()
                base_weight = decay ** depth
                if depth == depth_true:  # node is a leaf
                    w = base_weight
                else:  # node is internal
                    w = base_weight * (1.0 - decay)
                score -= w * math.log(max(P_j, 1e-300))
            out[row] = score

        return out
