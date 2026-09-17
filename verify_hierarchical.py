"""
Regression and analytical tests for HierarchicalScorer.

Run from the repository root with:
    python verify_hierarchical.py
"""

import math
import warnings
import numpy as np
import networkx as nx

from propnetscore.scorer import GraphScorer
from propnetscore.scorer_hierarchical import (
    HierarchicalScorer, tree_from_edges, tree_from_parent_dict
)

TOL = 1e-8


def check_close(name, val_new, val_ref, tol=TOL):
    err = abs(val_new - val_ref)
    status = "OK" if err < tol else "FAIL"
    print(f"  [{status}] {name}: got={val_new:.10f}, ref={val_ref:.10f}, err={err:.2e}")
    assert err < tol, f"FAILED: {name}"


def check_array_close(name, arr_new, arr_ref, tol=TOL):
    err = float(np.max(np.abs(arr_new - arr_ref)))
    status = "OK" if err < tol else "FAIL"
    print(f"  [{status}] {name}: max_err={err:.2e}")
    assert err < tol, f"FAILED: {name}"


# -----------------------------------------------------------------
# Build a simple balanced binary tree:
# root -> A, B; A -> a1, a2; B -> b1, b2
# -----------------------------------------------------------------
#         root
#        /    \
#       A      B
#      / \    / \
#     a1  a2 b1  b2

edges = [
    ("root", "A"), ("root", "B"),
    ("A", "a1"), ("A", "a2"),
    ("B", "b1"), ("B", "b2"),
]
leaves = ["a1", "a2", "b1", "b2"]
tree = tree_from_edges(edges)
scorer = HierarchicalScorer(tree, leaves)


# -----------------------------------------------------------------
# 1. Tree construction and validity checks
# -----------------------------------------------------------------
print("\n=== Tree construction and validity checks ===")

parents = {"A": "root", "B": "root",
           "a1": "A", "a2": "A", "b1": "B", "b2": "B"}
tree2 = tree_from_parent_dict(parents)
assert set(tree2.edges()) == set(tree.edges()), "tree_from_parent_dict mismatch"
print("  [OK] tree_from_edges == tree_from_parent_dict")

for label, bad_edges in [
    ("cycle",        [("root","A"),("A","B"),("B","root")]),
    ("multi-root",   [("root1","A"),("root2","B")]),
    ("disconnected", [("root","A"),("B","C")]),
]:
    try:
        tree_from_edges(bad_edges)
        print(f"  [FAIL] should have raised ValueError for {label}")
        assert False
    except ValueError:
        print(f"  [OK] correctly rejects {label} graph")

# leaf mismatch check
try:
    HierarchicalScorer(tree, ["a1", "a2", "b1"])  # missing b2
    print("  [FAIL] should have raised ValueError for leaf mismatch")
    assert False
except ValueError:
    print("  [OK] correctly rejects mismatched leaves")


# -----------------------------------------------------------------
# 2. HXE(alpha=0) == log-loss
# -----------------------------------------------------------------
print("\n=== HXE(alpha=0) == log-loss ===")
rng = np.random.default_rng(42)
for trial in range(5):
    p = rng.dirichlet(np.ones(4))
    true_node = int(rng.integers(4))
    P = p[np.newaxis]
    i = np.array([true_node])
    hxe = float(scorer.hxe_score(P, i, alpha=0.0)[0])
    logloss = float(scorer.logarithmic_score(P, i)[0])
    check_close(f"trial {trial} (i={true_node})", hxe, logloss)


# -----------------------------------------------------------------
# 3. HXE matches log-ratio formula
# -----------------------------------------------------------------
print("\n=== HXE matches log-ratio formula ===")

leaf_to_idx = {l: k for k, l in enumerate(leaves)}

def hxe_logratio(p, true_leaf, alpha, tree, leaves, leaf_to_idx):
    """Reference: log-ratio formula from paper."""
    def subtree_prob(node):
        desc = nx.descendants(tree, node) | {node}
        return sum(p[leaf_to_idx[l]] for l in leaves if l in desc)

    root = next(n for n in tree.nodes() if tree.in_degree(n) == 0)
    depths = nx.single_source_shortest_path_length(tree.to_undirected(), root)

    # walk from leaf up to root, collect (node, parent) pairs
    anc = []
    cur = true_leaf
    while list(tree.predecessors(cur)):
        parent = list(tree.predecessors(cur))[0]
        anc.append((cur, parent))
        cur = parent

    s = 0.0
    for node, parent in anc:
        d = depths[node]
        P_j = subtree_prob(node)
        P_par = subtree_prob(parent)
        w = math.exp(-alpha * d)
        s -= w * math.log(max(P_j / max(P_par, 1e-300), 1e-300))
    return s

rng = np.random.default_rng(7)
for alpha in [0.0, 0.5, 1.0, 2.0]:
    for trial in range(3):
        p = rng.dirichlet(np.ones(4))
        true_node = int(rng.integers(4))
        P = p[np.newaxis]
        i = np.array([true_node])
        hxe_new = float(scorer.hxe_score(P, i, alpha=alpha)[0])
        hxe_ref = hxe_logratio(p, leaves[true_node], alpha,
                                tree, leaves, leaf_to_idx)
        check_close(f"alpha={alpha} trial {trial} (i={true_node})",
                    hxe_new, hxe_ref)


# -----------------------------------------------------------------
# 4. HXE == Urbani score with HXE weights
# -----------------------------------------------------------------
print("\n=== HXE == Urbani with HXE weights ===")

def hxe_weights(tree, alpha):
    """Build HXE weight dict for urbani_score."""
    root = next(n for n in tree.nodes() if tree.in_degree(n) == 0)
    depths = nx.single_source_shortest_path_length(tree.to_undirected(), root)
    decay = math.exp(-alpha)
    w = {}
    for node in tree.nodes():
        if node == root:
            continue
        d = depths[node]
        children = list(tree.successors(node))
        if not children:  # leaf
            w[node] = decay ** d
        else:  # internal
            w[node] = (decay ** d) * (1.0 - decay)
    return w

rng = np.random.default_rng(11)
for alpha in [0.5, 1.0]:
    weights = hxe_weights(tree, alpha)
    for trial in range(3):
        p = rng.dirichlet(np.ones(4))
        true_node = int(rng.integers(4))
        P = p[np.newaxis]
        i = np.array([true_node])
        hxe = float(scorer.hxe_score(P, i, alpha=alpha)[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            urb = float(scorer.urbani_score(P, i, weights)[0])
        check_close(f"alpha={alpha} trial {trial} (i={true_node})", hxe, urb)


# -----------------------------------------------------------------
# 5. Urbani warns on unbalanced weights
# -----------------------------------------------------------------
print("\n=== Urbani warns on unbalanced weights ===")
unbalanced = {"A": 1.0, "B": 2.0, "a1": 0.5, "a2": 0.5, "b1": 0.5, "b2": 0.5}
p = np.array([[0.25, 0.25, 0.25, 0.25]])
i = np.array([0])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    scorer.urbani_score(p, i, unbalanced)
    assert len(w) == 1 and issubclass(w[0].category, UserWarning)
    print("  [OK] correctly warns for unbalanced weights")


# -----------------------------------------------------------------
# 6. distance@1 == height(LCA(top1, true))
# -----------------------------------------------------------------
print("\n=== distance@1 == height(LCA(top1, true)) ===")

heights = {"root": 2, "A": 1, "B": 1,
           "a1": 0, "a2": 0, "b1": 0, "b2": 0}

def lca(u, v, tree):
    def ancestors(n):
        anc = {n}
        while list(tree.predecessors(n)):
            n = list(tree.predecessors(n))[0]
            anc.add(n)
        return anc
    root = next(n for n in tree.nodes() if tree.in_degree(n)==0)
    depths = nx.single_source_shortest_path_length(tree.to_undirected(), root)
    common = ancestors(u) & ancestors(v)
    return max(common, key=lambda x: depths[x])

rng = np.random.default_rng(13)
for trial in range(5):
    p = rng.dirichlet(np.ones(4))
    true_node = int(rng.integers(4))
    P = p[np.newaxis]
    i = np.array([true_node])
    d1 = float(scorer.distance_at_k(P, i, k=1)[0])
    top1_leaf = leaves[int(np.argmax(p))]
    true_leaf  = leaves[true_node]
    lca_node = lca(top1_leaf, true_leaf, tree)
    ref = float(heights[lca_node])
    check_close(f"trial {trial} (top1={top1_leaf}, true={true_leaf})", d1, ref)


# -----------------------------------------------------------------
# 7. resistance_score == GraphScorer on embedded distribution
# -----------------------------------------------------------------
print("\n=== HierarchicalScorer.resistance_score == GraphScorer on embedded P ===")

nodes = list(tree.nodes())
A_full = nx.to_numpy_array(tree.to_undirected(), nodelist=nodes)
gs_full = GraphScorer(A_full)
leaf_to_full = np.array([nodes.index(l) for l in leaves])

rng = np.random.default_rng(0)
for trial in range(5):
    p = rng.dirichlet(np.ones(4))
    true_node = int(rng.integers(4))
    P = p[np.newaxis]
    iv = np.array([true_node])

    res_hier = float(scorer.resistance_score(P, iv)[0])

    P_full = np.zeros((1, len(nodes)))
    P_full[0, leaf_to_full] = p
    true_full = np.array([leaf_to_full[true_node]])
    res_gs = float(gs_full.resistance_score(P_full, true_full)[0])

    check_close(f"trial {trial} (i={true_node})", res_hier, res_gs)


# -----------------------------------------------------------------
# 8. Input checking
# -----------------------------------------------------------------
print("\n=== Input checking ===")

rng = np.random.default_rng(99)
p = rng.dirichlet(np.ones(4))[np.newaxis]

for label, args in [
    ("3D P",           (np.ones((1,1,4)), np.array([0]))),
    ("wrong columns",  (np.ones((1,3)),   np.array([0]))),
    ("2D true_nodes",  (p,                np.array([[0]]))),
    ("row mismatch",   (p,                np.array([0,1]))),
    ("index too large",(p,                np.array([4]))),
    ("negative index", (p,                np.array([-1]))),
]:
    try:
        scorer.hxe_score(*args, alpha=0.5)
        print(f"  [FAIL] should have raised for: {label}")
        assert False
    except (ValueError, IndexError):
        print(f"  [OK] correctly rejects: {label}")

print("\n=== All hierarchical tests passed ===\n")