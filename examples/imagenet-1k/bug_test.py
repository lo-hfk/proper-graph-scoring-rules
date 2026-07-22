"""
Diagnostic: check whether tree.gpickle and ctree.gpickle are genuinely
different graphs, and if so, why they produce identical evaluation scores.

Run from the same directory as your other imagenet scripts:
    python check_tree_ctree.py
"""

import hashlib
import pickle
from pathlib import Path

import networkx as nx

GRAPH_DIR = Path("./wordnet_graph")


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load(name):
    path = GRAPH_DIR / name
    with open(path, "rb") as f:
        return pickle.load(f)


print("=" * 60)
print("STEP 1: Are the two files byte-identical?")
print("=" * 60)
h_tree  = file_hash(GRAPH_DIR / "tree.gpickle")
h_ctree = file_hash(GRAPH_DIR / "ctree.gpickle")
print(f"tree.gpickle  MD5: {h_tree}")
print(f"ctree.gpickle MD5: {h_ctree}")
identical = h_tree == h_ctree
print(f"Byte-identical: {identical}")
if identical:
    print("\nDIAGNOSIS: ctree.gpickle is a copy of tree.gpickle.")
    print("build_tree_graph.py --graph ctree was either not run, or wrote")
    print("to the wrong file.  Fix:")
    print("  1. Delete wordnet_graph/tree.gpickle and wordnet_graph/ctree.gpickle")
    print("  2. python build_tree_graph.py --graph tree")
    print("  3. python build_tree_graph.py --graph ctree")
    print("  4. Delete model_validation_outputs/graph_scorer_cache_tree.pkl")
    print("     and model_validation_outputs/graph_scorer_cache_ctree.pkl")
    print("  5. Rerun evaluate.py --all --graph tree  and  --graph ctree")
    raise SystemExit

print()
print("=" * 60)
print("STEP 2: Do they have the same node/edge sets?")
print("=" * 60)
t = load("tree.gpickle")
c = load("ctree.gpickle")
print(f"tree:  {t.number_of_nodes()} nodes, {t.number_of_edges()} edges")
print(f"ctree: {c.number_of_nodes()} nodes, {c.number_of_edges()} edges")

same_nodes = set(t.nodes()) == set(c.nodes())
same_edges = set(t.edges()) == set(c.edges())
print(f"Same node set: {same_nodes}")
print(f"Same edge set: {same_edges}")

if same_nodes and same_edges:
    print("\nDIAGNOSIS: The two files have DIFFERENT bytes but encode the")
    print("SAME graph.  This means collapse_single_child_nodes() was a")
    print("genuine no-op on the real WordNet path-selected tree --")
    print("there were zero single-child internal nodes to collapse.")
    print("See STEP 3 below for confirmation.")
elif not same_nodes:
    print("\nFiles encode DIFFERENT graphs (different node sets).")
    print("The identical scores are unexpected -- see STEP 3 and 4.")

print()
print("=" * 60)
print("STEP 3: How many single-child internal nodes does tree.gpickle have?")
print("(These are the nodes collapse_single_child_nodes() would remove.)")
print("=" * 60)

# Load leaf names to exclude them from the single-child check
with open(GRAPH_DIR / "tree_leaf_synsets.json") as f:
    import json
    leaf_info = json.load(f)
leaf_set = set(leaf_info["class_idx_to_synset_name"])

# In child->parent DiGraph:
#   predecessors(n) = children of n (nodes pointing TO n)
#   successors(n)   = parents of n  (nodes n points to)
single_child_nodes = [
    n for n in t.nodes()
    if n not in leaf_set
    and t.in_degree(n) == 1   # exactly one child
    and t.out_degree(n) == 1  # exactly one parent (not root)
]
print(f"Single-child internal nodes in tree.gpickle: {len(single_child_nodes)}")
if single_child_nodes:
    print("Examples:", single_child_nodes[:10])
else:
    print("\nDIAGNOSIS CONFIRMED: zero single-child internal nodes.")
    print("Bertinetto's longest-path selection leaves no relay chains in")
    print("the real WordNet graph.  collapse_single_child_nodes() is a")
    print("genuine no-op, so tree.gpickle == ctree.gpickle as graphs.")
    print("The identical scores are a real finding, not a caching bug.")
    print()
    print("Intuition: in a rich taxonomy like WordNet, almost every internal")
    print("node is a hypernym of MULTIPLE synsets, so after path selection")
    print("every surviving internal node has at least two children pointing")
    print("to it from different leaves.  Single-child relay nodes, if they")
    print("ever existed in the DAG, were either excluded during path selection")
    print("(because a longer path was available without them) or they are")
    print("genuine branching nodes in the WordNet sense.")

print()
print("=" * 60)
print("STEP 4: Sanity check -- are the graph_scorer_cache files also identical?")
print("=" * 60)
OUT_DIR = Path("./model_validation_outputs")
cache_tree  = OUT_DIR / "graph_scorer_cache_tree.pkl"
cache_ctree = OUT_DIR / "graph_scorer_cache_ctree.pkl"
if cache_tree.exists() and cache_ctree.exists():
    h_ct = file_hash(cache_tree)
    h_cc = file_hash(cache_ctree)
    print(f"graph_scorer_cache_tree.pkl  MD5: {h_ct}")
    print(f"graph_scorer_cache_ctree.pkl MD5: {h_cc}")
    print(f"Cache files identical: {h_ct == h_cc}")
    if h_ct == h_cc:
        print("(Expected if the underlying graphs are identical.)")
else:
    missing = [str(p) for p in [cache_tree, cache_ctree] if not p.exists()]
    print(f"Cache file(s) not found: {missing}")
    print("(Run evaluate.py --all --graph tree and --graph ctree to generate them.)")