"""
Build the WordNet hierarchy tree for ImageNet-1k evaluation.

Applies Bertinetto et al.'s (2020) path-selection rule to the WordNet
ancestor closure of the 1000 ImageNet-1k leaf synsets, processing leaves
in class-index order (0-999). Unlike Bertinetto et al., we retain
single-child internal nodes in order to preserve distances for the
Resistance score.

Edges are directed parent -> child (convention of HierarchicalScorer).

Outputs
-------
    data/tree/tree.gpickle
    data/tree/leaf_synsets.json

Usage
-----
    python data_prep/03_build_tree.py
"""

import json
import pickle
from pathlib import Path

import networkx as nx
from nltk.corpus import wordnet as wn

METADATA_DIR = Path(__file__).parent.parent / "data" / "metadata"
TREE_DIR = Path(__file__).parent.parent / "data" / "tree"
TREE_DIR.mkdir(parents=True, exist_ok=True)


def load_class_index():
    with open(METADATA_DIR / "imagenet_class_index.json") as f:
        data = json.load(f)
    wnids  = [data[str(i)][0] for i in range(1000)]
    labels = [data[str(i)][1] for i in range(1000)]
    return wnids, labels


def wnid_to_synset(wnid):
    """
    Convert an ImageNet-style wnid string like 'n01440764' into an nltk
    WordNet Synset object.
    """
    pos = wnid[0]  # 'n' for noun (all ImageNet-1k classes are nouns)
    offset = int(wnid[1:])
    assert pos == "n", f"Unexpected POS prefix in wnid {wnid}"
    return wn.synset_from_pos_and_offset("n", offset)


def select_path(synset, existing_nodes):
    """Select path from synset to root using Bertinetto's rule:

    Enumerates all hypernym paths from synset to the WordNet root,
    then selects the longest one. Ties are broken by fewest new nodes
    (nodes not yet in existing_nodes), so that the tree reuses
    already-selected paths as much as possible.

    Returns path as a list [synset, parent, grandparent, ..., root].
    """
    def collect_all_paths(s, path_so_far):
        # path_so_far is the path from the original synset up to s (excluding s)
        parents = s.hypernyms()
        if not parents:
            # s is the root, this path is complete
            return [path_so_far + [s]]
        # recurse into each parent and collect all resulting paths
        paths = []
        for parent in parents:
            paths.extend(collect_all_paths(parent, path_so_far + [s]))
        return paths

    all_paths = collect_all_paths(synset, [])

    # Tie-break: among longest paths, pick the one with fewest new nodes:
    max_len = max(len(p) for p in all_paths)
    longest = [p for p in all_paths if len(p) == max_len]
    return min(longest, key=lambda p: sum(1 for s in p if s.name() not in existing_nodes))


def build_tree(wnids):
    tree = nx.DiGraph()
    existing = set()
    for idx, wnid in enumerate(wnids):
        # Select path from leaf represented by (idx, wnid) to the root:
        synset = wnid_to_synset(wnid)
        path = select_path(synset, existing)   # [leaf, ..., root]
        # Add all nodes and edges of the path to the digraph:
        for i in range(len(path) - 1):
            child  = path[i].name()
            parent = path[i + 1].name()
            tree.add_edge(parent, child)
            existing.add(parent)
            existing.add(child)
    return tree


if __name__ == "__main__":
    print("Loading class index...")
    wnids, labels = load_class_index()

    print("Building tree...")
    tree = build_tree(wnids)

    n_leaves = sum(1 for n in tree.nodes() if tree.out_degree(n) == 0)
    n_roots  = sum(1 for n in tree.nodes() if tree.in_degree(n) == 0)
    print(f"  Nodes: {tree.number_of_nodes()}, Leaves: {n_leaves}, Roots: {n_roots}")
    assert n_leaves == 1000
    assert n_roots  == 1

    with open(TREE_DIR / "tree.gpickle", "wb") as f:
        pickle.dump(tree, f)

    synset_names = [wnid_to_synset(w).name() for w in wnids]
    with open(TREE_DIR / "leaf_synsets.json", "w") as f:
        json.dump({
            "wnids": wnids,
            "synset_name": synset_names,
            "label": labels
        }, f, indent=2)

    print(f"Saved to {TREE_DIR}")
