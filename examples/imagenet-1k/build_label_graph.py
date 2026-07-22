"""
Step 1: Build the WordNet hypernym DAG restricted to (the ancestor closure of)
the 1000 ImageNet-1k leaf classes.

This produces:
  - graph.gpickle      : networkx.DiGraph, edges point child -> parent (is-a)
  - leaf_synsets.json  : list of the 1000 leaf synset IDs, in ImageNet class-index order
  - graph_stats.json   : diagnostics (depth, diamond count, etc.)

Run this AFTER downloading the ImageNet devkit (for the class index -> synset
mapping) but you can also run it standalone using torchvision's bundled
class index list, which we use below (no devkit parsing needed for this step).
"""

import json
import pickle
from pathlib import Path

import networkx as nx
import nltk

# ---------------------------------------------------------------------------
# 0. Make sure WordNet is available locally (one-time download, ~10MB)
# ---------------------------------------------------------------------------
nltk.download("wordnet")
from nltk.corpus import wordnet as wn  # noqa: E402

OUT_DIR = Path("./wordnet_graph")
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Get the 1000 ImageNet-1k class -> WordNet synset mapping.
#
# torchvision ships the official ILSVRC2012 class index -> (wnid, label) JSON.
# We fetch it once and cache it locally.
# ---------------------------------------------------------------------------
def get_imagenet_class_index():
    """
    Returns a dict: {class_idx (str "0".."999") -> [wnid, human_readable_label]}
    e.g. {"0": ["n01440764", "tench"], ...}

    torchvision/PyTorch hub provide this file at a stable URL. We download
    once and cache it to disk so re-runs don't need network access.
    """
    cache_path = OUT_DIR / "imagenet_class_index.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    import urllib.request

    url = (
        "https://raw.githubusercontent.com/pytorch/hub/master/"
        "imagenet_classes.txt"
    )
    # NOTE: imagenet_classes.txt is just human labels, not wnids.
    # The canonical idx->wnid->label JSON used throughout the literature is this one
    # (originally published by Keras / commonly mirrored):
    url_json = (
        "https://s3.amazonaws.com/deep-learning-models/image-models/"
        "imagenet_class_index.json"
    )
    try:
        with urllib.request.urlopen(url_json) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        raise RuntimeError(
            f"Could not download imagenet_class_index.json automatically ({e}).\n"
            "Download it manually from:\n"
            "  https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json\n"
            f"and place it at {cache_path}"
        )

    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def wnid_to_synset(wnid):
    """
    Convert an ImageNet-style wnid string like 'n01440764' into an nltk
    WordNet Synset object.
    """
    pos = wnid[0]  # 'n' for noun (all ImageNet-1k classes are nouns)
    offset = int(wnid[1:])
    assert pos == "n", f"Unexpected POS prefix in wnid {wnid}"
    return wn.synset_from_pos_and_offset("n", offset)


# ---------------------------------------------------------------------------
# 2. Build the full ancestor-closure DAG.
#
# For each of the 1000 leaf synsets, walk ALL hypernym paths to the root
# (entity.n.01), adding every node and every edge encountered. Because we do
# NOT select a single path (unlike Bertinetto et al.'s pruning), nodes with
# multiple hypernyms naturally produce diamonds.
# ---------------------------------------------------------------------------
def build_full_ancestor_dag(leaf_synsets):
    """
    leaf_synsets: list of nltk Synset objects (length 1000, ImageNet leaves)

    Returns: networkx.DiGraph with edges child -> parent (is-a direction),
             every node labeled with its synset name for readability.
    """
    G = nx.DiGraph()

    def add_synset_node(syn):
        if not G.has_node(syn.name()):
            G.add_node(syn.name(), gloss=syn.definition())

    def walk_up(syn, visited):
        """Recursively add edges from syn to all its hypernyms, all the way up."""
        add_synset_node(syn)
        for parent in syn.hypernyms():
            add_synset_node(parent)
            if not G.has_edge(syn.name(), parent.name()):
                G.add_edge(syn.name(), parent.name())
            if parent.name() not in visited:
                visited.add(parent.name())
                walk_up(parent, visited)

    visited_global = set()
    for leaf in leaf_synsets:
        add_synset_node(leaf)
        walk_up(leaf, visited_global)

    return G


# ---------------------------------------------------------------------------
# 3. Diagnostics: confirm DAG-ness, count "diamonds" (multi-parent nodes),
#    measure depth, etc. This is exactly the sanity check we discussed.
# ---------------------------------------------------------------------------
def compute_diagnostics(G, leaf_names):
    assert nx.is_directed_acyclic_graph(G), "Graph is not a DAG! Something is wrong."

    # out-degree here = number of parents, since edges point child -> parent
    multi_parent_nodes = [n for n in G.nodes if G.out_degree(n) > 1]

    # depth: longest path length from any leaf to the root, in this DAG
    root_candidates = [n for n in G.nodes if G.out_degree(n) == 0]

    # longest path overall (networkx handles DAG longest path natively)
    longest_path_len = nx.dag_longest_path_length(G)

    diagnostics = {
        "n_nodes_total": G.number_of_nodes(),
        "n_edges_total": G.number_of_edges(),
        "n_leaf_classes": len(leaf_names),
        "n_multi_parent_nodes": len(multi_parent_nodes),
        "pct_multi_parent_nodes": round(
            100 * len(multi_parent_nodes) / G.number_of_nodes(), 2
        ),
        "multi_parent_node_examples": multi_parent_nodes[:15],
        "root_candidates": root_candidates,
        "longest_leaf_to_root_path_length": longest_path_len,
        "is_dag_confirmed": True,
    }
    return diagnostics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading ImageNet class index -> WordNet synset mapping...")
    class_index = get_imagenet_class_index()  # {"0": [wnid, label], ...}

    # Order matters: this MUST match the class index order your model outputs
    # (torchvision ResNet/ViT models output logits in this exact 0..999 order).
    n_classes = len(class_index)
    assert n_classes == 1000, f"Expected 1000 classes, got {n_classes}"

    leaf_wnids = [class_index[str(i)][0] for i in range(1000)]
    leaf_labels = [class_index[str(i)][1] for i in range(1000)]
    leaf_synsets = [wnid_to_synset(w) for w in leaf_wnids]
    leaf_names = [s.name() for s in leaf_synsets]

    print("Building full ancestor-closure DAG (no pruning to a tree)...")
    G = build_full_ancestor_dag(leaf_synsets)

    print("Computing diagnostics...")
    diag = compute_diagnostics(G, leaf_names)
    for k, v in diag.items():
        print(f"  {k}: {v}")

    # Save everything
    with open(OUT_DIR / "graph.gpickle", "wb") as f:
        pickle.dump(G, f)

    with open(OUT_DIR / "leaf_synsets.json", "w") as f:
        json.dump(
            {
                "class_idx_to_wnid": leaf_wnids,       # index = ImageNet class idx
                "class_idx_to_label": leaf_labels,
                "class_idx_to_synset_name": leaf_names,
            },
            f,
            indent=2,
        )

    with open(OUT_DIR / "graph_stats.json", "w") as f:
        json.dump(diag, f, indent=2)

    print(f"\nSaved graph with {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges to {OUT_DIR / 'graph.gpickle'}")