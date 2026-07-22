"""
Step 3: Utility functions for evaluating a graph-aware scoring rule over the
label DAG. These are the building blocks your scoring rule S(p, v) will need:

  - ancestors(v): the upward closure of v (all nodes "more general" than v)
  - lowest_common_ancestor(u, v): for measuring semantic distance
  - shortest_path_length(u, v): graph distance, accounting for the DAG (not
    just a tree) -- this matters BECAUSE of diamonds: two nodes might have a
    shorter connecting path through one branch of a diamond than another.

This module is intentionally decoupled from torch/PIL so it can be unit
tested standalone (as we do below) without any of the heavy ML dependencies.
"""

import pickle
from pathlib import Path

import networkx as nx

DATA_DIR = Path("./wordnet_graph")


def load_graph(path=None):
    path = path or (DATA_DIR / "graph.gpickle")
    with open(path, "rb") as f:
        return pickle.load(f)


def ancestors_inclusive(G, v):
    """
    {v} union all ancestors of v (nodes reachable by following child->parent
    edges out of v). This is the "upset" U(v) discussed for the true-path-rule
    case in GO; for ImageNet it's just used as the support for partial credit.
    """
    return {v} | nx.descendants(G, v)
    # NOTE: in a DiGraph with edges child->parent, "descendants" in networkx's
    # sense (nodes reachable FROM v) are v's ANCESTORS in the semantic sense,
    # because our edges point child->parent. This naming mismatch is a classic
    # gotcha -- verified explicitly in the test below.


def lowest_common_ancestor(G, u, v):
    """
    For a DAG, there can be multiple lowest common ancestors. networkx's
    all_pairs_lowest_common_ancestor / lowest_common_ancestor function is
    defined for trees/forests by default in some versions; for a general DAG
    we implement this explicitly: the LCA(s) are the nodes in the intersection
    of ancestors(u) and ancestors(v) that are themselves not an ancestor of
    any other node in that intersection (i.e., the "lowest" ones).
    """
    anc_u = ancestors_inclusive(G, u)
    anc_v = ancestors_inclusive(G, v)
    common = anc_u & anc_v
    if not common:
        return set()

    # A common ancestor c is "lowest" if no OTHER common ancestor is a
    # descendant of c in the semantic sense (i.e. closer to u/v than c).
    lowest = set()
    for c in common:
        # descendants of c within the 'common' set, semantic sense:
        # nodes reachable BY GOING DOWN from c, i.e. nodes that have c as
        # their ancestor. We check: is there another common node c2 such that
        # c is an ancestor of c2 (c2 != c)? If yes, c is not lowest.
        is_lowest = True
        for c2 in common:
            if c2 == c:
                continue
            if c in ancestors_inclusive(G, c2):
                is_lowest = False
                break
        if is_lowest:
            lowest.add(c)
    return lowest


def graph_distance(G, u, v):
    """
    Shortest path length between u and v in the UNDIRECTED version of the
    DAG. We use the undirected version because semantic distance should not
    care about direction (going up to a common ancestor and back down is a
    valid "path" for measuring closeness, exactly as in WordNet-distance
    papers like Bertinetto et al.).

    Using the DAG (not a pruned tree) matters here: a diamond gives genuinely
    SHORTER paths between some node pairs than a tree-pruned version would,
    because there are two routes to the common ancestor instead of one fixed
    route. This is precisely the richness we want to capture.
    """
    G_undirected = G.to_undirected()
    return nx.shortest_path_length(G_undirected, u, v)


# ---------------------------------------------------------------------------
# Verification against the synthetic diamond example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Rebuild the same synthetic example used in the DAG-building test:
    #
    #            entity
    #               |
    #             mammal
    #            /        \
    #  aquatic_mammal    cetacean
    #            \        /
    #            dolphin
    #
    #   entity -> mammal -> canine -> dog   (separate branch)
    #
    # Edges: child -> parent
    G = nx.DiGraph()
    edges = [
        ("dolphin", "aquatic_mammal"),
        ("dolphin", "cetacean"),
        ("aquatic_mammal", "mammal"),
        ("cetacean", "mammal"),
        ("mammal", "entity"),
        ("dog", "canine"),
        ("canine", "mammal"),
    ]
    G.add_edges_from(edges)

    print("=== Test 1: ancestors_inclusive ===")
    anc_dolphin = ancestors_inclusive(G, "dolphin")
    print("ancestors_inclusive(dolphin):", sorted(anc_dolphin))
    assert anc_dolphin == {
        "dolphin", "aquatic_mammal", "cetacean", "mammal", "entity"
    }
    print("PASSED")

    print("\n=== Test 2: lowest_common_ancestor with a diamond in the way ===")
    lca = lowest_common_ancestor(G, "dolphin", "dog")
    print("LCA(dolphin, dog):", lca)
    # dolphin's ancestors: {dolphin, aquatic_mammal, cetacean, mammal, entity}
    # dog's ancestors:     {dog, canine, mammal, entity}
    # common: {mammal, entity}; mammal is "lower" (entity is mammal's ancestor)
    assert lca == {"mammal"}
    print("PASSED")

    print("\n=== Test 3: graph_distance through the diamond's shortest branch ===")
    # dolphin -> aquatic_mammal -> mammal : 2 hops either branch (symmetric here)
    # dog -> canine -> mammal : 2 hops
    # so dolphin to dog: 2 + 2 = 4 hops via mammal
    d = graph_distance(G, "dolphin", "dog")
    print("graph_distance(dolphin, dog):", d)
    assert d == 4
    print("PASSED")

    print("\n=== Test 4: distance WITHIN the diamond (this is where DAG != tree matters) ===")
    # aquatic_mammal and cetacean are SIBLINGS connected only through their
    # shared child (dolphin) or shared parent (mammal). In a tree-pruned
    # version, only ONE of {aquatic_mammal, cetacean} would exist as dolphin's
    # parent, so this comparison wouldn't even be possible to make.
    d2 = graph_distance(G, "aquatic_mammal", "cetacean")
    print("graph_distance(aquatic_mammal, cetacean):", d2)
    # path: aquatic_mammal -> mammal -> cetacean (2 hops) OR
    #       aquatic_mammal -> dolphin -> cetacean (2 hops)
    # both length 2 -- the diamond gives two EQUALLY short alternate routes
    assert d2 == 2
    print("PASSED")

    print("\nALL GRAPH UTILITY TESTS PASSED.")