import numpy as np
import networkx as nx
from scipy.linalg import expm


def coarse_grain_graph(G: nx.Graph, tau: float):
    """
    Source code for Laplacian coarse-graining paper [Villegas et al., 2023]
    taken from https://github.com/pvgongora/Laplacian-RG

    Returns
    -------
        G_cg:        coarse-grained graph (nodes relabelled 0,...,K-1)
        node_to_supernode: dict mapping each original node index to its new supernode index
    """
    # Select the Kadanoff supernodes
    L = nx.laplacian_matrix(G)
    L1 = L.todense()
    num = expm((-tau * L1))
    den = np.trace(num)
    rho = num / den
    n = len(G.nodes())
    adj2 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (rho[i][j] >= rho[j][j]) or (rho[i][j] >= rho[i][i]):
                adj2[i][j] = 1

    # Build old-node -> supernode-representative mapping.
    # Each connected component of adj2 becomes one supernode; we use the
    # component's sorted-first element as the representative (matching the
    # contraction loop below).
    G1 = nx.from_numpy_array(adj2)
    old_to_rep = {}  # old node → representative of its supernode
    for supernode in nx.connected_components(G1):
        nodes = sorted(list(supernode))
        rep = nodes[0]
        for node in nodes:
            old_to_rep[node] = rep

    # Contract nodes in G
    for supernode in nx.connected_components(G1):
        nodes = sorted(list(supernode))
        for node in nodes[1:]:
            G = nx.contracted_nodes(G, nodes[0], node, self_loops=False)

    # Relabel contracted graph to 0,...,K-1
    mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # Compose: original node -> representative -> new integer label
    node_to_supernode = {
        old: mapping[rep] for old, rep in old_to_rep.items()
    }

    return G, node_to_supernode


def coarse_grain_prediction(prediction: np.ndarray, node_to_supernode: dict) -> np.ndarray:
    """
    Aggregate a per-node probability vector to the coarse-grained graph.
    The probability of each supernode is the sum of its constituent nodes' probabilities.

    Parameters
    ----------
        prediction:        1-D array of length N (original graph), sums to 1.
        node_to_supernode: mapping returned by coarse_grain().

    Returns
    -------
        cg_prediction: 1-D array of length K (coarse-grained graph), sums to 1.
    """
    n_supernodes = max(node_to_supernode.values()) + 1
    cg_prediction = np.zeros(n_supernodes)
    for old_node, prob in enumerate(prediction):
        cg_prediction[node_to_supernode[old_node]] += prob
    return cg_prediction


def coarse_grain_true_node(true_node: int, node_to_supernode: dict) -> int:
    """
    Map a single node index to its supernode index in the coarse-grained graph.

    Parameters
    ----------
        true_node:         original node index.
        node_to_supernode: mapping returned by coarse_grain().

    Returns
    -------
        Supernode index in the coarse-grained graph.
    """
    return node_to_supernode[true_node]


def get_supernode_sizes(node_to_supernode: dict) -> np.ndarray:
    """
    Returns an array of length K where entry i is the number of original
    nodes that were merged into supernode i.
    """
    n_supernodes = max(node_to_supernode.values()) + 1
    return np.bincount(list(node_to_supernode.values()), minlength=n_supernodes)
