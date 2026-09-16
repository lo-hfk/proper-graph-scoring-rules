"""
Regression tests comparing GraphScorer against the original NodeSelectionTask.

Run from the repository root with:
    python verify.py

All assertions compare new (GraphScorer) against old (NodeSelectionTask) on
the same graph, probability vector, and true node. Tolerance is 1e-10.

Also checks two analytical results from the paper:
  - On K_N: resistance_score == brier_score
  - On P_N: resistance_score == RPS
"""

import numpy as np
import networkx as nx

from propnetscore.scorer import GraphScorer
import propnetscore.node_selection as ns

TOL = 1e-10


def make_adjacency(G):
    return nx.to_numpy_array(G)


def check_close(name, val_new, val_old, tol=TOL):
    err = abs(val_new - val_old)
    status = "OK" if err < tol else "FAIL"
    print(f"  [{status}] {name}: new={val_new:.10f}, old={val_old:.10f}, err={err:.2e}")
    assert err < tol, f"FAILED: {name}"


# -----------------------------------------------------------------
# Graph 1: Karate Club — regression against NodeSelectionTask
# -----------------------------------------------------------------
print("\n=== Karate Club: regression vs NodeSelectionTask ===")
G = nx.karate_club_graph()
A = make_adjacency(G)
N = A.shape[0]
rng = np.random.default_rng(42)
p = rng.dirichlet(np.ones(N))
true_node = 7

scorer = GraphScorer(A)
task = ns.NodeSelectionTask(A, p, true_node)
P = p[np.newaxis]
i = np.array([true_node])

check_close("brier_score",
            float(scorer.brier_score(P, i)[0]),
            task.brier_score())
check_close("logarithmic_score",
            float(scorer.logarithmic_score(P, i)[0]),
            task.logarithmic_score())
check_close("resistance_score",
            float(scorer.resistance_score(P, i)[0]),
            task.resistance_score() / N)   # old code had * N normalisation factor
check_close("laplace_score",
            float(scorer.laplace_score(P, i)[0]),
            task.laplace_score(convention="positive") * N / 2)  # old code had * 2/N
check_close("rank_score",
            float(scorer.rank_score(P, i)[0]),
            task.rank_score())
check_close("error_distance",
            float(scorer.error_distance(P, i)[0]),
            task.error_distance())
check_close("diffusion_score(tau=0.5)",
            float(scorer.diffusion_score(P, i, tau=0.5)[0]),
            task.diffusion_score(tau=0.5, convention="positive") * 4)  # old: * 0.5, new: * 2
check_close("diffusion_score(tau=2.0)",
            float(scorer.diffusion_score(P, i, tau=2.0)[0]),
            task.diffusion_score(tau=2.0, convention="positive") * 4)


# -----------------------------------------------------------------
# Graph 2: P_5 — resistance score should equal RPS
# -----------------------------------------------------------------
print("\n=== Path graph P_5: resistance == RPS ===")
N = 5
A = make_adjacency(nx.path_graph(N))
scorer = GraphScorer(A)

def rps(p, i):
    cdf = np.cumsum(p)
    return float(sum((cdf[j] - float(i <= j))**2 for j in range(len(p))))

rng = np.random.default_rng(0)
for trial in range(5):
    p = rng.dirichlet(np.ones(N))
    true_node = int(rng.integers(N))
    res = float(scorer.resistance_score(p[np.newaxis], np.array([true_node]))[0])
    check_close(f"P_5 trial {trial} (i={true_node})", res, rps(p, true_node))


# -----------------------------------------------------------------
# Graph 3: K_5 — resistance score should equal Brier score
# -----------------------------------------------------------------
print("\n=== Complete graph K_5: resistance == Brier ===")
N = 5
A = make_adjacency(nx.complete_graph(N))
scorer = GraphScorer(A)

rng = np.random.default_rng(1)
for trial in range(5):
    p = rng.dirichlet(np.ones(N))
    true_node = int(rng.integers(N))
    P = p[np.newaxis]
    iv = np.array([true_node])
    check_close(f"K_5 trial {trial} (i={true_node})",
                float(scorer.resistance_score(P, iv)[0]),
                float(scorer.brier_score(P, iv)[0]) / N)


# -----------------------------------------------------------------
# Graph 4: Batched vs single-instance loop
# -----------------------------------------------------------------
print("\n=== Batched scoring consistency ===")
G = nx.karate_club_graph()
A = make_adjacency(G)
N = A.shape[0]
scorer = GraphScorer(A)
rng = np.random.default_rng(99)
m = 20
P_batch = rng.dirichlet(np.ones(N), size=m)
nodes_batch = rng.integers(N, size=m)

for name, method, kwargs in [
    ("resistance", scorer.resistance_score, {}),
    ("brier",      scorer.brier_score,      {}),
    ("diffusion",  scorer.diffusion_score,  {"tau": 1.0}),
]:
    batch = method(P_batch, nodes_batch, **kwargs)
    loop  = np.array([
        float(method(P_batch[r:r+1], nodes_batch[r:r+1], **kwargs)[0])
        for r in range(m)
    ])
    check_close(f"{name} batch vs loop (max err)",
                float(np.max(np.abs(batch - loop))), 0.0)

print("\n=== All tests passed ===\n")