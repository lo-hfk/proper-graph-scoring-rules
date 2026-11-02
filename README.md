
# propnetscore

Strictly proper, graph-aware scoring rules for probabilistic node selection (the Resistance score and
the Diffusion score) plus the hierarchy-aware scores (Resistance score on trees, HXE, Urbani score) used
for hierarchical classification. Reference implementation for the paper *"Strictly proper scoring rules
for node selection tasks on graphs."*

## Installation

Requires Python >= 3.8. Clone the repository and install with:

```bash
git clone https://github.com/lo-hfk/proper-graph-scoring-rules.git
cd proper-graph-scoring-rules
pip install .
```

This installs the package under the import name `propnetscore` with its core dependencies (`numpy`,
`scipy`). `networkx` is not required by the package itself, but the examples below and in `examples/` use
it to build graphs, so you will likely want it too (`pip install networkx`).


### Optional: fast scorer for large sparse graphs

`FastGraphScorer` (in `scorer_fast.py`) computes the Resistance and Diffusion scores without ever forming
the dense Laplacian pseudoinverse or matrix exponential, which is what makes the scores usable on graphs
with more than a few hundred nodes. It needs two things beyond the base install:

1. A working [Julia](https://julialang.org/) installation, on your `PATH`.
2. The Julia package [`Laplacians.jl`](https://github.com/danspielman/Laplacians.jl), installed once from
   within Julia:

   ```bash
   julia -e 'using Pkg; Pkg.add("Laplacians")'
   ```

3. The `juliacall` Python package, which bridges the two: `pip install juliacall`.

If you only work with small graphs (roughly up to a thousand nodes, e.g. the toy and Karate Club examples
below), the plain `GraphScorer` is fast enough and none of this is needed.

## Usage

All scorers take a probability matrix `P` of shape `(m, n)` — `m` predicted distributions over the `n`
nodes/leaves of the graph — and an integer array `true_nodes` of shape `(m,)` giving the true node index
for each row. This batched signature is shared by every scoring method, so you can score many instances
(e.g. one row per image, per simulated outbreak, or per candidate node) in a single call.

### General graphs: `GraphScorer`

```python
import networkx as nx
import numpy as np
from propnetscore import GraphScorer

G = nx.karate_club_graph()
A = nx.to_numpy_array(G)          # GraphScorer takes a plain adjacency matrix
scorer = GraphScorer(A)

# two predicted distributions over the 34 nodes, both scored against true node 0
# (each row must sum to 1)
P = np.array([
    [0.8, 0.1, 0.1] + [0.0] * 31,          # sharp: most mass on/near true node 0
    [0.25, 0.25, 0.25, 0.25] + [0.0] * 30, # diffuse: spread over 4 nodes incl. true node 0
])
true_nodes = np.array([0, 0])

scorer.brier_score(P, true_nodes)                    # classical, graph-blind
scorer.resistance_score(P, true_nodes)                # parameter-free, graph-aware
scorer.diffusion_score(P, true_nodes, tau=1.0)         # graph-aware, scale controlled by tau
```

For large, sparse graphs, swap in `FastGraphScorer` (same interface, see Installation above):

```python
from propnetscore import FastGraphScorer
scorer = FastGraphScorer(A)
scorer.resistance_score(P, true_nodes)   # via a Laplacian solver (Julia)
scorer.diffusion_score(P, true_nodes, tau=1.0)  # via Krylov-subspace expm_multiply (scipy)
```

### Hierarchical classification: `HierarchicalScorer`

For node selection on a rooted tree (e.g. a label taxonomy), build the tree with `tree_from_edges` or
`tree_from_parent_dict` and use `HierarchicalScorer`, which scores distributions over the **leaves** only:

```python
import numpy as np
from propnetscore import HierarchicalScorer
from propnetscore.scorer_hierarchical import tree_from_parent_dict

parents = {"v": "root", "a": "v", "b": "v", "c": "root"}  # {child: parent}
tree = tree_from_parent_dict(parents)
leaves = ["a", "b", "c"]
scorer = HierarchicalScorer(tree, leaves)

P = np.array([[0.5, 0.3, 0.2]])
true_nodes = np.array([0])  # leaf "a"

scorer.resistance_score(P, true_nodes)
scorer.hxe_score(P, true_nodes, alpha=0.5)
```

See the docstrings in `propnetscore/scorer.py`, `scorer_fast.py`, and `scorer_hierarchical.py` for the full
list of scores (including `laplace_score`, `weighted_brier_score`, `logarithmic_score`, and the
graph-blind/improper baselines `rank_score`, `top_k_error`, `error_distance`, `distance_at_k` kept for
comparison) and their parameters.

## Examples

Examples from the paper:

### Toy example
`examples/toy/toy_example.py`: small synthetic toy graph for illustrative purposes.

### Karate Club source detection
`examples/karate/`: source detection for SI epidemic spreading on the Zachary Karate Club network.

### ImageNet hierarchical classification
`examples/imagenet/`: ImageNet-1k with the WordNet class hierarchy (pruned to a tree) as the underlying graph.

## Citation

If you use this code, please cite:

```bibtex
@misc{hilfiker2026propnetscore,
  author = {Hilfiker, Lorenz and Cincera Brack, Nathan and Sterchi, Martin},
  title  = {Strictly proper scoring rules for node selection tasks on graphs},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

See `LICENSE`.