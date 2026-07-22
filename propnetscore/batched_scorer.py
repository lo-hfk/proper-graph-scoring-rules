"""
Batched, vectorized version of NodeSelectionTask.

Design principle
-----------------
Everything that depends ONLY on the graph (Laplacian, its pseudoinverse,
resistance distances, diffusion/random-walk kernels, shortest paths) is
computed once and cached on the GraphScorer instance -- exactly as before,
just renamed/reorganised so it's clear these are graph-level quantities.

Everything that depends on the DATA (a batch of probability vectors P and
true nodes v) is a method that takes P (shape (m, n)) and true_nodes (shape
(m,)) and returns an array of m scores. No per-instance objects, no Python
loops over examples for the kernel-based scores.

Kernel scores share a single vectorised identity (derived below) that
batches cleanly; the few non-kernel scores (top-k, rank, error_distance)
either vectorise trivially via argsort/argpartition or, for error_distance's
multi-armax tie handling, keep a short per-row loop since ties are rare and
the loop is O(m) not O(m*n).
"""

import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import expm


class GraphScorer:
    def __init__(self, adjacency_matrix: np.ndarray):
        """
        Parameters
        ----------
        adjacency_matrix : np.ndarray (n, n)
            Adjacency matrix of the graph. Shared across all scored instances.
        """
        self.adjacency_matrix = adjacency_matrix
        self.size = adjacency_matrix.shape[0]

        # Graph-only quantities, computed lazily and cached -- identical in
        # spirit to the original class, just no longer tangled with
        # per-instance (p, v) data.
        self._degree_matrix = None
        self._laplacian = None
        self._normalised_laplacian = None
        self._laplacian_pinv = None
        self._resistance = None
        self._shortest_paths = None

    # -----------------------------------------------------------------
    # Graph-only quantities (cached, computed once regardless of how many
    # (P, true_nodes) batches you score against this graph)
    # -----------------------------------------------------------------
    @property
    def degree_matrix(self):
        if self._degree_matrix is None:
            self._degree_matrix = np.diag(np.sum(self.adjacency_matrix, axis=1))
        return self._degree_matrix

    @property
    def laplacian(self):
        if self._laplacian is None:
            self._laplacian = self.degree_matrix - self.adjacency_matrix
        return self._laplacian

    @property
    def normalised_laplacian(self):
        if self._normalised_laplacian is None:
            sqrt_deg = np.sqrt(self.degree_matrix)
            self._normalised_laplacian = sqrt_deg @ self.laplacian @ sqrt_deg
        return self._normalised_laplacian

    @property
    def laplacian_pinv(self):
        if self._laplacian_pinv is None:
            self._laplacian_pinv = np.linalg.pinv(self.laplacian)
        return self._laplacian_pinv

    @property
    def resistance(self):
        if self._resistance is None:
            diag = np.diag(self.laplacian_pinv)
            self._resistance = diag[:, None] + diag[None, :] - 2 * self.laplacian_pinv
        return self._resistance

    @property
    def shortest_paths(self):
        """All-pairs shortest paths via vectorised Floyd-Warshall."""
        if self._shortest_paths is None:
            dist = self.adjacency_matrix.astype(float).copy()
            dist[dist == 0] = np.inf
            np.fill_diagonal(dist, 0)
            for k in range(self.size):
                dist = np.minimum(dist, dist[:, k, np.newaxis] + dist[np.newaxis, k, :])
            self._shortest_paths = dist
        return self._shortest_paths

    def diffusion_kernel(self, tau: float, mu: float = 0.0, normalise: bool = False):
        """Not cached by default since it depends on (tau, mu, normalise);
        cheap to recompute relative to the matrix exponential cost itself
        only if called with many different tau values -- add an explicit
        cache keyed by (tau, mu, normalise) if that becomes a bottleneck."""
        K = expm(-0.5 * tau * (self.laplacian - mu * np.eye(self.size)))
        if normalise:
            K = K / np.trace(K)
        return K

    def random_walk_kernel(self, alpha: float, p: int):
        return matrix_power(
            np.eye(self.size) - (1 - alpha) * self.normalised_laplacian, p
        )

    # -----------------------------------------------------------------
    # Batched kernel score
    # -----------------------------------------------------------------
    def _kernel_score_batch(self, P: np.ndarray, true_nodes: np.ndarray,
                             kernel: np.ndarray, convention: str | None = None):
        """
        Vectorised batch version of:
            S(p, i) = -0.5 (p - e_i)^T G (p - e_i) + extra_term

        Expanding the quadratic form (exactly as the original code already
        implicitly does, just written out termwise) gives three pieces that
        each batch cleanly over m rows of P:

            -0.5 (p - e_i)^T G (p - e_i)
          = -0.5 p^T G p   +   p^T G e_i   -   0.5 e_i^T G e_i
          =  -regularisation  +  expected_residual  +  (extra_term, possibly 0)

        Each piece becomes one vectorised expression:
          - p^T G p  for every row p of P  ->  einsum('mi,ij,mj->m', P, G, P)
          - p^T G e_i = (G @ p)[i]  for every row -> take (P @ G.T) then index
            column true_nodes[row] per row, i.e. (P @ G)[arange(m), true_nodes]
            (G is symmetric for every kernel used here, so G.T == G; written
            with G explicitly for clarity rather than relying on that)
          - e_i^T G e_i = G[i, i]  ->  fancy-indexed diagonal lookup

        Parameters
        ----------
        P : np.ndarray (m, n)
            Batch of m probability vectors over n nodes.
        true_nodes : np.ndarray (m,)
            Batch of m true node indices, one per row of P.
        kernel : np.ndarray (n, n)
            Conditionally negative definite kernel matrix (e.g. -laplacian).
        convention : str or None
            None -> [Gneiting/Raftery 2007] convention.
            "positive" -> [Ziegel/Ginsbourger/Duembgen 2022] non-negative scores.

        Returns
        -------
        np.ndarray (m,)
            One score per row.

        Performance note
        -----------------
        Both terms reuse the single (m, n) product `P @ kernel`, computed
        once: O(m n^2). Computing p^T @ kernel @ p naively via a 3-tensor
        einsum (`einsum('mi,ij,mj->m', ...)`) without `optimize=True` can be
        an order of magnitude slower, because NumPy's default contraction
        path does not always pick the cheap (P @ kernel) @ P ordering on its
        own -- worth knowing if you extend this function, since the naive
        einsum form looks equally "vectorised" but performs very differently.
        """
        m = P.shape[0]
        rows = np.arange(m)

        PK = P @ kernel  # (m, n), shared by both terms below
        expected_residual = PK[rows, true_nodes]
        regularisation = 0.5 * np.sum(PK * P, axis=1)

        if convention == "positive":
            extra_term = -0.5 * kernel[true_nodes, true_nodes]
        else:
            extra_term = 0.0

        return expected_residual - regularisation + extra_term

    # -----------------------------------------------------------------
    # Public batched scoring methods -- same names/semantics as before,
    # now operating on (P, true_nodes) batches instead of single instances.
    # -----------------------------------------------------------------
    def error_distance(self, P: np.ndarray, true_nodes: np.ndarray):
        """Warning: not a proper scoring rule! Kept for comparison purposes.

        Handles ties in the argmax (multiple nodes with equal max
        probability) the same way the original did: average distance to
        ALL tied argmax nodes. Ties are rare so this stays a short Python
        loop over the m rows rather than a fully vectorised ragged op.
        """
        m = P.shape[0]
        out = np.empty(m, dtype=float)
        sp = self.shortest_paths
        for row in range(m):
            argmax_all = np.where(P[row] == P[row].max())[0]
            out[row] = sp[true_nodes[row], argmax_all].mean()
        return out

    def top_k_accuracy(self, P: np.ndarray, true_nodes: np.ndarray, k: int = 1):
        # argpartition is O(n) per row vs O(n log n) for a full argsort,
        # and we only need the top-k SET, not their order, for accuracy.
        top_k_idx = np.argpartition(-P, kth=k - 1, axis=1)[:, :k]
        return (top_k_idx == true_nodes[:, None]).any(axis=1).astype(float)

    def top_k_error_rate(self, P: np.ndarray, true_nodes: np.ndarray, k: int = 1):
        return 1.0 - self.top_k_accuracy(P, true_nodes, k=k)

    def rank_score(self, P: np.ndarray, true_nodes: np.ndarray):
        """1-indexed rank of the true node under P, per row.

        NOTE on tie-breaking -- this is a deliberate behaviour CHANGE from
        the original implementation, not an oversight:

        The original used `np.argsort(-p)` with NumPy's default
        kind='quicksort', which is NOT a stable sort. When probabilities are
        exactly tied, the relative order among tied nodes is an artifact of
        quicksort's internal partitioning -- it is deterministic for a given
        NumPy build, but has no clean mathematical characterisation (it is
        NOT simply "lower index wins", despite that appearing to hold in
        some small examples by coincidence). Replicating it exactly would
        mean depending on undocumented sort-algorithm internals.

        Instead, this version uses MIDRANK tie-breaking (the standard
        convention in rank statistics, e.g. scipy.stats.rankdata's default):
        all nodes tied with the true node's probability receive the AVERAGE
        of the ranks they would jointly occupy. E.g. if nodes are tied for
        what would be positions 2, 3, 4, all three receive rank 3.

        This only affects rows where P has an EXACT tie at the true node's
        probability value -- vanishingly rare for continuous model outputs,
        but can matter for synthetic/discrete examples or near-degenerate
        probability vectors. If you need the original's exact quicksort-
        artifact tie order for backward compatibility, that is achievable
        but not recommended; ask and I can add it as an alternative method.
        """
        m = P.shape[0]
        rows = np.arange(m)
        true_probs = P[rows, true_nodes]

        strictly_greater = np.sum(P > true_probs[:, None], axis=1)
        tied = np.sum(P == true_probs[:, None], axis=1)  # includes true_node itself

        # Midrank: average of the 'tied' consecutive ranks starting right
        # after all strictly-greater entries.
        # e.g. strictly_greater=1, tied=3  ->  ranks {2,3,4} -> midrank 3
        return strictly_greater + (tied + 1) / 2.0

    def inverse_rank_score(self, P: np.ndarray, true_nodes: np.ndarray):
        return 1.0 / self.rank_score(P, true_nodes)

    def logarithmic_score(self, P: np.ndarray, true_nodes: np.ndarray):
        return -np.log(P[np.arange(P.shape[0]), true_nodes])

    def brier_score(self, P: np.ndarray, true_nodes: np.ndarray):
        m, n = P.shape
        onehot = np.zeros((m, n))
        onehot[np.arange(m), true_nodes] = 1.0
        residuals = onehot - P
        return np.einsum('mi,mi->m', residuals, residuals)

    def weighted_brier_score(self, P: np.ndarray, true_nodes: np.ndarray, weights: np.ndarray):
        m, n = P.shape
        onehot = np.zeros((m, n))
        onehot[np.arange(m), true_nodes] = 1.0
        residuals = onehot - P
        return np.einsum('mi,i,mi->m', residuals, weights, residuals)

    def laplace_score(self, P: np.ndarray, true_nodes: np.ndarray, convention: str | None = None):
        normalisation_factor = 2 / self.size
        return self._kernel_score_batch(P, true_nodes, -self.laplacian, convention) * normalisation_factor

    def resistance_score(self, P: np.ndarray, true_nodes: np.ndarray):
        normalisation_factor = self.size
        return self._kernel_score_batch(P, true_nodes, self.resistance) * normalisation_factor

    def diffusion_score(self, P: np.ndarray, true_nodes: np.ndarray, tau: float,
                         convention: str | None = None, mu: float = 0.0, normalise: bool = False):
        K = self.diffusion_kernel(tau=tau, mu=mu, normalise=normalise)
        normalisation_factor = 0.5
        return self._kernel_score_batch(P, true_nodes, -K, convention) * normalisation_factor

    def random_walk_score(self, P: np.ndarray, true_nodes: np.ndarray, alpha: float, p: int,
                           convention: str | None = None):
        K = self.random_walk_kernel(alpha=alpha, p=p)
        return self._kernel_score_batch(P, true_nodes, -K, convention)