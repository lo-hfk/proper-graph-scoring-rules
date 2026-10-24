"""
Vectorised batched implementation of the node selection scoring rules.
"""

import numpy as np
from scipy.linalg import expm, pinvh


class GraphScorer:
    """
    Scoring rules for probabilistic node selection on a fixed graph.

    All graph-level quantities (Laplacian, pseudoinverse, resistance distances,
    shortest paths) are computed lazily on first access and cached on the
    instance, so they are computed at most once regardless of how many
    scoring calls are made.

    All public scoring methods share the same signature:

    Parameters
    ----------
    P : np.ndarray (m, n)
        Batch of m probability distributions over the n nodes.
        Each row must be a valid probability vector.
    true_nodes : np.ndarray (m,), dtype int
        True node index for each row of P. Values in [0, n).

    Returns
    -------
    np.ndarray (m,)
        One score per row. For all scores, lower values indicate better performance.
    """
    def __init__(self, adjacency_matrix: np.ndarray):
        """
        Parameters
        ----------
        adjacency_matrix : np.ndarray (n, n)
            Adjacency matrix of an undirected unweighted graph without self-loops.
            Shared across all scored instances.
        """
        # Check adjacency satisfies the assumptions
        if adjacency_matrix.ndim != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("adjacency_matrix must be square.")
        if not np.allclose(adjacency_matrix, adjacency_matrix.T):
            raise ValueError("adjacency_matrix must be symmetric (undirected graph).")
        if np.any(np.diag(adjacency_matrix) != 0):
            raise ValueError("adjacency_matrix must have zero diagonal (no self-loops).")

        self.adjacency_matrix = adjacency_matrix
        self.size = adjacency_matrix.shape[0]

        # Graph-only quantities, computed lazily and cached
        self._degree_matrix = None
        self._laplacian = None
        self._laplacian_pinv = None
        self._resistance = None
        self._shortest_paths = None

    # -----------------------------------------------------------------
    # Graph-only quantities
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
    def laplacian_pinv(self):
        if self._laplacian_pinv is None:
            # L is symmetric PSD with an exact zero eigenvalue (constant vector).
            # np.linalg.pinv (SVD, default rcond=1e-15) can fail to discard that
            # eigenvalue when rounding puts it at ~1e-14 * s_max, which injects a
            # ~1e12 * 11^T/n term and destroys symmetry after cancellation in R.
            # pinvh uses eigh and the more conservative cutoff n * eps * |lambda|_max.
            self._laplacian_pinv = pinvh(self.laplacian)
        return self._laplacian_pinv

    @property
    def resistance(self):
        if self._resistance is None:
            diag = np.diag(self.laplacian_pinv)
            R = diag[:, None] + diag[None, :] - 2 * self.laplacian_pinv
            if not np.allclose(R, R.T, atol=1e-10):
                raise FloatingPointError("Resistance matrix is not symmetric; Laplacian pseudoinverse is inaccurate.")
            self._resistance = 0.5 * (R + R.T)
        return self._resistance

    @property
    def shortest_paths(self):
        """
        All-pairs shortest paths via vectorised Floyd-Warshall.
        Vectorised implementation adapted from:
        https://gist.github.com/mosco/11178777
        """
        if self._shortest_paths is None:
            dist = self.adjacency_matrix.astype(float).copy()
            dist[dist == 0] = np.inf
            np.fill_diagonal(dist, 0)
            for k in range(self.size):
                dist = np.minimum(dist, dist[:, k, np.newaxis] + dist[np.newaxis, k, :])
            self._shortest_paths = dist
        return self._shortest_paths

    def diffusion_kernel(self, tau: float, mu: float = 0.0, normalise: bool = False):
        """ Standard pd diffusion kernel.
        Not cached by default since it depends on (tau, mu, normalise)

        Parameters
        ----------
        tau : float
            Diffusion parameter.
        mu : float
            Regularisation parameter (to avoid score deflation at large tau).
        normalise : bool
            Whether to normalise the diffusion kernel by its trace.

        Returns
        -------
        np.ndarray (n, n)
            Kernel matrix.
        """
        H = expm(-0.5 * tau * (self.laplacian - mu * np.eye(self.size)))
        if normalise:
            H = H / np.trace(H)
        return H

    # -----------------------------------------------------------------
    # Check input consistency
    # -----------------------------------------------------------------
    def _check_inputs(self, P: np.ndarray, true_nodes: np.ndarray):
        """Check that P and true_nodes are compatible with the graph."""
        if P.ndim != 2:
            raise ValueError(f"P must be 2D, got shape {P.shape}.")
        if P.shape[1] != self.size:
            raise ValueError(
                f"P has {P.shape[1]} columns but graph has {self.size} nodes."
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
    # Batched kernel score
    # -----------------------------------------------------------------
    def _kernel_score_batch(
            self,
            P: np.ndarray,
            true_nodes: np.ndarray,
            kernel: np.ndarray,
            convention: str = "positive"
    ):
        """ General kernel score.

        Vectorised batch version of:
            S(p, i) = -0.5 (p - e_i)^T K (p - e_i) + extra_term
            = -0.5 p^T K p + p^T K e_i - 0.5 e_i^T K e_i + (extra_term, possibly 0)
            = -regularisation + expected_residual + (extra_term, possibly 0)

        Parameters
        ----------
        P : np.ndarray (m, n)
            Batch of m probability vectors over n nodes.
        true_nodes : np.ndarray (m,)
            Batch of m true node indices, one per row of P.
        kernel : np.ndarray (n, n)
            Conditionally negative definite kernel matrix K (e.g. -laplacian).
        convention : str or None
            "positive" (default)
                -> [Ziegel/Ginsbourger/Duembgen 2022] non-negative scores (extra term = - K_ii/2)
            "classic"
                -> [Gneiting/Raftery 2007] convention (extra term = 0)

        Returns
        -------
        np.ndarray (m,)
            One score per row.
        """
        self._check_inputs(P, true_nodes)
        m = P.shape[0]
        rows = np.arange(m)

        PK = P @ kernel
        expected_residual = PK[rows, true_nodes]
        regularisation = 0.5 * np.sum(PK * P, axis=1)

        if convention == "positive":
            extra_term = -0.5 * kernel[true_nodes, true_nodes]
        elif convention == "classic":
            extra_term = 0.0
        else:
            raise ValueError(f"Unknown kernel score convention '{convention}'. Use 'positive' or 'classic'.")

        return expected_residual - regularisation + extra_term

    # -----------------------------------------------------------------
    # Public batched scores
    # -----------------------------------------------------------------
    def error_distance(self, P: np.ndarray, true_nodes: np.ndarray):
        """ Graph error distance.

        Warning: not a proper scoring rule! Kept for comparison purposes.

        In case of ties in the argmax (multiple nodes with equal max
        probability) returns the average distance to all tied argmax nodes.
        """
        self._check_inputs(P, true_nodes)
        m = P.shape[0]
        out = np.empty(m, dtype=float)
        sp = self.shortest_paths
        for row in range(m):
            argmax_all = np.where(P[row] == P[row].max())[0]
            out[row] = sp[true_nodes[row], argmax_all].mean()
        return out

    def top_k_error(self, P: np.ndarray, true_nodes: np.ndarray, k: int = 1):
        """ Top-k miss indicator:
        Returns 1 if the true node is not in the top-k nodes according to P; 0 else.
        """
        self._check_inputs(P, true_nodes)
        top_k_idx = np.argpartition(-P, kth=k - 1, axis=1)[:, :k]  # argpartition is more efficient than a full argsort
        return (~(top_k_idx == true_nodes[:, None]).any(axis=1)).astype(float)

    def rank_score(self, P: np.ndarray, true_nodes: np.ndarray):
        """1-indexed rank of the true node.

        In case of tied probabilities, ties are broken using midrank:
        all nodes tied with the true node's probability receive the average
        of the ranks they would jointly occupy. (e.g. nodes tied for
        positions 2, 3 and 4 all receive rank 3).
        """
        self._check_inputs(P, true_nodes)
        m = P.shape[0]
        rows = np.arange(m)
        true_probs = P[rows, true_nodes]

        strictly_greater = np.sum(P > true_probs[:, None], axis=1)
        tied = np.sum(P == true_probs[:, None], axis=1)

        # The ranks of all the nodes tied with the true node are
        #   strictly_greater + 1, ..., strictly_greater + tied
        # The average of these ranks is the average of the first and last:
        return strictly_greater + (tied + 1) / 2.0

    def logarithmic_score(self, P: np.ndarray, true_nodes: np.ndarray):
        """Logarithmic score, also known as cross-entropy."""
        self._check_inputs(P, true_nodes)
        return -np.log(P[np.arange(P.shape[0]), true_nodes])

    def brier_score(self, P: np.ndarray, true_nodes: np.ndarray):
        self._check_inputs(P, true_nodes)
        m = P.shape[0]
        onehot = np.zeros((m, self.size))
        onehot[np.arange(m), true_nodes] = 1.0
        residuals = onehot - P
        return np.einsum('mi,mi->m', residuals, residuals)

    def weighted_brier_score(self, P: np.ndarray, true_nodes: np.ndarray, weights: np.ndarray):
        """Brier score with different weights for each node."""
        self._check_inputs(P, true_nodes)
        m = P.shape[0]
        onehot = np.zeros((m, self.size))
        onehot[np.arange(m), true_nodes] = 1.0
        residuals = onehot - P
        return np.einsum('mi,i,mi->m', residuals, weights, residuals)

    def laplace_score(self, P: np.ndarray, true_nodes: np.ndarray, convention: str = "positive"):
        return self._kernel_score_batch(P, true_nodes, -self.laplacian, convention)

    def resistance_score(self, P: np.ndarray, true_nodes: np.ndarray):
        return self._kernel_score_batch(P, true_nodes, self.resistance) # for this score, the Gneiting/Raftery and Ziegel et al. conventions coincide

    def diffusion_score(
            self,
            P: np.ndarray,
            true_nodes: np.ndarray,
            tau: float,
            convention: str = "positive",
            mu: float = 0.0,
            normalise: bool = False
    ):
        H = self.diffusion_kernel(tau=tau, mu=mu, normalise=normalise)
        return 2 * self._kernel_score_batch(P, true_nodes, -H, convention)