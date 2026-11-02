"""
Fast approximations of the node selection scoring rules for large sparse graphs.
"""

import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import expm_multiply

from .scorer import GraphScorer


class FastGraphScorer(GraphScorer):
    """GraphScorer with fast approximations for large sparse graphs.

    Inherits all scores from GraphScorer. Overrides:
      - resistance_score: ApproxChol Laplacian solver (Julia/Laplacians.jl)
      - diffusion_score:  expm_multiply (scipy.sparse.linalg)
    """

    def __init__(self, adjacency_matrix: np.ndarray):
        super().__init__(adjacency_matrix)
        self._julia_solver = None
        self._sparse_adjacency = None
        self._sparse_laplacian = None

    # -----------------------------------------------------------------
    # Sparse graph-only quantities
    # -----------------------------------------------------------------
    @property
    def sparse_adjacency(self):
        """Sparse CSC representation of the adjacency matrix."""
        if self._sparse_adjacency is None:
            self._sparse_adjacency = csc_matrix(self.adjacency_matrix)
        return self._sparse_adjacency

    @property
    def sparse_laplacian(self):
        """Sparse CSC representation of the Laplacian."""
        if self._sparse_laplacian is None:
            self._sparse_laplacian = csc_matrix(self.laplacian)
        return self._sparse_laplacian

    # -----------------------------------------------------------------
    # Julia solver (for resistance score)
    # -----------------------------------------------------------------
    def _get_julia_solver(self):
        """Build and cache the ApproxChol Laplacian solver.

        Loads Julia packages on first call, then builds the solver
        from the sparse adjacency matrix. Subsequent calls return the
        cached solver.

        The solver is a Julia function f such that f(v) \approx L^+ v for any
        v in 1^perp. It uses approximate Cholesky factorisation,
        which runs in near-linear time in |E|.
        """
        if self._julia_solver is None:
            from juliacall import Main as jl

            jl.seval("using Laplacians")
            jl.seval("using LinearAlgebra")
            jl.seval("using SparseArrays")

            # transfer adjacency matrix to Julia in CSC sparse format
            sparse_adj = self.sparse_adjacency
            indptr_jl = jl.Vector((sparse_adj.indptr + 1).astype(np.int64))  # julia uses 1-based indexing
            indices_jl = jl.Vector((sparse_adj.indices + 1).astype(np.int64))  # julia uses 1-based indexing
            data_jl = jl.Vector(sparse_adj.data.astype(np.float64))
            adj_jl = jl.SparseMatrixCSC(self.size, self.size, indptr_jl, indices_jl, data_jl)

            self._julia_solver = jl.approxchol_lap(adj_jl)

        return self._julia_solver

    # -----------------------------------------------------------------
    # Overridden scoring methods
    # -----------------------------------------------------------------
    def resistance_score(self, P: np.ndarray, true_nodes: np.ndarray):
        """Resistance score via ApproxChol Laplacian solver.

        For each row p of P and corresponding true node i, solves
            L x = (p - e_i)
        and returns (p - e_i)^T x = (p - e_i)^T L^+ (p - e_i).

        This avoids forming the full (n, n) pseudoinverse L^+ and is efficient
        for large sparse graphs.
        """
        self._check_inputs(P, true_nodes)
        solver = self._get_julia_solver()
        m = P.shape[0]
        out = np.empty(m)
        e = np.zeros(self.size)
        for row in range(m):
            e[true_nodes[row]] = 1.0
            v = P[row] - e  # v = p - e_i, lies in 1^perp
            out[row] = float(v @ solver(v))
            e[true_nodes[row]] = 0.0  # reset
        return out

    def diffusion_score(
            self,
            P: np.ndarray,
            true_nodes: np.ndarray,
            tau: float,
            convention: str = "positive",
            mu: float = 0.0,
            normalise: bool = False
    ):
        """Diffusion score via expm_multiply (no full matrix exponential).

        For each row p of P and corresponding true node i, computes
            (p - e_i)^T H (p - e_i)
        where H = exp(-tau/2 * (L - mu * I)), using a Krylov subspace
        approximation via scipy.sparse.linalg.expm_multiply.

        Parameters
        ----------
        P : np.ndarray (m, n)
        true_nodes : np.ndarray (m,)
        tau : float
            Diffusion parameter.
        convention : str
            Accepted for API compatibility; only 'positive' is supported.
        mu : float
            Regularisation parameter.
        normalise : bool
            Not supported in FastGraphScorer (requires full matrix exponential
            to compute the trace). Raises NotImplementedError if True.

        Returns
        -------
        np.ndarray (m,)
        """
        if normalise:
            raise NotImplementedError(
                "normalise=True is not supported in FastGraphScorer: computing "
                "tr(H) requires the full matrix exponential. Use GraphScorer "
                "instead, or compute the trace separately."
            )
        if convention == "classic":
            raise NotImplementedError(
                "convention='classic' is not supported in FastGraphScorer: computing "
                "the diagonal correction H_ii requires the full matrix exponential. "
                "Use GraphScorer instead."
            )
        if convention != "positive":
            raise ValueError(
                f"Unknown convention '{convention}'. Only 'positive' (default) is accepted."
            )
        self._check_inputs(P, true_nodes)

        # A is the matrix in the exponent: H = expm(A)
        # scipy.sparse.linalg.expm_multiply(A, v) approximates H @ v
        if mu == 0:
            A = -0.5 * tau * self.sparse_laplacian
        else:
            A = -0.5 * tau * (self.sparse_laplacian - mu * eye(self.size, format="csc"))

        m = P.shape[0]
        out = np.empty(m)
        e = np.zeros(self.size)
        for row in range(m):
            e[true_nodes[row]] = 1.0
            v = P[row] - e  # v = p - e_i
            Hv = expm_multiply(A, v)  # H @ v via Krylov subspace
            out[row] = float(v @ Hv)
            e[true_nodes[row]] = 0.0
        return out
