from propnetscore.node_selection import NodeSelectionTask
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply
from propnetscore.utils import standard_basis_vec
from juliacall import Main as jl
import time


class NodeSelectionTaskFast(NodeSelectionTask):
    def __init__(self, adjacency_matrix: np.ndarray, node_probs: np.ndarray, true_node:int):
        super().__init__(adjacency_matrix, node_probs, true_node)
        self.approx_laplacian_inv = None # compute only when necessary
        self.approx_resistance = None # compute only when necessary
        self.sparse_adjacency_matrix = None # compute only when necessary
        self.sparse_laplacian = None # compute only when necessary

    def _julia_approx_laplacian_inverse(self, sparse_adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Julia implementation of the approximate laplacian inverse.
        """
        # load julia packages
        start = time.time()
        jl.seval("using Laplacians")
        jl.seval("using LinearAlgebra")
        jl.seval("using SparseArrays")
        print(f" --- Loaded Julia packages in {time.time() - start:.3f}s")

        # transport sparse adjacency matrix to julia
        start = time.time()
        n = sparse_adjacency_matrix.shape[0]
        indptr_py = (sparse_adjacency_matrix.indptr + 1).astype(np.int64) # julia uses 1-based indexing
        indices_py = (sparse_adjacency_matrix.indices + 1).astype(np.int64) # julia uses 1-based indexing
        data_py = sparse_adjacency_matrix.data.astype(np.float64)
        indptr_jl = jl.Vector(indptr_py) # convert to julia vector
        indices_jl = jl.Vector(indices_py) # convert to julia vector
        data_jl = jl.Vector(data_py) # convert to julia vector
        adjacency_jl = jl.SparseMatrixCSC(n, n, indptr_jl, indices_jl, data_jl)
        print(f" --- Transferred sparse adjacency matrix to Julia in {time.time() - start:.3f}s")

        # initialize laplacian solver and compute laplacian inverse
        start = time.time()
        lap_solver = jl.approxchol_lap(adjacency_jl)
        I_py = np.eye(n, dtype=np.float64) # unit matrix
        L_inverse_list = [lap_solver(I_py[:, i]) for i in range(n)]
        L_inverse_py = np.column_stack([np.array(col) for col in L_inverse_list])
        print(f" --- Computed approximate laplacian inverse in Julia in {time.time() - start:.3f}s")
        return L_inverse_py

    def resistance_score_fast(self):
        """
        Proper scoring rule based on resistance distance, but with a fast julia implementation of the laplacian inverse.
        """
        if self.laplacian is None:
            self.laplacian = self._get_laplacian()
        if self.sparse_adjacency_matrix is None:
            self.sparse_adjacency_matrix = csc_matrix(self.adjacency_matrix)
        if self.approx_laplacian_inv is None:
            self.approx_laplacian_inv = self._julia_approx_laplacian_inverse(self.sparse_adjacency_matrix)
        if self.approx_resistance is None:
            diagonal = np.diag(self.approx_laplacian_inv)
            self.approx_resistance = diagonal[:, None] + diagonal[None, :] - 2 * self.approx_laplacian_inv
        return float(self._kernel_score(self.approx_resistance))

    def _kernel_score_fast(self, cnd_kernel_x_node_probs: np.ndarray, convention: str | None = None, true_diag: float | None = None):
        """
        Parameters
        ----------
        cnd_kernel_x_node_probs : np.ndarray (n,)
            Product of conditionally negative definite kernel matrix with node_probs
        convention : str
            Defaults to the definition by [Gneiting/Raftery, 2007].
            If set to "positive", subtract cnd_kernel[true_node, true_node]/2 which ensures non-negative scores
                [Ziegel/Ginsbourger/Duembgen, 2022].
        true_diag : float
            Diagonal element of the kernel matrix corresponding to the true node.

        Returns
        -------
        float
            Score
        """
        expected_residual = (cnd_kernel_x_node_probs)[self.true_node]
        regularisation = 0.5 * self.node_probs @ cnd_kernel_x_node_probs
        if convention == "positive":
            extra_term = - 0.5 * true_diag
        else:
            extra_term = 0
        return expected_residual - regularisation + extra_term

    def diffusion_score_fast(self, kappa2: float, convention: str | None = None):
        """
        Proper scoring rule based on the diffusion kernel but with a fast implementation based on expm_multiply
        (only matrix-vector products instead of full matrix exponentiation).

        Parameters
        ----------
        kappa2 : float (non-negative)
            Inverse length scale
        convention : str
            Kernel score convention (None or "positive")
        """
        if self.laplacian is None:
            self.laplacian = self._get_laplacian()
        if self.sparse_laplacian is None:
            self.sparse_laplacian = csc_matrix(self.laplacian)
        if convention == "positive":
            true_diag = - expm_multiply(-0.5 * kappa2 * self.sparse_laplacian,
                                        standard_basis_vec(size=len(self.node_probs), i=self.true_node))[self.true_node]
        else:
            true_diag = None
        minus_diffusion_kernel_x_node_probs = - expm_multiply(-0.5 * kappa2 * self.sparse_laplacian, self.node_probs)
        return float(self._kernel_score_fast(minus_diffusion_kernel_x_node_probs, convention, true_diag))

