import numpy as np
from scipy.linalg import expm
from .utils import standard_basis_vec

class NodeSelectionTask:
    def __init__(
            self, adjacency_matrix: np.ndarray,
            node_probs: np.ndarray, true_node:int
    ):
        """
        Parameters
        ----------
        adjacency_matrix : np.ndarray (n, n)
            Adjacency matrix of the graph
        node_probs : np.ndarray (n,)
            Predicted probability distribution over the nodes
        true_node : int
            Index of the true node from 0..n-1
        """
        self.adjacency_matrix = adjacency_matrix
        self.node_probs = node_probs
        self.true_node = true_node
        self.laplacian = None  # compute only when necessary

    def _get_laplacian(self):
        degree_matrix = np.diag(np.sum(self.adjacency_matrix, axis=1))
        return degree_matrix - self.adjacency_matrix

    def _kernel_score(self, cnd_kernel: np.ndarray):
        """
        Parameters
        ----------
        cnd_kernel : np.ndarray (n, n)
            Conditionally negative definite kernel matrix.

        Returns
        -------
        float
            Score
        """
        expected_residual = (cnd_kernel @ self.node_probs)[self.true_node]
        regularisation = 0.5 * self.node_probs @ cnd_kernel @ self.node_probs
        return expected_residual - regularisation

    def logarithmic_score(self):
        return float(-np.log(self.node_probs[self.true_node]))

    def brier_score(self):
        outcome = standard_basis_vec(
            size=len(self.node_probs), i=self.true_node
        )
        residuals = outcome - self.node_probs
        return float(np.dot(residuals, residuals))

    def resistance_score(self):
        if self.laplacian is None:
            self.laplacian = self._get_laplacian()
        laplacian_inv = np.linalg.pinv(self.laplacian)
        n = self.laplacian.shape[0]
        resistance = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                resistance[i, j] = laplacian_inv[i, i] + laplacian_inv[j, j] \
                                   - 2 * laplacian_inv[i, j]
        return float(self._kernel_score(resistance))

    def diffusion_score(self, kappa2: float):
        if self.laplacian is None:
            self.laplacian = self._get_laplacian()
        diffusion_kernel = expm(-0.5 * kappa2 * self.laplacian)
        return float(self._kernel_score(1 - diffusion_kernel))
