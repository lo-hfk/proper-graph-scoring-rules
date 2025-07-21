import numpy as np
from scipy.linalg import expm, matrix_power
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
        self.size = len(node_probs)
        self.degree_matrix = None  # compute only when necessary
        self.laplacian = None  # compute only when necessary
        self.normalised_laplacian = None  # compute only when necessary

    def _get_degree_matrix(self):
        return np.diag(np.sum(self.adjacency_matrix, axis=1))

    def _get_laplacian(self):
        if self.degree_matrix is None:
            self.degree_matrix = self._get_degree_matrix()
        return self.degree_matrix - self.adjacency_matrix

    def _get_normalised_laplacian(self):
        if self.laplacian is None:
            self.laplacian = self._get_laplacian()
        if self.degree_matrix is None:
            self.degree_matrix = self._get_degree_matrix()
        sqrt_degree_matrix = np.sqrt(self.degree_matrix)
        return sqrt_degree_matrix @ self.laplacian @ sqrt_degree_matrix

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
        """
        Proper scoring rule based on resistance distance
        """
        if self.laplacian is None:
            self.laplacian = self._get_laplacian()
        laplacian_inv = np.linalg.pinv(self.laplacian)
        resistance = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                resistance[i, j] = laplacian_inv[i, i] + laplacian_inv[j, j] \
                                   - 2 * laplacian_inv[i, j]
        return float(self._kernel_score(resistance))

    def diffusion_score(self, kappa2: float):
        """
        Proper scoring rule based on the diffusion kernel

        Parameters
        ----------
        kappa2 : float (non-negative)
            Inverse length scale
        """
        if self.laplacian is None:
            self.laplacian = self._get_laplacian()
        diffusion_kernel = expm(-0.5 * kappa2 * self.laplacian)
        return float(self._kernel_score(-diffusion_kernel))

    def random_walk_score(self, alpha: float, p: int):
        """
        Proper scoring rule based on the lazy random walk kernel
            [Smola/Kondor, 2003]

        Parameters
        ----------
        alpha : float (in [0,1))
            Laziness parameter
        p : int (non-negative)
            Number of steps of the random walk
        """
        if self.normalised_laplacian is None:
            self.normalised_laplacian = self._get_normalised_laplacian()
        random_walk_kernel = matrix_power(
            np.eye(self.size) - (1 - alpha) * self.normalised_laplacian, p
        )
        return float(self._kernel_score(-random_walk_kernel))
