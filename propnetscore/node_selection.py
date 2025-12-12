import numpy as np
from numpy.linalg import matrix_power
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
        self.size = len(node_probs)
        self.degree_matrix = None  # compute only when necessary
        self.laplacian = None  # compute only when necessary
        self.normalised_laplacian = None  # compute only when necessary
        self.laplacian_inv = None  # compute only when necessary
        self.resistance = None  # compute only when necessary

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

    def _kernel_score(self, cnd_kernel: np.ndarray, convention: str | None = None):
        """
        Parameters
        ----------
        cnd_kernel : np.ndarray (n, n)
            Conditionally negative definite kernel matrix.
        convention : str
            Defaults to the definition by [Gneiting/Raftery, 2007].
            If set to "positive", subtract cnd_kernel[true_node, true_node]/2 which ensures non-negative scores
                [Ziegel/Ginsbourger/Duembgen, 2022].

        Returns
        -------
        float
            Score
        """
        expected_residual = (cnd_kernel @ self.node_probs)[self.true_node]
        regularisation = 0.5 * self.node_probs @ cnd_kernel @ self.node_probs
        if convention == "positive":
            extra_term = - 0.5 * cnd_kernel[self.true_node, self.true_node]
        else:
            extra_term = 0
        return expected_residual - regularisation + extra_term

    def top_k_accuracy(self, k=1):
        top_k_indices = np.argsort(-self.node_probs)[:k]
        return float(self.true_node in top_k_indices)

    def top_k_error_rate(self, k=1):
        top_k_indices = np.argsort(-self.node_probs)[:k]
        return 1 - self.top_k_accuracy(k=k)

    def rank_score(self):
        sorted_indices = np.argsort(-self.node_probs)
        rank = np.where(sorted_indices == self.true_node)[0][0] + 1
        return rank

    def inverse_rank_score(self):
        return 1. / self.rank_score()

    def logarithmic_score(self):
        return float(-np.log(self.node_probs[self.true_node]))

    def brier_score(self):
        outcome = standard_basis_vec(
            size=self.size, i=self.true_node
        )
        residuals = outcome - self.node_probs
        return float(np.dot(residuals, residuals))

    def laplace_score(self, convention: str | None = None):
        """
        Proper scoring rule based on the Laplacian

        Parameters
        ----------
        convention : str
            Kernel score convention (None or "positive")
        """
        if self.laplacian is None:
            self.laplacian = self._get_laplacian()
        normalisation_factor = 2 / self.size
        return float(self._kernel_score(-self.laplacian, convention) * normalisation_factor)

    def resistance_score(self):
        """
        Proper scoring rule based on resistance distance
        """
        if self.laplacian is None:
            self.laplacian = self._get_laplacian()
        if self.laplacian_inv is None:
            self.laplacian_inv = np.linalg.pinv(self.laplacian)
        if self.resistance is None:
            diagonal = np.diag(self.laplacian_inv)
            self.resistance = diagonal[:, None] + diagonal[None, :] - 2 * self.laplacian_inv
        normalisation_factor = self.size
        return float(self._kernel_score(self.resistance) * normalisation_factor)

    def diffusion_score(self, kappa2: float, convention: str | None = None):
        """
        Proper scoring rule based on the diffusion kernel

        Parameters
        ----------
        kappa2 : float (non-negative)
            Inverse length scale
        convention : str
            Kernel score convention (None or "positive")
        """
        if self.laplacian is None:
            self.laplacian = self._get_laplacian()
        diffusion_kernel = expm(-0.5 * kappa2 * (self.laplacian - np.eye(self.size)))
        normalisation_factor = 0.5
        return float(self._kernel_score(-diffusion_kernel, convention) * normalisation_factor)

    def random_walk_score(self, alpha: float, p: int, convention: str | None = None):
        """
        Proper scoring rule based on the lazy random walk kernel
            [Smola/Kondor, 2003]

        Parameters
        ----------
        alpha : float (in [0,1))
            Laziness parameter
        p : int (non-negative)
            Number of steps of the random walk
        convention : str
            Kernel score convention (None or "positive")
        """
        if self.normalised_laplacian is None:
            self.normalised_laplacian = self._get_normalised_laplacian()
        random_walk_kernel = matrix_power(
            np.eye(self.size) - (1 - alpha) * self.normalised_laplacian, p
        )
        return float(self._kernel_score(-random_walk_kernel, convention))
