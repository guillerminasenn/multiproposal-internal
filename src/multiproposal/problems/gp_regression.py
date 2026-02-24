# GP regression problem

# problems/gp_regression.py
import numpy as np
from .base import GaussianPriorProblem
from ..kernels.stationary import stationary_kernel

class GaussianProcessRegression(GaussianPriorProblem):
    def __init__(
        self,
        X,
        y,
        length_scale=1.0,
        noise_variance=0.09,
        jitter=1e-9,
    ):
        """Class that implements the Gaussian process regression in Murray et al. (2010).
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)-> n_samples=#data points, n_features=covariates.
        y : np.ndarray
            Output data of shape (n_samples,).
        length_scale : float
            Length scale parameter for the stationary kernel.
        noise_variance : float
            Variance of the observation noise.
        jitter : float
            Jitter term added to the diagonal of the covariance matrix for numerical stability."""
        self.X = X
        self.y = y
        self.noise_variance = noise_variance

        # Build prior
        K = stationary_kernel(X, X, length_scale)
        K += jitter * np.eye(len(y))
        mu = np.zeros(len(y))

        super().__init__(mu, K)

    def log_likelihood(self, f):
        r = self.y - f
        return -0.5 / self.noise_variance * np.dot(r, r)
