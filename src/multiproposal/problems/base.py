# GaussianPriorProblem

# problems/base.py
import numpy as np
from scipy.linalg import solve_triangular

class GaussianPriorProblem:
    def __init__(self, mu, cov):
        self.mu = mu
        self.dim = len(mu)
        self.L = np.linalg.cholesky(cov)

    def log_prior(self, x):
        """Unnormalized log-density of the Gaussian prior at x."""
        dx = x - self.mu
        # L is lower-triangular from Cholesky, so use a triangular solve.
        z = solve_triangular(self.L, dx, lower=True, check_finite=False)
        return -0.5 * np.dot(z, z)

    def log_posterior(self, x):
        """Unnormalized log-density of the posterior at x."""
        return self.log_prior(x) + self.log_likelihood(x)

    def sample_prior(self, rng):
        z = rng.standard_normal(self.dim)
        return self.mu + self.L @ z

    def prior_mean(self):
        return self.mu

