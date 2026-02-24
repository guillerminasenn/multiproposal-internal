import numpy as np

from .base import GaussianPriorProblem


class ToyCustomLikelihood2D(GaussianPriorProblem):
    """2D toy posterior with a user-specified log-likelihood function."""

    def __init__(self, log_likelihood_fn, prior_mean=None, prior_cov=None):
        if prior_mean is None:
            prior_mean = np.zeros(2)
        if prior_cov is None:
            prior_cov = np.eye(2)

        prior_mean = np.asarray(prior_mean, dtype=float)
        prior_cov = np.asarray(prior_cov, dtype=float)

        if prior_mean.shape != (2,):
            raise ValueError("prior_mean must have shape (2,)")
        if prior_cov.shape != (2, 2):
            raise ValueError("prior_cov must have shape (2, 2)")

        self.log_likelihood_fn = log_likelihood_fn
        super().__init__(prior_mean, prior_cov)

    def log_likelihood(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (2,):
            raise ValueError("x must have shape (2,)")
        return float(self.log_likelihood_fn(x))
