# RBF + exponential kernel

# src/multiproposal/kernels/stationary.py
import numpy as np

def stationary_kernel(x1, x2, length_scale=1.0, p=2):
    """
    Stationary isotropic kernel.

    p = 2 : Gaussian / squared exponential
    p = 1 : exponential 

    K(x, x') = exp( - ||x - x'||^p / (2 * l^p) )

    Parameters
    ----------
    x1 : np.ndarray
        First set of input points of shape (n_samples_1, n_features).
    x2 : np.ndarray
        Second set of input points of shape (n_samples_2, n_features).
    length_scale : float
        Length scale parameter.
    p : int
        Exponent parameter (1 or 2).
    """
    # Matrices containing the features as rows, each row is a data point
    x1 = x1.T
    x2 = x2.T

    # Compute the Euclidean distance in feature space
    diff = (
        np.sum(x1**2, axis=1)[:, None]
        + np.sum(x2**2, axis=1)[None, :]
        - 2 * x1 @ x2.T
    )
    dist = np.sqrt(np.maximum(diff, 0.0))

    return np.exp(-(dist ** p) / (2 * length_scale ** p))
