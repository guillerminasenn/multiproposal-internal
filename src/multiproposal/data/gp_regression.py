# Data simulation for GP regression

import numpy as np
from multiproposal.kernels.stationary import stationary_kernel

def generate_gp_regression_data(
    num_data=200,
    num_dims=1,
    length_scale=1.0,
    noise_variance=0.09,
    jitter=1e-9,
    seed=0,
):
    """
    Generate synthetic GP regression data following
    Murray et al. (2010), ESS paper, Example 1.

    Parameters
    ----------
    num_data : int
        Number of observations.
    num_dims : int
        Input dimension (1 to 10 in the original paper).
    length_scale : float
        Kernel length-scale.
    noise_variance : float
        Observation noise variance.
    jitter : float
        Jitter added to the covariance matrix.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing data, prior quantities,
        and a shared initial state.
    """
    rng = np.random.default_rng(seed)

    # Inputs on the unit hypercube
    X = rng.uniform(0.0, 1.0, size=(num_dims, num_data))

    # Prior covariance (RBF kernel: p = 2)
    K = stationary_kernel(
        X, X, length_scale=length_scale, p=2
    )
    K += jitter * np.eye(num_data)

    # Cholesky decomposition
    chol_K = np.linalg.cholesky(K)

    # Latent function (true)
    f_true = chol_K @ rng.standard_normal(num_data)

    # Initial state shared across algorithms
    f_init = chol_K @ rng.standard_normal(num_data)

    # Observations with noise
    y = f_true + np.sqrt(noise_variance) * rng.standard_normal(num_data)

    return {
        "X": X,
        "y": y,
        "K": K,
        "chol_K": chol_K,
        "f_true": f_true,
        "f_init": f_init,
        "noise_variance": noise_variance,
    }

