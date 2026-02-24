# Data simulation for semi-blind deconvolution

import numpy as np
from scipy.linalg import toeplitz


def generate_sbd_data(
    n_v=50,
    n_h=50,
    kernel_length=5,
    prior_var=1.0,
    noise_variance=0.1,
    seed=0,
):
    """
    Generate synthetic data for semi-blind deconvolution.
    
    The forward model is: d = W @ c + e
    where:
    - c is the true image (vectorized), sampled from N(0, prior_var * I)
    - W is a 1D convolution matrix formed by blur kernel w
    - e is observation noise, sampled from N(0, noise_variance * I)
    - d is the observed blurred and noisy image
    
    Parameters
    ----------
    n_v : int
        Number of rows in the lattice (image height).
    n_h : int
        Number of columns in the lattice (image width).
    kernel_length : int
        Length of the blur kernel (k).
    prior_var : float
        Prior variance for the image pixels.
    noise_variance : float
        Observation noise variance.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'c_true': true image (vectorized), shape (n,)
        - 'c_init': initial state for MCMC, shape (n,)
        - 'd': observed data, shape (n,)
        - 'w': blur kernel, shape (k,)
        - 'W': convolution matrix, shape (n, n)
        - 'n_v': number of rows
        - 'n_h': number of columns
        - 'prior_var': prior variance
        - 'noise_variance': noise variance
    """
    rng = np.random.default_rng(seed)
    n = n_v * n_h
    k = kernel_length
    
    assert k <= n_v, f"Kernel length {k} must be <= n_v = {n_v}"
    
    # Generate a normalized Gaussian blur kernel
    w = rng.standard_normal(k)
    w = w / np.sum(np.abs(w))  # Normalize to sum to 1
    
    # Build the convolution matrix W
    col = np.zeros(n_v)
    col[:k] = w
    row = np.zeros(n_v)
    row[0] = w[0]
    W_single = toeplitz(col, row)
    
    # Build block diagonal matrix for all columns
    W = np.zeros((n, n))
    for i in range(n_h):
        start_idx = i * n_v
        end_idx = (i + 1) * n_v
        W[start_idx:end_idx, start_idx:end_idx] = W_single
    
    # Sample true image from prior
    c_true = np.sqrt(prior_var) * rng.standard_normal(n)
    
    # Generate initial state for MCMC
    c_init = np.sqrt(prior_var) * rng.standard_normal(n)
    
    # Generate observations: d = W @ c + e
    e = np.sqrt(noise_variance) * rng.standard_normal(n)
    d = W @ c_true + e
    
    return {
        "c_true": c_true,
        "c_init": c_init,
        "d": d,
        "w": w,
        "W": W,
        "n_v": n_v,
        "n_h": n_h,
        "prior_var": prior_var,
        "noise_variance": noise_variance,
    }
