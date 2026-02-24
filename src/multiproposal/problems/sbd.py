# Semi-blind deconvolution problem

import numpy as np
from scipy.linalg import toeplitz
from .base import GaussianPriorProblem

class SemiBlindDeconvolution(GaussianPriorProblem):
    """Semi-blind deconvolution with Gaussian prior on the image.
    
    Forward model: d = W @ c + e
    
    where:
    - W is a 1D convolutional matrix (n x n) formed by the blur kernel w
    - w is a length k vector (k <= n_v), the 1D blur kernel
    - c is the vectorized true image (length n = n_v * n_h)
    - e is observation noise (length n)
    - d is the observations (length n)
    - n_v is the number of rows in the lattice
    - n_h is the number of columns in the lattice
    
    The convolution is applied along the vertical direction (rows) for each column.
    """
    
    def __init__(self, d, w, n_v, n_h, prior_var=1.0, noise_variance=1.0, jitter=1e-9):
        """
        Parameters
        ----------
        d : np.ndarray
            Observed data (blurred and noisy image), vectorized, shape (n,)
            where n = n_v * n_h
        w : np.ndarray
            Blur kernel, shape (k,) where k <= n_v
        n_v : int
            Number of rows in the lattice (image height)
        n_h : int
            Number of columns in the lattice (image width)
        prior_var : float
            Variance of the Gaussian prior on the image pixels
        noise_variance : float
            Variance of the observation noise
        jitter : float
            Small value for numerical stability
        """
        self.d = d
        self.w = w
        self.n_v = n_v
        self.n_h = n_h
        self.k = len(w)
        self.n = n_v * n_h
        self.noise_variance = noise_variance
        
        assert len(d) == self.n, f"Data length {len(d)} must equal n_v * n_h = {self.n}"
        assert self.k <= n_v, f"Kernel length {self.k} must be <= n_v = {n_v}"
        
        # Build the convolution matrix W
        self.W = self._build_convolution_matrix(w, n_v, n_h)
        
        # Gaussian prior: N(0, prior_var * I)
        mu = np.zeros(self.n)
        cov = prior_var * np.eye(self.n) + jitter * np.eye(self.n)
        
        super().__init__(mu, cov)
    
    def _build_convolution_matrix(self, w, n_v, n_h):
        """Build the 1D convolution matrix W.
        
        The convolution is applied along columns (vertical direction).
        Each column of the image is convolved independently with the kernel w.
        
        Parameters
        ----------
        w : np.ndarray
            Blur kernel of shape (k,)
        n_v : int
            Number of rows in the lattice
        n_h : int
            Number of columns in the lattice
        
        Returns
        -------
        W : np.ndarray
            Convolution matrix of shape (n, n) where n = n_v * n_h
        """
        k = len(w)
        
        # Create a Toeplitz matrix for a single column
        # The first column of the Toeplitz matrix is [w[0], w[1], ..., w[k-1], 0, 0, ...]
        # The first row is [w[0], 0, 0, ...]
        col = np.zeros(n_v)
        col[:k] = w
        row = np.zeros(n_v)
        row[0] = w[0]
        
        W_single = toeplitz(col, row)
        
        # Build block diagonal matrix for all columns
        W = np.zeros((self.n, self.n))
        for i in range(n_h):
            start_idx = i * n_v
            end_idx = (i + 1) * n_v
            W[start_idx:end_idx, start_idx:end_idx] = W_single
        
        return W
    
    def log_likelihood(self, c):
        """
        Log likelihood for the deconvolution problem.
        
        p(d | c, w) = N(d | W @ c, noise_variance * I)
        log p(d | c, w) = -1/(2*noise_variance) * ||d - W @ c||^2 + const
        
        Parameters
        ----------
        c : np.ndarray
            Image vector of shape (n,) where n = n_v * n_h
        
        Returns
        -------
        float
            Log-likelihood value (up to constant)
        """
        residual = self.d - self.W @ c
        return -0.5 / self.noise_variance * np.dot(residual, residual)
