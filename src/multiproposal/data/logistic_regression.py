# Data simulation for logistic regression
import numpy as np


def generate_logistic_regression_data(
    n_samples=100,
    n_features=10,
    beta_true=None,
    noise_scale=0.0,
    seed=None
):
    """Generate synthetic binary classification data for logistic regression.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features (dimension)
    beta_true : np.ndarray, optional
        True coefficient vector. If None, randomly generated.
    noise_scale : float
        Standard deviation of Gaussian noise added to linear predictor
        (for soft labels)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'X': Feature matrix of shape (n_samples, n_features)
        - 'y': Binary labels of shape (n_samples,) with values in {0, 1}
        - 'beta_true': True coefficient vector
        - 'y_proba': True class probabilities
    """
    rng = np.random.default_rng(seed)
    
    # Generate features from standard normal
    X = rng.standard_normal((n_samples, n_features))
    
    # Generate or use provided true coefficients
    if beta_true is None:
        beta_true = rng.standard_normal(n_features) * 0.5
    
    # Compute linear predictor
    eta = X @ beta_true
    
    # Add noise if specified
    if noise_scale > 0:
        eta = eta + noise_scale * rng.standard_normal(n_samples)
    
    # Compute true probabilities
    y_proba = 1.0 / (1.0 + np.exp(-eta))
    
    # Generate binary labels
    y = (rng.uniform(size=n_samples) < y_proba).astype(int)
    
    return {
        'X': X,
        'y': y,
        'beta_true': beta_true,
        'y_proba': y_proba,
    }