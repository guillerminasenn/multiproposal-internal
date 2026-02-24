# Logistic regression problem
import numpy as np
from .base import GaussianPriorProblem


class BayesianLogisticRegression(GaussianPriorProblem):
    """Bayesian logistic regression with Gaussian prior on coefficients.
    
    Implements binary classification with likelihood:
        p(y_i | x_i, beta) = sigmoid(y_i * x_i^T beta)
    
    where y_i ∈ {-1, +1} and prior is N(0, prior_var * I).
    """
    
    def __init__(self, X, y, prior_var=1.0, jitter=1e-9):
        """
        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features)
        y : np.ndarray
            Binary labels of shape (n_samples,) with values in {-1, +1} or {0, 1}
        prior_var : float
            Variance of the Gaussian prior on coefficients
        jitter : float
            Small value for numerical stability
        """
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        
        # Convert labels to {-1, +1} if they are {0, 1}
        if np.all(np.isin(y, [0, 1])):
            self.y = 2 * y - 1
        
        # Gaussian prior: N(0, prior_var * I)
        mu = np.zeros(self.n_features)
        cov = prior_var * np.eye(self.n_features) + jitter * np.eye(self.n_features)
        
        super().__init__(mu, cov)
    
    def log_likelihood(self, beta):
        """
        Log likelihood for logistic regression.
        
        p(y | X, beta) = prod_i sigmoid(y_i * X_i @ beta)
        log p(y | X, beta) = sum_i log(sigmoid(y_i * X_i @ beta))
                           = sum_i -log(1 + exp(-y_i * X_i @ beta))
        
        Parameters
        ----------
        beta : np.ndarray
            Coefficient vector of shape (n_features,)
        
        Returns
        -------
        float
            Log-likelihood value
        """
        eta = self.y * (self.X @ beta)  # Element-wise product with labels
        # Use numerically stable computation: log(sigmoid(z)) = -log(1 + exp(-z))
        log_lik = np.sum(-np.logaddexp(0, -eta))
        return log_lik