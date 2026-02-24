# Advection-diffusion toy problem

import numpy as np
from .base import GaussianPriorProblem


def skew_from_params(dim, params):
    """Build a skew-symmetric matrix A from upper-triangular params."""
    A = np.zeros((dim, dim), dtype=float)
    iju = np.triu_indices(dim, k=1)
    A[iju] = params
    A[(iju[1], iju[0])] = -params
    return A


def params_from_skew(A):
    """Extract upper-triangular (k=1) entries of a skew matrix into a vector."""
    dim = A.shape[0]
    iju = np.triu_indices(dim, k=1)
    return A[iju].copy()


def ij_to_k(i, j, dim):
    """Map matrix index (i, j) to parameter index k in the upper triangle."""
    if i == j:
        raise ValueError("Diagonal entries are not parameterized.")
    if i < j:
        iu, ju = i, j
    else:
        iu, ju = j, i
    k = iu * (dim - 1) - iu * (iu - 1) // 2 + (ju - iu - 1)
    return k


def make_omegas_power(dim, beta=0.5, c=1.0, offset=1.0):
    """Power-law decay for nearest-neighbor couplings."""
    i = np.arange(dim - 1, dtype=float)
    return c * (offset + i) ** (-beta)


def make_Astar_nn(dim, omegas):
    """Nearest-neighbor skew-symmetric A* with A[i,i+1]=omegas[i]."""
    omegas = np.asarray(omegas, dtype=float)
    if dim < 2:
        raise ValueError("dim must be >= 2")
    if omegas.shape != (dim - 1,):
        raise ValueError(f"omegas must have shape ({dim-1},), got {omegas.shape}")

    A = np.zeros((dim, dim), dtype=float)
    idx = np.arange(dim - 1)
    A[idx, idx + 1] = omegas
    A[idx + 1, idx] = -omegas
    return A


def make_Astar_from_atrue(dim, atrue):
    """Build skew-symmetric A from row-wise upper-triangle vector atrue."""
    atrue = np.asarray(atrue[:, ], dtype=float)
    expected = dim * (dim - 1) // 2
    if atrue.shape != (expected,):
        raise ValueError(f"atrue must have shape ({expected},), got {atrue.shape}")

    A = np.zeros((dim, dim), dtype=float)
    k = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            A[i, j] = atrue[k]
            A[j, i] = -atrue[k]
            k += 1
    return A


def prior_diag_from_powerlaw(dim, alpha=1.0, gamma=2.0, tau2=1.0, offset=1.0):
    """Diagonal prior variances q_ij = tau2 * (ij)^(-alpha) * |i-j|^(-gamma)."""
    iju = np.triu_indices(dim, k=1)
    i = iju[0].astype(float) + offset
    j = iju[1].astype(float) + offset
    r = np.abs(iju[0] - iju[1]).astype(float) + offset
    return tau2 * (i * j) ** (-alpha) * (r) ** (-gamma)


def solve_theta(dim, params, g, kappa):
    """Solve (A + kappa I) theta = g for theta given upper-triangular params."""
    A = skew_from_params(dim, params)
    A_plus = A + kappa * np.eye(dim)
    return np.linalg.solve(A_plus, g)


class AdvectionDiffusionToy(GaussianPriorProblem):
    """Toy advection-diffusion inverse problem with Gaussian prior on A params."""

    def __init__(
        self,
        dim,
        kappa,
        sigma,
        y,
        obs_indices,
        g=None,
        prior_diag=None,
        alpha=1.0,
        gamma=2.0,
        tau2=1.0,
        offset=1.0,
    ):
        self.dim_state = dim
        self.kappa = float(kappa)
        self.sigma = float(sigma)
        self.obs_indices = np.asarray(obs_indices, dtype=int)
        self.y = np.asarray(y, dtype=float)

        if g is None:
            g = np.zeros(dim, dtype=float)
            g[0] = 1.0
        self.g = np.asarray(g, dtype=float)

        if prior_diag is None:
            prior_diag = prior_diag_from_powerlaw(
                dim, alpha=alpha, gamma=gamma, tau2=tau2, offset=offset
            )

        cov = np.diag(prior_diag)
        mu = np.zeros(cov.shape[0])
        super().__init__(mu, cov)

    def theta_from_params(self, params):
        return solve_theta(self.dim_state, params, self.g, self.kappa)

    def log_likelihood(self, params):
        theta = self.theta_from_params(params)
        resid = self.y - theta[self.obs_indices]
        return -0.5 / (self.sigma ** 2) * np.dot(resid, resid)
