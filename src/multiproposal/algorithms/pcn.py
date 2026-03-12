# Single-proposal pCN sampler

import numpy as np


def pcn_step(x, problem, rng, rho=0.9, return_log_alpha=False):
    """Single pCN step with standard MH accept/reject."""
    mu = problem.prior_mean()
    eta = np.sqrt(1.0 - rho * rho)

    z = rng.standard_normal(problem.dim)
    y = mu + rho * (x - mu) + eta * (problem.L @ z)

    log_alpha = problem.log_likelihood(y) - problem.log_likelihood(x)
    if np.log(rng.uniform()) < min(0.0, log_alpha):
        if return_log_alpha:
            return y, True, log_alpha
        return y, True

    if return_log_alpha:
        return x, False, log_alpha
    return x, False


def pcn_chain(x0, problem, rng, n_iters, rho=0.9, return_acceptance=False):
    """Run a pCN chain for n_iters steps."""
    chain = np.zeros((n_iters + 1, problem.dim), dtype=float)
    chain[0] = x0
    x = x0
    n_acc = 0
    for t in range(n_iters):
        x, accepted = pcn_step(x, problem, rng, rho=rho)
        n_acc += int(accepted)
        chain[t + 1] = x
    if return_acceptance:
        return chain, n_acc / n_iters
    return chain
