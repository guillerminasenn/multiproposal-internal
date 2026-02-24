# Multi-proposal pCN sampler

import numpy as np


def mpcn_step(x, problem, rng, rho=0.9, n_props=10):
    """Single multi-proposal pCN step using likelihood-only weights."""
    mu = problem.prior_mean()
    eta = np.sqrt(1.0 - rho * rho)

    x_center = mu + rho * (x - mu) + eta * problem.L @ rng.standard_normal(problem.dim)

    z = rng.standard_normal((problem.dim, n_props))
    props = (mu[:, None] + rho * (x_center - mu)[:, None] + eta * problem.L @ z).T

    candidates = np.vstack([x[None, :], props])
    log_w = np.array([problem.log_likelihood(c) for c in candidates])
    log_w -= np.max(log_w)
    weights = np.exp(log_w)

    idx = rng.choice(n_props + 1, p=weights / weights.sum())
    return candidates[idx]


def mpcn_chain(x0, problem, rng, n_iters, rho=0.9, n_props=10):
    """Run a multi-proposal pCN chain for n_iters steps."""
    chain = np.zeros((n_iters + 1, problem.dim), dtype=float)
    chain[0] = x0
    x = x0
    for t in range(n_iters):
        x = mpcn_step(x, problem, rng, rho=rho, n_props=n_props)
        chain[t + 1] = x
    return chain
