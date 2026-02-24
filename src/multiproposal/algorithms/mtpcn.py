# Multiple-try pCN sampler with MTM correction

import numpy as np


def _logsumexp(log_w):
    m = np.max(log_w)
    return m + np.log(np.sum(np.exp(log_w - m)))


def _log_q_pcn(x, y, problem, rho):
    """Log q(x|y) for pCN up to a constant."""
    mu = problem.prior_mean()
    eta = np.sqrt(1.0 - rho * rho)
    mean = mu + rho * (y - mu)
    diff = x - mean
    z = np.linalg.solve(problem.L, diff) / eta
    return -0.5 * np.dot(z, z)


def mtpcn_step(x, problem, rng, rho=0.9, n_props=10):
    """Single multiple-try pCN step with MTM correction."""
    mu = problem.prior_mean()
    eta = np.sqrt(1.0 - rho * rho)

    z = rng.standard_normal((problem.dim, n_props))
    props = (mu[:, None] + rho * (x - mu)[:, None] + eta * problem.L @ z).T

    log_w = np.array(
        [problem.log_posterior(p) + _log_q_pcn(x, p, problem, rho) for p in props]
    )
    log_w -= np.max(log_w)
    weights = np.exp(log_w)
    idx = rng.choice(n_props, p=weights / weights.sum())
    y_star = props[idx]

    z_back = rng.standard_normal((problem.dim, n_props - 1))
    back_props = (
        mu[:, None] + rho * (y_star - mu)[:, None] + eta * problem.L @ z_back
    ).T
    back_candidates = np.vstack([x[None, :], back_props])

    log_w_back = np.array(
        [
            problem.log_posterior(p) + _log_q_pcn(y_star, p, problem, rho)
            for p in back_candidates
        ]
    )

    log_alpha = _logsumexp(log_w) - _logsumexp(log_w_back)
    if np.log(rng.uniform()) < min(0.0, log_alpha):
        return y_star, True

    return x, False


def mtpcn_chain(x0, problem, rng, n_iters, rho=0.9, n_props=10):
    """Run a multiple-try pCN chain for n_iters steps."""
    chain = np.zeros((n_iters + 1, problem.dim), dtype=float)
    chain[0] = x0
    x = x0
    n_acc = 0
    for t in range(n_iters):
        x, accepted = mtpcn_step(x, problem, rng, rho=rho, n_props=n_props)
        n_acc += int(accepted)
        chain[t + 1] = x
    return chain, n_acc / n_iters
