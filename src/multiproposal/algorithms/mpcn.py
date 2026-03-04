# Multi-proposal pCN sampler

import numpy as np


def mpcn_step(x, problem, rng, rho=0.9, n_props=10, return_idx=False, return_diagnostics=False):
    """Single multi-proposal pCN step using likelihood-only weights."""
    mu = problem.prior_mean()
    eta = np.sqrt(1.0 - rho * rho)

    nu_c = problem.L @ rng.standard_normal(problem.dim)
    x_center = mu + rho * (x - mu) + eta * nu_c

    z = rng.standard_normal((problem.dim, n_props))
    props = (mu[:, None] + rho * (x_center - mu)[:, None] + eta * problem.L @ z).T

    candidates = np.vstack([x[None, :], props])
    log_w = np.array([problem.log_likelihood(c) for c in candidates])
    log_w -= np.max(log_w)
    weights = np.exp(log_w)

    idx = rng.choice(n_props + 1, p=weights / weights.sum())

    if return_diagnostics:
        diagnostics = {
            "x_center": x_center,
            "nu_c": nu_c,
            "props": props,
            "candidates": candidates,
            "log_w": log_w,
            "weights": weights,
        }
        if return_idx:
            return candidates[idx], idx, diagnostics
        return candidates[idx], diagnostics

    if return_idx:
        return candidates[idx], idx
    return candidates[idx]


def mpcn_chain(
    x0,
    problem,
    rng,
    n_iters,
    rho=0.9,
    n_props=10,
    return_indices=False,
    return_diagnostics=False,
    diag_indices=None,
):
    """Run a multi-proposal pCN chain for n_iters steps."""
    chain = np.zeros((n_iters + 1, problem.dim), dtype=float)
    accepted_index = np.zeros(n_iters, dtype=int) if return_indices else None
    diagnostics = [] if return_diagnostics else None
    diag_set = None
    if return_diagnostics and diag_indices is not None:
        diag_set = set(int(i) for i in diag_indices)
    chain[0] = x0
    x = x0
    for t in range(n_iters):
        capture_diag = return_diagnostics and (diag_set is None or t in diag_set)
        if return_indices or capture_diag:
            if capture_diag:
                x_new, idx, diag = mpcn_step(
                    x,
                    problem,
                    rng,
                    rho=rho,
                    n_props=n_props,
                    return_idx=True,
                    return_diagnostics=True,
                )
                diagnostics.append(
                    {
                        "iter": t,
                        "x": x.copy(),
                        "accepted_idx": int(idx),
                        **diag,
                    }
                )
            else:
                x_new, idx = mpcn_step(x, problem, rng, rho=rho, n_props=n_props, return_idx=True)
            if return_indices:
                accepted_index[t] = idx
        else:
            x_new = mpcn_step(x, problem, rng, rho=rho, n_props=n_props)
        x = x_new
        chain[t + 1] = x
    if return_diagnostics and return_indices:
        return chain, accepted_index, diagnostics
    if return_diagnostics:
        return chain, diagnostics
    if return_indices:
        return chain, accepted_index
    return chain
