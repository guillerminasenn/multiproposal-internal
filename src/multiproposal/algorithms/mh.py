# Random-walk Metropolis-Hastings sampler

import numpy as np


def mh_step(x, problem, rng, proposal_std=0.1, proposal_cov="isotropic"):
    """Single MH step with Gaussian random-walk proposal.

    proposal_cov: "isotropic" (default) or "prior" (uses problem.L).
    """
    if proposal_cov == "prior":
        if not hasattr(problem, "L"):
            raise ValueError("proposal_cov='prior' requires problem.L (Cholesky of prior covariance).")
        z = rng.standard_normal(problem.dim)
        x_prop = x + proposal_std * (problem.L @ z)
    elif proposal_cov == "isotropic":
        x_prop = x + proposal_std * rng.standard_normal(problem.dim)
    else:
        raise ValueError("proposal_cov must be 'isotropic' or 'prior'.")
    log_alpha = problem.log_posterior(x_prop) - problem.log_posterior(x)
    if np.log(rng.uniform()) < min(0.0, log_alpha):
        return x_prop, True
    return x, False


def mh_chain(x0, problem, rng, n_iters, proposal_std=0.1, proposal_cov="isotropic"):
    """Run an MH chain for n_iters steps.

    proposal_cov: "isotropic" (default) or "prior" (uses problem.L).
    """
    chain = np.zeros((n_iters + 1, problem.dim), dtype=float)
    chain[0] = x0
    x = x0
    log_post = problem.log_posterior(x)
    n_acc = 0
    for t in range(n_iters):
        if proposal_cov == "prior":
            if not hasattr(problem, "L"):
                raise ValueError("proposal_cov='prior' requires problem.L (Cholesky of prior covariance).")
            z = rng.standard_normal(problem.dim)
            x_prop = x + proposal_std * (problem.L @ z)
        elif proposal_cov == "isotropic":
            x_prop = x + proposal_std * rng.standard_normal(problem.dim)
        else:
            raise ValueError("proposal_cov must be 'isotropic' or 'prior'.")

        log_post_prop = problem.log_posterior(x_prop)
        log_alpha = log_post_prop - log_post
        if np.log(rng.uniform()) < min(0.0, log_alpha):
            x = x_prop
            log_post = log_post_prop
            n_acc += 1
        chain[t + 1] = x
    return chain, n_acc / n_iters
