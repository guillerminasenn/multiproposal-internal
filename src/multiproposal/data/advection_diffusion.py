# Data simulation for the advection-diffusion toy model

import numpy as np
from multiproposal.problems.advection_diffusion import (
    make_omegas_power,
    make_Astar_nn,
    make_Astar_from_atrue,
    params_from_skew,
    prior_diag_from_powerlaw,
    solve_theta,
)


def generate_advection_diffusion_data(
    dim=11,
    kappa=0.1,
    sigma=0.1,
    obs_indices=None,
    alpha=1.0,
    gamma=2.0,
    tau2=1.0,
    offset=1.0,
    a_mode="nearest_neighbor",
    seed=0,
):
    """Generate synthetic data for the AD toy problem.

    Returns a dict with prior pieces, true parameters, and observations.
    """
    rng = np.random.default_rng(seed)

    if obs_indices is None:
        obs_indices = np.arange(max(0, dim - 3), dim - 1, dtype=int)
    obs_indices = np.asarray(obs_indices, dtype=int)

    prior_diag = prior_diag_from_powerlaw(
        dim, alpha=alpha, gamma=gamma, tau2=tau2, offset=offset
    )

    if a_mode == "nearest_neighbor":
        omegas = make_omegas_power(dim, beta=alpha, c=2.0 ** (-gamma), offset=offset)
        A_true = make_Astar_nn(dim, omegas)
        a_true = params_from_skew(A_true)
    elif a_mode == "prior":
        a_true = rng.standard_normal(prior_diag.shape[0]) * np.sqrt(prior_diag)
        # print(f"a_true: {a_true}, prior_diag: {prior_diag}, a_true.shape: {a_true.shape}, prior_diag.shape: {prior_diag.shape}")
        A_true = make_Astar_from_atrue(dim, a_true)
    else:
        raise ValueError("a_mode must be 'nearest_neighbor' or 'prior'")

    g = np.zeros(dim, dtype=float)
    g[0] = 1.0
    theta_true = solve_theta(dim, a_true, g, kappa)
    y = theta_true[obs_indices] + sigma * rng.standard_normal(len(obs_indices))

    a_init = rng.standard_normal(prior_diag.shape[0]) * np.sqrt(prior_diag)

    return {
        "dim": dim,
        "kappa": kappa,
        "alpha": alpha,
        "gamma": gamma,
        "tau2": tau2,
        "sigma": sigma,
        "obs_indices": obs_indices,
        "prior_diag": prior_diag,
        "a_true": a_true,
        "A_true": A_true,
        "g": g,
        "theta_true": theta_true,
        "y": y,
        "a_init": a_init,
    }
