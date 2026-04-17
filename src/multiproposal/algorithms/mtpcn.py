# Multiple-try pCN sampler with MTM correction

import time

import numpy as np


def _logsumexp(log_w):
    m = np.max(log_w)
    return m + np.log(np.sum(np.exp(log_w - m)))


def mtpcn_step(
    x,
    problem,
    rng,
    rho=0.9,
    n_props=10,
    return_idx=False,
    return_diagnostics=False,
    collect_timing=False,
):
    """Single multiple-try pCN step with MTM correction."""
    t_start = time.perf_counter() if collect_timing else None
    mu = problem.prior_mean()
    eta = np.sqrt(1.0 - rho * rho)

    z = rng.standard_normal((problem.dim, n_props))
    props = (mu[:, None] + rho * (x - mu)[:, None] + eta * problem.L @ z).T

    t_after_props = time.perf_counter() if collect_timing else None

    log_w = np.array([problem.log_likelihood(p) for p in props])
    t_after_llh = time.perf_counter() if collect_timing else None
    log_w_sel = log_w - _logsumexp(log_w)
    weights = np.exp(log_w_sel)
    t_after_weights = time.perf_counter() if collect_timing else None
    idx = rng.choice(n_props, p=weights)
    t_after_choice = time.perf_counter() if collect_timing else None
    y_star = props[idx]
    idx_is_max = bool(idx == int(np.argmax(log_w)))

    z_back = rng.standard_normal((problem.dim, n_props - 1))
    back_props = (
        mu[:, None] + rho * (y_star - mu)[:, None] + eta * problem.L @ z_back
    ).T
    t_after_back_props = time.perf_counter() if collect_timing else None
    back_candidates = np.vstack([x[None, :], back_props])

    log_w_back = np.array([problem.log_likelihood(p) for p in back_candidates])
    t_after_back_llh = time.perf_counter() if collect_timing else None

    log_alpha = _logsumexp(log_w) - _logsumexp(log_w_back)
    accept_u = rng.uniform()
    log_u = np.log(accept_u)
    accepted = bool(log_u < min(0.0, log_alpha))
    x_new = y_star if accepted else x

    timing = None
    if collect_timing and t_start is not None:
        timing = {
            "props": (t_after_props - t_start) if t_after_props else 0.0,
            "llh": (t_after_llh - t_after_props) if t_after_llh else 0.0,
            "weights": (t_after_weights - t_after_llh) if t_after_weights else 0.0,
            "choice": (t_after_choice - t_after_weights) if t_after_choice else 0.0,
            "back_props": (t_after_back_props - t_after_choice) if t_after_back_props else 0.0,
            "back_llh": (t_after_back_llh - t_after_back_props) if t_after_back_llh else 0.0,
            "total": (t_after_back_llh - t_start) if t_after_back_llh else 0.0,
        }

    if return_diagnostics:
        diagnostics = {
            "props": props,
            "back_props": back_props,
            "back_candidates": back_candidates,
            "y_star": y_star,
            "log_w": log_w,
            "weights": weights,
            "idx": int(idx),
            "idx_is_max": idx_is_max,
            "log_w_back": log_w_back,
            "log_alpha": float(log_alpha),
            "accept_u": float(accept_u),
            "log_u": float(log_u),
            "accepted": accepted,
        }
        if timing is not None:
            diagnostics["timing"] = timing
        if return_idx:
            return x_new, int(idx), diagnostics
        return x_new, diagnostics

    if return_idx:
        return x_new, int(idx)
    return x_new, accepted


def mtpcn_chain(
    x0,
    problem,
    rng,
    n_iters,
    rho=0.9,
    n_props=10,
    return_indices=False,
    return_diagnostics=False,
    diag_indices=None,
    collect_timing=False,
):
    """Run a multiple-try pCN chain for n_iters steps."""
    chain = np.zeros((n_iters + 1, problem.dim), dtype=float)
    chain[0] = x0
    x = x0
    n_acc = 0
    selected_index = np.zeros(n_iters, dtype=int) if return_indices else None
    diagnostics = [] if return_diagnostics else None
    diag_set = None
    if return_diagnostics and diag_indices is not None:
        diag_set = set(int(i) for i in diag_indices)
    for t in range(n_iters):
        capture_diag = return_diagnostics and (diag_set is None or t in diag_set)
        if return_indices or capture_diag:
            if capture_diag:
                x_new, idx, diag = mtpcn_step(
                    x,
                    problem,
                    rng,
                    rho=rho,
                    n_props=n_props,
                    return_idx=True,
                    return_diagnostics=True,
                    collect_timing=collect_timing,
                )
                diagnostics.append({"iter": t, "x": x.copy(), **diag})
                accepted = bool(diag["accepted"])
            else:
                x_new, idx = mtpcn_step(
                    x,
                    problem,
                    rng,
                    rho=rho,
                    n_props=n_props,
                    return_idx=True,
                )
                accepted = not np.allclose(x_new, x)
            if return_indices:
                selected_index[t] = int(idx)
        else:
            x_new, accepted = mtpcn_step(x, problem, rng, rho=rho, n_props=n_props)
        n_acc += int(accepted)
        x = x_new
        chain[t + 1] = x

    accept_rate = n_acc / n_iters
    if return_diagnostics and return_indices:
        return chain, accept_rate, selected_index, diagnostics
    if return_diagnostics:
        return chain, accept_rate, diagnostics
    if return_indices:
        return chain, accept_rate, selected_index
    return chain, accept_rate
