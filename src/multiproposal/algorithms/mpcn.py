# Multi-proposal pCN sampler

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat

import numpy as np


def _resolve_n_jobs(n_jobs, n_props, core_frac):
    if n_jobs is None or n_jobs <= 0:
        cores = os.cpu_count() or 1
        max_workers = int(np.floor(core_frac * cores))
        max_workers = max(1, max_workers)
        return min(n_props, max_workers)
    return min(n_props, int(n_jobs))


def _evaluate_log_likelihoods(problem, props, executor=None):
    if executor is None:
        return np.array([problem.log_likelihood(p) for p in props])
    return np.array(list(executor.map(problem.log_likelihood, props)))


def _evaluate_log_likelihoods_chunk(problem, props_chunk):
    return np.array([problem.log_likelihood(p) for p in props_chunk])


def _chunk_slices(n_items, chunk_size):
    for start in range(0, n_items, chunk_size):
        stop = min(n_items, start + chunk_size)
        yield slice(start, stop)


def _evaluate_log_likelihoods_parallel(problem, props, executor, chunk_size):
    if chunk_size is None or chunk_size <= 0 or chunk_size >= len(props):
        return np.array(list(executor.map(problem.log_likelihood, props)))
    chunks = [props[slc] for slc in _chunk_slices(len(props), chunk_size)]
    results = list(executor.map(_evaluate_log_likelihoods_chunk, repeat(problem), chunks))
    if not results:
        return np.array([], dtype=float)
    return np.concatenate(results, axis=0)


def _spawn_seeds(rng, n_seeds):
    seed = int(rng.integers(0, 2**32, dtype=np.uint32))
    seed_seq = np.random.SeedSequence(seed)
    return seed_seq.spawn(n_seeds)


def _propose_and_log_likelihood(seed_seq, problem, mu, x_center, rho, eta):
    rng_local = np.random.default_rng(seed_seq)
    z = rng_local.standard_normal(problem.dim)
    prop = mu + rho * (x_center - mu) + eta * problem.L @ z
    return prop, problem.log_likelihood(prop)


def mpcn_step(
    x,
    problem,
    rng,
    rho=0.9,
    n_props=10,
    return_idx=False,
    return_diagnostics=False,
    executor=None,
    parallelize_props=False,
    collect_timing=False,
    llh_chunk_size=None,
):
    """Run one mpCN step."""
    t_start = time.perf_counter() if collect_timing else None
    mu = problem.prior_mean()
    eta = np.sqrt(1.0 - rho * rho)

    nu_c = problem.L @ rng.standard_normal(problem.dim)
    x_center = mu + rho * (x - mu) + eta * nu_c
    t_after_center = time.perf_counter() if collect_timing else None

    if executor is not None and parallelize_props:
        seeds = _spawn_seeds(rng, n_props)
        results = list(
            executor.map(
                _propose_and_log_likelihood,
                seeds,
                repeat(problem),
                repeat(mu),
                repeat(x_center),
                repeat(rho),
                repeat(eta),
            )
        )
        props = np.array([item[0] for item in results])
        log_w = np.empty(n_props + 1, dtype=float)
        log_w[0] = problem.log_likelihood(x)
        log_w[1:] = np.array([item[1] for item in results])
        t_after_props = time.perf_counter() if collect_timing else None
        t_after_llh = t_after_props
    else:
        z = rng.standard_normal((problem.dim, n_props))
        props = (mu[:, None] + rho * (x_center - mu)[:, None] + eta * problem.L @ z).T
        t_after_props = time.perf_counter() if collect_timing else None
        log_w = np.empty(n_props + 1, dtype=float)
        log_w[0] = problem.log_likelihood(x)
        if executor is None:
            log_w[1:] = _evaluate_log_likelihoods(problem, props, executor=None)
        else:
            log_w[1:] = _evaluate_log_likelihoods_parallel(
                problem,
                props,
                executor,
                llh_chunk_size,
            )
        t_after_llh = time.perf_counter() if collect_timing else None

    candidates = np.vstack([x[None, :], props])
    log_w -= np.max(log_w)
    weights = np.exp(log_w)
    t_after_weights = time.perf_counter() if collect_timing else None

    idx = rng.choice(n_props + 1, p=weights / weights.sum())
    t_after_choice = time.perf_counter() if collect_timing else None

    timing = None
    if collect_timing and t_start is not None:
        if executor is not None and parallelize_props:
            props_time = (t_after_props - t_after_center) if t_after_props else 0.0
            llh_time = 0.0
        else:
            props_time = (t_after_props - t_after_center) if t_after_props else 0.0
            llh_time = (t_after_llh - t_after_props) if t_after_llh else 0.0
        timing = {
            "center": (t_after_center - t_start) if t_after_center else 0.0,
            "props": props_time,
            "llh": llh_time,
            "weights": (t_after_weights - t_after_llh) if t_after_weights else 0.0,
            "choice": (t_after_choice - t_after_weights) if t_after_choice else 0.0,
            "total": (t_after_choice - t_start) if t_after_choice else 0.0,
            "parallel_props": bool(executor is not None and parallelize_props),
        }

    if return_diagnostics:
        diagnostics = {
            "x_center": x_center,
            "nu_c": nu_c,
            "props": props,
            "candidates": candidates,
            "log_w": log_w,
            "weights": weights,
        }
        if timing is not None:
            diagnostics["timing"] = timing
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
    n_jobs=1,
    core_frac=0.7,
    parallel_backend="auto",
    return_indices=False,
    return_diagnostics=False,
    diag_indices=None,
    parallelize_props=False,
    collect_timing=False,
    llh_chunk_size=None,
):
    """Run one mpCN chain for n_iters steps."""
    chain = np.zeros((n_iters + 1, problem.dim), dtype=float)
    accepted_index = np.zeros(n_iters, dtype=int) if return_indices else None
    diagnostics = [] if return_diagnostics else None
    diag_set = None
    if return_diagnostics and diag_indices is not None:
        diag_set = set(int(i) for i in diag_indices)
    chain[0] = x0
    x = x0

    n_jobs_resolved = _resolve_n_jobs(n_jobs, n_props, core_frac)
    use_parallel = n_jobs_resolved > 1
    if parallel_backend in {"auto", "thread", "threads"}:
        executor_factory = ThreadPoolExecutor
    elif parallel_backend in {"process", "processes"}:
        executor_factory = ProcessPoolExecutor
    else:
        raise ValueError("parallel_backend must be 'auto', 'thread(s)', or 'process(es)'")

    if use_parallel:
        with executor_factory(max_workers=n_jobs_resolved) as executor:
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
                            executor=executor,
                            parallelize_props=parallelize_props,
                            collect_timing=collect_timing,
                            llh_chunk_size=llh_chunk_size,
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
                        x_new, idx = mpcn_step(
                            x,
                            problem,
                            rng,
                            rho=rho,
                            n_props=n_props,
                            return_idx=True,
                            executor=executor,
                            parallelize_props=parallelize_props,
                            llh_chunk_size=llh_chunk_size,
                        )
                    if return_indices:
                        accepted_index[t] = idx
                else:
                    x_new = mpcn_step(
                        x,
                        problem,
                        rng,
                        rho=rho,
                        n_props=n_props,
                        executor=executor,
                        parallelize_props=parallelize_props,
                        llh_chunk_size=llh_chunk_size,
                    )
                x = x_new
                chain[t + 1] = x
    else:
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
                        parallelize_props=parallelize_props,
                        collect_timing=collect_timing,
                        llh_chunk_size=llh_chunk_size,
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
                    x_new, idx = mpcn_step(
                        x,
                        problem,
                        rng,
                        rho=rho,
                        n_props=n_props,
                        return_idx=True,
                        parallelize_props=parallelize_props,
                        llh_chunk_size=llh_chunk_size,
                    )
                if return_indices:
                    accepted_index[t] = idx
            else:
                x_new = mpcn_step(
                    x,
                    problem,
                    rng,
                    rho=rho,
                    n_props=n_props,
                    parallelize_props=parallelize_props,
                    llh_chunk_size=llh_chunk_size,
                )
            x = x_new
            chain[t + 1] = x
    if return_diagnostics and return_indices:
        return chain, accepted_index, diagnostics
    if return_diagnostics:
        return chain, diagnostics
    if return_indices:
        return chain, accepted_index
    return chain
