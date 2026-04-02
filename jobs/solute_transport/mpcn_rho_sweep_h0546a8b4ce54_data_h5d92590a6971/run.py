import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from multiproposal.algorithms.effective_sample_size import (
    estimate_effective_sample_size,
    summarize_squared_jumping_distance,
)
from multiproposal.algorithms.mess import mess_step
from multiproposal.algorithms.mpcn import mpcn_chain
from multiproposal.algorithms.pcn import pcn_chain
from multiproposal.problems.advection_diffusion import (
    AdvectionDiffusionToy,
    make_Astar_from_atrue,
    make_Astar_nn,
    make_omegas_power,
    params_from_skew,
    prior_diag_from_powerlaw,
    solve_theta,
)
from multiproposal.utils.run_paths import format_float_tag


def _resolve_repo_root():
    env_root = os.environ.get("MULTIPROPOSAL_RUN_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    root = Path.cwd().resolve()
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    return root


def _canonicalize_payload(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: _canonicalize_payload(val) for key, val in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize_payload(val) for val in obj]
    return obj


def _stable_hash(payload, length=12):
    data = json.dumps(
        _canonicalize_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def _parse_int_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_float_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def compute_msjd_per_param(chain):
    if chain.shape[0] < 2:
        return np.zeros(chain.shape[1])
    jumps = np.diff(chain, axis=0)
    msjd = np.mean(jumps * jumps, axis=0)
    return msjd


def compute_ess_per_param(chain, max_lag):
    if chain.shape[0] < 2:
        return np.zeros(chain.shape[1])
    variances = np.var(chain, axis=0)
    if np.all(variances == 0):
        return np.zeros(chain.shape[1])
    ess_vals = estimate_effective_sample_size(chain, max_lag=max_lag)
    ess_vals = np.asarray(ess_vals, dtype=float)
    ess_vals[variances == 0] = 0.0
    return ess_vals


def summarize_chain_metrics(chain, runtime_sec, burn_in, max_lag, include_msjd_distribution=False, parallel_info=None):
    post = chain[burn_in:]
    ess_vals = compute_ess_per_param(post, max_lag=max_lag)
    msjd_vals = compute_msjd_per_param(post)
    msjd_summary = summarize_squared_jumping_distance(
        post,
        percentiles=(0.5, 2.5, 50.0, 97.5, 99.5),
        return_distribution=include_msjd_distribution,
    )
    ess_mean = float(np.nanmean(ess_vals)) if ess_vals.size else 0.0
    msjd_mean = float(np.nanmean(msjd_vals)) if msjd_vals.size else 0.0
    runtime_min = runtime_sec / 60.0
    ess_per_min = ess_mean / runtime_min if runtime_min > 0 else np.nan
    return {
        "runtime_sec": runtime_sec,
        "runtime_min": runtime_min,
        "ess_mean": ess_mean,
        "msjd_mean": msjd_mean,
        "msjd_summary": msjd_summary,
        "ess_per_min": ess_per_min,
        "ess_per_param": ess_vals.tolist(),
        "msjd_per_param": msjd_vals.tolist(),
        "parallel": parallel_info,
    }


def rho_to_tag(rho):
    return format_float_tag(rho)


def chain_cache_paths(estimations_dir, method, rho, seed_base, P=None):
    rho_tag = rho_to_tag(rho)
    if P is None:
        stem = f"{method}_rho{rho_tag}_seed{seed_base}"
    else:
        stem = f"{method}_P{P}_rho{rho_tag}_seed{seed_base}"
    chains_dir = estimations_dir / "chains"
    samples_path = chains_dir / f"{stem}.npz"
    metrics_path = chains_dir / f"{stem}_metrics.json"
    return samples_path, metrics_path


def mess_cache_paths(estimations_dir, method, P, seed_base):
    stem = f"{method}_M{P}_seed{seed_base}"
    chains_dir = estimations_dir / "chains"
    samples_path = chains_dir / f"{stem}.npz"
    metrics_path = chains_dir / f"{stem}_metrics.json"
    return samples_path, metrics_path


def mpcn_diag_path(estimations_dir, P, rho, seed_base):
    rho_tag = rho_to_tag(rho)
    diag_dir = estimations_dir / "diagnostics"
    return diag_dir / f"mpcn_P{P}_rho{rho_tag}_seed{seed_base}_diag.npz"


def independent_pcn_dir(estimations_dir):
    return estimations_dir / "chains" / "independent_chains"


def independent_pcn_index_path(estimations_dir, P, rho, seed_base):
    rho_tag = rho_to_tag(rho)
    chains_dir = independent_pcn_dir(estimations_dir)
    return chains_dir / f"pcn_independent_P{P}_rho{rho_tag}_seed{seed_base}_index.json"


def independent_pcn_chain_path(estimations_dir, P, rho, seed_base, chain_idx):
    rho_tag = rho_to_tag(rho)
    chains_dir = independent_pcn_dir(estimations_dir)
    stem = f"pcn_independent_P{P}_rho{rho_tag}_seed{seed_base}_chain{chain_idx:04d}"
    return chains_dir / f"{stem}.npz"


def load_independent_index(index_path):
    if not index_path.exists():
        return None
    with open(index_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_independent_index(index_path, payload):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_metrics_json(metrics_path, metrics, accept_rate, runtime_sec):
    payload = dict(metrics)
    payload["accept_rate"] = None if accept_rate is None else float(accept_rate)
    payload["runtime_sec"] = float(runtime_sec)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_metrics_json(metrics_path):
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_chain_bundle(samples_path, metrics_path, chain, accept_rate, runtime_sec, metrics):
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    accept_val = np.nan if accept_rate is None else float(accept_rate)
    np.savez_compressed(
        samples_path,
        chain=chain,
        accept_rate=accept_val,
        runtime_sec=float(runtime_sec),
    )
    save_metrics_json(metrics_path, metrics, accept_rate, runtime_sec)


def load_chain_bundle(samples_path, metrics_path):
    if not samples_path.exists():
        return None
    data = np.load(samples_path, allow_pickle=False)
    chain = data["chain"]
    accept_rate = float(data["accept_rate"]) if "accept_rate" in data else np.nan
    if np.isnan(accept_rate):
        accept_rate = None
    runtime_sec = float(data["runtime_sec"]) if "runtime_sec" in data else 0.0
    metrics = load_metrics_json(metrics_path)
    return chain, accept_rate, runtime_sec, metrics


def save_mpcn_diagnostics(diag_path, snapshot, mean_dist_samples, mean_sq_dist_samples):
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_payload = np.array(snapshot, dtype=object) if snapshot is not None else None
    np.savez_compressed(
        diag_path,
        snapshot=snapshot_payload,
        mean_dist_samples=np.asarray(mean_dist_samples, dtype=float),
        mean_sq_dist_samples=np.asarray(mean_sq_dist_samples, dtype=float),
    )


def load_mpcn_diagnostics(diag_path):
    if not diag_path.exists():
        return None, [], []
    data = np.load(diag_path, allow_pickle=True)
    snapshot = data["snapshot"].item() if "snapshot" in data and data["snapshot"] is not None else None
    mean_dist_samples = data.get("mean_dist_samples", [])
    mean_sq_dist_samples = data.get("mean_sq_dist_samples", [])
    return snapshot, mean_dist_samples, mean_sq_dist_samples


def _write_progress(progress_path, payload):
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_partial_chain(samples_path, chain, accept_rate, runtime_sec, n_iters_completed, n_iters_total):
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        samples_path,
        chain=chain,
        accept_rate=float(accept_rate),
        runtime_sec=float(runtime_sec),
        n_iters_completed=int(n_iters_completed),
        n_iters_total=int(n_iters_total),
    )


def run_mpcn_chain(
    problem,
    x0,
    n_iters,
    rho,
    n_props,
    seed,
    diag_indices=None,
    parallel_backend="process",
    n_jobs=1,
    parallelize_props=False,
    llh_chunk_size=None,
):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    return_diag = diag_indices is not None
    if return_diag:
        chain, accepted_index, diagnostics = mpcn_chain(
            x0,
            problem,
            rng,
            n_iters,
            rho=rho,
            n_props=n_props,
            return_indices=True,
            return_diagnostics=True,
            diag_indices=diag_indices,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallelize_props=parallelize_props,
            llh_chunk_size=llh_chunk_size,
        )
    else:
        chain, accepted_index = mpcn_chain(
            x0,
            problem,
            rng,
            n_iters,
            rho=rho,
            n_props=n_props,
            return_indices=True,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallelize_props=parallelize_props,
            llh_chunk_size=llh_chunk_size,
        )
        diagnostics = None
    runtime_sec = time.perf_counter() - t0
    accept_rate = float(np.mean(accepted_index != 0))
    return chain, runtime_sec, accept_rate, diagnostics


def run_mpcn_chain_with_checkpoints(
    problem,
    x0,
    n_iters,
    rho,
    n_props,
    seed,
    diag_indices=None,
    parallel_backend="process",
    n_jobs=1,
    parallelize_props=False,
    llh_chunk_size=None,
    checkpoint_interval=10000,
    progress_path=None,
    partial_samples_path=None,
    progress_payload_base=None,
):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    diag_set = set(int(i) for i in diag_indices) if diag_indices is not None else None
    chain_blocks = [x0[None, :]]
    accepted_blocks = []
    diagnostics = [] if diag_set is not None else None
    iter_completed = 0
    x = x0.copy()

    while iter_completed < n_iters:
        block_iters = min(checkpoint_interval, n_iters - iter_completed)
        block_diag_indices = None
        block_diags = None
        if diag_set is not None:
            block_diag_indices = [
                i - iter_completed
                for i in diag_set
                if iter_completed <= i < iter_completed + block_iters
            ]

        if block_diag_indices:
            chain_block, accepted_block, block_diags = mpcn_chain(
                x,
                problem,
                rng,
                block_iters,
                rho=rho,
                n_props=n_props,
                return_indices=True,
                return_diagnostics=True,
                diag_indices=block_diag_indices,
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
                parallelize_props=parallelize_props,
                llh_chunk_size=llh_chunk_size,
            )
        else:
            chain_block, accepted_block = mpcn_chain(
                x,
                problem,
                rng,
                block_iters,
                rho=rho,
                n_props=n_props,
                return_indices=True,
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
                parallelize_props=parallelize_props,
                llh_chunk_size=llh_chunk_size,
            )

        if block_diags:
            for diag in block_diags:
                diag["iter"] = int(diag["iter"]) + iter_completed
                diagnostics.append(diag)

        chain_blocks.append(chain_block[1:])
        accepted_blocks.append(accepted_block)
        iter_completed += block_iters
        x = chain_block[-1]

        runtime_sec = time.perf_counter() - t0
        if accepted_blocks:
            accept_rate = float(np.mean(np.concatenate(accepted_blocks) != 0))
        else:
            accept_rate = 0.0

        if partial_samples_path is not None:
            _write_partial_chain(
                partial_samples_path,
                np.vstack(chain_blocks),
                accept_rate,
                runtime_sec,
                iter_completed,
                n_iters,
            )

        if progress_path is not None:
            payload = dict(progress_payload_base or {})
            payload.update(
                {
                    "n_iters": int(n_iters),
                    "completed_iters": int(iter_completed),
                    "percent_complete": float(iter_completed / n_iters),
                    "runtime_sec": float(runtime_sec),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            _write_progress(progress_path, payload)

    chain = np.vstack(chain_blocks)
    accepted_index = np.concatenate(accepted_blocks) if accepted_blocks else np.array([], dtype=int)
    runtime_sec = time.perf_counter() - t0
    accept_rate = float(np.mean(accepted_index != 0)) if accepted_index.size else 0.0
    return chain, runtime_sec, accept_rate, diagnostics


def run_mess_chain(problem, x0, n_iters, M, seed, use_lp=False, distance_metric="angular", lam=0.0):
    rng = np.random.default_rng(seed)
    chain = np.zeros((n_iters + 1, problem.dim), dtype=float)
    chain[0] = x0
    x = x0.copy()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        x, _, _ = mess_step(
            x,
            problem,
            rng,
            M=M,
            use_lp=use_lp,
            distance_metric=distance_metric,
            lam=lam,
        )
        chain[_ + 1] = x
    runtime_sec = time.perf_counter() - t0
    return chain, runtime_sec


def get_obs_indices(dim_value, highest_freq, bandwidth):
    highest_freq = min(highest_freq, dim_value)
    bandwidth = min(bandwidth, dim_value)
    start = max(0, highest_freq - bandwidth + 1)
    return np.arange(start, highest_freq + 1, dtype=int)


def get_param_indices_for_dim(dim, shared_draws):
    cache = shared_draws.setdefault("param_indices_cache", {})
    if dim not in cache:
        iju = shared_draws["param_iju"]
        mask = (iju[0] < dim) & (iju[1] < dim)
        cache[dim] = np.nonzero(mask)[0]
    return cache[dim]


def build_shared_draws(
    d_max,
    kappa,
    sigma,
    alpha,
    gamma,
    tau2,
    offset,
    a_mode,
    seed,
):
    rng = np.random.default_rng(seed)
    m_max = d_max * (d_max - 1) // 2
    prior_diag_max = prior_diag_from_powerlaw(
        d_max, alpha=alpha, gamma=gamma, tau2=tau2, offset=offset
    )
    if prior_diag_max.shape != (m_max,):
        raise ValueError(f"prior_diag_max must have shape ({m_max},), got {prior_diag_max.shape}")
    if a_mode == "nearest_neighbor":
        omegas = make_omegas_power(d_max, beta=alpha, c=2.0 ** (-gamma), offset=offset)
        A_true_max = make_Astar_nn(d_max, omegas)
        a_true_max = params_from_skew(A_true_max)
    elif a_mode == "prior":
        z_prior = rng.standard_normal(m_max)
        a_true_max = z_prior * np.sqrt(prior_diag_max)
        A_true_max = make_Astar_from_atrue(d_max, a_true_max)
    else:
        raise ValueError("a_mode must be 'nearest_neighbor' or 'prior'")
    g_max = np.zeros(d_max, dtype=float)
    g_max[0] = 1.0
    theta_true_max = solve_theta(d_max, a_true_max, g_max, kappa)
    noise_max = rng.standard_normal(d_max)
    z_init = rng.standard_normal(m_max)
    a_init_max = z_init * np.sqrt(prior_diag_max)
    return {
        "d_max": d_max,
        "m_max": m_max,
        "kappa": kappa,
        "sigma": sigma,
        "alpha": alpha,
        "gamma": gamma,
        "tau2": tau2,
        "offset": offset,
        "a_mode": a_mode,
        "param_iju": np.triu_indices(d_max, k=1),
        "param_indices_cache": {},
        "prior_diag": prior_diag_max,
        "a_true": a_true_max,
        "A_true": A_true_max,
        "g": g_max,
        "theta_true": theta_true_max,
        "noise": noise_max,
        "a_init": a_init_max,
    }


def generate_advection_diffusion_data_shared(dim, obs_indices, shared_draws):
    a_mode_local = shared_draws["a_mode"]
    param_idx = get_param_indices_for_dim(dim, shared_draws)
    prior_diag = shared_draws["prior_diag"][param_idx]
    g = shared_draws["g"][:dim]
    if a_mode_local == "nearest_neighbor":
        omegas = make_omegas_power(
            dim,
            beta=shared_draws["alpha"],
            c=2.0 ** (-shared_draws["gamma"]),
            offset=shared_draws["offset"],
        )
        A_true = make_Astar_nn(dim, omegas)
        a_true = params_from_skew(A_true)
        theta_true = solve_theta(dim, a_true, g, shared_draws["kappa"])
    elif a_mode_local == "prior":
        a_true = shared_draws["a_true"][param_idx]
        A_true = make_Astar_from_atrue(dim, a_true)
        theta_true = shared_draws["theta_true"][:dim]
    else:
        raise ValueError("a_mode must be 'nearest_neighbor' or 'prior'")
    noise = shared_draws["noise"][:dim]
    y = theta_true[obs_indices] + shared_draws["sigma"] * noise[obs_indices]
    a_init = shared_draws["a_init"][param_idx]
    return {
        "dim": dim,
        "kappa": shared_draws["kappa"],
        "alpha": shared_draws["alpha"],
        "gamma": shared_draws["gamma"],
        "tau2": shared_draws["tau2"],
        "sigma": shared_draws["sigma"],
        "obs_indices": obs_indices,
        "prior_diag": prior_diag,
        "a_true": a_true,
        "A_true": A_true,
        "g": g,
        "theta_true": theta_true,
        "y": y,
        "a_init": a_init,
    }


def build_problem_for_dim(dim, shared_draws, obs_highest_freq, obs_bandwidth):
    obs_indices = get_obs_indices(dim, obs_highest_freq, obs_bandwidth)
    data = generate_advection_diffusion_data_shared(dim, obs_indices, shared_draws)
    problem = AdvectionDiffusionToy(
        dim=dim,
        kappa=shared_draws["kappa"],
        sigma=shared_draws["sigma"],
        y=data["y"],
        obs_indices=obs_indices,
        g=data["g"],
        prior_diag=data["prior_diag"],
    )
    return problem, data["a_init"], data


def main():
    parser = argparse.ArgumentParser(
        description="Run solute transport mPCN rho sweep and cache chains/metrics."
    )
    parser.add_argument("--P-list", type=str, default=None)
    parser.add_argument("--rho-list", type=str, default=None)
    parser.add_argument("--grid-count", type=int, default=1)
    parser.add_argument("--grid-index", type=int, default=0)
    parser.add_argument("--refresh-metrics-only", action="store_true")
    parser.add_argument("--skip-pcn", action="store_true")
    parser.add_argument("--skip-mess", action="store_true")
    parser.add_argument("--skip-mpcn", action="store_true")
    parser.add_argument("--skip-independent-pcn", action="store_true")
    parser.add_argument("--independent-pcn-count", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    args = parser.parse_args()

    repo_root = _resolve_repo_root()

    # Data configuration
    seed_data = 0
    seed_mcmc = 0

    d = 10
    kappa = 0.02
    sigma = 0.5
    alpha = 3.0
    gamma = 2.0
    tau2 = 2.0
    a_mode = "nearest_neighbor"
    use_prior_A = True
    shared_draws_seed = seed_data

    obs_highest_freq = 6
    obs_bandwidth = 3
    obs_config = "central_modes"

    # Sweep configuration
    n_iters = 300000
    P_list = [2000]
    M_list = P_list
    rho_list = [round(val, 3) for val in np.arange(0, 1.01, 0.05)]
    seed_base = 202
    run_pcn = True
    run_mess = False
    run_mpcn = True
    independent_pcn_count = 0
    run_independent_pcn = False

    # Parallel execution (mPCN)
    mpcn_parallel_n_jobs = 12
    mpcn_parallel_backend = "process"
    mpcn_parallelize_props = False
    mpcn_llh_chunk_size = 0

    # Metrics config
    max_lag = 5000
    n_diag_samples = 100
    burn_in = 5000
    checkpoint_interval = max(0, int(args.checkpoint_interval))

    # Optional overrides to reuse an existing run directory exactly.
    data_id_override = "data_h5d92590a6971"
    run_id_override = "mpcn_rho_sweep_h0546a8b4ce54"

    if args.P_list:
        P_list = _parse_int_list(args.P_list)
        M_list = list(P_list)
    if args.rho_list:
        rho_list = _parse_float_list(args.rho_list)

    run_pcn = run_pcn and not args.skip_pcn
    run_mess = run_mess and not args.skip_mess
    run_mpcn = run_mpcn and not args.skip_mpcn
    if args.independent_pcn_count is not None:
        independent_pcn_count = int(args.independent_pcn_count)
    run_independent_pcn = independent_pcn_count > 0
    run_independent_pcn = run_independent_pcn and not args.skip_independent_pcn

    if args.grid_count < 1:
        raise ValueError("grid-count must be >= 1")
    if args.grid_index < 0 or args.grid_index >= args.grid_count:
        raise ValueError("grid-index must be in [0, grid-count)")

    if isinstance(obs_config, dict):
        obs_config_serializable = {
            key: (val.tolist() if isinstance(val, np.ndarray) else val)
            for key, val in obs_config.items()
        }
    else:
        obs_config_serializable = obs_config
    data_id_config = {
        "seed_data": seed_data,
        "seed_mcmc": seed_mcmc,
        "kappa": kappa,
        "sigma": sigma,
        "alpha": alpha,
        "gamma": gamma,
        "tau2": tau2,
        "a_mode": a_mode,
        "use_prior_A": use_prior_A,
        "shared_draws_seed": shared_draws_seed,
        "obs_highest_freq": obs_highest_freq,
        "obs_bandwidth": obs_bandwidth,
        "obs_config": obs_config_serializable,
    }
    data_config = dict(data_id_config)
    data_config.update({"d": d})

    algo_config = {
        "n_iters": n_iters,
        "burn_in": burn_in,
        "max_lag": max_lag,
        "n_diag_samples": n_diag_samples,
    }
    execution_config = {
        "mpcn_parallel_n_jobs": mpcn_parallel_n_jobs,
        "mpcn_parallel_backend": mpcn_parallel_backend,
        "mpcn_parallelize_props": mpcn_parallelize_props,
        "mpcn_llh_chunk_size": mpcn_llh_chunk_size,
        "checkpoint_interval": checkpoint_interval,
        "independent_pcn": {
            "enabled": run_independent_pcn,
            "count": int(independent_pcn_count),
        },
    }
    sweep_config = {
        "P_list": P_list,
        "M_list": M_list,
        "rho_list": rho_list,
        "seed_base": seed_base,
        "run_pcn": run_pcn,
        "run_mess": run_mess,
    }

    data_id = data_id_override or f"data_h{_stable_hash(data_id_config)}"
    run_id = run_id_override or f"mpcn_rho_sweep_h{_stable_hash({
        "algorithm": "mpcn_rho_sweep",
        "algorithm_config": algo_config,
        "sweep": sweep_config,
    })}"
    estimations_dir = repo_root / "estimations" / "solute_transport" / data_id / "sweep" / run_id
    reports_dir = repo_root / "reports" / "solute_transport" / data_id / "sweep" / run_id
    for path in (estimations_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    run_config = {
        "dataset": "solute_transport",
        "algorithm": "mpcn_rho_sweep",
        "data": data_config,
        "algorithm_config": algo_config,
        "execution": execution_config,
        "sweep": sweep_config,
    }
    config_path = estimations_dir / "config.json"
    if not config_path.exists():
        payload = dict(run_config)
        payload["data_id"] = data_id
        payload["run_id"] = run_id
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    print("P_list:", P_list)
    print("rho_list:", rho_list)
    print("n_iters:", n_iters)
    print("n_diag_samples:", n_diag_samples)
    print("run_pcn:", run_pcn)
    print("run_mess:", run_mess)
    print("run_mpcn:", run_mpcn)
    print("run_independent_pcn:", run_independent_pcn)
    print("independent_pcn_count:", independent_pcn_count)
    print("mpcn_parallel_backend:", mpcn_parallel_backend)
    print("mpcn_parallel_n_jobs:", mpcn_parallel_n_jobs)
    print("mpcn_parallelize_props:", mpcn_parallelize_props)
    print("mpcn_llh_chunk_size:", mpcn_llh_chunk_size)
    print("checkpoint_interval:", checkpoint_interval)
    print("data_id:", data_id)
    print("run_id:", run_id)
    print("Run directory:", estimations_dir)

    shared_draws = build_shared_draws(
        d_max=d,
        kappa=kappa,
        sigma=sigma,
        alpha=alpha,
        gamma=gamma,
        tau2=tau2,
        offset=1.0,
        a_mode="prior" if use_prior_A else a_mode,
        seed=shared_draws_seed,
    )
    problem, a_init, _ = build_problem_for_dim(
        d, shared_draws, obs_highest_freq=obs_highest_freq, obs_bandwidth=obs_bandwidth
    )
    x0 = a_init.copy()

    if run_pcn:
        for rho in rho_list:
            samples_path, metrics_path = chain_cache_paths(
                estimations_dir, "pcn", rho=rho, seed_base=seed_base
            )
            loaded = load_chain_bundle(samples_path, metrics_path)
            if loaded is not None:
                chain, accept_rate, runtime_sec, metrics = loaded
                if args.refresh_metrics_only or metrics is None or "ess_per_param" not in metrics:
                    metrics = summarize_chain_metrics(
                        chain, runtime_sec, burn_in=burn_in, max_lag=max_lag
                    )
                    save_metrics_json(metrics_path, metrics, accept_rate, runtime_sec)
                accept_display = np.nan if accept_rate is None else accept_rate
                print(
                    f"pCN loaded: rho={rho:.3f}, accept={accept_display:.3f}, runtime={runtime_sec:.2f}s"
                )
                continue
            if args.refresh_metrics_only:
                print(f"pCN missing chain: rho={rho:.3f} (skipping, refresh_metrics_only=True)")
                continue
            seed = seed_base + int(round(rho * 100))
            rng = np.random.default_rng(seed)
            t0 = time.perf_counter()
            chain, accept_rate = pcn_chain(
                x0, problem, rng, n_iters, rho=rho, return_acceptance=True
            )
            runtime_sec = time.perf_counter() - t0
            metrics = summarize_chain_metrics(chain, runtime_sec, burn_in=burn_in, max_lag=max_lag)
            save_chain_bundle(samples_path, metrics_path, chain, accept_rate, runtime_sec, metrics)
            print(
                f"pCN done: rho={rho:.3f}, accept={accept_rate:.3f}, runtime={runtime_sec:.2f}s"
            )
    else:
        print("pCN disabled (run_pcn=False).")

    if run_independent_pcn:
        independent_P = int(independent_pcn_count)
        if independent_P < 1:
            raise ValueError("independent-pcn-count must be >= 1")
        rho_grid = rho_list[args.grid_index:: args.grid_count]
        for rho in rho_grid:
            index_path = independent_pcn_index_path(
                estimations_dir, independent_P, rho=rho, seed_base=seed_base
            )
            index_payload = load_independent_index(index_path)
            if index_payload is None:
                index_payload = {
                    "metadata": {
                        "P": int(independent_P),
                        "rho": float(rho),
                        "seed_base": int(seed_base),
                        "n_iters": int(n_iters),
                        "dim": int(problem.dim),
                    },
                    "chains": [],
                }
            chains = index_payload.get("chains", [])
            index_payload["chains"] = chains
            completed = len(chains)
            if completed >= independent_P:
                print(
                    f"pCN independent loaded: rho={rho:.3f}, "
                    f"P={independent_P}, completed={completed}"
                )
                continue

            rho_seed = int(round(rho * 1000))
            for idx in range(completed, independent_P):
                seed = seed_base + rho_seed * 10000 + idx
                chain_path = independent_pcn_chain_path(
                    estimations_dir, independent_P, rho=rho, seed_base=seed_base, chain_idx=idx
                )
                if chain_path.exists():
                    chains.append(
                        {
                            "chain_idx": int(idx),
                            "file": chain_path.name,
                            "seed": int(seed),
                            "accept_rate": None,
                            "runtime_sec": None,
                        }
                    )
                    save_independent_index(index_path, index_payload)
                    continue

                rng_chain = np.random.default_rng(seed)
                x0_chain = problem.sample_prior(rng_chain)
                t0 = time.perf_counter()
                chain, accept_rate = pcn_chain(
                    x0_chain, problem, rng_chain, n_iters, rho=rho, return_acceptance=True
                )
                runtime_sec = time.perf_counter() - t0
                chain_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    chain_path,
                    chain=chain,
                    accept_rate=float(accept_rate),
                    runtime_sec=float(runtime_sec),
                    seed=int(seed),
                    rho=float(rho),
                    n_iters=int(n_iters),
                    chain_idx=int(idx),
                )
                chains.append(
                    {
                        "chain_idx": int(idx),
                        "file": chain_path.name,
                        "seed": int(seed),
                        "accept_rate": float(accept_rate),
                        "runtime_sec": float(runtime_sec),
                    }
                )
                save_independent_index(index_path, index_payload)
                print(
                    f"pCN independent chain {idx + 1}/{independent_P} done: "
                    f"rho={rho:.3f}, accept={accept_rate:.3f}, runtime={runtime_sec:.2f}s"
                )
            print(
                f"pCN independent done: rho={rho:.3f}, P={independent_P}, "
                f"completed={len(chains)}"
            )
    else:
        print("pCN independent disabled (run_independent_pcn=False).")

    if run_mpcn:
        grid = [(P, float(rho)) for P in P_list for rho in rho_list]
        grid = grid[args.grid_index:: args.grid_count]

        for P, rho in grid:
            if mpcn_llh_chunk_size and mpcn_llh_chunk_size > 0:
                llh_chunk_size = int(mpcn_llh_chunk_size)
            else:
                llh_chunk_size = max(1, int(np.ceil(P / max(1, mpcn_parallel_n_jobs))))
            parallel_info = {
                "backend": mpcn_parallel_backend,
                "n_jobs": mpcn_parallel_n_jobs,
                "parallelize_props": mpcn_parallelize_props,
                "llh_chunk_size": llh_chunk_size,
            }

            samples_path, metrics_path = chain_cache_paths(
                estimations_dir, "mpcn", rho=rho, seed_base=seed_base, P=P
            )
            diag_path = mpcn_diag_path(estimations_dir, P, rho, seed_base)
            loaded = load_chain_bundle(samples_path, metrics_path)
            if loaded is not None:
                chain, accept_rate, runtime_sec, metrics = loaded
                if args.refresh_metrics_only or metrics is None or "ess_per_param" not in metrics:
                    metrics = summarize_chain_metrics(
                        chain,
                        runtime_sec,
                        burn_in=burn_in,
                        max_lag=max_lag,
                        parallel_info=parallel_info,
                    )
                    save_metrics_json(metrics_path, metrics, accept_rate, runtime_sec)
                accept_display = np.nan if accept_rate is None else accept_rate
                print(
                    f"mPCN loaded: P={P}, rho={rho:.3f}, accept={accept_display:.3f}, runtime={runtime_sec:.2f}s"
                )
                continue
            if args.refresh_metrics_only:
                print(
                    f"mPCN missing chain: P={P}, rho={rho:.3f} (skipping, refresh_metrics_only=True)"
                )
                continue

            seed = seed_base + int(P * 1000 + round(rho * 100))
            rng_diag = np.random.default_rng(seed)
            diag_pool = np.arange(burn_in, n_iters)
            replace = n_diag_samples > diag_pool.size
            diag_indices = rng_diag.choice(diag_pool, size=n_diag_samples, replace=replace)

            progress_path = samples_path.with_suffix(".progress.json")
            partial_samples_path = samples_path.with_name(f"{samples_path.stem}_partial.npz")
            progress_payload_base = {
                "dataset": "solute_transport",
                "run_id": run_id,
                "data_id": data_id,
                "P": int(P),
                "rho": float(rho),
                "seed": int(seed),
            }

            if checkpoint_interval > 0:
                chain, runtime_sec, accept_rate, diagnostics = run_mpcn_chain_with_checkpoints(
                    problem,
                    x0,
                    n_iters,
                    rho=rho,
                    n_props=P,
                    seed=seed,
                    diag_indices=diag_indices,
                    parallel_backend=mpcn_parallel_backend,
                    n_jobs=mpcn_parallel_n_jobs,
                    parallelize_props=mpcn_parallelize_props,
                    llh_chunk_size=llh_chunk_size,
                    checkpoint_interval=checkpoint_interval,
                    progress_path=progress_path,
                    partial_samples_path=partial_samples_path,
                    progress_payload_base=progress_payload_base,
                )
            else:
                chain, runtime_sec, accept_rate, diagnostics = run_mpcn_chain(
                    problem,
                    x0,
                    n_iters,
                    rho=rho,
                    n_props=P,
                    seed=seed,
                    diag_indices=diag_indices,
                    parallel_backend=mpcn_parallel_backend,
                    n_jobs=mpcn_parallel_n_jobs,
                    parallelize_props=mpcn_parallelize_props,
                    llh_chunk_size=llh_chunk_size,
                )
            metrics = summarize_chain_metrics(
                chain,
                runtime_sec,
                burn_in=burn_in,
                max_lag=max_lag,
                parallel_info=parallel_info,
            )

            mean_dist_samples = []
            mean_sq_dist_samples = []
            snapshot = None
            if diagnostics:
                for diag in diagnostics:
                    x_diag = diag["x"]
                    log_l0 = problem.log_likelihood(x_diag)
                    logy = log_l0
                    x_center = diag["x_center"]
                    nu_c = diag["nu_c"]
                    props = diag["props"]
                    candidates = diag["candidates"]
                    diff = props - x_center[None, :]
                    mean_dist = float(np.mean(np.linalg.norm(diff, axis=1)))
                    mean_sq_dist = float(np.mean(np.sum(diff * diff, axis=1)))
                    mean_dist_samples.append(mean_dist)
                    mean_sq_dist_samples.append(mean_sq_dist)
                    if snapshot is None:
                        snapshot = {
                            "iter": int(diag["iter"]),
                            "x": x_diag.copy(),
                            "x_center": x_center.copy(),
                            "nu_c": nu_c.copy(),
                            "props": props.copy(),
                            "candidates": candidates.copy(),
                            "logy": logy,
                            "accepted_idx": int(diag["accepted_idx"]),
                            "mean_dist": mean_dist,
                            "mean_sq_dist": mean_sq_dist,
                        }

            save_chain_bundle(samples_path, metrics_path, chain, accept_rate, runtime_sec, metrics)
            save_mpcn_diagnostics(diag_path, snapshot, mean_dist_samples, mean_sq_dist_samples)

            print(
                f"mPCN done: P={P}, rho={rho:.3f}, accept={accept_rate:.3f}, runtime={runtime_sec:.2f}s"
            )
    else:
        print("mPCN disabled (run_mpcn=False).")

    if run_mess:
        for P in P_list:
            seed_mess = seed_base + P
            samples_path, metrics_path = mess_cache_paths(
                estimations_dir, "mess_uniform", P=P, seed_base=seed_base
            )
            loaded = load_chain_bundle(samples_path, metrics_path)
            if loaded is not None:
                mess_chain, _, mess_runtime, mess_metrics = loaded
                if args.refresh_metrics_only or mess_metrics is None or "ess_per_param" not in mess_metrics:
                    mess_metrics = summarize_chain_metrics(
                        mess_chain, mess_runtime, burn_in=burn_in, max_lag=max_lag
                    )
                    save_metrics_json(metrics_path, mess_metrics, None, mess_runtime)
                print(f"MESS uniform loaded: M={P}, runtime={mess_runtime:.2f}s")
            elif args.refresh_metrics_only:
                print(f"MESS uniform missing chain: M={P} (skipping, refresh_metrics_only=True)")
            else:
                mess_chain, mess_runtime = run_mess_chain(
                    problem,
                    x0,
                    n_iters,
                    M=P,
                    seed=seed_mess,
                    use_lp=False,
                    distance_metric="angular",
                    lam=0.0,
                )
                mess_metrics = summarize_chain_metrics(
                    mess_chain, mess_runtime, burn_in=burn_in, max_lag=max_lag
                )
                save_chain_bundle(samples_path, metrics_path, mess_chain, None, mess_runtime, mess_metrics)
                print(f"MESS uniform done: M={P}, runtime={mess_runtime:.2f}s")

            seed_mess_lp = seed_base + P
            samples_path, metrics_path = mess_cache_paths(
                estimations_dir, "mess_euclid_sq", P=P, seed_base=seed_base
            )
            loaded = load_chain_bundle(samples_path, metrics_path)
            if loaded is not None:
                mess_lp_chain, _, mess_lp_runtime, mess_lp_metrics = loaded
                if args.refresh_metrics_only or mess_lp_metrics is None or "ess_per_param" not in mess_lp_metrics:
                    mess_lp_metrics = summarize_chain_metrics(
                        mess_lp_chain, mess_lp_runtime, burn_in=burn_in, max_lag=max_lag
                    )
                    save_metrics_json(metrics_path, mess_lp_metrics, None, mess_lp_runtime)
                print(f"MESS euclid_sq loaded: M={P}, runtime={mess_lp_runtime:.2f}s")
            elif args.refresh_metrics_only:
                print(f"MESS euclid_sq missing chain: M={P} (skipping, refresh_metrics_only=True)")
            else:
                mess_lp_chain, mess_lp_runtime = run_mess_chain(
                    problem,
                    x0,
                    n_iters,
                    M=P,
                    seed=seed_mess_lp,
                    use_lp=True,
                    distance_metric="euclidean_squared",
                    lam=0.0,
                )
                mess_lp_metrics = summarize_chain_metrics(
                    mess_lp_chain, mess_lp_runtime, burn_in=burn_in, max_lag=max_lag
                )
                save_chain_bundle(samples_path, metrics_path, mess_lp_chain, None, mess_lp_runtime, mess_lp_metrics)
                print(f"MESS euclid_sq done: M={P}, runtime={mess_lp_runtime:.2f}s")
    else:
        print("MESS disabled (run_mess=False).")


if __name__ == "__main__":
    main()
