import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

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
    """Return repo root, honoring MULTIPROPOSAL_RUN_ROOT when set."""
    env_root = os.environ.get("MULTIPROPOSAL_RUN_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    root = Path.cwd().resolve()
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    return root


def _canonicalize_payload(obj):
    """Convert numpy/path objects into JSON-serializable payloads."""
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
    """Return a deterministic short hash for a JSON payload."""
    data = json.dumps(
        _canonicalize_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def get_obs_indices(dim_value, highest_freq, bandwidth):
    """Return observation indices for a target dimension and frequency band."""
    highest_freq = min(highest_freq, dim_value)
    bandwidth = min(bandwidth, dim_value)
    start = max(0, highest_freq - bandwidth + 1)
    return np.arange(start, highest_freq + 1, dtype=int)


def get_param_indices_for_dim(dim, shared_draws):
    """Return cached parameter indices for a given dimension."""
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
    """Build shared draws for multiple dimensions and reuse across jobs."""
    rng = np.random.default_rng(seed)
    m_max = d_max * (d_max - 1) // 2
    prior_diag_max = prior_diag_from_powerlaw(
        d_max, alpha=alpha, gamma=gamma, tau2=tau2, offset=offset
    )
    if prior_diag_max.shape != (m_max,):
        raise ValueError(
            f"prior_diag_max must have shape ({m_max},), got {prior_diag_max.shape}"
        )
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
    """Generate advection-diffusion data using shared draws."""
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


def build_problem_for_dim(dim, shared_draws, obs_highest_freq, obs_bandwidth, kappa, sigma):
    """Build the advection-diffusion toy problem for a target dimension."""
    obs_indices = get_obs_indices(dim, obs_highest_freq, obs_bandwidth)
    data = generate_advection_diffusion_data_shared(dim, obs_indices, shared_draws)
    problem = AdvectionDiffusionToy(
        dim=dim,
        kappa=kappa,
        sigma=sigma,
        y=data["y"],
        obs_indices=obs_indices,
        g=data["g"],
        prior_diag=data["prior_diag"],
    )
    return problem, data["a_init"], data


def sample_prior_points(rng, prior_diag, count):
    """Sample independent draws from the Gaussian prior."""
    z = rng.standard_normal((count, prior_diag.shape[0]))
    return z * np.sqrt(prior_diag)[None, :]


def pcn_chain_path(chains_dir, rho, seed_base, chain_idx):
    """Return the expected pCN independent chain path."""
    rho_tag = format_float_tag(rho)
    return chains_dir / f"pcn_independent_rho{rho_tag}_seed{seed_base}_chain{chain_idx:03d}.npz"


def pcn_index_path(chains_dir, rho, seed_base):
    """Return the pCN independent index path."""
    rho_tag = format_float_tag(rho)
    return chains_dir / f"pcn_independent_rho{rho_tag}_seed{seed_base}_index.json"


def mpcn_chain_path(chains_dir, rho, P, seed_base, chain_idx):
    """Return the expected mPCN independent chain path."""
    rho_tag = format_float_tag(rho)
    return (
        chains_dir / f"mpcn_P{P}_rho{rho_tag}_seed{seed_base}_chain{chain_idx:03d}.npz"
    )


def mpcn_index_path(chains_dir, rho, P, seed_base):
    """Return the mPCN independent index path."""
    rho_tag = format_float_tag(rho)
    return chains_dir / f"mpcn_P{P}_rho{rho_tag}_seed{seed_base}_index.json"


def load_chain(path):
    """Load a saved chain bundle from disk."""
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    chain = data["chain"]
    accept_rate = float(data["accept_rate"]) if "accept_rate" in data else np.nan
    if np.isnan(accept_rate):
        accept_rate = None
    runtime_sec = float(data["runtime_sec"]) if "runtime_sec" in data else 0.0
    return chain, accept_rate, runtime_sec


def save_chain(path, chain, accept_rate, runtime_sec, extra=None):
    """Save a chain bundle to disk."""
    payload = {
        "chain": chain,
        "accept_rate": np.nan if accept_rate is None else float(accept_rate),
        "runtime_sec": float(runtime_sec),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_index(path):
    """Load a JSON index file if present."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_index(path, payload):
    """Save a JSON index file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_metrics(path, payload):
    """Save per-chain metrics payload under diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


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


def build_index_from_files(chains_dir, pattern, expected_meta):
    """Build an index payload by scanning chain files."""
    payload = {
        "metadata": dict(expected_meta),
        "chains": [],
    }
    for path in sorted(chains_dir.glob(pattern)):
        stem = path.stem
        parts = stem.split("_chain")
        if len(parts) != 2:
            continue
        try:
            chain_idx = int(parts[1])
        except ValueError:
            continue
        payload["chains"].append(
            {
                "chain_idx": int(chain_idx),
                "file": path.name,
                "seed": None,
                "start_index": int(chain_idx),
            }
        )
    payload["chains"].sort(key=lambda x: x["chain_idx"])
    return payload


def select_indices_for_worker(count, grid_count, grid_index):
    """Return chain indices assigned to a worker shard."""
    return [idx for idx in range(count) if idx % grid_count == grid_index]


def run_pcn_chain(problem, x0, n_iters, rho, seed):
    """Run a single pCN chain and return (chain, accept_rate, runtime_sec)."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    chain, acc_rate = pcn_chain(
        x0,
        problem,
        rng,
        n_iters,
        rho=rho,
        return_acceptance=True,
    )
    runtime_sec = time.perf_counter() - t0
    return chain, acc_rate, runtime_sec


def run_pcn_chain_with_checkpoints(
    problem,
    x0,
    n_iters,
    rho,
    seed,
    checkpoint_interval=10000,
    progress_path=None,
    partial_samples_path=None,
    progress_payload_base=None,
):
    """Run pCN with periodic checkpoints and progress updates."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    chain_blocks = [x0[None, :]]
    accepted_counts = []
    iter_completed = 0
    x = x0.copy()

    while iter_completed < n_iters:
        block_iters = min(checkpoint_interval, n_iters - iter_completed)
        chain_block, acc_rate_block = pcn_chain(
            x,
            problem,
            rng,
            block_iters,
            rho=rho,
            return_acceptance=True,
        )
        chain_blocks.append(chain_block[1:])
        accepted_counts.append(acc_rate_block * block_iters)
        iter_completed += block_iters
        x = chain_block[-1]

        runtime_sec = time.perf_counter() - t0
        accept_rate = float(sum(accepted_counts) / iter_completed) if iter_completed else 0.0

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
    runtime_sec = time.perf_counter() - t0
    accept_rate = float(sum(accepted_counts) / n_iters) if n_iters else 0.0
    return chain, accept_rate, runtime_sec


def run_mpcn_chain(problem, x0, n_iters, rho, n_props, seed):
    """Run a single mPCN chain and return (chain, accept_rate, runtime_sec)."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    chain, accepted_index = mpcn_chain(
        x0,
        problem,
        rng,
        n_iters,
        rho=rho,
        n_props=n_props,
        return_indices=True,
        n_jobs=1,
        parallel_backend="auto",
        parallelize_props=False,
        llh_chunk_size=0,
    )
    runtime_sec = time.perf_counter() - t0
    acc_rate = float(np.mean(accepted_index != 0))
    return chain, acc_rate, runtime_sec


def run_mpcn_chain_with_checkpoints(
    problem,
    x0,
    n_iters,
    rho,
    n_props,
    seed,
    checkpoint_interval=100,
    progress_path=None,
    partial_samples_path=None,
    progress_payload_base=None,
):
    """Run mPCN with periodic checkpoints and progress updates."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    chain_blocks = [x0[None, :]]
    accepted_blocks = []
    iter_completed = 0
    x = x0.copy()

    while iter_completed < n_iters:
        block_iters = min(checkpoint_interval, n_iters - iter_completed)
        chain_block, accepted_block = mpcn_chain(
            x,
            problem,
            rng,
            block_iters,
            rho=rho,
            n_props=n_props,
            return_indices=True,
            n_jobs=1,
            parallel_backend="auto",
            parallelize_props=False,
            llh_chunk_size=0,
        )
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
    runtime_sec = time.perf_counter() - t0
    if accepted_blocks:
        accept_rate = float(np.mean(np.concatenate(accepted_blocks) != 0))
    else:
        accept_rate = 0.0
    return chain, accept_rate, runtime_sec


def parse_args():
    """Parse command-line arguments for independent chain jobs."""
    parser = argparse.ArgumentParser(description="Run independent pCN/mPCN chains for solute transport.")
    parser.add_argument("--pcn-count", type=int, default=100)
    parser.add_argument("--mpcn-count", type=int, default=100)
    parser.add_argument("--grid-count", type=int, default=1)
    parser.add_argument("--grid-index", type=int, default=0)
    parser.add_argument("--skip-pcn", action="store_true")
    parser.add_argument("--skip-mpcn", action="store_true")
    return parser.parse_args()


def main():
    """Entry point for independent chain job execution."""
    args = parse_args()
    if args.grid_count < 1:
        raise ValueError("grid_count must be >= 1")
    if not (0 <= args.grid_index < args.grid_count):
        raise ValueError("grid_index must be in [0, grid_count)")

    seed_data = 0
    seed_mcmc = 202
    config_id = 2

    d = 40
    obs_highest_freq = 12
    obs_bandwidth = 7
    kappa = 0.02
    sigma = 0.5
    alpha = 3.0
    gamma = 2.0
    tau2 = 2.0
    a_mode = "nearest_neighbor"
    use_prior_A = True
    shared_draws_seed = seed_data
    obs_config = "central_modes"

    n_iters = 100000
    rho = 0.9
    P = 100
    burn_in = 0

    data_id_config = {
        "seed_data": seed_data,
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
        "obs_config": obs_config,
        "d": d,
    }
    run_id_config = {
        "n_iters": n_iters,
        "rho": rho,
        "P": P,
        "seed_mcmc": seed_mcmc,
        "burn_in": burn_in,
        "config_id": config_id,
    }
    data_id = f"data_h{_stable_hash(data_id_config)}"
    run_id = f"mpcn_pcn_convergence_h{_stable_hash(run_id_config)}"

    repo_root = _resolve_repo_root()
    estimations_dir = repo_root / "estimations" / "solute_transport" / data_id / "fixed" / run_id
    reports_dir = repo_root / "reports" / "solute_transport" / data_id / "fixed" / run_id
    estimations_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "dataset": "solute_transport",
        "algorithm": "mpcn_pcn_convergence",
        "data": dict(data_id_config),
        "algorithm_config": dict(run_id_config),
        "execution_config": {
            "num_pcn_chains": int(args.pcn_count),
            "num_mpcn_chains": int(args.mpcn_count),
            "grid_count": int(args.grid_count),
            "grid_index": int(args.grid_index),
        },
    }
    config_path = estimations_dir / "config.json"
    if not config_path.exists():
        payload = dict(run_config)
        payload["data_id"] = data_id
        payload["run_id"] = run_id
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

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
    problem, a_init, data = build_problem_for_dim(
        d, shared_draws, obs_highest_freq, obs_bandwidth, kappa, sigma
    )
    prior_diag = data["prior_diag"]

    max_num_chains = max(args.pcn_count, args.mpcn_count)
    rng_starts = np.random.default_rng(seed_mcmc)
    all_start_points = sample_prior_points(rng_starts, prior_diag, max_num_chains)
    pcn_start_points = all_start_points[: args.pcn_count]
    mpcn_start_points = all_start_points[: args.mpcn_count]

    pcn_chains_dir = estimations_dir / "chains" / "independent_chains"
    mpcn_chains_dir = estimations_dir / "chains" / "mpcn_independent"
    diagnostics_dir = estimations_dir / "diagnostics" / "independent_chains"
    pcn_chains_dir.mkdir(parents=True, exist_ok=True)
    mpcn_chains_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_interval = 10000

    if not args.skip_pcn:
        pcn_indices = select_indices_for_worker(args.pcn_count, args.grid_count, args.grid_index)
        pcn_expected_meta = {
            "rho": float(rho),
            "seed_mcmc": int(seed_mcmc),
            "n_iters": int(n_iters),
            "data_id": data_id,
            "run_id": run_id,
        }
        pcn_generated = 0
        pcn_skipped = 0
        for chain_idx in pcn_indices:
            chain_path = pcn_chain_path(pcn_chains_dir, rho, seed_mcmc, chain_idx)
            if chain_path.exists():
                pcn_skipped += 1
                continue
            seed = seed_mcmc + 2000 + chain_idx
            progress_path = chain_path.with_suffix(".progress.json")
            partial_samples_path = chain_path.with_name(f"{chain_path.stem}_partial.npz")
            progress_payload_base = {
                "method": "pcn_independent",
                "chain_idx": int(chain_idx),
                "seed": int(seed),
                "rho": float(rho),
            }
            chain, acc_rate, runtime_sec = run_pcn_chain_with_checkpoints(
                problem,
                pcn_start_points[chain_idx],
                n_iters,
                rho,
                seed,
                checkpoint_interval=checkpoint_interval,
                progress_path=progress_path,
                partial_samples_path=partial_samples_path,
                progress_payload_base=progress_payload_base,
            )
            save_chain(
                chain_path,
                chain,
                acc_rate,
                runtime_sec,
                extra={"start_index": int(chain_idx)},
            )
            metrics_path = diagnostics_dir / f"{chain_path.stem}_metrics.json"
            save_metrics(
                metrics_path,
                {
                    "method": "pcn_independent",
                    "rho": float(rho),
                    "seed": int(seed),
                    "chain_idx": int(chain_idx),
                    "n_iters": int(n_iters),
                    "accept_rate": None if acc_rate is None else float(acc_rate),
                    "runtime_sec": float(runtime_sec),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            pcn_generated += 1
        if pcn_generated > 0:
            index_payload = build_index_from_files(
                pcn_chains_dir,
                f"pcn_independent_rho{format_float_tag(rho)}_seed{seed_mcmc}_chain*.npz",
                pcn_expected_meta,
            )
            save_index(pcn_index_path(pcn_chains_dir, rho, seed_mcmc), index_payload)
        print(
            f"pCN worker {args.grid_index}: generated {pcn_generated}, skipped {pcn_skipped}."
        )

    if not args.skip_mpcn:
        mpcn_indices = select_indices_for_worker(args.mpcn_count, args.grid_count, args.grid_index)
        mpcn_expected_meta = {
            "rho": float(rho),
            "P": int(P),
            "seed_mcmc": int(seed_mcmc),
            "n_iters": int(n_iters),
            "data_id": data_id,
            "run_id": run_id,
        }
        mpcn_generated = 0
        mpcn_skipped = 0
        for chain_idx in mpcn_indices:
            seed = seed_mcmc + 5000 + chain_idx
            chain_path = mpcn_chain_path(mpcn_chains_dir, rho, P, seed, chain_idx)
            if chain_path.exists():
                mpcn_skipped += 1
                continue
            progress_path = chain_path.with_suffix(".progress.json")
            partial_samples_path = chain_path.with_name(f"{chain_path.stem}_partial.npz")
            progress_payload_base = {
                "method": "mpcn_independent",
                "chain_idx": int(chain_idx),
                "seed": int(seed),
                "rho": float(rho),
                "P": int(P),
            }
            chain, acc_rate, runtime_sec = run_mpcn_chain_with_checkpoints(
                problem,
                mpcn_start_points[chain_idx],
                n_iters,
                rho,
                P,
                seed,
                checkpoint_interval=checkpoint_interval,
                progress_path=progress_path,
                partial_samples_path=partial_samples_path,
                progress_payload_base=progress_payload_base,
            )
            save_chain(
                chain_path,
                chain,
                acc_rate,
                runtime_sec,
                extra={"start_index": int(chain_idx)},
            )
            metrics_path = diagnostics_dir / f"{chain_path.stem}_metrics.json"
            save_metrics(
                metrics_path,
                {
                    "method": "mpcn_independent",
                    "rho": float(rho),
                    "P": int(P),
                    "seed": int(seed),
                    "chain_idx": int(chain_idx),
                    "n_iters": int(n_iters),
                    "accept_rate": None if acc_rate is None else float(acc_rate),
                    "runtime_sec": float(runtime_sec),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            mpcn_generated += 1
        if mpcn_generated > 0:
            index_payload = build_index_from_files(
                mpcn_chains_dir,
                f"mpcn_P{P}_rho{format_float_tag(rho)}_seed*_chain*.npz",
                mpcn_expected_meta,
            )
            save_index(mpcn_index_path(mpcn_chains_dir, rho, P, seed_mcmc), index_payload)
        print(
            f"mPCN worker {args.grid_index}: generated {mpcn_generated}, skipped {mpcn_skipped}."
        )


if __name__ == "__main__":
    main()
