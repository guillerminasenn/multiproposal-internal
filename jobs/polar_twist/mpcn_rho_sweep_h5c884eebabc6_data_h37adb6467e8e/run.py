import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from multiproposal.algorithms.effective_sample_size import estimate_effective_sample_size
from multiproposal.algorithms.mess import mess_step
from multiproposal.algorithms.mpcn import mpcn_chain
from multiproposal.algorithms.pcn import pcn_chain
from multiproposal.problems.toy_custom_likelihood import ToyCustomLikelihood2D
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


def f_polar_twist(x, alpha, weight_x=1.0, weight_y=1.0):
    x1, x2 = x
    radius = np.sqrt(x1**2 + x2**2)
    comp1 = x1 * np.cos(alpha * weight_x * radius) - x2 * np.sin(alpha * weight_y * radius)
    comp2 = x1 * np.sin(alpha * weight_x * radius) + x2 * np.cos(alpha * weight_y * radius)
    return np.array([comp1, comp2])


def log_likelihood_polar_twist(x, y_obs, sigma=0.3, alpha=0.5, weight_x=1.0, weight_y=1.0):
    r = f_polar_twist(x, alpha=alpha, weight_x=weight_x, weight_y=weight_y) - y_obs
    return -0.5 * np.dot(r, r) / (sigma**2)


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


def summarize_chain_metrics(chain, runtime_sec, burn_in, max_lag):
    post = chain[burn_in:]
    ess_vals = compute_ess_per_param(post, max_lag=max_lag)
    msjd_vals = compute_msjd_per_param(post)
    ess_mean = float(np.nanmean(ess_vals)) if ess_vals.size else 0.0
    msjd_mean = float(np.nanmean(msjd_vals)) if msjd_vals.size else 0.0
    runtime_min = runtime_sec / 60.0
    ess_per_min = ess_mean / runtime_min if runtime_min > 0 else np.nan
    return {
        "runtime_sec": runtime_sec,
        "runtime_min": runtime_min,
        "ess_mean": ess_mean,
        "msjd_mean": msjd_mean,
        "ess_per_min": ess_per_min,
        "ess_per_param": ess_vals.tolist(),
        "msjd_per_param": msjd_vals.tolist(),
    }


def rho_to_tag(rho):
    return format_float_tag(rho, precision=5)


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


def independent_pcn_paths(estimations_dir, P, rho, seed_base):
    rho_tag = rho_to_tag(rho)
    chains_dir = estimations_dir / "chains" / "independent_chains"
    stem = f"pcn_independent_P{P}_rho{rho_tag}_seed{seed_base}"
    samples_path = chains_dir / f"{stem}.npz"
    metrics_path = chains_dir / f"{stem}_metrics.json"
    return samples_path, metrics_path


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
    try:
        data = np.load(samples_path, allow_pickle=False)
    except (EOFError, OSError, ValueError):
        return None
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


def main():
    parser = argparse.ArgumentParser(
        description="Run polar twist mPCN rho sweep and cache chains/metrics."
    )
    parser.add_argument("--P-list", type=str, default=None)
    parser.add_argument("--rho-list", type=str, default=None)
    parser.add_argument("--grid-count", type=int, default=1)
    parser.add_argument("--grid-index", type=int, default=0)
    parser.add_argument("--refresh-metrics-only", action="store_true")
    parser.add_argument("--skip-pcn", action="store_true")
    parser.add_argument("--skip-mess", action="store_true")
    parser.add_argument("--skip-independent-pcn", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    args = parser.parse_args()

    repo_root = _resolve_repo_root()

    # Data configuration
    alpha = 2
    sigma_noise = 1.0
    prior_std = 2.0
    prior_cov = prior_std**2 * np.array([[1.0, 0.3], [0.3, 0.5]])
    prior_mean = np.zeros(2)
    weight_x = 1
    weight_y = 1
    data_seed = 202

    # Sweep configuration
    n_iters = 300000
    P_list = [10, 30, 50, 100]
    M_list = list(P_list)
    rho_list = [round(val, 3) for val in np.arange(0, 1.01, 0.025)]
    seed_base = 202
    run_pcn = False
    run_mess = False
    run_independent_pcn = True

    # Metrics config
    max_lag = 5000
    n_diag_samples = 100
    burn_in = 5000
    checkpoint_interval = max(0, int(args.checkpoint_interval))

    # Optional overrides to reuse an existing run directory exactly.
    data_id_override = "data_h37adb6467e8e"
    run_id_override = "mpcn_rho_sweep_h5c884eebabc6"

    if args.P_list:
        P_list = _parse_int_list(args.P_list)
        M_list = list(P_list)
    if args.rho_list:
        rho_list = _parse_float_list(args.rho_list)

    run_pcn = run_pcn and not args.skip_pcn
    run_mess = run_mess and not args.skip_mess
    run_independent_pcn = run_independent_pcn and not args.skip_independent_pcn
    if args.grid_count > 1 and args.grid_index != 0:
        run_pcn = False
        run_mess = False
        run_independent_pcn = False

    if args.grid_count < 1:
        raise ValueError("grid-count must be >= 1")
    if args.grid_index < 0 or args.grid_index >= args.grid_count:
        raise ValueError("grid-index must be in [0, grid-count)")

    data_id_config = {
        "model": "polar_twist",
        "alpha": alpha,
        "weight_x": weight_x,
        "weight_y": weight_y,
        "sigma_noise": sigma_noise,
        "prior_std": prior_std,
        "prior_cov": prior_cov.tolist(),
        "prior_mean": prior_mean.tolist(),
        "data_seed": data_seed,
    }
    data_config = dict(data_id_config)
    algo_config = {
        "n_iters": n_iters,
        "burn_in": burn_in,
        "max_lag": max_lag,
        "n_diag_samples": n_diag_samples,
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

    estimations_dir = repo_root / "estimations" / "polar_twist" / data_id / "sweep" / run_id
    reports_dir = repo_root / "reports" / "polar_twist" / data_id / "sweep" / run_id
    for path in (estimations_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    run_config = {
        "dataset": "polar_twist",
        "algorithm": "mpcn_rho_sweep",
        "data": data_config,
        "algorithm_config": algo_config,
        "execution": {
            "checkpoint_interval": checkpoint_interval,
            "independent_pcn": {
                "enabled": run_independent_pcn,
                "P": int(max(P_list)) if P_list else None,
            },
        },
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
    print("run_independent_pcn:", run_independent_pcn)
    print("checkpoint_interval:", checkpoint_interval)
    print("data_id:", data_id)
    print("run_id:", run_id)
    print("Run directory:", estimations_dir)

    rng = np.random.default_rng(data_seed)
    prior_sample = rng.multivariate_normal(prior_mean, prior_cov)
    theta_true = f_polar_twist(prior_sample, alpha=alpha, weight_x=weight_x, weight_y=weight_y)
    y_obs = theta_true + rng.normal(0.0, sigma_noise, size=theta_true.shape)

    def log_likelihood(x):
        return log_likelihood_polar_twist(
            x,
            y_obs,
            sigma=sigma_noise,
            alpha=alpha,
            weight_x=weight_x,
            weight_y=weight_y,
        )

    problem = ToyCustomLikelihood2D(
        log_likelihood_fn=log_likelihood,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
    )
    rng_init = np.random.default_rng(seed_base)
    x0 = problem.sample_prior(rng_init)

    independent_P = max(P_list)

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
            print(f"pCN done: rho={rho:.3f}, accept={accept_rate:.3f}, runtime={runtime_sec:.2f}s")
    else:
        print("pCN disabled (run_pcn=False).")

    if run_independent_pcn:
        for rho in rho_list:
            samples_path, metrics_path = independent_pcn_paths(
                estimations_dir, independent_P, rho=rho, seed_base=seed_base
            )
            if samples_path.exists():
                metrics = load_metrics_json(metrics_path)
                with np.load(samples_path, allow_pickle=False) as data:
                    runtimes = data.get("runtimes_sec", np.asarray([]))
                runtime_sec = float(np.sum(runtimes)) if runtimes.size else 0.0
                if args.refresh_metrics_only and metrics is None:
                    metrics = {
                        "P": int(independent_P),
                        "rho": float(rho),
                        "seed_base": int(seed_base),
                        "n_iters": int(n_iters),
                    }
                    save_metrics_json(metrics_path, metrics, None, runtime_sec)
                print(
                    f"pCN independent loaded: P={independent_P}, rho={rho:.3f}, "
                    f"runtime={runtime_sec:.2f}s"
                )
                continue
            if args.refresh_metrics_only:
                print(
                    f"pCN independent missing: P={independent_P}, rho={rho:.3f} "
                    "(skipping, refresh_metrics_only=True)"
                )
                continue

            chains = np.zeros((independent_P, n_iters + 1, problem.dim), dtype=float)
            accept_rates = np.zeros(independent_P, dtype=float)
            runtimes = np.zeros(independent_P, dtype=float)
            seeds = []
            rho_seed = int(round(rho * 1000))
            for idx in range(independent_P):
                seed = seed_base + rho_seed * 10000 + idx
                rng_chain = np.random.default_rng(seed)
                x0_chain = problem.sample_prior(rng_chain)
                t0 = time.perf_counter()
                chain, accept_rate = pcn_chain(
                    x0_chain, problem, rng_chain, n_iters, rho=rho, return_acceptance=True
                )
                runtime_sec = time.perf_counter() - t0
                chains[idx] = chain
                accept_rates[idx] = accept_rate
                runtimes[idx] = runtime_sec
                seeds.append(seed)
                print(
                    f"pCN independent chain {idx + 1}/{independent_P} done: "
                    f"rho={rho:.3f}, accept={accept_rate:.3f}, runtime={runtime_sec:.2f}s"
                )

            metrics = {
                "P": int(independent_P),
                "rho": float(rho),
                "seed_base": int(seed_base),
                "n_iters": int(n_iters),
                "chain_seeds": [int(val) for val in seeds],
            }
            accept_rate = float(np.mean(accept_rates)) if accept_rates.size else np.nan
            runtime_sec = float(np.sum(runtimes)) if runtimes.size else 0.0
            samples_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                samples_path,
                chains=chains,
                accept_rates=accept_rates,
                runtimes_sec=runtimes,
                seeds=np.asarray(seeds, dtype=int),
                n_iters=int(n_iters),
                rho=float(rho),
                P=int(independent_P),
            )
            save_metrics_json(metrics_path, metrics, accept_rate, runtime_sec)
            print(
                f"pCN independent done: P={independent_P}, rho={rho:.3f}, "
                f"total_runtime={runtime_sec:.2f}s"
            )
    else:
        print("pCN independent disabled (run_independent_pcn=False).")

    grid = [(P, float(rho)) for P in P_list for rho in rho_list]
    grid = grid[args.grid_index:: args.grid_count]

    for P, rho in grid:
        samples_path, metrics_path = chain_cache_paths(
            estimations_dir, "mpcn", rho=rho, seed_base=seed_base, P=P
        )
        diag_path = mpcn_diag_path(estimations_dir, P, rho, seed_base)
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
            "dataset": "polar_twist",
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
            )
        metrics = summarize_chain_metrics(
            chain, runtime_sec, burn_in=burn_in, max_lag=max_lag
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
