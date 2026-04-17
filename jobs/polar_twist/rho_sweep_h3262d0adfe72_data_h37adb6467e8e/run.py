import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from multiproposal.algorithms.effective_sample_size import estimate_effective_sample_size
from multiproposal.algorithms.mtpcn import mtpcn_chain
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


def run_mtpcn_chain_with_checkpoints(
    problem,
    x0,
    n_iters,
    rho,
    n_props,
    seed,
    checkpoint_interval=10000,
    progress_path=None,
    partial_samples_path=None,
    progress_payload_base=None,
):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    chain_blocks = [x0[None, :]]
    accept_weighted_sum = 0.0
    iter_completed = 0
    x = x0.copy()

    while iter_completed < n_iters:
        block_iters = min(checkpoint_interval, n_iters - iter_completed)
        chain_block, accept_rate = mtpcn_chain(
            x,
            problem,
            rng,
            block_iters,
            rho=rho,
            n_props=n_props,
        )
        chain_blocks.append(chain_block[1:])
        accept_weighted_sum += float(accept_rate) * block_iters
        iter_completed += block_iters
        x = chain_block[-1]

        runtime_sec = time.perf_counter() - t0
        if partial_samples_path is not None:
            _write_partial_chain(
                partial_samples_path,
                np.vstack(chain_blocks),
                accept_weighted_sum / max(iter_completed, 1),
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
    accept_rate = accept_weighted_sum / max(iter_completed, 1)
    return chain, runtime_sec, accept_rate


def main():
    parser = argparse.ArgumentParser(
        description="Run polar twist mTPCN rho sweep and cache chains/metrics."
    )
    parser.add_argument("--P-list", type=str, default=None)
    parser.add_argument("--rho-list", type=str, default=None)
    parser.add_argument("--grid-count", type=int, default=1)
    parser.add_argument("--grid-index", type=int, default=0)
    parser.add_argument("--refresh-metrics-only", action="store_true")
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
    rho_list = [round(val, 3) for val in np.arange(0, 1.01, 0.025)]
    seed_base = 202
    run_pcn = False
    run_mpcn = True
    run_mtpcn = False
    run_mess = False

    # Metrics config
    max_lag = 5000
    n_diag_samples = 100
    burn_in = 5000
    checkpoint_interval = max(0, int(args.checkpoint_interval))

    # Optional overrides to reuse an existing run directory exactly.
    data_id_override = "data_h37adb6467e8e"
    run_id_override = "rho_sweep_h3262d0adfe72"

    if args.P_list:
        P_list = _parse_int_list(args.P_list)
    if args.rho_list:
        rho_list = _parse_float_list(args.rho_list)

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
        "rho_list": rho_list,
        "seed_base": seed_base,
        "run_pcn": run_pcn,
        "run_mpcn": run_mpcn,
        "run_mtpcn": run_mtpcn,
        "run_mess": run_mess,
    }

    data_id = data_id_override or f"data_h{_stable_hash(data_id_config)}"
    run_id = run_id_override or f"rho_sweep_h{_stable_hash({
        "algorithm": "polar_twist_rho_sweep",
        "algorithm_config": algo_config,
        "sweep": sweep_config,
    })}"

    estimations_dir = repo_root / "estimations" / "polar_twist" / data_id / "sweep" / run_id
    reports_dir = repo_root / "reports" / "polar_twist" / data_id / "sweep" / run_id
    for path in (estimations_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    run_config = {
        "dataset": "polar_twist",
        "algorithm": "polar_twist_rho_sweep",
        "methods": ["mtpcn"],
        "data": data_config,
        "algorithm_config": algo_config,
        "execution": {
            "checkpoint_interval": checkpoint_interval,
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
    print("run_mtpcn:", run_mtpcn)
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

    grid = [(P, float(rho)) for P in P_list for rho in rho_list]
    grid = grid[args.grid_index :: args.grid_count]

    for P, rho in grid:
        samples_path, metrics_path = chain_cache_paths(
            estimations_dir, "mtpcn", rho=rho, seed_base=seed_base, P=P
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
                f"mTPCN loaded: P={P}, rho={rho:.3f}, accept={accept_display:.3f}, runtime={runtime_sec:.2f}s"
            )
            continue
        if args.refresh_metrics_only:
            print(
                f"mTPCN missing chain: P={P}, rho={rho:.3f} (skipping, refresh_metrics_only=True)"
            )
            continue

        seed = seed_base + int(P * 1000 + round(rho * 100))
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
            chain, runtime_sec, accept_rate = run_mtpcn_chain_with_checkpoints(
                problem,
                x0,
                n_iters,
                rho=rho,
                n_props=P,
                seed=seed,
                checkpoint_interval=checkpoint_interval,
                progress_path=progress_path,
                partial_samples_path=partial_samples_path,
                progress_payload_base=progress_payload_base,
            )
        else:
            rng_chain = np.random.default_rng(seed)
            t0 = time.perf_counter()
            chain, accept_rate = mtpcn_chain(
                x0,
                problem,
                rng_chain,
                n_iters,
                rho=rho,
                n_props=P,
            )
            runtime_sec = time.perf_counter() - t0

        metrics = summarize_chain_metrics(
            chain, runtime_sec, burn_in=burn_in, max_lag=max_lag
        )
        save_chain_bundle(samples_path, metrics_path, chain, accept_rate, runtime_sec, metrics)

        print(
            f"mTPCN done: P={P}, rho={rho:.3f}, accept={accept_rate:.3f}, runtime={runtime_sec:.2f}s"
        )


if __name__ == "__main__":
    main()
