"""Observable and MSE utilities for diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Observable:
    obs_id: int
    name: str
    label: str
    value_fn: Callable[[np.ndarray], float]
    series_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None


def upper_triangle_values(A: np.ndarray) -> np.ndarray:
    """Return upper-triangular entries (i < j) flattened."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    iju = np.triu_indices(A.shape[0], k=1)
    return A[iju]


def band_limited_energy(A: np.ndarray, k: int) -> float:
    """Energy within |i-j| <= k over i < j."""
    if k < 0:
        raise ValueError("k must be non-negative.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    i, j = np.triu_indices(A.shape[0], k=1)
    mask = (np.abs(i - j) <= k)
    vals = A[i[mask], j[mask]]
    return float(np.sum(vals ** 2))


def row_norms_upper(A: np.ndarray) -> np.ndarray:
    """Row norms using only the upper triangle (j > i)."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    d = A.shape[0]
    norms = np.zeros(d, dtype=float)
    for i in range(d):
        row_vals = A[i, i + 1 :]
        norms[i] = float(np.sum(row_vals ** 2))
    return norms


def row_norms_full(A: np.ndarray) -> np.ndarray:
    """Row norms using all off-diagonal entries (j != i)."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    d = A.shape[0]
    norms = np.zeros(d, dtype=float)
    for i in range(d):
        row_vals = np.concatenate([A[i, :i], A[i, i + 1 :]])
        norms[i] = float(np.sum(row_vals ** 2))
    return norms


def row_norm_i(A: np.ndarray, row: int, full: bool = False) -> float:
    """Row norm for a single row."""
    if row < 0 or row >= A.shape[0]:
        raise ValueError("row index out of bounds.")
    if full:
        return float(row_norms_full(A)[row])
    return float(row_norms_upper(A)[row])


def max_entry(A: np.ndarray) -> float:
    """Maximum absolute entry over the upper triangle."""
    vals = np.abs(upper_triangle_values(A))
    if vals.size == 0:
        return 0.0
    return float(np.max(vals))


def top_k_average(A: np.ndarray, k: int) -> float:
    """Average of the k largest absolute upper-triangular entries."""
    if k < 1:
        raise ValueError("k must be >= 1.")
    vals = np.abs(upper_triangle_values(A))
    if vals.size == 0:
        return 0.0
    if k > vals.size:
        raise ValueError("k exceeds number of upper-triangular entries.")
    idx = np.argpartition(vals, -k)[-k:]
    return float(np.mean(vals[idx]))


def leading_singular_value(
    A: np.ndarray,
    method: str = "svd",
    n_iter: int = 50,
    tol: float = 1e-8,
    random_state: Optional[int] = None,
) -> float:
    """Largest singular value via SVD or power iteration."""
    if method == "svd":
        s = np.linalg.svd(A, compute_uv=False)
        return float(s[0]) if s.size else 0.0
    if method != "power":
        raise ValueError("method must be 'svd' or 'power'.")
    rng = np.random.default_rng(random_state)
    v = rng.standard_normal(A.shape[1])
    v /= np.linalg.norm(v)
    last = 0.0
    for _ in range(n_iter):
        w = A.T @ (A @ v)
        norm_w = np.linalg.norm(w)
        if norm_w == 0.0:
            return 0.0
        v = w / norm_w
        if abs(norm_w - last) < tol:
            break
        last = norm_w
    return float(np.sqrt(last))


def support_size(A: np.ndarray, eps: float, normalize: bool = False) -> float:
    """Count upper-triangular entries exceeding eps in magnitude."""
    if eps < 0:
        raise ValueError("eps must be non-negative.")
    vals = np.abs(upper_triangle_values(A))
    count = float(np.sum(vals > eps))
    if not normalize:
        return count
    denom = float(vals.size) if vals.size else 1.0
    return count / denom


def make_antisymmetric_A_observables(
    k_band: int,
    top_k: int,
    eps: float,
    row_index: Optional[int] = None,
    full_row: bool = False,
    row_indices: Optional[Sequence[int]] = None,
) -> List[Observable]:
    """Factory for A-matrix observables with configured parameters."""
    observables: List[Observable] = [
        Observable(
            101,
            f"BandEnergy_k{k_band}",
            rf"$E_{{{k_band}}}(A)$",
            lambda A, kk=k_band: band_limited_energy(A, kk),
        ),
        Observable(
            102,
            "MaxEntry",
            r"$\max_{i<j}|a_{ij}|$",
            max_entry,
        ),
        Observable(
            103,
            f"TopKAvg_k{top_k}",
            rf"$T_{{{top_k}}}(A)$",
            lambda A, kk=top_k: top_k_average(A, kk),
        ),
        Observable(
            104,
            "SpectralNorm",
            r"$\|A\|_2$",
            leading_singular_value,
        ),
        Observable(
            105,
            f"Support_eps{eps:g}",
            rf"$S_{{{eps:g}}}(A)$",
            lambda A, ee=eps: support_size(A, ee, normalize=False),
        ),
    ]
    row_label = "full" if full_row else "upper"
    if row_index is not None:
        observables.append(
            Observable(
                106,
                f"RowNorm_{row_index}_{row_label}",
                rf"$R_{{{row_index}}}(A)$",
                lambda A, ii=row_index, full=full_row: row_norm_i(A, ii, full=full),
            )
        )
    if row_indices:
        for idx in row_indices:
            observables.append(
                Observable(
                    106,
                    f"RowNorm_{idx}_{row_label}",
                    rf"$R_{{{idx}}}(A)$",
                    lambda A, ii=idx, full=full_row: row_norm_i(A, ii, full=full),
                )
            )
    return observables


def make_parameter_observables(problem, dim: int) -> List[Observable]:
    """Factory for parameter-space observables used in solute transport."""
    if dim < 2:
        raise ValueError("dim must be >= 2 for first-row observables.")

    def _potential_scalar(params: np.ndarray) -> float:
        theta = problem.theta_from_params(params)
        resid = problem.y - theta[problem.obs_indices]
        return float(0.5 * np.dot(resid, resid) / (problem.sigma ** 2))

    def _potential_series(chain: np.ndarray) -> np.ndarray:
        series = np.empty(chain.shape[0], dtype=float)
        for idx, params in enumerate(chain):
            series[idx] = _potential_scalar(params)
        return series

    def _mean_first_row(params: np.ndarray) -> float:
        return float(np.mean(params[: dim - 1]))

    def _mean_first_row_series(chain: np.ndarray) -> np.ndarray:
        return np.mean(chain[:, : dim - 1], axis=1)

    def _var_first_row(params: np.ndarray) -> float:
        return float(np.var(params[: dim - 1], ddof=1))

    def _var_first_row_series(chain: np.ndarray) -> np.ndarray:
        return np.var(chain[:, : dim - 1], axis=1, ddof=1)

    def _mean_series(chain: np.ndarray) -> np.ndarray:
        return np.mean(chain, axis=1)

    def _var_series(chain: np.ndarray) -> np.ndarray:
        return np.var(chain, axis=1, ddof=1)

    def _norm_series(chain: np.ndarray) -> np.ndarray:
        return np.linalg.norm(chain, axis=1)

    return [
        Observable(1, "FirstRowMean", r"$\bar{x}_{1:d-1}$", _mean_first_row, _mean_first_row_series),
        Observable(2, "FirstRowVar", r"$\mathrm{Var}(x_{1:d-1})$", _var_first_row, _var_first_row_series),
        Observable(3, "FirstComponent", r"$x_{0}$", lambda p: float(p[0]), lambda c: c[:, 0]),
        Observable(4, "Min", r"$\min_i x_i$", lambda p: float(np.min(p)), lambda c: np.min(c, axis=1)),
        Observable(5, "Mean", r"$\bar{x}=\frac{1}{d}\sum_i x_i$", lambda p: float(np.mean(p)), _mean_series),
        Observable(6, "Variance", r"$\mathrm{Var}(x)$", lambda p: float(np.var(p, ddof=1)), _var_series),
        Observable(7, "Norm", r"$\|x\|_2$", lambda p: float(np.linalg.norm(p)), _norm_series),
        Observable(8, "Potential", r"$\Phi(x)=\frac{1}{2\sigma^2}\|y - f(x)\|^2$", _potential_scalar, _potential_series),
    ]


def select_observables(observables: Sequence[Observable], obs_ids: Sequence[int]) -> List[Observable]:
    """Filter observables by ID, preserving order."""
    obs_ids_set = set(obs_ids)
    return [obs for obs in observables if obs.obs_id in obs_ids_set]


def observable_series(chain: np.ndarray, obs: Observable, n_iter: Optional[int] = None) -> np.ndarray:
    """Compute observable series for a chain."""
    if obs.series_fn is None:
        series = np.array([obs.value_fn(params) for params in chain])
    else:
        series = obs.series_fn(chain)
    if n_iter is not None:
        return series[:n_iter]
    return series


def stack_observable_series(
    chains: Sequence[np.ndarray], obs: Observable, n_iter: Optional[int] = None
) -> np.ndarray:
    """Stack observable series across chains."""
    return np.stack([observable_series(chain, obs, n_iter) for chain in chains], axis=0)


def running_mean(series: np.ndarray) -> np.ndarray:
    """Running mean of a 1D series."""
    denom = np.arange(1, series.shape[0] + 1, dtype=float)
    return np.cumsum(series) / denom


def chain_running_mean(chain: np.ndarray, obs: Observable, n_iter: int) -> np.ndarray:
    """Running mean of an observable over a single chain."""
    return running_mean(observable_series(chain, obs, n_iter))


def ep_running_mean(group: Sequence[np.ndarray], obs: Observable, n_iter: int) -> np.ndarray:
    """Running mean for EP groups (average over chains at each iter)."""
    obs_stack = stack_observable_series(group, obs, n_iter)
    mean_over_p = np.mean(obs_stack, axis=0)
    return running_mean(mean_over_p)


def min_chain_len(chains: Sequence[np.ndarray]) -> int:
    """Minimum chain length."""
    return min(chain.shape[0] for chain in chains)


def compute_observable_targets(
    chains: Sequence[np.ndarray],
    observables: Sequence[Observable],
    burn_in: int,
) -> List[float]:
    """Posterior-mean targets computed from per-chain means."""
    targets = []
    for obs in observables:
        chain_means = []
        for chain in chains:
            if chain.shape[0] <= burn_in:
                raise ValueError("burn_in exceeds chain length.")
            series = observable_series(chain, obs)[burn_in:]
            chain_means.append(float(np.mean(series)))
        targets.append(float(np.mean(chain_means)))
    return targets


def compute_running_mse(
    mpcn_chains: Sequence[np.ndarray],
    pcn_chains: Sequence[np.ndarray],
    observables: Sequence[Observable],
    targets: Sequence[float],
    n_iter: int,
    ep_groups: Optional[Sequence[Sequence[np.ndarray]]] = None,
) -> dict:
    """Compute running MSE series for mPCN, pCN, and EP."""
    if len(observables) != len(targets):
        raise ValueError("observables and targets must be the same length.")

    results = {}
    for obs, target in zip(observables, targets):
        mpcn_series = stack_observable_series(mpcn_chains, obs, n_iter)
        pcn_series = stack_observable_series(pcn_chains, obs, n_iter)
        mpcn_rm = np.stack([running_mean(series) for series in mpcn_series], axis=0)
        pcn_rm = np.stack([running_mean(series) for series in pcn_series], axis=0)
        mpcn_mse = np.mean((mpcn_rm - target) ** 2, axis=0)
        pcn_mse = np.mean((pcn_rm - target) ** 2, axis=0)

        ep_mse = None
        if ep_groups:
            ep_series = np.stack([ep_running_mean(group, obs, n_iter) for group in ep_groups], axis=0)
            ep_mse = np.mean((ep_series - target) ** 2, axis=0)

        results[obs.name] = {
            "target": float(target),
            "mpcn_mse": mpcn_mse,
            "pcn_mse": pcn_mse,
            "ep_mse": ep_mse,
        }
    return results
