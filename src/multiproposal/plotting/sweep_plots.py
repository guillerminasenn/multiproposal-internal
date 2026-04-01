"""Plotting utilities for mPCN/pCN sweep notebooks."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .figure_style import apply_pub_style


def _iter_axes(axes):
    if isinstance(axes, np.ndarray):
        return axes.ravel().tolist()
    if isinstance(axes, (list, tuple)):
        return list(axes)
    return [axes]


def _unique_legend(fig, axes, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, frameon=False):
    handles = []
    labels = []
    for ax in _iter_axes(axes):
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                labels.append(ll)
                handles.append(hh)
    if handles:
        fig.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol, frameon=frameon)


def _post_sample_count(entry, burn_in):
    chain = entry.get("chain")
    if chain is None:
        return None
    return max(int(chain.shape[0]) - int(burn_in), 1)


def _normalize_ess(values, count):
    if count is None or count <= 0:
        return values
    return [val / count if np.isfinite(val) else val for val in values]


def _shared_ylim(ax_list):
    y_vals = []
    for ax in ax_list:
        for line in ax.get_lines():
            data = np.asarray(line.get_ydata(), dtype=float)
            data = data[np.isfinite(data)]
            if data.size:
                y_vals.append(data)
    if not y_vals:
        return
    y_all = np.concatenate(y_vals)
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    for ax in ax_list:
        ax.set_ylim(y_min, y_max)


def plot_trace_grid(
    results,
    P_list,
    rho_list_plot,
    burn_in,
    reports_dir,
    seed_base,
    file_name_fmt,
    file_name_kwargs=None,
    max_iters=30000,
    ncols=3,
):
    """Plot trace subplots for each P with a shared legend."""
    apply_pub_style()

    rho_list_plot = list(rho_list_plot)
    nrows = int(np.ceil(len(rho_list_plot) / ncols))

    for P in P_list:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True)
        axes = np.array(axes).reshape(-1)
        for ax, rho in zip(axes, rho_list_plot):
            chain = results["mpcn"][P][rho]["chain"]
            start = min(burn_in, chain.shape[0])
            end = min(start + max_iters, chain.shape[0])
            segment = chain[start:end]
            if segment.size == 0:
                continue
            ax.plot(segment[:, 0], color="#1f77b4", linewidth=0.5, label=r"$x_1$", alpha=0.8)
            ax.plot(segment[:, 1], color="red", linewidth=0.5, label=r"$x_2$", alpha=0.4)
            ax.set_title(fr"$P={P}, \rho={rho:.2f}$")
            ax.grid(alpha=0.3)

        for ax in axes[len(rho_list_plot):]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(1.02, 0.92),
                ncol=2,
                frameon=False,
            )

        fig.suptitle(f"mpCN trace plots (P={P})")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        reports_dir.mkdir(parents=True, exist_ok=True)
        format_kwargs = {"P": P, "seed_base": seed_base}
        if file_name_kwargs:
            format_kwargs.update(file_name_kwargs)
        fig.savefig(
            reports_dir / file_name_fmt.format(**format_kwargs),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()


def plot_ess_msjd_vs_rho(
    results,
    rho_list,
    P_list,
    reports_dir,
    seed_base,
    file_name_fmt,
    run_pcn,
    run_mess,
    burn_in,
    file_name_kwargs=None,
    show_mess=False,
    show_pcn=True,
    title_prefix="Multiwell",
    normalize_ess=False,
    share_y_metrics=False,
    show_independent=False,
    independent_label_prefix="pCN indep",
):
    """Plot ESS/MSJD vs rho curves for each P."""
    apply_pub_style()

    P_sorted = sorted(P_list)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(P_sorted)))
    color_by_P = {P: colors[i] for i, P in enumerate(P_sorted)}

    pcn_ess = None
    pcn_msjd = None
    if run_pcn and results.get("pcn"):
        pcn_ess = [results["pcn"][rho]["metrics"]["ess_mean"] for rho in rho_list]
        pcn_msjd = [results["pcn"][rho]["metrics"]["msjd_mean"] for rho in rho_list]
        if normalize_ess:
            pcn_counts = [
                _post_sample_count(results["pcn"][rho], burn_in) for rho in rho_list
            ]
            pcn_ess = [
                val / count if count else val for val, count in zip(pcn_ess, pcn_counts)
            ]

    reports_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharex=True)
    ax_ess, ax_msjd = axes

    for P in P_sorted:
        ess_vals = [results["mpcn"][P][rho]["metrics"]["ess_mean"] for rho in rho_list]
        if normalize_ess:
            counts = [
                _post_sample_count(results["mpcn"][P][rho], burn_in) for rho in rho_list
            ]
            ess_vals = [
                val / count if count else val for val, count in zip(ess_vals, counts)
            ]
        ax_ess.plot(rho_list, ess_vals, marker="o", markersize=3, color=color_by_P[P], label=f"mpCN (P={P})")
        if show_independent and results.get("pcn_independent"):
            indep_entries = results["pcn_independent"].get(P)
            if indep_entries:
                key = "ess_mean_sum_norm" if normalize_ess else "ess_mean_sum"
                indep_vals = [indep_entries[rho]["metrics"].get(key, np.nan) for rho in rho_list]
                ax_ess.plot(
                    rho_list,
                    indep_vals,
                    marker="s",
                    markersize=3,
                    linestyle=":",
                    color=color_by_P[P],
                    label=f"{independent_label_prefix} (P={P})",
                )
    if show_pcn and pcn_ess is not None:
        ax_ess.plot(rho_list, pcn_ess, color="black", marker="s", markersize=3, linestyle="--", label="pCN")
    if show_mess and run_mess:
        for P in P_sorted:
            mess_uniform = results["mess_uniform"][P]["metrics"]["ess_mean"]
            mess_euclid = results["mess_euclid_sq"][P]["metrics"]["ess_mean"]
            if normalize_ess:
                mess_count = _post_sample_count(results["mess_uniform"][P], burn_in)
                if mess_count:
                    mess_uniform = mess_uniform / mess_count
                mess_count = _post_sample_count(results["mess_euclid_sq"][P], burn_in)
                if mess_count:
                    mess_euclid = mess_euclid / mess_count
            ax_ess.plot(rho_list, [mess_uniform] * len(rho_list), linestyle="--", color=color_by_P[P], alpha=0.6)
            ax_ess.plot(rho_list, [mess_euclid] * len(rho_list), linestyle=":", color=color_by_P[P], alpha=0.6)
    ax_ess.set_xlabel(r"$\rho$")
    ax_ess.set_ylabel("Effective Sample Size")
    if normalize_ess:
        ax_ess.set_ylabel(r"ESS per sample (IACT$^{-1}$)")
    ax_ess.grid(alpha=0.25)

    for P in P_sorted:
        msjd_vals = [results["mpcn"][P][rho]["metrics"]["msjd_mean"] for rho in rho_list]
        ax_msjd.plot(rho_list, msjd_vals, marker="o", markersize=3, color=color_by_P[P], label=f"mpCN (P={P})")
        if show_independent and results.get("pcn_independent"):
            indep_entries = results["pcn_independent"].get(P)
            if indep_entries:
                msjd_mean_vals = [
                    indep_entries[rho]["metrics"].get("msjd_mean_mean", np.nan)
                    for rho in rho_list
                ]
                msjd_max_vals = [
                    indep_entries[rho]["metrics"].get("msjd_mean_max", np.nan)
                    for rho in rho_list
                ]
                ax_msjd.plot(
                    rho_list,
                    msjd_mean_vals,
                    marker="s",
                    markersize=3,
                    linestyle=":",
                    color=color_by_P[P],
                    label=f"{independent_label_prefix} mean (P={P})",
                )
                ax_msjd.plot(
                    rho_list,
                    msjd_max_vals,
                    marker="s",
                    markersize=3,
                    linestyle="--",
                    color=color_by_P[P],
                    label=f"{independent_label_prefix} max (P={P})",
                )
    if show_pcn and pcn_msjd is not None:
        ax_msjd.plot(rho_list, pcn_msjd, color="black", marker="s", markersize=3, linestyle="--", label="pCN")
    if show_mess and run_mess:
        for P in P_sorted:
            mess_uniform = results["mess_uniform"][P]["metrics"]["msjd_mean"]
            mess_euclid = results["mess_euclid_sq"][P]["metrics"]["msjd_mean"]
            ax_msjd.plot(rho_list, [mess_uniform] * len(rho_list), linestyle="--", color=color_by_P[P], alpha=0.6)
            ax_msjd.plot(rho_list, [mess_euclid] * len(rho_list), linestyle=":", color=color_by_P[P], alpha=0.6)
    ax_msjd.set_xlabel(r"$\rho$")
    ax_msjd.set_ylabel("MSJD")
    ax_msjd.grid(alpha=0.25)

    if share_y_metrics:
        _shared_ylim([ax_ess, ax_msjd])

    _unique_legend(fig, axes, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(fr"{title_prefix}: ESS/MSJD vs $\rho$")
    fig.tight_layout()
    format_kwargs = {"seed_base": seed_base}
    if file_name_kwargs:
        format_kwargs.update(file_name_kwargs)
    fig.savefig(reports_dir / file_name_fmt.format(**format_kwargs), bbox_inches="tight")
    plt.show()


def _get_param_metric(results_dict, P, rho, metric_key, param_index):
    entry = results_dict.get(P, {}).get(float(rho))
    if entry is None:
        return np.nan
    values = entry["metrics"].get(metric_key)
    if values is None or len(values) <= param_index:
        return np.nan
    return values[param_index]


def plot_ess_msjd_per_param_vs_rho(
    results,
    rho_list,
    P_list,
    reports_dir,
    seed_base,
    file_name_fmt,
    run_pcn,
    burn_in,
    file_name_kwargs=None,
    show_pcn=True,
    title_prefix="Multiwell",
    pcn_scale=None,
    normalize_ess=False,
    share_y_metrics=False,
    show_independent=False,
    independent_label_prefix="pCN indep",
):
    """Plot ESS/MSJD per-parameter vs rho."""
    apply_pub_style()

    P_sorted = sorted(P_list)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(P_sorted)))
    color_by_P = {P: colors[i] for i, P in enumerate(P_sorted)}

    reports_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True)
    for P in P_sorted:
        ess_x1 = [_get_param_metric(results["mpcn"], P, rho, "ess_per_param", 0) for rho in rho_list]
        ess_x2 = [_get_param_metric(results["mpcn"], P, rho, "ess_per_param", 1) for rho in rho_list]
        msjd_x1 = [_get_param_metric(results["mpcn"], P, rho, "msjd_per_param", 0) for rho in rho_list]
        msjd_x2 = [_get_param_metric(results["mpcn"], P, rho, "msjd_per_param", 1) for rho in rho_list]
        if normalize_ess:
            counts = [
                _post_sample_count(results["mpcn"][P][rho], burn_in) for rho in rho_list
            ]
            ess_x1 = [val / count if count else val for val, count in zip(ess_x1, counts)]
            ess_x2 = [val / count if count else val for val, count in zip(ess_x2, counts)]
        axes[0, 0].plot(rho_list, ess_x1, marker="o", markersize=3, color=color_by_P[P], label=f"mpCN (P={P})")
        axes[1, 0].plot(rho_list, ess_x2, marker="o", markersize=3, color=color_by_P[P])
        axes[0, 1].plot(rho_list, msjd_x1, marker="o", markersize=3, color=color_by_P[P])
        axes[1, 1].plot(rho_list, msjd_x2, marker="o", markersize=3, color=color_by_P[P])
        if show_independent and results.get("pcn_independent"):
            indep_entries = results["pcn_independent"].get(P)
            if indep_entries:
                key = "ess_per_param_sum_norm" if normalize_ess else "ess_per_param_sum"
                ess_x1_i = [
                    indep_entries[rho]["metrics"].get(key, [np.nan, np.nan])[0]
                    for rho in rho_list
                ]
                ess_x2_i = [
                    indep_entries[rho]["metrics"].get(key, [np.nan, np.nan])[1]
                    for rho in rho_list
                ]
                msjd_x1_mean = [
                    indep_entries[rho]["metrics"].get("msjd_per_param_mean", [np.nan, np.nan])[0]
                    for rho in rho_list
                ]
                msjd_x2_mean = [
                    indep_entries[rho]["metrics"].get("msjd_per_param_mean", [np.nan, np.nan])[1]
                    for rho in rho_list
                ]
                msjd_x1_max = [
                    indep_entries[rho]["metrics"].get("msjd_per_param_max", [np.nan, np.nan])[0]
                    for rho in rho_list
                ]
                msjd_x2_max = [
                    indep_entries[rho]["metrics"].get("msjd_per_param_max", [np.nan, np.nan])[1]
                    for rho in rho_list
                ]
                axes[0, 0].plot(
                    rho_list,
                    ess_x1_i,
                    marker="s",
                    markersize=3,
                    linestyle=":",
                    color=color_by_P[P],
                    label=f"{independent_label_prefix} (P={P})",
                )
                axes[1, 0].plot(
                    rho_list,
                    ess_x2_i,
                    marker="s",
                    markersize=3,
                    linestyle=":",
                    color=color_by_P[P],
                )
                axes[0, 1].plot(
                    rho_list,
                    msjd_x1_mean,
                    marker="s",
                    markersize=3,
                    linestyle=":",
                    color=color_by_P[P],
                    label=f"{independent_label_prefix} mean (P={P})",
                )
                axes[1, 1].plot(
                    rho_list,
                    msjd_x2_mean,
                    marker="s",
                    markersize=3,
                    linestyle=":",
                    color=color_by_P[P],
                )
                axes[0, 1].plot(
                    rho_list,
                    msjd_x1_max,
                    marker="s",
                    markersize=3,
                    linestyle="--",
                    color=color_by_P[P],
                    label=f"{independent_label_prefix} max (P={P})",
                )
                axes[1, 1].plot(
                    rho_list,
                    msjd_x2_max,
                    marker="s",
                    markersize=3,
                    linestyle="--",
                    color=color_by_P[P],
                )

    if show_pcn and run_pcn and results.get("pcn"):
        pcn_ess_x1 = [results["pcn"][rho]["metrics"]["ess_per_param"][0] for rho in rho_list]
        pcn_ess_x2 = [results["pcn"][rho]["metrics"]["ess_per_param"][1] for rho in rho_list]
        pcn_msjd_x1 = [results["pcn"][rho]["metrics"]["msjd_per_param"][0] for rho in rho_list]
        pcn_msjd_x2 = [results["pcn"][rho]["metrics"]["msjd_per_param"][1] for rho in rho_list]
        if normalize_ess:
            counts = [
                _post_sample_count(results["pcn"][rho], burn_in) for rho in rho_list
            ]
            pcn_ess_x1 = [val / count if count else val for val, count in zip(pcn_ess_x1, counts)]
            pcn_ess_x2 = [val / count if count else val for val, count in zip(pcn_ess_x2, counts)]
        axes[0, 0].plot(rho_list, pcn_ess_x1, color="black", marker="s", markersize=3, linestyle="--", label="pCN")
        axes[1, 0].plot(rho_list, pcn_ess_x2, color="black", marker="s", markersize=3, linestyle="--")
        axes[0, 1].plot(rho_list, pcn_msjd_x1, color="black", marker="s", markersize=3, linestyle="--")
        axes[1, 1].plot(rho_list, pcn_msjd_x2, color="black", marker="s", markersize=3, linestyle="--")
        if pcn_scale:
            axes[0, 0].plot(
                rho_list,
                [x * pcn_scale for x in pcn_ess_x1],
                color="red",
                marker="s",
                markersize=3,
                linestyle="-",
                label=f"pCN x {pcn_scale}",
            )
            axes[1, 0].plot(
                rho_list,
                [x * pcn_scale for x in pcn_ess_x2],
                color="red",
                marker="s",
                markersize=3,
                linestyle="-",
                label=f"pCN x {pcn_scale}",
            )

    axes[0, 0].set_title(r"$x_1$")
    axes[0, 1].set_title(r"$x_1$")
    axes[1, 0].set_title(r"$x_2$")
    axes[1, 1].set_title(r"$x_2$")
    for ax in axes[1, :]:
        ax.set_xlabel(r"$\rho$")
    if normalize_ess:
        axes[0, 0].set_ylabel("ESS per sample (IACT$^{-1}$)")
        axes[1, 0].set_ylabel("ESS per sample (IACT$^{-1}$)")
    else:
        axes[0, 0].set_ylabel("ESS")
        axes[1, 0].set_ylabel("ESS")
    axes[0, 1].set_ylabel("MSJD")
    axes[1, 1].set_ylabel("MSJD")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)

    if share_y_metrics:
        _shared_ylim([axes[0, 0], axes[1, 0]])
        _shared_ylim([axes[0, 1], axes[1, 1]])

    _unique_legend(fig, axes, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(fr"{title_prefix}: ESS/MSJD per-parameter vs $\rho$")
    fig.tight_layout()
    format_kwargs = {"seed_base": seed_base}
    if file_name_kwargs:
        format_kwargs.update(file_name_kwargs)
    fig.savefig(reports_dir / file_name_fmt.format(**format_kwargs), bbox_inches="tight")
    plt.show()


def plot_rejection_vs_rho(
    results,
    rho_list,
    P_list,
    reports_dir,
    seed_base,
    file_name_fmt,
    run_pcn,
    file_name_kwargs=None,
    show_pcn=True,
    title_prefix="Multiwell",
):
    """Plot rejection rate vs rho curves for each P."""
    apply_pub_style()

    P_sorted = sorted(P_list)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(P_sorted)))
    color_by_P = {P: colors[i] for i, P in enumerate(P_sorted)}

    pcn_reject = None
    if run_pcn and results.get("pcn"):
        pcn_reject = [1.0 - results["pcn"][rho]["accept_rate"] for rho in rho_list]

    reports_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2), sharex=True)

    for P in P_sorted:
        reject_vals = [1.0 - results["mpcn"][P][rho]["accept_rate"] for rho in rho_list]
        ax.plot(rho_list, reject_vals, marker="o", color=color_by_P[P], label=f"mpCN (P={P})")
    if show_pcn and pcn_reject is not None:
        ax.plot(rho_list, pcn_reject, color="black", marker="s", linestyle="--", label="pCN")

    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Rejection rate")
    ax.grid(alpha=0.25)

    _unique_legend(fig, [ax], loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(fr"{title_prefix}: rejection rate vs $\rho$")
    fig.tight_layout()
    format_kwargs = {"seed_base": seed_base}
    if file_name_kwargs:
        format_kwargs.update(file_name_kwargs)
    fig.savefig(reports_dir / file_name_fmt.format(**format_kwargs), bbox_inches="tight")
    plt.show()
