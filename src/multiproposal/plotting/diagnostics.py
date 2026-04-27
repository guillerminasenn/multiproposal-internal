"""Diagnostics plotting utilities."""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm as normdist


def write_csv(filename, array, new_file=False):
	"""Write a 2D array to CSV, optionally creating a new file."""
	mode = "w" if new_file else "a"
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	with open(filename, mode, newline="") as handle:
		np.savetxt(handle, array, delimiter=",", fmt="%.18e")


def _get_component(samples, index):
	return [item[index] for item in samples]


def make_hist_grid_comps(
	R,
	dr,
	samples,
	comp_list,
	save_path,
	C=None,
	beta=0.95,
	hide_plot=True,
	label_map=None,
	font_size=14,
	tick_label_size=None,
	axis_label_size=None,
	x_label_size=None,
	y_label_size=None,
	title=None,
	figsize=(15, 15),
	true_values=None,
):
	"""Grid of 1D/2D histograms for selected components."""
	samples = np.asarray(samples)
	comp_samples = [_get_component(samples, j) for j in comp_list]
	resolved_tick_size = font_size if tick_label_size is None else tick_label_size
	if x_label_size is not None:
		resolved_x_label_size = x_label_size
	elif axis_label_size is not None:
		resolved_x_label_size = axis_label_size
	else:
		resolved_x_label_size = font_size
	if y_label_size is not None:
		resolved_y_label_size = y_label_size
	elif axis_label_size is not None:
		resolved_y_label_size = axis_label_size
	else:
		resolved_y_label_size = font_size

	numbins = int(2 * R / dr)
	x_bins = np.linspace(-R, R, numbins)
	y_bins = np.linspace(-R, R, numbins)

	def _format_label(comp):
		if label_map and comp in label_map:
			return label_map[comp]
		return f"$a_{{{comp}}}$"

	def _get_true_value(comp):
		if true_values is None:
			return None
		if isinstance(true_values, dict):
			return true_values.get(comp)
		return true_values[comp]

	num_params = len(comp_list)
	fig, axs = plt.subplots(num_params, num_params, figsize=figsize)
	for i in range(num_params):
		for j in range(num_params):
			if i == j:
				axs[i, j].hist(comp_samples[i], density=True, bins=x_bins)
				true_val = _get_true_value(comp_list[i])
				if true_val is not None:
					axs[i, j].axvline(x=true_val, color="black", linestyle="--", linewidth=1.2)
				if C is not None:
					z = normdist.ppf((1 + beta) / 2)
					bound = z * np.sqrt(C[comp_list[i], comp_list[i]])
					for x_pos in [-bound, bound]:
						axs[i, j].axvline(x=x_pos, ymin=0, ymax=0.08, color="red", linewidth=2.5)
			else:
				axs[i, j].hist2d(comp_samples[j], comp_samples[i], bins=[x_bins, y_bins])
				if C is not None:
					Sigma = np.array(
						[
							[C[comp_list[j], comp_list[j]], C[comp_list[j], comp_list[i]]],
							[C[comp_list[i], comp_list[j]], C[comp_list[i], comp_list[i]]],
						]
					)
					chi2_val = -2 * np.log(1 - beta)
					eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
					order = eigenvalues.argsort()[::-1]
					eigenvalues = eigenvalues[order]
					eigenvectors = eigenvectors[:, order]
					width = 2 * np.sqrt(eigenvalues[0] * chi2_val)
					height = 2 * np.sqrt(eigenvalues[1] * chi2_val)
					angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
					ellipse = Ellipse(
						xy=(0, 0),
						width=width,
						height=height,
						angle=angle,
						edgecolor="red",
						facecolor="none",
						linewidth=2,
						linestyle="-",
						zorder=10,
					)
					axs[i, j].add_patch(ellipse)
		for ax in axs.flat:
			ax.label_outer()
			ax.tick_params(labelsize=resolved_tick_size)

	for i, comp in enumerate(comp_list):
		x_label = _format_label(comp)
		axs[-1, i].set_xlabel(x_label, fontsize=resolved_x_label_size)
		axs[i, 0].set_ylabel(x_label, fontsize=resolved_y_label_size)

	if title is None:
		title = "Posterior Marginals and Pairwise Densities"
	fig.suptitle(title, fontsize=font_size + 2)
	fig.tight_layout(rect=(0, 0, 1, 0.97))

	if save_path is not None:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path)
	if hide_plot:
		plt.close(fig)
	return fig


def plot_timeseries(samples, pot_samples, components, filename, mcmc_type, burn_in=0):
	"""Plot trace plots for selected components and the potential."""
	samples = np.asarray(samples)
	pot_samples = np.asarray(pot_samples).reshape(-1)

	n_iter, _ = samples.shape
	iters = np.arange(burn_in, n_iter)
	subplot_dim = len(components) + 1

	fig, axes = plt.subplots(subplot_dim, 1, figsize=(8, 2.0 * subplot_dim), sharex=True)
	axes[0].plot(iters, pot_samples[burn_in:])
	axes[0].set_ylabel("$\\phi(x)$")
	axes[0].grid(alpha=0.3)

	for d, comp in enumerate(components):
		axes[d + 1].plot(iters, samples[burn_in:, comp])
		axes[d + 1].set_ylabel(f"$x_{comp + 1}$")
		axes[d + 1].grid(alpha=0.3)

	axes[-1].set_xlabel("Iteration")
	fig.suptitle(f"Trace Plots for {mcmc_type}", y=0.99)
	fig.tight_layout()

	os.makedirs(os.path.dirname(filename), exist_ok=True)
	fig.savefig(filename, dpi=300, bbox_inches="tight")
	plt.close(fig)
